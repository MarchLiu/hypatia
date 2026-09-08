//! usearch-backed ANN index for one catalog (`vectors/<catalog>.usearch`).
//!
//! Semantics per docs/sqlite-refactor-plan.md §4:
//! - The index is a **rebuildable cache**, not a source of truth — the
//!   embedding BLOBs in SQLite are authoritative.
//! - `doc_id` (docs.id) is the usearch key, linking vectors to metadata.
//! - Concurrency: single writer, many readers. Saves are atomic
//!   (`.tmp` + rename); a lost race just means the next open reconciles
//!   via `synced_with_store` / rebuild.

use std::path::Path;

use usearch::{Index, IndexOptions, MetricKind, ScalarKind, new_index};

use crate::error::{Result, StorageError};
pub struct VectorFileIndex {
    index: Index,
    dimensions: usize,
    path: std::path::PathBuf,
    dirty: bool,
}

impl VectorFileIndex {
    /// Build a fresh in-memory index and insert every item.
    pub fn build(path: &Path, dimensions: usize, items: &[(i64, Vec<f32>)]) -> Result<Self> {
        let index =
            new_index(&options(dimensions)).map_err(|e| StorageError::Vector(e.to_string()))?;
        index
            .reserve(items.len().max(64))
            .map_err(|e| StorageError::Vector(e.to_string()))?;
        for (key, vector) in items {
            if vector.len() != dimensions {
                continue; // dimension mismatch: skip (model changed)
            }
            index
                .add(*key as u64, vector)
                .map_err(|e| StorageError::Vector(e.to_string()))?;
        }
        Ok(Self {
            index,
            dimensions,
            path: path.to_path_buf(),
            dirty: true,
        })
    }

    /// Load an existing index file. Errors if absent/corrupt — caller falls
    /// back to `build`.
    pub fn load(path: &Path, dimensions: usize) -> Result<Self> {
        let index =
            new_index(&options(dimensions)).map_err(|e| StorageError::Vector(e.to_string()))?;
        index
            .load(path.to_str().unwrap_or(""))
            .map_err(|e| StorageError::Vector(e.to_string()))?;
        Ok(Self {
            index,
            dimensions,
            path: path.to_path_buf(),
            dirty: false,
        })
    }

    pub fn exists(path: &Path) -> bool {
        path.exists()
    }

    pub fn size(&self) -> usize {
        self.index.size()
    }

    pub fn capacity(&self) -> usize {
        self.index.capacity()
    }

    /// Insert or replace a vector. usearch reuses the slot of a deleted key.
    pub fn upsert(&mut self, doc_id: i64, vector: &[f32]) -> Result<()> {
        if vector.len() != self.dimensions {
            return Ok(()); // dimension mismatch: skip
        }
        // usearch does not auto-expand; reserve more capacity when full.
        if self.index.size() >= self.index.capacity() {
            let new_capacity = (self.index.capacity() * 2).max(64);
            self.index
                .reserve(new_capacity)
                .map_err(|e| StorageError::Vector(e.to_string()))?;
        }
        // remove first so re-adding the same key is legal
        let _ = self.index.remove(doc_id as u64);
        self.index
            .add(doc_id as u64, vector)
            .map_err(|e| StorageError::Vector(e.to_string()))?;
        self.dirty = true;
        Ok(())
    }

    pub fn remove(&mut self, doc_id: i64) -> Result<()> {
        let _ = self.index.remove(doc_id as u64);
        self.dirty = true;
        Ok(())
    }

    /// ANN search returning `(doc_id, distance)` pairs, best first
    /// (cosine distance, smaller = closer).
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<(i64, f64)>> {
        if self.index.size() == 0 {
            return Ok(Vec::new());
        }
        let k = k.min(self.index.size());
        let results = self
            .index
            .search(query, k)
            .map_err(|e| StorageError::Vector(e.to_string()))?;
        Ok(results
            .keys
            .iter()
            .zip(results.distances.iter())
            .map(|(key, dist)| (*key as i64, *dist as f64))
            .collect())
    }

    /// Atomic snapshot with a uniquely reserved temporary file per writer.
    pub fn save(&mut self) -> Result<()> {
        if !self.dirty {
            return Ok(());
        }
        let (tmp, file) = reserve_temporary(&self.path)?;
        drop(file);
        let result = (|| -> Result<()> {
            self.index
                .save(tmp.to_str().unwrap_or(""))
                .map_err(|e| StorageError::Vector(e.to_string()))?;
            std::fs::rename(&tmp, &self.path)?;
            Ok(())
        })();
        if result.is_err() {
            let _ = std::fs::remove_file(&tmp);
        }
        result?;
        self.dirty = false;
        Ok(())
    }

    pub fn is_dirty(&self) -> bool {
        self.dirty
    }
}

/// Reserve a distinct sibling temporary file even for simultaneous writers.
fn reserve_temporary(path: &Path) -> Result<(std::path::PathBuf, std::fs::File)> {
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    loop {
        let seq = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let tmp = suffix_path(path, &format!(".{}.{}.{seq}.tmp", std::process::id(), time));
        match std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&tmp)
        {
            Ok(file) => return Ok((tmp, file)),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(e.into()),
        }
    }
}

pub(crate) fn atomic_write(path: &Path, bytes: &[u8]) -> Result<()> {
    use std::io::Write;
    let (tmp, mut file) = reserve_temporary(path)?;
    let result = (|| -> Result<()> {
        file.write_all(bytes)?;
        file.sync_all()?;
        drop(file);
        std::fs::rename(&tmp, path)?;
        Ok(())
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
    result
}

fn options(dimensions: usize) -> IndexOptions {
    IndexOptions {
        dimensions,
        metric: MetricKind::Cos,
        quantization: ScalarKind::F32,
        connectivity: 0,
        expansion_add: 0,
        expansion_search: 0,
        multi: false,
    }
}

fn suffix_path(path: &Path, suffix: &str) -> std::path::PathBuf {
    let mut s = path.as_os_str().to_os_string();
    s.push(suffix);
    std::path::PathBuf::from(s)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn concurrent_atomic_saves_do_not_share_temporary_files() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("shared.usearch");
        let barrier = std::sync::Arc::new(std::sync::Barrier::new(4));
        let handles: Vec<_> = (0..4)
            .map(|i| {
                let path = path.clone();
                let barrier = barrier.clone();
                std::thread::spawn(move || {
                    let mut index = VectorFileIndex::build(&path, 2, &[(i, vec![1., 0.])]).unwrap();
                    barrier.wait();
                    index.save().unwrap();
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
        assert_eq!(VectorFileIndex::load(&path, 2).unwrap().size(), 1);
        assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 1);
    }

    #[test]
    fn build_search_save_load_roundtrip() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("knowledge.usearch");

        let items = vec![(1i64, vec![1.0f32, 0.0, 0.0]), (2i64, vec![0.0, 1.0, 0.0])];
        let mut index = VectorFileIndex::build(&path, 3, &items).unwrap();
        assert_eq!(index.size(), 2);

        let results = index.search(&[1.0, 0.0, 0.0], 1).unwrap();
        assert_eq!(results[0].0, 1);

        index.save().unwrap();
        assert!(!index.is_dirty());

        let reloaded = VectorFileIndex::load(&path, 3).unwrap();
        assert_eq!(reloaded.size(), 2);
        let results = reloaded.search(&[0.0, 1.0, 0.0], 1).unwrap();
        assert_eq!(results[0].0, 2);
    }

    #[test]
    fn upsert_and_remove() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("t.usearch");
        let mut index = VectorFileIndex::build(&path, 3, &[(1i64, vec![1.0, 0.0, 0.0])]).unwrap();

        index.remove(1).unwrap();
        index.upsert(1, &[0.0, 0.0, 1.0]).unwrap();
        assert_eq!(index.size(), 1);

        let results = index.search(&[0.0, 0.0, 1.0], 5).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, 1);
    }

    #[test]
    fn upsert_fails_when_capacity_exhausted() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("t.usearch");
        // build reserves max(items.len(), 64) = 64
        let mut index = VectorFileIndex::build(
            &path,
            3,
            &[(1i64, vec![1.0, 0.0, 0.0]), (2i64, vec![0.0, 1.0, 0.0])],
        )
        .unwrap();

        // fill up to capacity with fresh keys
        for i in 3..=64 {
            index.upsert(i, &[0.0, 0.0, 1.0]).unwrap();
        }
        assert_eq!(index.size(), 64);

        // after the fix, the next insertion auto-expands capacity and succeeds
        index.upsert(65, &[0.0, 0.0, 1.0]).unwrap();
        assert_eq!(index.size(), 65);
        assert!(index.capacity() >= 65);
    }
}
