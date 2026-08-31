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

use usearch::{new_index, Index, IndexOptions, MetricKind, ScalarKind};

use crate::error::{Result, StorageError};
pub struct VectorFileIndex {
    index: Index,
    dimensions: usize,
    path: std::path::PathBuf,
    dirty: bool,
}

impl VectorFileIndex {
    /// Build a fresh in-memory index and insert every item.
    pub fn build(
        path: &Path,
        dimensions: usize,
        items: &[(i64, Vec<f32>)],
    ) -> Result<Self> {
        let index = new_index(&options(dimensions)).map_err(|e| StorageError::Vector(e.to_string()))?;
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
        let index = new_index(&options(dimensions)).map_err(|e| StorageError::Vector(e.to_string()))?;
        index.load(path.to_str().unwrap_or("")).map_err(|e| StorageError::Vector(e.to_string()))?;
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

    /// Insert or replace a vector. usearch reuses the slot of a deleted key.
    pub fn upsert(&mut self, doc_id: i64, vector: &[f32]) -> Result<()> {
        if vector.len() != self.dimensions {
            return Ok(()); // dimension mismatch: skip
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
        let results = self.index.search(query, k).map_err(|e| StorageError::Vector(e.to_string()))?;
        Ok(results
            .keys
            .iter()
            .zip(results.distances.iter())
            .map(|(key, dist)| (*key as i64, *dist as f64))
            .collect())
    }

    /// Atomic snapshot: save to `<path>.tmp` then rename into place.
    pub fn save(&mut self) -> Result<()> {
        if !self.dirty {
            return Ok(());
        }
        let tmp = suffix_path(&self.path, ".tmp");
        self.index.save(tmp.to_str().unwrap_or("")).map_err(|e| StorageError::Vector(e.to_string()))?;
        std::fs::rename(&tmp, &self.path)?;
        self.dirty = false;
        Ok(())
    }

    pub fn is_dirty(&self) -> bool {
        self.dirty
    }
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
    fn build_search_save_load_roundtrip() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("knowledge.usearch");

        let items = vec![
            (1i64, vec![1.0f32, 0.0, 0.0]),
            (2i64, vec![0.0, 1.0, 0.0]),
        ];
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
}
