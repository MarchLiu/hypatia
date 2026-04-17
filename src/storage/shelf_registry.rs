use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::error::Result;

/// Persistent registry of named shelf paths, stored as `~/.hypatia/shelves.json`.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ShelfRegistry {
    pub shelves: HashMap<String, PathBuf>,
}

impl ShelfRegistry {
    /// Load registry from a JSON file. Returns an empty registry if the file does not exist.
    pub fn load(path: &Path) -> Result<Self> {
        if !path.exists() {
            return Ok(Self::default());
        }
        let content = std::fs::read_to_string(path)?;
        let registry: Self = serde_json::from_str(&content)?;
        Ok(registry)
    }

    /// Save registry to a JSON file (pretty-printed).
    pub fn save(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Register a shelf name with its filesystem path. Overwrites if name already exists.
    pub fn register(&mut self, name: &str, path: &PathBuf) {
        self.shelves.insert(name.to_string(), path.clone());
    }

    /// Unregister a shelf by name.
    pub fn unregister(&mut self, name: &str) {
        self.shelves.remove(name);
    }

    /// List all registered shelves as (name, path) pairs.
    pub fn list(&self) -> Vec<(&str, &PathBuf)> {
        let mut entries: Vec<_> = self.shelves.iter().map(|(k, v)| (k.as_str(), v)).collect();
        entries.sort_by_key(|(name, _)| *name);
        entries
    }

    /// Look up a shelf path by name.
    pub fn get(&self, name: &str) -> Option<&PathBuf> {
        self.shelves.get(name)
    }

    /// Check whether a shelf name is registered.
    pub fn contains(&self, name: &str) -> bool {
        self.shelves.contains_key(name)
    }

    /// Return the canonical path for the registry file: `~/.hypatia/shelves.json`.
    pub fn registry_path() -> PathBuf {
        dirs_home().join(".hypatia").join("shelves.json")
    }
}

fn dirs_home() -> PathBuf {
    std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("."))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn load_nonexistent_returns_empty() {
        let registry = ShelfRegistry::load(Path::new("/tmp/no_such_file_shelves.json")).unwrap();
        assert!(registry.shelves.is_empty());
    }

    #[test]
    fn save_and_load_roundtrip() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("shelves.json");

        let mut registry = ShelfRegistry::default();
        registry.register("default", &PathBuf::from("/tmp/default"));
        registry.register("project-a", &PathBuf::from("/tmp/project-a"));
        registry.save(&path).unwrap();

        let loaded = ShelfRegistry::load(&path).unwrap();
        assert_eq!(loaded.shelves.len(), 2);
        assert_eq!(loaded.get("default"), Some(&PathBuf::from("/tmp/default")));
        assert_eq!(loaded.get("project-a"), Some(&PathBuf::from("/tmp/project-a")));
    }

    #[test]
    fn register_overwrites() {
        let mut registry = ShelfRegistry::default();
        registry.register("test", &PathBuf::from("/old"));
        registry.register("test", &PathBuf::from("/new"));
        assert_eq!(registry.get("test"), Some(&PathBuf::from("/new")));
    }

    #[test]
    fn unregister_removes_entry() {
        let mut registry = ShelfRegistry::default();
        registry.register("test", &PathBuf::from("/tmp/test"));
        registry.unregister("test");
        assert!(!registry.contains("test"));
    }

    #[test]
    fn list_sorted_by_name() {
        let mut registry = ShelfRegistry::default();
        registry.register("zeta", &PathBuf::from("/z"));
        registry.register("alpha", &PathBuf::from("/a"));
        registry.register("mid", &PathBuf::from("/m"));

        let list = registry.list();
        let names: Vec<&str> = list.iter().map(|(n, _)| *n).collect();
        assert_eq!(names, vec!["alpha", "mid", "zeta"]);
    }

    #[test]
    fn save_creates_parent_directory() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("nested").join("dir").join("shelves.json");

        let mut registry = ShelfRegistry::default();
        registry.register("test", &PathBuf::from("/tmp/test"));
        registry.save(&path).unwrap();

        assert!(path.exists());
        let loaded = ShelfRegistry::load(&path).unwrap();
        assert_eq!(loaded.shelves.len(), 1);
    }
}
