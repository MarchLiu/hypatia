use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
pub struct ShelfId {
    pub name: String,
    pub path: PathBuf,
}

impl ShelfId {
    pub fn new(path: PathBuf) -> Self {
        let name = path
            .file_name()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "default".to_string());
        Self { name, path }
    }

    pub fn with_name(path: PathBuf, name: String) -> Self {
        Self { name, path }
    }
}

#[derive(Debug, Clone)]
pub struct ShelfConfig {
    pub id: ShelfId,
    pub duckdb_path: PathBuf,
    pub sqlite_path: PathBuf,
    pub model_path: PathBuf,
    pub tokenizer_path: PathBuf,
}

impl ShelfConfig {
    pub fn from_shelf_id(id: ShelfId) -> Self {
        let duckdb_path = id.path.join("data.duckdb");
        let sqlite_path = id.path.join("index.sqlite");
        let model_path = id.path.join("embedding_model.onnx");
        let tokenizer_path = id.path.join("tokenizer.json");
        Self {
            id,
            duckdb_path,
            sqlite_path,
            model_path,
            tokenizer_path,
        }
    }

    pub fn from_path(path: &Path, name: Option<&str>) -> Self {
        let id = match name {
            Some(n) => ShelfId::with_name(path.to_path_buf(), n.to_string()),
            None => ShelfId::new(path.to_path_buf()),
        };
        Self::from_shelf_id(id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shelf_id_name_from_path() {
        let id = ShelfId::new(PathBuf::from("/path/to/my-project"));
        assert_eq!(id.name, "my-project");
    }

    #[test]
    fn shelf_id_custom_name() {
        let id = ShelfId::with_name(PathBuf::from("/tmp/x"), "custom".to_string());
        assert_eq!(id.name, "custom");
    }

    #[test]
    fn shelf_config_paths() {
        let config = ShelfConfig::from_path(Path::new("/tmp/test-shelf"), None);
        assert_eq!(config.duckdb_path, PathBuf::from("/tmp/test-shelf/data.duckdb"));
        assert_eq!(config.sqlite_path, PathBuf::from("/tmp/test-shelf/index.sqlite"));
        assert_eq!(config.id.name, "test-shelf");
    }
}
