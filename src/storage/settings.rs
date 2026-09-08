//! Strict shelf configuration: validate before creating any backend resources.
use crate::embedding::{EmbeddingConfig, config::EmbeddingToml};
use crate::error::{HypatiaError, Result};
use serde::Deserialize;
use std::path::Path;

#[derive(Debug, Clone)]
pub struct ShelfSettings {
    pub storage: StorageSettings,
    pub embedding: EmbeddingConfig,
}
#[derive(Debug, Clone, Deserialize, Default)]
#[serde(default, deny_unknown_fields)]
pub struct StorageSettings {
    pub backend: BackendKind,
    pub postgres: Option<PostgresSettings>,
    pub vector: VectorSettings,
}
#[derive(Debug, Clone, Copy, Deserialize, Default, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum BackendKind {
    #[default]
    Sqlite,
    Pgvector,
}
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PostgresSettings {
    pub url_env: Option<String>,
    pub url: Option<String>,
    pub schema: String,
    #[serde(default = "connect_timeout")]
    pub connect_timeout_seconds: u64,
    #[serde(default = "statement_timeout")]
    pub statement_timeout_ms: u64,
}
impl std::fmt::Debug for PostgresSettings {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PostgresSettings")
            .field("url_env", &self.url_env)
            .field("url", &self.url.as_ref().map(|_| "[REDACTED]"))
            .field("schema", &self.schema)
            .field("connect_timeout_seconds", &self.connect_timeout_seconds)
            .field("statement_timeout_ms", &self.statement_timeout_ms)
            .finish()
    }
}
impl PostgresSettings {
    pub fn validate_connection_source(&self) -> Result<()> {
        match (&self.url, &self.url_env) {
            (Some(url), None) if !url.trim().is_empty() => Ok(()),
            (None, Some(name))
                if !name.is_empty()
                    && name.bytes().all(|c| c.is_ascii_alphanumeric() || c == b'_') =>
            {
                Ok(())
            }
            (Some(_), Some(_)) => Err(HypatiaError::Config(
                "postgres.url and postgres.url_env are mutually exclusive; configure exactly one"
                    .into(),
            )),
            (Some(_), None) => Err(HypatiaError::Config(
                "postgres.url must be a nonempty connection string".into(),
            )),
            (None, Some(_)) => Err(HypatiaError::Config(
                "postgres.url_env must name an environment variable".into(),
            )),
            (None, None) => Err(HypatiaError::Config(
                "PostgreSQL requires exactly one of postgres.url or postgres.url_env".into(),
            )),
        }
    }
    /// Resolve only at connection time; never include a connection string in errors.
    pub fn connection_url(&self) -> Result<String> {
        self.validate_connection_source()?;
        if let Some(url) = &self.url {
            return Ok(url.clone());
        }
        let name = self.url_env.as_ref().expect("validated connection source");
        std::env::var(name)
            .ok()
            .filter(|s| !s.trim().is_empty())
            .ok_or_else(|| {
                HypatiaError::Config(
                    "PostgreSQL URL environment variable is missing, empty or invalid".into(),
                )
            })
    }
}
fn connect_timeout() -> u64 {
    5
}
fn statement_timeout() -> u64 {
    30000
}
#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct VectorSettings {
    pub index: String,
    pub metric: String,
}
impl Default for VectorSettings {
    fn default() -> Self {
        Self {
            index: "hnsw".into(),
            metric: "cosine".into(),
        }
    }
}
#[derive(Deserialize, Default)]
#[serde(default, deny_unknown_fields)]
struct SettingsToml {
    storage: StorageSettings,
    embedding: EmbeddingToml,
}
impl ShelfSettings {
    pub fn load(dir: &Path) -> Result<Self> {
        let path = dir.join("shelf.toml");
        let text = match std::fs::read_to_string(&path) {
            Ok(s) => s,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => String::new(),
            Err(e) => {
                return Err(HypatiaError::Config(format!(
                    "cannot read {}: {e}",
                    path.display()
                )));
            }
        };
        Self::parse(&text, dir)
    }
    pub fn parse(text: &str, dir: &Path) -> Result<Self> {
        // Do not echo TOML source, which might contain a mistakenly pasted secret.
        let parsed: SettingsToml = toml::from_str(text)
            .map_err(|e| HypatiaError::Config(format!("invalid shelf.toml: {}", e.message())))?;
        if !matches!(parsed.embedding.provider.as_str(), "local" | "remote") {
            return Err(HypatiaError::Config(
                "embedding.provider must be local or remote".into(),
            ));
        }
        let embedding = EmbeddingConfig::from_parsed(parsed.embedding, dir);
        let settings = Self {
            storage: parsed.storage,
            embedding,
        };
        settings.validate()?;
        Ok(settings)
    }
    pub fn validate(&self) -> Result<()> {
        let fail = |s: &str| Err(HypatiaError::Config(s.into()));
        if self.embedding.dimensions() == 0 {
            return fail("embedding.dimensions must be positive");
        }
        if self.storage.vector.metric != "cosine" {
            return fail("storage.vector.metric must be cosine");
        }
        if !matches!(self.storage.vector.index.as_str(), "hnsw" | "none") {
            return fail("storage.vector.index must be hnsw or none");
        }
        if self.storage.backend == BackendKind::Pgvector {
            if !self.embedding.model_identity_trusted {
                return fail(
                    "pgvector requires an explicit embedding.model (or readable local model and tokenizer files) to identify the vector space; model inference may remain unavailable",
                );
            }
            let pg = self.storage.postgres.as_ref().ok_or_else(|| {
                HypatiaError::Config(
                    "pgvector requires [storage.postgres] with schema and exactly one of url or url_env".into(),
                )
            })?;
            validate_schema(&pg.schema)?;
            pg.validate_connection_source()?;
            if pg.connect_timeout_seconds == 0
                || pg.statement_timeout_ms == 0
                || pg.statement_timeout_ms > i32::MAX as u64
            {
                return fail(
                    "PostgreSQL timeouts must be positive; statement_timeout_ms must fit int32",
                );
            }
            if self.embedding.dimensions() > 16000 {
                return fail("pgvector vector dimensions must not exceed 16000");
            }
            if self.storage.vector.index == "hnsw" && self.embedding.dimensions() > 2000 {
                return fail(
                    "vector HNSW supports at most 2000 dimensions; explicitly choose index = 'none' for exact search",
                );
            }
        } else if self.storage.postgres.is_some() {
            return fail("storage.postgres requires backend = 'pgvector'");
        }
        Ok(())
    }
}
pub fn validate_schema(schema: &str) -> Result<()> {
    if schema.is_empty()
        || schema.len() > 63
        || !schema
            .bytes()
            .next()
            .is_some_and(|b| b.is_ascii_alphabetic() || b == b'_')
        || !schema
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_')
        || schema.starts_with("pg_")
        || matches!(schema, "public" | "information_schema")
    {
        return Err(HypatiaError::Config("postgres.schema must be a dedicated 1–63 byte ASCII identifier, excluding public, information_schema and pg_*".into()));
    }
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn strict_config_and_defaults() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(
            ShelfSettings::load(dir.path()).unwrap().storage.backend,
            BackendKind::Sqlite
        );
        for text in [
            "[storage]\nbackend='typo'",
            "[storage]\nbacked='pgvector'",
            "[storage]\nbackend='pgvector'",
            "[storag]",
            "[storage",
            "[embedding]\nprovider='typo'",
        ] {
            assert!(ShelfSettings::parse(text, dir.path()).is_err(), "{text}");
        }
    }
    #[test]
    fn connection_sources_and_redaction() {
        let dir = tempfile::tempdir().unwrap();
        let parse = |connection: &str| {
            ShelfSettings::parse(
                &format!(
                    "[embedding]\nmodel='test-model'\n[storage]\nbackend='pgvector'\n[storage.postgres]\nschema='test_schema'\n{connection}"
                ),
                dir.path(),
            )
        };
        let url = "postgres://user:secret-test-password@localhost/db";
        let settings = parse(&format!("url='{url}'")).unwrap();
        let pg = settings.storage.postgres.as_ref().unwrap();
        assert_eq!(pg.connection_url().unwrap(), url);
        assert!(pg.url_env.is_none());
        let debug = format!("{settings:?}");
        assert!(debug.contains("[REDACTED]"));
        assert!(!debug.contains("secret-test-password"));
        assert!(!debug.contains(url));
        let env = parse("url_env='HYPATIA_TEST_MISSING_URL_9455722'").unwrap();
        assert!(env.storage.postgres.unwrap().connection_url().is_err());
        for source in [
            "".to_string(),
            "url=''".into(),
            "url='   '".into(),
            "url_env=''".into(),
            "url_env='invalid-name'".into(),
            format!("url='{url}'\nurl_env='HYPATIA_POSTGRES_URL'"),
            format!("url='{url}'\nurl_env=''"),
            "url=''\nurl_env='HYPATIA_POSTGRES_URL'".into(),
        ] {
            let error = parse(&source).unwrap_err().to_string();
            assert!(!error.contains(url));
            assert!(!error.contains("secret-test-password"));
        }
    }
    #[test]
    fn validates_schema_names() {
        for s in ["", "public", "pg_catalog", "x;drop schema y", "a.b", "a-b"] {
            assert!(validate_schema(s).is_err());
        }
        assert!(validate_schema("team_memory").is_ok());
    }
}
