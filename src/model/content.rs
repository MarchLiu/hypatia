use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// Synonyms for FTS indexing.
///
/// - Knowledge entries use `Flat(Vec<String>)`: a simple list of synonym strings.
/// - Statement entries use `Positional(HashMap)`: keys are "subject", "predicate", "object",
///   values are synonym lists for each triple position.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum Synonyms {
    Flat(Vec<String>),
    Positional(HashMap<String, Vec<String>>),
}

/// Decomposed fields for multi-column FTS indexing.
pub struct FtsFields {
    pub key: String,
    pub data: String,
    pub tags: String,
    pub synonyms: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Content {
    pub format: Format,
    pub data: String,
    pub tags: Vec<String>,
    #[serde(default)]
    pub synonyms: Option<Synonyms>,
    /// References to archive files stored in the shelf's `archives/` directory.
    /// Uses the `archive://<relative_path>` convention.
    #[serde(default)]
    pub figures: Option<Vec<String>>,
    /// Scope labels for filtering knowledge by project or global context.
    /// Empty string `""` represents global scope.
    #[serde(default)]
    pub scopes: Option<Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum Format {
    Markdown,
    Json,
    Plain,
}

impl Default for Content {
    fn default() -> Self {
        Self {
            format: Format::Markdown,
            data: String::new(),
            tags: Vec::new(),
            synonyms: None,
            figures: None,
            scopes: None,
        }
    }
}

impl Content {
    pub fn new(data: impl Into<String>) -> Self {
        Self {
            data: data.into(),
            ..Self::default()
        }
    }

    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.tags = tags;
        self
    }

    pub fn with_format(mut self, format: Format) -> Self {
        self.format = format;
        self
    }

    pub fn with_synonyms(mut self, synonyms: Option<Synonyms>) -> Self {
        self.synonyms = synonyms;
        self
    }

    pub fn with_figures(mut self, figures: Vec<String>) -> Self {
        self.figures = if figures.is_empty() {
            None
        } else {
            Some(figures)
        };
        self
    }

    pub fn with_scopes(mut self, scopes: Vec<String>) -> Self {
        self.scopes = if scopes.is_empty() {
            None
        } else {
            Some(scopes)
        };
        self
    }

    /// Resolve an `archive://` reference to a filesystem path relative to a shelf root.
    /// Returns the relative path without the `archive://` prefix, or None if not an archive ref.
    pub fn resolve_figure_path(figure: &str) -> Option<&str> {
        figure.strip_prefix("archive://")
    }

    pub fn to_json_string(&self) -> String {
        serde_json::to_string(self).expect("Content serialization should not fail")
    }

    pub fn from_json_str(s: &str) -> std::result::Result<Self, serde_json::Error> {
        serde_json::from_str(s)
    }

    /// Build decomposed fields for multi-column FTS indexing.
    pub fn fts_fields(&self, name: &str) -> FtsFields {
        use crate::text::segment_for_fts;

        let synonyms_text = match &self.synonyms {
            Some(Synonyms::Flat(list)) => list.join(" "),
            Some(Synonyms::Positional(map)) => map
                .values()
                .flat_map(|v| v.iter())
                .cloned()
                .collect::<Vec<_>>()
                .join(" "),
            None => String::new(),
        };
        FtsFields {
            key: segment_for_fts(name),
            data: segment_for_fts(&self.data),
            tags: segment_for_fts(&self.tags.join(" ")),
            synonyms: segment_for_fts(&synonyms_text),
        }
    }

    /// Build the text used for embedding vector generation.
    /// Combines tags, name, and data into a single string for semantic search.
    pub fn embedding_text(&self, name: &str) -> String {
        let mut parts = Vec::new();
        if !self.tags.is_empty() {
            parts.push(self.tags.join(", "));
        }
        parts.push(name.to_string());
        parts.push(self.data.clone());
        parts.join("\n")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_content() {
        let c = Content::default();
        assert_eq!(c.format, Format::Markdown);
        assert!(c.data.is_empty());
        assert!(c.tags.is_empty());
        assert!(c.synonyms.is_none());
    }

    #[test]
    fn serialization_roundtrip() {
        let c = Content::new("hello world")
            .with_tags(vec!["test".to_string()])
            .with_format(Format::Json);
        let json = c.to_json_string();
        let c2 = Content::from_json_str(&json).unwrap();
        assert_eq!(c, c2);
    }

    #[test]
    fn backward_compat_no_synonyms() {
        let json = r#"{"format":"markdown","data":"hello","tags":[]}"#;
        let c = Content::from_json_str(json).unwrap();
        assert_eq!(c.data, "hello");
        assert!(c.synonyms.is_none());
        assert!(c.figures.is_none());
        assert!(c.scopes.is_none());
    }

    #[test]
    fn figures_roundtrip() {
        let c = Content::new("desc")
            .with_figures(vec!["archive://euclid/fig1.png".to_string()]);
        let json = c.to_json_string();
        let c2 = Content::from_json_str(&json).unwrap();
        assert_eq!(c, c2);
        assert_eq!(c2.figures.unwrap().len(), 1);
    }

    #[test]
    fn resolve_figure_path_test() {
        assert_eq!(
            Content::resolve_figure_path("archive://euclid/fig1.png"),
            Some("euclid/fig1.png")
        );
        assert_eq!(Content::resolve_figure_path("/some/abs/path"), None);
    }

    #[test]
    fn scopes_roundtrip() {
        let c = Content::new("data").with_scopes(vec!["project-a".to_string(), "".to_string()]);
        let json = c.to_json_string();
        let c2 = Content::from_json_str(&json).unwrap();
        assert_eq!(c, c2);
        assert_eq!(c2.scopes.unwrap(), vec!["project-a", ""]);
    }

    #[test]
    fn scopes_empty_becomes_none() {
        let c = Content::new("data").with_scopes(vec![]);
        assert!(c.scopes.is_none());
    }

    #[test]
    fn synonyms_flat_roundtrip() {
        let c = Content::new("data").with_synonyms(Some(Synonyms::Flat(vec![
            "DB".to_string(),
            "database".to_string(),
        ])));
        let json = c.to_json_string();
        let c2 = Content::from_json_str(&json).unwrap();
        assert_eq!(c, c2);
    }

    #[test]
    fn synonyms_positional_roundtrip() {
        let mut map = HashMap::new();
        map.insert("subject".to_string(), vec!["Alice A.".to_string()]);
        map.insert("predicate".to_string(), vec!["leads".to_string(), "manages".to_string()]);
        let c = Content::new("data").with_synonyms(Some(Synonyms::Positional(map)));
        let json = c.to_json_string();
        let c2 = Content::from_json_str(&json).unwrap();
        assert_eq!(c, c2);
    }

    #[test]
    fn fts_fields_includes_all() {
        let c = Content::new("some data")
            .with_tags(vec!["tag1".to_string(), "tag2".to_string()])
            .with_synonyms(Some(Synonyms::Flat(vec!["syn1".to_string()])));
        let f = c.fts_fields("my-knowledge");
        assert_eq!(f.key, "my-knowledge");
        assert_eq!(f.data, "some data");
        assert_eq!(f.tags, "tag1 tag2");
        assert_eq!(f.synonyms, "syn1");
    }

    #[test]
    fn fts_fields_positional_synonyms() {
        let mut map = HashMap::new();
        map.insert("subject".to_string(), vec!["Bob".to_string(), "Robert".to_string()]);
        map.insert("object".to_string(), vec!["DB".to_string()]);
        let c = Content::new("data").with_synonyms(Some(Synonyms::Positional(map)));
        let f = c.fts_fields("triple_key");
        assert!(f.synonyms.contains("Bob"));
        assert!(f.synonyms.contains("Robert"));
        assert!(f.synonyms.contains("DB"));
    }
}
