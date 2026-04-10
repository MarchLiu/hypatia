use crate::error::Result;
use crate::model::{Content, Knowledge};
use crate::storage::{FtsDoc, OpenShelf};

pub struct KnowledgeService<'a> {
    shelf: &'a mut OpenShelf,
}

impl<'a> KnowledgeService<'a> {
    pub fn new(shelf: &'a mut OpenShelf) -> Self {
        Self { shelf }
    }

    fn build_fts_doc(content: &Content, name: &str) -> FtsDoc {
        let fields = content.fts_fields(name);
        FtsDoc {
            content: content.to_json_string(),
            fts_key: fields.key,
            fts_data: fields.data,
            fts_tags: fields.tags,
            fts_synonyms: fields.synonyms,
        }
    }

    pub fn create(&mut self, name: &str, content: Content) -> Result<Knowledge> {
        // Insert into DuckDB
        self.shelf.duckdb.insert_knowledge(name, &content)?;

        // Insert into SQLite FTS
        let doc = Self::build_fts_doc(&content, name);
        self.shelf.sqlite.upsert_doc("knowledge", name, &doc)?;

        // Read back to get the generated timestamp
        let knowledge = self.shelf.duckdb.get_knowledge(name)?.ok_or_else(|| {
            crate::error::HypatiaError::NotFound {
                kind: "knowledge".to_string(),
                key: name.to_string(),
            }
        })?;
        Ok(knowledge)
    }

    pub fn get(&self, name: &str) -> Result<Option<Knowledge>> {
        self.shelf.duckdb.get_knowledge(name)
    }

    pub fn update(&mut self, name: &str, content: Content) -> Result<Knowledge> {
        self.shelf.duckdb.update_knowledge(name, &content)?;

        // Update FTS
        let doc = Self::build_fts_doc(&content, name);
        self.shelf.sqlite.upsert_doc("knowledge", name, &doc)?;

        let knowledge = self.shelf.duckdb.get_knowledge(name)?.ok_or_else(|| {
            crate::error::HypatiaError::NotFound {
                kind: "knowledge".to_string(),
                key: name.to_string(),
            }
        })?;
        Ok(knowledge)
    }

    pub fn delete(&mut self, name: &str) -> Result<()> {
        self.shelf.duckdb.delete_knowledge(name)?;
        self.shelf.sqlite.delete_doc("knowledge", name)?;
        Ok(())
    }
}
