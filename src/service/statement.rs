use chrono::NaiveDateTime;

use crate::error::Result;
use crate::model::{Content, Statement, StatementKey};
use crate::storage::{FtsDoc, OpenShelf};

pub struct StatementService<'a> {
    shelf: &'a mut OpenShelf,
}

impl<'a> StatementService<'a> {
    pub fn new(shelf: &'a mut OpenShelf) -> Self {
        Self { shelf }
    }

    fn build_fts_doc(content: &Content, csv_key: &str) -> FtsDoc {
        let fields = content.fts_fields(csv_key);
        FtsDoc {
            content: content.to_json_string(),
            fts_key: fields.key,
            fts_data: fields.data,
            fts_tags: fields.tags,
            fts_synonyms: fields.synonyms,
        }
    }

    pub fn create(
        &mut self,
        key: &StatementKey,
        content: Content,
        tr_start: Option<NaiveDateTime>,
        tr_end: Option<NaiveDateTime>,
    ) -> Result<Statement> {
        self.shelf.duckdb.insert_statement(key, &content, tr_start, tr_end)?;

        // Insert into FTS with CSV-formatted key
        let csv_key = key.to_csv_key();
        let doc = Self::build_fts_doc(&content, &csv_key);
        self.shelf.sqlite.upsert_doc("statement", &csv_key, &doc)?;

        // Generate embedding and store in DuckDB (best-effort)
        if let Some(vector) = self.shelf.embedder.maybe_embed(&content.embedding_text(&csv_key))? {
            self.shelf.duckdb.upsert_statement_embedding(&csv_key, &vector)?;
        }

        let statement = self.shelf.duckdb.get_statement(key)?.ok_or_else(|| {
            crate::error::HypatiaError::NotFound {
                kind: "statement".to_string(),
                key: csv_key,
            }
        })?;
        Ok(statement)
    }

    pub fn get(&self, key: &StatementKey) -> Result<Option<Statement>> {
        self.shelf.duckdb.get_statement(key)
    }

    pub fn update(
        &mut self,
        key: &StatementKey,
        content: Content,
        tr_start: Option<NaiveDateTime>,
        tr_end: Option<NaiveDateTime>,
    ) -> Result<Statement> {
        self.shelf.duckdb.update_statement(key, &content, tr_start, tr_end)?;

        // Update FTS and vector
        let csv_key = key.to_csv_key();
        let doc = Self::build_fts_doc(&content, &csv_key);
        self.shelf.sqlite.upsert_doc("statement", &csv_key, &doc)?;

        if let Some(vector) = self.shelf.embedder.maybe_embed(&content.embedding_text(&csv_key))? {
            self.shelf.duckdb.upsert_statement_embedding(&csv_key, &vector)?;
        }

        let statement = self.shelf.duckdb.get_statement(key)?.ok_or_else(|| {
            crate::error::HypatiaError::NotFound {
                kind: "statement".to_string(),
                key: csv_key,
            }
        })?;
        Ok(statement)
    }

    pub fn delete(&mut self, key: &StatementKey) -> Result<()> {
        let csv_key = key.to_csv_key();
        self.shelf.duckdb.delete_statement(key)?;
        self.shelf.sqlite.delete_doc("statement", &csv_key)?;
        Ok(())
    }
}
