use chrono::NaiveDateTime;

use crate::error::Result;
use crate::model::{Content, Statement, StatementKey};
use crate::storage::OpenShelf;

pub struct StatementService<'a> {
    shelf: &'a mut OpenShelf,
}

impl<'a> StatementService<'a> {
    pub fn new(shelf: &'a mut OpenShelf) -> Self {
        Self { shelf }
    }

    pub fn create(
        &mut self,
        key: &StatementKey,
        content: Content,
        tr_start: Option<NaiveDateTime>,
        tr_end: Option<NaiveDateTime>,
    ) -> Result<Statement> {
        let csv_key = key.to_csv_key();
        // Source row + FTS doc are written in one store transaction.
        self.shelf.store.insert_statement(key, &content, tr_start, tr_end)?;

        // Generate embedding and store the BLOB (best-effort)
        if let Some(vector) = self.shelf.embedder.maybe_embed(&content.embedding_text(&csv_key))? {
            self.shelf.store.upsert_statement_embedding(&csv_key, &vector)?;
        }

        let statement = self.shelf.store.get_statement(key)?.ok_or_else(|| {
            crate::error::HypatiaError::NotFound {
                kind: "statement".to_string(),
                key: csv_key,
            }
        })?;
        Ok(statement)
    }

    pub fn get(&self, key: &StatementKey) -> Result<Option<Statement>> {
        self.shelf.store.get_statement(key)
    }

    pub fn update(
        &mut self,
        key: &StatementKey,
        content: Content,
        tr_start: Option<NaiveDateTime>,
        tr_end: Option<NaiveDateTime>,
    ) -> Result<Statement> {
        let csv_key = key.to_csv_key();
        self.shelf.store.update_statement(key, &content, tr_start, tr_end)?;

        if let Some(vector) = self.shelf.embedder.maybe_embed(&content.embedding_text(&csv_key))? {
            self.shelf.store.upsert_statement_embedding(&csv_key, &vector)?;
        }

        let statement = self.shelf.store.get_statement(key)?.ok_or_else(|| {
            crate::error::HypatiaError::NotFound {
                kind: "statement".to_string(),
                key: csv_key,
            }
        })?;
        Ok(statement)
    }

    pub fn delete(&mut self, key: &StatementKey) -> Result<()> {
        self.shelf.store.delete_statement(key)
    }
}
