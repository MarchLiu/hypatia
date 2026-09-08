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
        let version = self
            .shelf
            .backend
            .insert_statement(key, &content, tr_start, tr_end)?;

        // Generate embedding and store the BLOB (best-effort)
        self.shelf
            .embed_saved("statement", &csv_key, &content, version);

        let statement = self.shelf.backend.get_statement(key)?.ok_or_else(|| {
            crate::error::HypatiaError::NotFound {
                kind: "statement".to_string(),
                key: csv_key,
            }
        })?;
        Ok(statement)
    }

    pub fn get(&self, key: &StatementKey) -> Result<Option<Statement>> {
        self.shelf.backend.get_statement(key)
    }

    pub fn update(
        &mut self,
        key: &StatementKey,
        content: Content,
        tr_start: Option<NaiveDateTime>,
        tr_end: Option<NaiveDateTime>,
    ) -> Result<Statement> {
        let csv_key = key.to_csv_key();
        let version = self
            .shelf
            .backend
            .update_statement(key, &content, tr_start, tr_end)?;

        self.shelf
            .embed_saved("statement", &csv_key, &content, version);

        let statement = self.shelf.backend.get_statement(key)?.ok_or_else(|| {
            crate::error::HypatiaError::NotFound {
                kind: "statement".to_string(),
                key: csv_key,
            }
        })?;
        Ok(statement)
    }

    pub fn delete(&mut self, key: &StatementKey) -> Result<()> {
        self.shelf.backend.delete_statement(key)
    }
}
