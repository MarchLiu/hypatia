use crate::error::Result;
use crate::model::{Content, Knowledge};
use crate::storage::OpenShelf;

pub struct KnowledgeService<'a> {
    shelf: &'a mut OpenShelf,
}

impl<'a> KnowledgeService<'a> {
    pub fn new(shelf: &'a mut OpenShelf) -> Self {
        Self { shelf }
    }

    pub fn create(&mut self, name: &str, content: Content) -> Result<Knowledge> {
        // Source row + FTS doc are written in one store transaction.
        let version = self.shelf.backend.insert_knowledge(name, &content)?;

        // Generate embedding and store the BLOB (best-effort: skip if model unavailable)
        self.shelf.embed_saved("knowledge", name, &content, version);

        // Read back to get the generated timestamp
        let knowledge = self.shelf.backend.get_knowledge(name)?.ok_or_else(|| {
            crate::error::HypatiaError::NotFound {
                kind: "knowledge".to_string(),
                key: name.to_string(),
            }
        })?;
        Ok(knowledge)
    }

    pub fn get(&self, name: &str) -> Result<Option<Knowledge>> {
        self.shelf.backend.get_knowledge(name)
    }

    pub fn update(&mut self, name: &str, content: Content) -> Result<Knowledge> {
        let version = self.shelf.backend.update_knowledge(name, &content)?;

        self.shelf.embed_saved("knowledge", name, &content, version);

        let knowledge = self.shelf.backend.get_knowledge(name)?.ok_or_else(|| {
            crate::error::HypatiaError::NotFound {
                kind: "knowledge".to_string(),
                key: name.to_string(),
            }
        })?;
        Ok(knowledge)
    }

    pub fn delete(&mut self, name: &str) -> Result<()> {
        self.shelf.backend.delete_knowledge(name)
    }
}
