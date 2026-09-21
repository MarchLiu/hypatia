use crate::error::Result;
use crate::model::{Content, Knowledge, Synonyms};
use crate::storage::OpenShelf;

/// Fields to change in an existing knowledge entry. `None` keeps the stored value,
/// `Some` replaces it, and an empty value clears the field. The format is always kept.
/// Values are stored as given: callers other than the CLI, such as the MCP layer, should
/// parse their inputs the way the CLI does.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct KnowledgePatch {
    pub data: Option<String>,
    pub tags: Option<Vec<String>>,
    pub synonyms: Option<Option<Synonyms>>,
    pub figures: Option<Vec<String>>,
    pub scopes: Option<Vec<String>>,
    /// `Some(false)` takes the entry out of the vector index, `Some(true)` puts it back.
    pub embed: Option<bool>,
}

impl KnowledgePatch {
    /// True when the patch names no field at all.
    pub fn is_empty(&self) -> bool {
        *self == Self::default()
    }

    /// The content after applying this patch to `current`.
    pub fn apply(&self, current: &Content) -> Content {
        let mut next = current.clone();
        if let Some(data) = &self.data {
            next.data = data.clone();
        }
        if let Some(tags) = &self.tags {
            next.tags = tags.clone();
        }
        if let Some(synonyms) = &self.synonyms {
            next.synonyms = synonyms.clone();
        }
        // The builders normalise an empty list to "absent", as `knowledge-create` does.
        if let Some(figures) = &self.figures {
            next = next.with_figures(figures.clone());
        }
        if let Some(scopes) = &self.scopes {
            next = next.with_scopes(scopes.clone());
        }
        if let Some(embed) = self.embed {
            next = next.with_embed(Some(embed));
        }
        next
    }
}

/// Outcome of [`KnowledgeService::patch`].
#[derive(Debug, Clone)]
pub struct UpdatedKnowledge {
    pub knowledge: Knowledge,
    /// `false` when the patch left the content as it was, so nothing was written.
    pub changed: bool,
}

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

    /// Change only the fields a patch names. A result equal to the stored content writes
    /// nothing, so the version and the stored vector stay as they are. Reading and writing
    /// are two steps, and the write replaces the whole entry: any change another writer makes
    /// in between is lost, even to a field this patch does not name. The window is two
    /// statements with no model call inside, so no version check is made.
    pub fn patch(&mut self, name: &str, patch: &KnowledgePatch) -> Result<UpdatedKnowledge> {
        let current = self.shelf.backend.get_knowledge(name)?.ok_or_else(|| {
            crate::error::HypatiaError::NotFound {
                kind: "knowledge".to_string(),
                key: name.to_string(),
            }
        })?;
        let next = patch.apply(&current.content);
        if normalized(&next) == normalized(&current.content) {
            return Ok(UpdatedKnowledge {
                knowledge: current,
                changed: false,
            });
        }
        let knowledge = self.update(name, next)?;
        Ok(UpdatedKnowledge {
            knowledge,
            changed: true,
        })
    }

    pub fn delete(&mut self, name: &str) -> Result<()> {
        self.shelf.backend.delete_knowledge(name)
    }
}

/// Content with the "absent" spellings unified, so rewriting an empty list as absent (or
/// the reverse) does not count as a change and cost a re-embedding.
fn normalized(content: &Content) -> Content {
    let mut c = content.clone();
    if c.figures.as_ref().is_some_and(|f| f.is_empty()) {
        c.figures = None;
    }
    if c.scopes.as_ref().is_some_and(|s| s.is_empty()) {
        c.scopes = None;
    }
    let empty_synonyms = match &c.synonyms {
        Some(Synonyms::Flat(list)) => list.is_empty(),
        Some(Synonyms::Positional(map)) => map.is_empty(),
        None => false,
    };
    if empty_synonyms {
        c.synonyms = None;
    }
    // Embedding is the default, so asking for it again is not a change.
    if c.embed == Some(true) {
        c.embed = None;
    }
    c
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rewriting_an_empty_list_as_absent_is_not_a_change() {
        let mut current = Content::new("x");
        current.figures = Some(vec![]);
        current.scopes = Some(vec![]);
        current.synonyms = Some(Synonyms::Flat(vec![]));
        let next = KnowledgePatch {
            figures: Some(vec![]),
            scopes: Some(vec![]),
            synonyms: Some(None),
            ..Default::default()
        }
        .apply(&current);
        assert_ne!(next, current);
        assert_eq!(normalized(&next), normalized(&current));
    }

    fn stored() -> Content {
        Content::new("original")
            .with_tags(vec!["a".into(), "b".into()])
            .with_synonyms(Some(Synonyms::Flat(vec!["alias".into()])))
            .with_figures(vec!["archive://fig.png".into()])
            .with_scopes(vec!["p".into(), String::new()])
    }

    #[test]
    fn omitted_fields_keep_their_stored_values() {
        let patch = KnowledgePatch {
            data: Some("changed".into()),
            ..Default::default()
        };
        let next = patch.apply(&stored());
        assert_eq!(next.data, "changed");
        assert_eq!(
            Content {
                data: "original".into(),
                ..next
            },
            stored()
        );
    }

    #[test]
    fn empty_values_clear_fields() {
        let patch = KnowledgePatch {
            tags: Some(vec![]),
            synonyms: Some(None),
            figures: Some(vec![]),
            scopes: Some(vec![]),
            ..Default::default()
        };
        let next = patch.apply(&stored());
        assert!(next.tags.is_empty());
        assert_eq!(next.synonyms, None);
        assert_eq!(next.figures, None);
        assert_eq!(next.scopes, None);
        assert_eq!(next.data, "original");
    }

    #[test]
    fn an_empty_patch_is_detected_but_empty_data_is_a_change() {
        assert!(KnowledgePatch::default().is_empty());
        assert!(
            !KnowledgePatch {
                data: Some(String::new()),
                ..Default::default()
            }
            .is_empty()
        );
    }
}
