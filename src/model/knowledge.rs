use chrono::NaiveDateTime;

use super::Content;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
pub struct Knowledge {
    pub name: String,
    pub content: Content,
    pub created_at: NaiveDateTime,
}
