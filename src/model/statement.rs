use chrono::NaiveDateTime;

use super::Content;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct StatementKey {
    pub head: String,
    pub relation: String,
    pub tail: String,
}

impl StatementKey {
    pub fn new(head: impl Into<String>, relation: impl Into<String>, tail: impl Into<String>) -> Self {
        Self {
            head: head.into(),
            relation: relation.into(),
            tail: tail.into(),
        }
    }

    /// Format as CSV row for FTS key / triple column (handles commas and quotes).
    pub fn to_csv_key(&self) -> String {
        let fields = [&self.head, &self.relation, &self.tail];
        fields
            .iter()
            .map(|f| csv_escape(f))
            .collect::<Vec<_>>()
            .join(",")
    }

    /// Parse a CSV-formatted triple back into a StatementKey.
    pub fn from_csv(csv: &str) -> Option<Self> {
        let fields = csv_split(csv);
        if fields.len() == 3 {
            Some(Self {
                head: fields[0].clone(),
                relation: fields[1].clone(),
                tail: fields[2].clone(),
            })
        } else {
            None
        }
    }
}

#[derive(Debug, Clone)]
pub struct Statement {
    pub key: StatementKey,
    pub content: Content,
    pub created_at: NaiveDateTime,
    pub tr_start: Option<NaiveDateTime>,
    pub tr_end: Option<NaiveDateTime>,
}

/// Escape a field for CSV: wrap in quotes if it contains comma, quote, or newline.
fn csv_escape(s: &str) -> String {
    if s.contains(',') || s.contains('"') || s.contains('\n') {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}

/// Split a CSV line respecting quoted fields.
pub fn csv_split(s: &str) -> Vec<String> {
    let mut result = Vec::new();
    let mut current = String::new();
    let mut in_quotes = false;
    let mut chars = s.chars().peekable();

    while let Some(ch) = chars.next() {
        if in_quotes {
            if ch == '"' {
                if chars.peek() == Some(&'"') {
                    chars.next();
                    current.push('"');
                } else {
                    in_quotes = false;
                }
            } else if ch == '"' {
                in_quotes = true;
            } else {
                current.push(ch);
            }
        } else if ch == '"' {
            in_quotes = true;
        } else if ch == ',' {
            result.push(std::mem::take(&mut current));
        } else {
            current.push(ch);
        }
    }
    result.push(current);
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn statement_key_csv_simple() {
        let key = StatementKey::new("Alice", "knows", "Bob");
        assert_eq!(key.to_csv_key(), "Alice,knows,Bob");
    }

    #[test]
    fn statement_key_csv_with_comma() {
        let key = StatementKey::new("Alice, Jr.", "knows", "Bob");
        assert_eq!(key.to_csv_key(), "\"Alice, Jr.\",knows,Bob");
    }

    #[test]
    fn statement_key_csv_with_quote() {
        let key = StatementKey::new("Alice \"Al\"", "knows", "Bob");
        assert_eq!(key.to_csv_key(), "\"Alice \"\"Al\"\"\",knows,Bob");
    }

    #[test]
    fn statement_key_equality() {
        let k1 = StatementKey::new("a", "b", "c");
        let k2 = StatementKey::new("a", "b", "c");
        let k3 = StatementKey::new("a", "b", "d");
        assert_eq!(k1, k2);
        assert_ne!(k1, k3);
    }

    #[test]
    fn from_csv_roundtrip() {
        let key = StatementKey::new("Alice", "knows", "Bob");
        let csv = key.to_csv_key();
        let parsed = StatementKey::from_csv(&csv).unwrap();
        assert_eq!(parsed.head, "Alice");
        assert_eq!(parsed.relation, "knows");
        assert_eq!(parsed.tail, "Bob");
    }

    #[test]
    fn from_csv_with_comma() {
        let key = StatementKey::new("Alice, Jr.", "knows", "Bob");
        let csv = key.to_csv_key();
        let parsed = StatementKey::from_csv(&csv).unwrap();
        assert_eq!(parsed.head, "Alice, Jr.");
    }

    #[test]
    fn from_csv_invalid() {
        assert!(StatementKey::from_csv("only,two").is_none());
    }

    #[test]
    fn csv_split_simple() {
        assert_eq!(csv_split("Alice,knows,Bob"), vec!["Alice", "knows", "Bob"]);
    }

    #[test]
    fn csv_split_quoted() {
        assert_eq!(csv_split("\"Alice, Jr.\",knows,Bob"), vec!["Alice, Jr.", "knows", "Bob"]);
    }
}
