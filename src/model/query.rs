use serde_json::Map;

#[derive(Debug, Clone)]
pub struct QueryOpts {
    pub catalog: Option<String>,
    pub offset: i64,
    pub limit: i64,
}

impl Default for QueryOpts {
    fn default() -> Self {
        Self {
            catalog: None,
            offset: 0,
            limit: 100,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SearchOpts {
    pub catalog: Option<String>,
    pub offset: i64,
    pub limit: i64,
}

impl Default for SearchOpts {
    fn default() -> Self {
        Self {
            catalog: None,
            offset: 0,
            limit: 100,
        }
    }
}

pub type ResultSetRow = Map<String, serde_json::Value>;

#[derive(Debug)]
pub struct QueryResult {
    pub rows: Vec<ResultSetRow>,
    pub total_count: Option<i64>,
}

impl QueryResult {
    pub fn empty() -> Self {
        Self {
            rows: Vec::new(),
            total_count: Some(0),
        }
    }

    pub fn new(rows: Vec<ResultSetRow>) -> Self {
        let count = rows.len() as i64;
        Self {
            rows,
            total_count: Some(count),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum QueryTarget {
    Knowledge,
    Statement,
}

impl QueryTarget {
    pub fn table_name(&self) -> &'static str {
        match self {
            QueryTarget::Knowledge => "knowledge",
            QueryTarget::Statement => "statement",
        }
    }

    /// Primary key column name for this target, used for FTS/vector key matching.
    pub fn key_column(&self) -> &'static str {
        match self {
            QueryTarget::Knowledge => "name",
            QueryTarget::Statement => "triple",
        }
    }
}
