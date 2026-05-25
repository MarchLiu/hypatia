use crate::error::{HypatiaError, Result};
use crate::model::{QueryOpts, QueryResult, QueryTarget, SearchOpts};
use crate::storage::Storage;
use super::ast::AstNode;
use super::operators::OperatorResult;
use super::parser::Parser;
use super::sql_builder::SqlBuilder;

pub struct Evaluator;

impl Evaluator {
    /// Parse a JSE JSON expression and evaluate it against storage.
    pub fn execute(json: &serde_json::Value, store: &dyn Storage) -> Result<QueryResult> {
        let ast = Parser::parse(json)?;
        Self::eval(&ast, store)
    }

    fn eval(ast: &AstNode, store: &dyn Storage) -> Result<QueryResult> {
        match ast {
            AstNode::Operator { operator, operands, metadata } => {
                match operator.as_str() {
                    "$knowledge" => Self::eval_query(QueryTarget::Knowledge, operands, metadata, store),
                    "$statement" => Self::eval_query(QueryTarget::Statement, operands, metadata, store),
                    "$not-summaried" => Self::eval_not_summaried(operands, metadata, store),
                        _ => Err(HypatiaError::Eval(format!(
                            "top-level operator must be $knowledge, $statement, or $not-summaried, got {operator}"
                        ))),
                    }
                }
                AstNode::Quote(_inner) => {
                    Err(HypatiaError::Eval("quoted expression is not a valid query".to_string()))
                }
                _ => Err(HypatiaError::Eval(
                    "top-level expression must be an operator ($knowledge, $statement, or $not-summaried)".to_string(),
                )),
        }
    }

    fn eval_query(
        target: QueryTarget,
        operands: &[AstNode],
        metadata: &serde_json::Map<String, serde_json::Value>,
        store: &dyn Storage,
    ) -> Result<QueryResult> {
        let opts = extract_query_opts(metadata);
        let mut builder = SqlBuilder::new(target);
        builder.set_limit(opts.limit);
        builder.set_offset(opts.offset);

        // Evaluate conditions from operands
        for operand in operands {
            let result = Self::eval_condition(operand)?;
            match result {
                OperatorResult::SqlCondition { fragment, params } => {
                    builder.add_condition(fragment, params);
                }
                OperatorResult::FtsQuery { query } => {
                    // $search inside $knowledge/$statement: execute FTS, then use
                    // the resulting keys to build SQL IN conditions.
                    let search_opts = query_opts_to_search_opts(&opts, target);
                    let search_result = store.execute_search(&query, &search_opts)?;
                    let keys: Vec<String> = search_result.rows.iter()
                        .filter_map(|row| row.get("key").and_then(|v| v.as_str()).map(String::from))
                        .collect();
                    if keys.is_empty() {
                        // No FTS matches — add an impossible condition
                        builder.add_condition("1=0".to_string(), Vec::new());
                    } else {
                        let (fragment, params) = build_key_match_condition(target, &keys);
                        builder.add_condition(fragment, params);
                    }
                }
                OperatorResult::VectorQuery { query_text } => {
                    // $similar: execute vector search, then use resulting keys
                    // to build SQL IN conditions (same pattern as FTS).
                    let search_opts = query_opts_to_search_opts(&opts, target);
                    let search_result = store.execute_similar(&query_text, &search_opts, target)?;
                    let keys: Vec<String> = search_result.rows.iter()
                        .filter_map(|row| row.get(target.key_column()).and_then(|v| v.as_str()).map(String::from))
                        .collect();
                    if keys.is_empty() {
                        builder.add_condition("1=0".to_string(), Vec::new());
                    } else {
                        let (fragment, params) = build_key_match_condition(target, &keys);
                        builder.add_condition(fragment, params);
                    }
                }
                OperatorResult::KHop { subject, predicate, depth } => {
                    // $k-hop: only valid inside $statement
                    if target != QueryTarget::Statement {
                        return Err(HypatiaError::Eval(
                            "$k-hop is only valid inside $statement".to_string(),
                        ));
                    }
                    let khop_result = store.execute_khop(&subject, predicate.as_deref(), depth)?;
                    let keys: Vec<String> = khop_result.rows.iter()
                        .filter_map(|row| row.get("triple").and_then(|v| v.as_str()).map(String::from))
                        .collect();
                    if keys.is_empty() {
                        builder.add_condition("1=0".to_string(), Vec::new());
                    } else {
                        let (fragment, params) = build_key_match_condition(target, &keys);
                        builder.add_condition(fragment, params);
                    }
                }
                OperatorResult::Value(_) => {
                    // Ignore literal values in condition context
                }
            }
        }

        let (sql, params) = builder.build();
        store.execute_query(target, &sql, params)
    }

    fn eval_not_summaried(
        operands: &[AstNode],
        metadata: &serde_json::Map<String, serde_json::Value>,
        store: &dyn Storage,
    ) -> Result<QueryResult> {
        use serde_json::Value;

        if operands.is_empty() {
            return Err(HypatiaError::Eval(
                "$not-summaried expects at least a tag argument (e.g. \"message\", \"summary-l1\")".to_string(),
            ));
        }

        let tag = match &operands[0] {
            AstNode::Literal(Value::String(s)) => s.clone(),
            AstNode::Symbol(s) => s.clone(),
            _ => {
                return Err(HypatiaError::Eval(
                    "$not-summaried first argument must be a tag string".to_string(),
                ));
            }
        };

        let opts = extract_query_opts(metadata);

        let mut conditions = Vec::new();
        let mut cond_params = Vec::new();

        for operand in &operands[1..] {
            let result = Self::eval_condition(operand)?;
            match result {
                OperatorResult::SqlCondition { fragment, params } => {
                    let aliased = alias_knowledge_columns(&fragment);
                    conditions.push(aliased);
                    cond_params.extend(params);
                }
                OperatorResult::FtsQuery { query } => {
                    let search_opts = query_opts_to_search_opts(&opts, QueryTarget::Knowledge);
                    let search_result = store.execute_search(&query, &search_opts)?;
                    let keys: Vec<String> = search_result.rows.iter()
                        .filter_map(|row| row.get("key").and_then(|v| v.as_str()).map(String::from))
                        .collect();
                    if keys.is_empty() {
                        conditions.push("1=0".to_string());
                    } else {
                        let (fragment, params) = build_key_match_condition(QueryTarget::Knowledge, &keys);
                        conditions.push(alias_knowledge_columns(&fragment));
                        cond_params.extend(params);
                    }
                }
                OperatorResult::VectorQuery { query_text } => {
                    let search_opts = query_opts_to_search_opts(&opts, QueryTarget::Knowledge);
                    let search_result = store.execute_similar(&query_text, &search_opts, QueryTarget::Knowledge)?;
                    let keys: Vec<String> = search_result.rows.iter()
                        .filter_map(|row| row.get("name").and_then(|v| v.as_str()).map(String::from))
                        .collect();
                    if keys.is_empty() {
                        conditions.push("1=0".to_string());
                    } else {
                        let (fragment, params) = build_key_match_condition(QueryTarget::Knowledge, &keys);
                        conditions.push(alias_knowledge_columns(&fragment));
                        cond_params.extend(params);
                    }
                }
                OperatorResult::KHop { .. } => {
                    return Err(HypatiaError::Eval(
                        "$k-hop is not valid inside $not-summaried".to_string(),
                    ));
                }
                OperatorResult::Value(_) => {}
            }
        }

        let tag_pattern = format!("%{}%", tag);
        let mut all_params: Vec<Value> = vec![Value::String(tag_pattern)];
        all_params.extend(cond_params);
        all_params.push(Value::Number(opts.limit.into()));
        all_params.push(Value::Number(opts.offset.into()));

        let extra_conditions = if conditions.is_empty() {
            String::new()
        } else {
            format!(" AND {}", conditions.join(" AND "))
        };

        let sql = format!(
            "SELECT knowledge.name, knowledge.content, CAST(knowledge.created_at AS VARCHAR) AS created_at \
             FROM knowledge \
             LEFT JOIN statement ON knowledge.name = statement.object AND statement.predicate = 'summarizes' \
             WHERE statement.subject IS NULL \
             AND json_extract_string(knowledge.content, '$.tags') LIKE ?{} \
             ORDER BY knowledge.created_at ASC \
             LIMIT CAST(? AS INTEGER) OFFSET CAST(? AS INTEGER)",
            extra_conditions
        );

        store.execute_query(QueryTarget::Knowledge, &sql, all_params)
    }

    fn eval_condition(ast: &AstNode) -> Result<OperatorResult> {
        match ast {
            AstNode::Operator { operator, operands, metadata } => {
                super::operators::evaluate_operator(
                    operator,
                    operands,
                    metadata,
                    &|node| Self::eval_condition(node),
                )
            }
            AstNode::Quote(inner) => {
                Ok(OperatorResult::Value(ast_to_value(inner)))
            }
            AstNode::Literal(v) => Ok(OperatorResult::Value(v.clone())),
            _ => Err(HypatiaError::Eval(format!(
                "unexpected node in condition context: {:?}", ast
            ))),
        }
    }
}

fn extract_query_opts(metadata: &serde_json::Map<String, serde_json::Value>) -> QueryOpts {
    let mut opts = QueryOpts::default();
    if let Some(serde_json::Value::String(catalog)) = metadata.get("catalog") {
        opts.catalog = Some(catalog.clone());
    }
    if let Some(serde_json::Value::Number(n)) = metadata.get("limit") {
        opts.limit = n.as_i64().unwrap_or(100);
    }
    if let Some(serde_json::Value::Number(n)) = metadata.get("offset") {
        opts.offset = n.as_i64().unwrap_or(0);
    }
    opts
}

/// Convert QueryOpts (from the parent $knowledge/$statement) to SearchOpts for FTS execution.
fn query_opts_to_search_opts(opts: &QueryOpts, target: QueryTarget) -> SearchOpts {
    SearchOpts {
        // Default catalog to the query target's table name if not explicitly set
        catalog: opts.catalog.clone().or_else(|| Some(target.table_name().to_string())),
        limit: opts.limit,
        offset: opts.offset,
    }
}

/// Build a SQL condition that matches keys from FTS results against the target table.
/// For Knowledge: `name IN (?, ?, ...)`
/// For Statement: `triple IN (?, ?, ...)`
fn build_key_match_condition(target: QueryTarget, keys: &[String]) -> (String, Vec<serde_json::Value>) {
    let pk_column = target.key_column();
    let params: Vec<serde_json::Value> = keys.iter()
        .map(|k| serde_json::Value::String(k.clone()))
        .collect();
    let placeholders: Vec<&str> = keys.iter().map(|_| "?").collect();
    let in_clause = placeholders.join(", ");
    (format!("{pk_column} IN ({in_clause})"), params)
}

/// Qualify knowledge column references for LEFT JOIN context.
/// Replaces bare `content` in json_extract_string calls and `created_at`
/// in CAST expressions with `knowledge.` prefixed versions.
fn alias_knowledge_columns(fragment: &str) -> String {
    fragment
        .replace("json_extract_string(content,", "json_extract_string(knowledge.content,")
        .replace("CAST(created_at", "CAST(knowledge.created_at")
}

/// Convert an AST node back to a JSON value (for $quote).
fn ast_to_value(node: &AstNode) -> serde_json::Value {
    match node {
        AstNode::Literal(v) => v.clone(),
        AstNode::Symbol(s) => serde_json::Value::String(s.clone()),
        AstNode::Array(nodes) => {
            serde_json::Value::Array(nodes.iter().map(ast_to_value).collect())
        }
        AstNode::Object(map) => serde_json::Value::Object(map.clone()),
        AstNode::Quote(_inner) => ast_to_value(_inner),
        AstNode::Operator { operator, operands, .. } => {
            let mut arr = vec![serde_json::Value::String(operator.clone())];
            arr.extend(operands.iter().map(ast_to_value));
            serde_json::Value::Array(arr)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// Mock storage for testing the evaluator without real databases.
    struct MockStorage {
        query_results: Vec<serde_json::Map<String, serde_json::Value>>,
        search_results: Vec<serde_json::Map<String, serde_json::Value>>,
    }

    impl MockStorage {
        fn new(results: Vec<serde_json::Map<String, serde_json::Value>>) -> Self {
            Self {
                query_results: results.clone(),
                search_results: results,
            }
        }

        fn with_search_results(
            query_results: Vec<serde_json::Map<String, serde_json::Value>>,
            search_results: Vec<serde_json::Map<String, serde_json::Value>>,
        ) -> Self {
            Self { query_results, search_results }
        }
    }

    impl Storage for MockStorage {
        fn execute_query(
            &self,
            _target: QueryTarget,
            sql: &str,
            _params: Vec<serde_json::Value>,
        ) -> Result<QueryResult> {
            // Just verify the SQL looks right and return mock results
            let _ = sql; // use the sql to avoid warning
            Ok(QueryResult::new(self.query_results.clone()))
        }

        fn execute_search(&self, _query: &str, _opts: &SearchOpts) -> Result<QueryResult> {
            Ok(QueryResult::new(self.search_results.clone()))
        }

        fn execute_similar(
            &self,
            _query_text: &str,
            _opts: &SearchOpts,
            _target: QueryTarget,
        ) -> Result<QueryResult> {
            Ok(QueryResult::new(self.search_results.clone()))
        }

        fn execute_khop(
            &self,
            _subject: &str,
            _predicate: Option<&str>,
            _depth: i64,
        ) -> Result<QueryResult> {
            Ok(QueryResult::new(self.search_results.clone()))
        }
    }

    #[test]
    fn eval_knowledge_query() {
        let mock = MockStorage::new(vec![{
            let mut m = serde_json::Map::new();
            m.insert("name".to_string(), json!("test"));
            m
        }]);
        let result = Evaluator::execute(
            &json!(["$knowledge", ["$eq", "name", "test"]]),
            &mock,
        ).unwrap();
        assert_eq!(result.rows.len(), 1);
        assert_eq!(result.rows[0]["name"], json!("test"));
    }

    #[test]
    fn eval_statement_query() {
        let mock = MockStorage::new(vec![]);
        let result = Evaluator::execute(
            &json!(["$statement"]),
            &mock,
        ).unwrap();
        assert_eq!(result.rows.len(), 0);
    }

    #[test]
    fn eval_search_inside_knowledge() {
        // $search inside $knowledge: FTS returns keys, which are used to filter knowledge by name
        let mut search_row = serde_json::Map::new();
        search_row.insert("key".to_string(), json!("rust"));
        search_row.insert("catalog".to_string(), json!("knowledge"));

        let mut query_row = serde_json::Map::new();
        query_row.insert("name".to_string(), json!("rust"));

        let mock = MockStorage::with_search_results(vec![query_row], vec![search_row]);
        let result = Evaluator::execute(
            &json!(["$knowledge", ["$search", "rust"]]),
            &mock,
        ).unwrap();
        assert_eq!(result.rows.len(), 1);
        assert_eq!(result.rows[0]["name"], json!("rust"));
    }

    #[test]
    fn eval_search_inside_statement() {
        // $search inside $statement: FTS returns keys as CSV triples,
        // which are matched via `triple IN (...)`
        let mut search_row = serde_json::Map::new();
        search_row.insert("key".to_string(), json!("Alice,knows,Bob"));
        search_row.insert("catalog".to_string(), json!("statement"));

        let mut query_row = serde_json::Map::new();
        query_row.insert("triple".to_string(), json!("Alice,knows,Bob"));

        let mock = MockStorage::with_search_results(vec![query_row], vec![search_row]);
        let result = Evaluator::execute(
            &json!(["$statement", ["$search", "Alice"]]),
            &mock,
        ).unwrap();
        assert_eq!(result.rows.len(), 1);
    }

    #[test]
    fn eval_search_not_top_level() {
        // $search can no longer be used as a top-level operator
        let mock = MockStorage::new(vec![]);
        let result = Evaluator::execute(
            &json!({"$search": "rust", "catalog": "knowledge"}),
            &mock,
        );
        assert!(result.is_err());
    }

    #[test]
    fn eval_invalid_top_level() {
        let mock = MockStorage::new(vec![]);
        let result = Evaluator::execute(
            &json!("not a query"),
            &mock,
        );
        assert!(result.is_err());
    }

    #[test]
    fn eval_similar_inside_knowledge() {
        // $similar returns rows with "name" field (not "key")
        let mut similar_row = serde_json::Map::new();
        similar_row.insert("name".to_string(), json!("rust"));
        similar_row.insert("distance".to_string(), json!(0.1));

        let mut query_row = serde_json::Map::new();
        query_row.insert("name".to_string(), json!("rust"));

        let mock = MockStorage::with_search_results(vec![query_row], vec![similar_row]);
        let result = Evaluator::execute(
            &json!(["$knowledge", ["$similar", "systems programming"]]),
            &mock,
        ).unwrap();
        assert_eq!(result.rows.len(), 1);
        assert_eq!(result.rows[0]["name"], json!("rust"));
    }

    #[test]
    fn eval_similar_inside_statement() {
        // $similar returns rows with "triple" field for statements
        let mut similar_row = serde_json::Map::new();
        similar_row.insert("triple".to_string(), json!("Alice,knows,Bob"));
        similar_row.insert("distance".to_string(), json!(0.2));

        let mut query_row = serde_json::Map::new();
        query_row.insert("triple".to_string(), json!("Alice,knows,Bob"));

        let mock = MockStorage::with_search_results(vec![query_row], vec![similar_row]);
        let result = Evaluator::execute(
            &json!(["$statement", ["$similar", "relationships"]]),
            &mock,
        ).unwrap();
        assert_eq!(result.rows.len(), 1);
    }

    #[test]
    fn build_key_match_knowledge() {
        let (sql, params) = build_key_match_condition(
            QueryTarget::Knowledge,
            &["rust".to_string(), "go".to_string()],
        );
        assert_eq!(sql, "name IN (?, ?)");
        assert_eq!(params.len(), 2);
    }

    #[test]
    fn build_key_match_statement() {
        let (sql, params) = build_key_match_condition(
            QueryTarget::Statement,
            &["Alice,knows,Bob".to_string(), "Charlie,likes,Rust".to_string()],
        );
        assert_eq!(sql, "triple IN (?, ?)");
        assert_eq!(params.len(), 2);
    }

    #[test]
    fn eval_not_summaried_basic() {
        let mut row = serde_json::Map::new();
        row.insert("name".to_string(), json!("msg-s1-001"));
        row.insert("content".to_string(), json!(r#"{"tags":["message","user"]}"#));
        row.insert("created_at".to_string(), json!("2026-01-01 00:00:00"));

        let mock = MockStorage::new(vec![row]);
        let result = Evaluator::execute(
            &json!(["$not-summaried", "message"]),
            &mock,
        ).unwrap();
        assert_eq!(result.rows.len(), 1);
        assert_eq!(result.rows[0]["name"], json!("msg-s1-001"));
    }

    #[test]
    fn eval_not_summaried_with_condition() {
        let mut row = serde_json::Map::new();
        row.insert("name".to_string(), json!("msg-s1-001"));
        row.insert("content".to_string(), json!(r#"{"tags":["message","user"]}"#));
        row.insert("created_at".to_string(), json!("2026-01-01 00:00:00"));

        let mock = MockStorage::new(vec![row]);
        let result = Evaluator::execute(
            &json!(["$not-summaried", "message", ["$contains", "scopes", "my-project"]]),
            &mock,
        ).unwrap();
        assert_eq!(result.rows.len(), 1);
    }

    #[test]
    fn eval_not_summaried_empty_operands() {
        let mock = MockStorage::new(vec![]);
        let result = Evaluator::execute(
            &json!(["$not-summaried"]),
            &mock,
        );
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("expects at least a tag"));
    }

    #[test]
    fn eval_not_summaried_bad_tag() {
        let mock = MockStorage::new(vec![]);
        let result = Evaluator::execute(
            &json!(["$not-summaried", 123]),
            &mock,
        );
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("must be a tag string"));
    }

    #[test]
    fn alias_knowledge_columns_replaces_content() {
        let input = "json_extract_string(content, '$.tags') LIKE ?";
        let result = alias_knowledge_columns(input);
        assert_eq!(result, "json_extract_string(knowledge.content, '$.tags') LIKE ?");
    }

    #[test]
    fn alias_knowledge_columns_replaces_created_at() {
        let input = "CAST(created_at AS VARCHAR) LIKE ?";
        let result = alias_knowledge_columns(input);
        assert_eq!(result, "CAST(knowledge.created_at AS VARCHAR) LIKE ?");
    }

    #[test]
    fn alias_knowledge_columns_preserves_name() {
        let input = "name = ?";
        let result = alias_knowledge_columns(input);
        assert_eq!(result, "name = ?");
    }

    #[test]
    fn alias_knowledge_columns_handles_combined() {
        let input = "json_extract_string(content, '$.scopes') LIKE ? AND name = ?";
        let result = alias_knowledge_columns(input);
        assert_eq!(result, "json_extract_string(knowledge.content, '$.scopes') LIKE ? AND name = ?");
    }
}
