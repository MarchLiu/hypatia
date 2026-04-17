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
                    _ => Err(HypatiaError::Eval(format!(
                        "top-level operator must be $knowledge or $statement, got {operator}"
                    ))),
                }
            }
            AstNode::Quote(_inner) => {
                // A quoted expression at the top level is not a valid query
                Err(HypatiaError::Eval("quoted expression is not a valid query".to_string()))
            }
            _ => Err(HypatiaError::Eval(
                "top-level expression must be an operator ($knowledge or $statement)".to_string(),
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
}
