use super::ast::AstNode;
use super::operators::{OpContext, OperatorResult};
use super::parser::Parser;
use super::sql_builder::SqlBuilder;
use crate::error::{HypatiaError, Result};
use crate::model::{QueryOpts, QueryResult, QueryTarget, SearchOpts};
use crate::storage::Storage;

pub struct Evaluator;

impl Evaluator {
    /// Parse a JSE JSON expression and evaluate it against storage.
    pub fn execute(json: &serde_json::Value, store: &dyn Storage) -> Result<QueryResult> {
        let ast = Parser::parse(json)?;
        if store.sql_dialect() == super::SqlDialect::Postgres {
            return super::postgres::execute(&ast, store);
        }
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
        let ctx = OpContext::for_target(target);

        // Evaluate conditions from operands
        for operand in operands {
            let result = Self::eval_condition(operand, &ctx)?;
            match result {
                OperatorResult::SqlCondition { fragment, params } => {
                    builder.add_condition(fragment, params);
                }
                OperatorResult::FtsQuery { query } => {
                    // $search inside $knowledge/$statement: execute FTS, then use
                    // the resulting keys to build SQL IN conditions.
                    let search_opts = query_opts_to_search_opts(&opts, target);
                    let search_result = store.execute_search(&query, &search_opts)?;
                    let keys: Vec<String> = search_result
                        .rows
                        .iter()
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
                    let keys: Vec<String> = search_result
                        .rows
                        .iter()
                        .filter_map(|row| {
                            row.get(target.key_column())
                                .and_then(|v| v.as_str())
                                .map(String::from)
                        })
                        .collect();
                    if keys.is_empty() {
                        builder.add_condition("1=0".to_string(), Vec::new());
                    } else {
                        let (fragment, params) = build_key_match_condition(target, &keys);
                        builder.add_condition(fragment, params);
                    }
                }
                OperatorResult::KHop {
                    subject,
                    predicate,
                    depth,
                } => {
                    // $k-hop: only valid inside $statement
                    if target != QueryTarget::Statement {
                        return Err(HypatiaError::Eval(
                            "$k-hop is only valid inside $statement".to_string(),
                        ));
                    }
                    let khop_result = store.execute_khop(&subject, predicate.as_deref(), depth)?;
                    let keys: Vec<String> = khop_result
                        .rows
                        .iter()
                        .filter_map(|row| {
                            row.get("triple").and_then(|v| v.as_str()).map(String::from)
                        })
                        .collect();
                    if keys.is_empty() {
                        builder.add_condition("1=0".to_string(), Vec::new());
                    } else {
                        let (fragment, params) = build_key_match_condition(target, &keys);
                        builder.add_condition(fragment, params);
                    }
                }
                OperatorResult::Value(_) => return Err(not_a_condition(operand)),
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
                "$not-summaried expects at least a tag argument (e.g. \"message\", \"summary-l1\")"
                    .to_string(),
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
        // The query below joins knowledge with statement, so `content` and
        // `created_at` are ambiguous. Operators emit them already qualified.
        let ctx = OpContext::for_target(QueryTarget::Knowledge).qualified();

        let mut conditions = Vec::new();
        let mut cond_params = Vec::new();

        for operand in &operands[1..] {
            let result = Self::eval_condition(operand, &ctx)?;
            match result {
                OperatorResult::SqlCondition { fragment, params } => {
                    conditions.push(fragment);
                    cond_params.extend(params);
                }
                OperatorResult::FtsQuery { query } => {
                    let search_opts = query_opts_to_search_opts(&opts, QueryTarget::Knowledge);
                    let search_result = store.execute_search(&query, &search_opts)?;
                    let keys: Vec<String> = search_result
                        .rows
                        .iter()
                        .filter_map(|row| row.get("key").and_then(|v| v.as_str()).map(String::from))
                        .collect();
                    if keys.is_empty() {
                        conditions.push("1=0".to_string());
                    } else {
                        let (fragment, params) =
                            build_key_match_condition(QueryTarget::Knowledge, &keys);
                        conditions.push(fragment);
                        cond_params.extend(params);
                    }
                }
                OperatorResult::VectorQuery { query_text } => {
                    let search_opts = query_opts_to_search_opts(&opts, QueryTarget::Knowledge);
                    let search_result =
                        store.execute_similar(&query_text, &search_opts, QueryTarget::Knowledge)?;
                    let keys: Vec<String> = search_result
                        .rows
                        .iter()
                        .filter_map(|row| {
                            row.get("name").and_then(|v| v.as_str()).map(String::from)
                        })
                        .collect();
                    if keys.is_empty() {
                        conditions.push("1=0".to_string());
                    } else {
                        let (fragment, params) =
                            build_key_match_condition(QueryTarget::Knowledge, &keys);
                        conditions.push(fragment);
                        cond_params.extend(params);
                    }
                }
                OperatorResult::KHop { .. } => {
                    return Err(HypatiaError::Eval(
                        "$k-hop is not valid inside $not-summaried".to_string(),
                    ));
                }
                OperatorResult::Value(_) => return Err(not_a_condition(operand)),
            }
        }

        // Exact tag membership over json_index (replaces LIKE substring).
        let tag_filter = format!(
            "EXISTS (SELECT 1 FROM docs d JOIN json_index j ON j.doc_id = d.id \
             WHERE d.catalog = 'knowledge' AND d.key = knowledge.name \
             AND j.path = 'tags' AND j.value = ?)"
        );
        let mut all_params: Vec<Value> = vec![Value::String(tag)];
        all_params.extend(cond_params);
        all_params.push(Value::Number(opts.limit.into()));
        all_params.push(Value::Number(opts.offset.into()));

        let extra_conditions = if conditions.is_empty() {
            String::new()
        } else {
            format!(" AND {}", conditions.join(" AND "))
        };

        let sql = format!(
            "SELECT knowledge.name, knowledge.content, knowledge.created_at \
             FROM knowledge \
             LEFT JOIN statement ON knowledge.name = statement.tail AND statement.relation = 'summary' \
             WHERE statement.head IS NULL \
             AND {tag_filter}{} \
             ORDER BY knowledge.created_at ASC \
             LIMIT ? OFFSET ?",
            extra_conditions
        );

        store.execute_query(QueryTarget::Knowledge, &sql, all_params)
    }

    /// Evaluate a node that must contribute a WHERE fragment. Only an
    /// operator carries a filter, so everything else is an error: a literal,
    /// symbol or quote accepted here would leave the WHERE clause narrower
    /// than written — or empty — and quietly return rows the caller excluded.
    fn eval_condition(ast: &AstNode, ctx: &OpContext) -> Result<OperatorResult> {
        match ast {
            AstNode::Operator {
                operator,
                operands,
                metadata,
            } => super::operators::evaluate_operator(operator, operands, metadata, ctx, &|node| {
                Self::eval_condition(node, ctx)
            }),
            _ => Err(not_a_condition(ast)),
        }
    }
}

/// A node that carries no filter turned up where a condition was required.
/// Both backends raise this, so a mis-shaped query fails identically on
/// SQLite and PostgreSQL instead of silently matching every row.
pub(super) fn not_a_condition(node: &AstNode) -> HypatiaError {
    // A whitelisted operator name arriving as a literal is proof that a call
    // lost its own array and flattened into sibling operands. Other
    // $-strings — a field reference like "$name", a wildcard, an escaped
    // "$$foo" — are not, so they get no hint rather than a misleading one.
    // (The parser drops the escape, so "$$contains" does get the hint; it is
    // indistinguishable here from a flattened "$contains".)
    let hint = match node {
        AstNode::Literal(serde_json::Value::String(s))
            if super::parser::OPERATORS.contains(&s.as_str()) =>
        {
            " — a nested call needs its own array, as in [\"$knowledge\", [\"$contains\", ...]]"
        }
        _ => "",
    };
    HypatiaError::Eval(format!(
        "unexpected node in condition context: {node:?}{hint}"
    ))
}

/// A nested `$knowledge`/`$statement` got more than one operand. If one of
/// them is not a condition, that is the real mistake — usually a call that
/// lost its array — so name it rather than suggest an `$and` that would only
/// fail again. Otherwise the caller meant several conditions.
pub(super) fn too_many_nested_operands(operator: &str, operands: &[AstNode]) -> HypatiaError {
    match operands
        .iter()
        .find(|n| !matches!(n, AstNode::Operator { .. }))
    {
        Some(node) => not_a_condition(node),
        None => HypatiaError::Eval(format!(
            "nested {operator} takes at most one condition, got {}; \
             combine them with [\"$and\", ...]",
            operands.len()
        )),
    }
}

/// Shapes that used to drop a filter and quietly widen the result, each with
/// the error it must now raise instead. One table for both backends, so the
/// SQLite and PostgreSQL paths are held to the same list rather than two
/// hand-synced copies (see `no_operand_is_silently_dropped` in each).
#[cfg(test)]
pub(super) fn dropped_filter_cases() -> Vec<(serde_json::Value, &'static str)> {
    use serde_json::json;
    const NOT_A_CONDITION: &str = "unexpected node in condition context";
    const FLATTENED: &str = "a nested call needs its own array";
    const NESTED: &str = "takes at most one condition";
    const ONE_QUERY: &str = "expects exactly one query argument";
    vec![
        // The issue: a call that lost its array flattens into literals. Every
        // operator name gets the hint (#28: $has, $json-contains and $triple
        // parsed as field references, so theirs went without).
        (
            json!({"$knowledge": ["$contains", "scopes", "zzz-absent"], "limit": -1}),
            FLATTENED,
        ),
        (
            json!({"$knowledge": ["$has", "scopes", "zzz-absent"], "limit": -1}),
            FLATTENED,
        ),
        (
            json!({"$knowledge": ["$json-contains", {"scopes": "zzz-absent"}], "limit": -1}),
            FLATTENED,
        ),
        (
            json!({"$statement": ["$triple", "alice", "knows", "$*"], "limit": -1}),
            FLATTENED,
        ),
        // Its control: any other $-string is a field reference, a symbol, and
        // no condition either.
        (json!(["$knowledge", "$name"]), NOT_A_CONDITION),
        (
            json!(["$knowledge", ["$and", ["$eq", "name", "a"], "stray"]]),
            NOT_A_CONDITION,
        ),
        (json!(["$knowledge", ["$or", "stray"]]), NOT_A_CONDITION),
        (json!(["$knowledge", ["$not", "stray"]]), NOT_A_CONDITION),
        (json!(["$knowledge", ["$quote", "stray"]]), NOT_A_CONDITION),
        (json!(["$statement", 42]), NOT_A_CONDITION),
        (
            json!(["$not-summaried", "message", "stray"]),
            NOT_A_CONDITION,
        ),
        // A nested query operator answered 1=1/TRUE and dropped every
        // condition; under $not that became "exclude everything".
        (
            json!([
                "$knowledge",
                [
                    "$knowledge",
                    ["$contains", "tags", "a"],
                    ["$contains", "tags", "b"]
                ]
            ]),
            NESTED,
        ),
        (
            json!([
                "$knowledge",
                [
                    "$not",
                    ["$knowledge", ["$eq", "name", "a"], ["$eq", "name", "b"]]
                ]
            ]),
            NESTED,
        ),
        // Flattened literals inside one: name the literal, don't suggest $and.
        (
            json!(["$knowledge", {"$knowledge": ["$contains", "scopes", "zzz"]}]),
            NOT_A_CONDITION,
        ),
        // $search/$similar kept the first operand and dropped the rest, a
        // filter included.
        (
            json!(["$knowledge", ["$search", "rust", ["$eq", "name", "a"]]]),
            ONE_QUERY,
        ),
        (
            json!(["$knowledge", {"$search": ["rust", "async"]}]),
            ONE_QUERY,
        ),
        (
            json!(["$statement", ["$similar", "rust", "async"]]),
            ONE_QUERY,
        ),
        (
            json!(["$not-summaried", "message", ["$search", "rust", "async"]]),
            ONE_QUERY,
        ),
    ]
}

pub(super) fn extract_query_opts(
    metadata: &serde_json::Map<String, serde_json::Value>,
) -> QueryOpts {
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
pub(super) fn query_opts_to_search_opts(opts: &QueryOpts, target: QueryTarget) -> SearchOpts {
    SearchOpts {
        // Default catalog to the query target's table name if not explicitly set
        catalog: opts
            .catalog
            .clone()
            .or_else(|| Some(target.table_name().to_string())),
        limit: opts.limit,
        offset: opts.offset,
    }
}

/// Build a SQL condition that matches keys from FTS results against the target table.
/// For Knowledge: `name IN (?, ?, ...)`
/// For Statement: `triple IN (?, ?, ...)`
fn build_key_match_condition(
    target: QueryTarget,
    keys: &[String],
) -> (String, Vec<serde_json::Value>) {
    let pk_column = target.key_column();
    let params: Vec<serde_json::Value> = keys
        .iter()
        .map(|k| serde_json::Value::String(k.clone()))
        .collect();
    let placeholders: Vec<&str> = keys.iter().map(|_| "?").collect();
    let in_clause = placeholders.join(", ");
    (format!("{pk_column} IN ({in_clause})"), params)
}

/// Convert an AST node back to a JSON value (for $quote).
pub(super) fn ast_to_value(node: &AstNode) -> serde_json::Value {
    match node {
        AstNode::Literal(v) => v.clone(),
        AstNode::Symbol(s) => serde_json::Value::String(s.clone()),
        AstNode::Array(nodes) => serde_json::Value::Array(nodes.iter().map(ast_to_value).collect()),
        AstNode::Object(map) => serde_json::Value::Object(map.clone()),
        AstNode::Quote(_inner) => ast_to_value(_inner),
        AstNode::Operator {
            operator, operands, ..
        } => {
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
        /// Every call that reached the store — SQL, searches, traversals —
        /// so a test can assert a rejected query touched nothing.
        executed: std::cell::RefCell<Vec<String>>,
    }

    impl MockStorage {
        fn new(results: Vec<serde_json::Map<String, serde_json::Value>>) -> Self {
            Self {
                query_results: results.clone(),
                search_results: results,
                executed: Default::default(),
            }
        }

        fn with_search_results(
            query_results: Vec<serde_json::Map<String, serde_json::Value>>,
            search_results: Vec<serde_json::Map<String, serde_json::Value>>,
        ) -> Self {
            Self {
                query_results,
                search_results,
                executed: Default::default(),
            }
        }
    }

    impl Storage for MockStorage {
        fn execute_query(
            &self,
            _target: QueryTarget,
            sql: &str,
            _params: Vec<serde_json::Value>,
        ) -> Result<QueryResult> {
            self.executed.borrow_mut().push(sql.to_string());
            Ok(QueryResult::new(self.query_results.clone()))
        }

        fn execute_search(&self, query: &str, _opts: &SearchOpts) -> Result<QueryResult> {
            self.executed.borrow_mut().push(format!("search: {query}"));
            Ok(QueryResult::new(self.search_results.clone()))
        }

        fn execute_similar(
            &self,
            query_text: &str,
            _opts: &SearchOpts,
            _target: QueryTarget,
        ) -> Result<QueryResult> {
            self.executed
                .borrow_mut()
                .push(format!("similar: {query_text}"));
            Ok(QueryResult::new(self.search_results.clone()))
        }

        fn execute_khop(
            &self,
            subject: &str,
            _predicate: Option<&str>,
            _depth: i64,
        ) -> Result<QueryResult> {
            self.executed.borrow_mut().push(format!("k-hop: {subject}"));
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
        let result =
            Evaluator::execute(&json!(["$knowledge", ["$eq", "name", "test"]]), &mock).unwrap();
        assert_eq!(result.rows.len(), 1);
        assert_eq!(result.rows[0]["name"], json!("test"));
    }

    #[test]
    fn eval_statement_query() {
        let mock = MockStorage::new(vec![]);
        let result = Evaluator::execute(&json!(["$statement"]), &mock).unwrap();
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
        let result =
            Evaluator::execute(&json!(["$knowledge", ["$search", "rust"]]), &mock).unwrap();
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
        let result =
            Evaluator::execute(&json!(["$statement", ["$search", "Alice"]]), &mock).unwrap();
        assert_eq!(result.rows.len(), 1);
    }

    #[test]
    fn eval_search_not_top_level() {
        // $search can no longer be used as a top-level operator
        let mock = MockStorage::new(vec![]);
        let result = Evaluator::execute(&json!({"$search": "rust", "catalog": "knowledge"}), &mock);
        assert!(result.is_err());
    }

    #[test]
    fn eval_invalid_top_level() {
        let mock = MockStorage::new(vec![]);
        let result = Evaluator::execute(&json!("not a query"), &mock);
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
        )
        .unwrap();
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
        let result =
            Evaluator::execute(&json!(["$statement", ["$similar", "relationships"]]), &mock)
                .unwrap();
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
            &[
                "Alice,knows,Bob".to_string(),
                "Charlie,likes,Rust".to_string(),
            ],
        );
        assert_eq!(sql, "triple IN (?, ?)");
        assert_eq!(params.len(), 2);
    }

    #[test]
    fn eval_not_summaried_basic() {
        let mut row = serde_json::Map::new();
        row.insert("name".to_string(), json!("msg-s1-001"));
        row.insert(
            "content".to_string(),
            json!(r#"{"tags":["message","user"]}"#),
        );
        row.insert("created_at".to_string(), json!("2026-01-01 00:00:00"));

        let mock = MockStorage::new(vec![row]);
        let result = Evaluator::execute(&json!(["$not-summaried", "message"]), &mock).unwrap();
        assert_eq!(result.rows.len(), 1);
        assert_eq!(result.rows[0]["name"], json!("msg-s1-001"));
    }

    #[test]
    fn eval_not_summaried_with_condition() {
        let mut row = serde_json::Map::new();
        row.insert("name".to_string(), json!("msg-s1-001"));
        row.insert(
            "content".to_string(),
            json!(r#"{"tags":["message","user"]}"#),
        );
        row.insert("created_at".to_string(), json!("2026-01-01 00:00:00"));

        let mock = MockStorage::new(vec![row]);
        let result = Evaluator::execute(
            &json!([
                "$not-summaried",
                "message",
                ["$contains", "scopes", "my-project"]
            ]),
            &mock,
        )
        .unwrap();
        assert_eq!(result.rows.len(), 1);
    }

    #[test]
    fn eval_not_summaried_empty_operands() {
        let mock = MockStorage::new(vec![]);
        let result = Evaluator::execute(&json!(["$not-summaried"]), &mock);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("expects at least a tag")
        );
    }

    #[test]
    fn eval_not_summaried_bad_tag() {
        let mock = MockStorage::new(vec![]);
        let result = Evaluator::execute(&json!(["$not-summaried", 123]), &mock);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("must be a tag string")
        );
    }

    /// Build the SQL fragment an operator emits under a qualifying context,
    /// the way `$not-summaried` splices it into its join.
    fn qualified_fragment(jse: serde_json::Value) -> (String, Vec<serde_json::Value>) {
        let ast = crate::engine::parser::Parser::parse(&jse).expect("parse");
        let ctx = OpContext::for_target(QueryTarget::Knowledge).qualified();
        match Evaluator::eval_condition(&ast, &ctx).expect("eval") {
            OperatorResult::SqlCondition { fragment, params } => (fragment, params),
            other => panic!("expected SqlCondition, got {other:?}"),
        }
    }

    #[test]
    fn qualified_context_qualifies_content() {
        let (fragment, params) = qualified_fragment(json!(["$like", "data", "%rust%"]));
        assert_eq!(fragment, "json_extract(knowledge.content, ?) LIKE ?");
        assert_eq!(params, vec![json!("$.data"), json!("%rust%")]);
    }

    #[test]
    fn qualified_context_qualifies_created_at() {
        let (fragment, _) = qualified_fragment(json!(["$gt", "created_at", "2025-01-01"]));
        assert_eq!(fragment, "knowledge.created_at > ?");
    }

    #[test]
    fn qualified_context_qualifies_temporal_columns() {
        let (fragment, _) = qualified_fragment(json!(["$eq", "tr_start", "2025-01-01"]));
        assert_eq!(fragment, "knowledge.tr_start = ?");
        let (fragment, _) = qualified_fragment(json!(["$eq", "tr_end", "2025-01-01"]));
        assert_eq!(fragment, "knowledge.tr_end = ?");
    }

    #[test]
    fn qualified_context_preserves_unambiguous_columns() {
        let (fragment, _) = qualified_fragment(json!(["$eq", "name", "rust"]));
        assert_eq!(fragment, "name = ?");
    }

    #[test]
    fn qualified_context_handles_combined() {
        let (fragment, params) = qualified_fragment(json!([
            "$and",
            ["$like", "scopes", "%proj%"],
            ["$eq", "name", "rust"]
        ]));
        assert_eq!(
            fragment,
            "(json_extract(knowledge.content, ?) LIKE ? AND name = ?)"
        );
        assert_eq!(
            params,
            vec![json!("$.scopes"), json!("%proj%"), json!("rust")]
        );
    }

    /// Regression: qualification used to be a post-hoc `String::replace` over
    /// finished SQL, so a Content field whose name contained `created_at`
    /// (or `tr_start`/`tr_end`) had that substring rewritten inside its own
    /// JSON path — `'$.doc_created_at'` became `'$.doc_knowledge.created_at'`,
    /// a valid path that silently matched nothing.
    #[test]
    fn qualified_context_does_not_rewrite_column_names_inside_field_paths() {
        for field in [
            "doc_created_at",
            "created_at_utc",
            "tr_start_ns",
            "my_tr_end",
        ] {
            let (fragment, params) = qualified_fragment(json!(["$like", field, "%x%"]));
            assert_eq!(
                fragment, "json_extract(knowledge.content, ?) LIKE ?",
                "field {field}"
            );
            assert_eq!(params[0], json!(format!("$.{field}")), "field {field}");
        }
    }

    /// Regression: the old textual rewrite only knew the string
    /// `"json_extract(content,"`, so `$json-contains` — which builds
    /// `json_contains(content, ?)` — kept a bare `content` and failed with
    /// "ambiguous column name" inside the `$not-summaried` join.
    #[test]
    fn qualified_context_qualifies_json_contains_recheck() {
        let (fragment, _) = qualified_fragment(json!(["$json-contains", {"tags": ["rust"]}]));
        assert!(
            fragment.ends_with("AND json_contains(knowledge.content, ?)"),
            "unqualified content in: {fragment}"
        );
    }

    /// Regression for #22: an operator call that loses its own array
    /// flattens into sibling operands — `{"$knowledge": ["$contains", ...]}`
    /// parses as three literals — and ignoring them left the WHERE clause
    /// empty, returning the whole table while looking filtered. The cases
    /// cover that and the neighbouring paths that dropped a filter the same
    /// way. Each must fail with its own message before touching the store.
    #[test]
    fn no_operand_is_silently_dropped() {
        let leaked = json!({"name": "everything"}).as_object().unwrap().clone();
        for (expression, expected) in super::dropped_filter_cases() {
            let mock = MockStorage::new(vec![leaked.clone()]);
            let msg = Evaluator::execute(&expression, &mock)
                .expect_err(&format!("must not match every row: {expression}"))
                .to_string();
            assert!(msg.contains(expected), "{expression}: {msg}");
            assert!(
                mock.executed.borrow().is_empty(),
                "{expression}: reached the store anyway: {:?}",
                mock.executed.borrow()
            );
        }
    }

    /// The empty nested form really does mean "no filter", so it must keep
    /// working — the arity check above must not swallow it.
    #[test]
    fn an_empty_nested_query_operator_is_still_no_filter() {
        let mock = MockStorage::new(vec![]);
        Evaluator::execute(&json!(["$knowledge", ["$knowledge"]]), &mock).expect("no filter");
        Evaluator::execute(
            &json!(["$knowledge", ["$knowledge", ["$eq", "name", "a"]]]),
            &mock,
        )
        .expect("one nested condition still passes through");
        assert!(
            mock.executed.borrow()[1].contains("name = ?"),
            "nested condition lost: {:?}",
            mock.executed.borrow()
        );
    }

    /// The error names the node so a mis-shaped query can be located, and
    /// points at the missing brackets when an operator name is the literal.
    #[test]
    fn condition_context_error_names_the_node() {
        let mock = MockStorage::new(vec![]);
        let err = Evaluator::execute(&json!({"$knowledge": ["$contains", "scopes", "x"]}), &mock)
            .expect_err("flattened condition");
        let msg = err.to_string();
        let (node, hint) = msg.split_once(" — ").expect("a flattened call gets a hint");
        assert!(node.contains(r#""$contains""#), "node not named: {msg}");
        assert!(hint.contains("a nested call needs its own array"), "{msg}");

        // Only a whitelisted operator name earns the hint. A field reference,
        // a wildcard and an escaped literal are not flattened calls, and
        // pointing them at brackets would send the caller the wrong way.
        for expression in [
            json!(["$knowledge", ["$not", "plain"]]),
            json!(["$knowledge", "$name"]),
            json!(["$knowledge", "$*"]),
            json!(["$knowledge", "$$foo"]),
        ] {
            let msg = Evaluator::execute(&expression, &mock)
                .expect_err(&format!("{expression}"))
                .to_string();
            assert!(
                msg.contains("unexpected node in condition context"),
                "{expression}: {msg}"
            );
            assert!(!msg.contains("needs its own array"), "{expression}: {msg}");
        }
    }

    #[test]
    fn not_summaried_rejects_injected_field_name() {
        let mock = MockStorage::new(vec![]);
        let result = Evaluator::execute(
            &json!(["$not-summaried", "message", ["$eq", "$a') OR 1=1 --", "x"]]),
            &mock,
        );
        let err = result.expect_err("injected field name must be rejected");
        assert!(
            err.to_string().contains("invalid field name"),
            "unexpected error: {err}"
        );
    }
}
