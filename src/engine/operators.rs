use crate::error::{HypatiaError, Result};
use super::ast::AstNode;

/// Result of evaluating an operator in the context of query building.
#[derive(Debug, Clone)]
pub enum OperatorResult {
    /// A SQL WHERE fragment with parameterized values.
    SqlCondition {
        fragment: String,
        params: Vec<serde_json::Value>,
    },
    /// A FTS search query string. Opts are inherited from the parent query.
    FtsQuery {
        query: String,
    },
    /// A semantic similarity query. The evaluator embeds the text and searches vectors.
    VectorQuery {
        query_text: String,
    },
    /// A k-hop forward graph traversal query on the statement graph.
    KHop {
        subject: String,
        predicate: Option<String>,
        depth: i64,
    },
    /// A literal value (from $quote or non-operator expressions).
    Value(serde_json::Value),
}

/// Evaluate an operator AST node against its operands.
/// Returns the SQL contribution of this operator.
pub fn evaluate_operator(
    operator: &str,
    operands: &[AstNode],
    _metadata: &serde_json::Map<String, serde_json::Value>,
    eval_fn: &dyn Fn(&AstNode) -> Result<OperatorResult>,
) -> Result<OperatorResult> {
    match operator {
        "$knowledge" | "$statement" => {
            // These are handled by the evaluator at the top level.
            // When evaluated as operators, they just pass through their first operand.
            if operands.len() == 1 {
                eval_fn(&operands[0])
            } else {
                // No conditions — return a tautology
                Ok(OperatorResult::SqlCondition {
                    fragment: "1=1".to_string(),
                    params: Vec::new(),
                })
            }
        }
        "$and" => {
            let mut fragments = Vec::new();
            let mut all_params = Vec::new();
            for operand in operands {
                match eval_fn(operand)? {
                    OperatorResult::SqlCondition { fragment, params } => {
                        fragments.push(fragment);
                        all_params.extend(params);
                    }
                    other => {
                        return Err(HypatiaError::Eval(format!(
                            "$and expects SQL conditions, got {:?}", other
                        )));
                    }
                }
            }
            if fragments.is_empty() {
                Ok(OperatorResult::SqlCondition {
                    fragment: "1=1".to_string(),
                    params: Vec::new(),
                })
            } else if fragments.len() == 1 {
                Ok(OperatorResult::SqlCondition {
                    fragment: fragments.into_iter().next().unwrap(),
                    params: all_params,
                })
            } else {
                Ok(OperatorResult::SqlCondition {
                    fragment: format!("({})", fragments.join(" AND ")),
                    params: all_params,
                })
            }
        }
        "$or" => {
            let mut fragments = Vec::new();
            let mut all_params = Vec::new();
            for operand in operands {
                match eval_fn(operand)? {
                    OperatorResult::SqlCondition { fragment, params } => {
                        fragments.push(fragment);
                        all_params.extend(params);
                    }
                    other => {
                        return Err(HypatiaError::Eval(format!(
                            "$or expects SQL conditions, got {:?}", other
                        )));
                    }
                }
            }
            if fragments.is_empty() {
                Ok(OperatorResult::SqlCondition {
                    fragment: "1=1".to_string(),
                    params: Vec::new(),
                })
            } else {
                Ok(OperatorResult::SqlCondition {
                    fragment: format!("({})", fragments.join(" OR ")),
                    params: all_params,
                })
            }
        }
        "$not" => {
            if operands.len() != 1 {
                return Err(HypatiaError::Eval("$not expects exactly one argument".to_string()));
            }
            match eval_fn(&operands[0])? {
                OperatorResult::SqlCondition { fragment, params } => {
                    Ok(OperatorResult::SqlCondition {
                        fragment: format!("NOT ({fragment})"),
                        params,
                    })
                }
                other => Err(HypatiaError::Eval(format!(
                    "$not expects SQL condition, got {:?}", other
                ))),
            }
        }
        "$eq" => comparison_op("=", operands, eval_fn),
        "$ne" => comparison_op("!=", operands, eval_fn),
        "$gt" => comparison_op(">", operands, eval_fn),
        "$lt" => comparison_op("<", operands, eval_fn),
        "$gte" => comparison_op(">=", operands, eval_fn),
        "$lte" => comparison_op("<=", operands, eval_fn),
        "$contains" => {
            if operands.len() != 2 {
                return Err(HypatiaError::Eval(
                    "$contains expects exactly two arguments (field, value)".to_string(),
                ));
            }
            let field = expect_symbol(&operands[0])?;
            let value = expect_literal(&operands[1])?;
            let search_str = match &value {
                serde_json::Value::String(s) => s.clone(),
                other => other.to_string(),
            };
            Ok(OperatorResult::SqlCondition {
                fragment: format!("json_extract_string(content, '$.{field}') LIKE ?"),
                params: vec![serde_json::Value::String(format!("%{search_str}%"))],
            })
        }
        "$like" => {
            if operands.len() != 2 {
                return Err(HypatiaError::Eval(
                    "$like expects exactly two arguments (field, pattern)".to_string(),
                ));
            }
            let field = expect_symbol(&operands[0])?;
            let value = expect_literal(&operands[1])?;
            let pattern = match &value {
                serde_json::Value::String(s) => s.clone(),
                other => other.to_string(),
            };
            let sql_field = resolve_field_like(&field);
            Ok(OperatorResult::SqlCondition {
                fragment: format!("{sql_field} LIKE ?"),
                params: vec![serde_json::Value::String(pattern)],
            })
        }
        "$content" => {
            if operands.len() != 1 {
                return Err(HypatiaError::Eval(
                    "$content expects exactly one argument (a JSON object)".to_string(),
                ));
            }
            let map = match &operands[0] {
                AstNode::Object(m) => m,
                _ => {
                    return Err(HypatiaError::Eval(
                        "$content expects a JSON object".to_string(),
                    ))
                }
            };
            if map.is_empty() {
                return Ok(OperatorResult::SqlCondition {
                    fragment: "1=1".to_string(),
                    params: Vec::new(),
                });
            }
            let mut fragments = Vec::new();
            let mut params = Vec::new();
            for (key, val) in map {
                let str_val = match val {
                    serde_json::Value::String(s) => s.clone(),
                    other => other.to_string(),
                };
                fragments.push(format!("json_extract_string(content, '$.{key}') = ?"));
                params.push(serde_json::Value::String(str_val));
            }
            Ok(OperatorResult::SqlCondition {
                fragment: fragments.join(" AND "),
                params,
            })
        }
        "$search" => {
            let query = if operands.is_empty() {
                return Err(HypatiaError::Eval("$search expects a query argument".to_string()));
            } else {
                expect_literal(&operands[0])?
            };
            let query_str = match &query {
                serde_json::Value::String(s) => s.clone(),
                other => other.to_string(),
            };
            Ok(OperatorResult::FtsQuery {
                query: query_str,
            })
        }
        "$similar" => {
            let query = if operands.is_empty() {
                return Err(HypatiaError::Eval("$similar expects a query argument".to_string()));
            } else {
                expect_literal(&operands[0])?
            };
            let query_str = match &query {
                serde_json::Value::String(s) => s.clone(),
                other => other.to_string(),
            };
            Ok(OperatorResult::VectorQuery {
                query_text: query_str,
            })
        }
        "$k-hop" => {
            if operands.len() != 3 {
                return Err(HypatiaError::Eval(
                    "$k-hop expects exactly 3 arguments (subject, predicate, depth)".to_string(),
                ));
            }
            // Arg 1: subject (required string)
            let subject_val = expect_literal(&operands[0])?;
            let subject = match &subject_val {
                serde_json::Value::String(s) => s.clone(),
                _ => {
                    return Err(HypatiaError::Eval(
                        "$k-hop subject must be a string".to_string(),
                    ));
                }
            };
            // Arg 2: predicate (string or "$*" for any)
            let predicate = match &operands[1] {
                AstNode::Symbol(s) if s == "$*" => None,
                AstNode::Literal(serde_json::Value::String(s)) if s == "$*" => None,
                other => {
                    let val = expect_literal(other)?;
                    match val {
                        serde_json::Value::String(s) => Some(s),
                        _ => {
                            return Err(HypatiaError::Eval(
                                "$k-hop predicate must be a string or $*".to_string(),
                            ));
                        }
                    }
                }
            };
            // Arg 3: depth (positive integer)
            let depth_val = expect_literal(&operands[2])?;
            let depth = match &depth_val {
                serde_json::Value::Number(n) => {
                    let d = n.as_i64().ok_or_else(|| HypatiaError::Eval(
                        "$k-hop depth must be an integer".to_string(),
                    ))?;
                    if d <= 0 {
                        return Err(HypatiaError::Eval(
                            "$k-hop depth must be a positive integer".to_string(),
                        ));
                    }
                    d
                }
                _ => {
                    return Err(HypatiaError::Eval(
                        "$k-hop depth must be a positive integer".to_string(),
                    ));
                }
            };
            Ok(OperatorResult::KHop { subject, predicate, depth })
        }
        "$quote" => {
            if operands.len() != 1 {
                return Err(HypatiaError::Eval("$quote expects exactly one argument".to_string()));
            }
            // Return the unevaluated operand as a literal value
            Ok(OperatorResult::Value(ast_to_value(&operands[0])))
        }
        "$triple" => {
            if operands.len() != 3 {
                return Err(HypatiaError::Eval(
                    "$triple expects exactly 3 arguments (subject, predicate, object)".to_string(),
                ));
            }
            // Parse each operand: "$*" means wildcard (None), otherwise exact match
            let patterns: Vec<Option<String>> = operands.iter().map(|op| {
                match op {
                    AstNode::Symbol(s) if s == "$*" => Ok(None),
                    AstNode::Literal(serde_json::Value::String(s)) if s == "$*" => Ok(None),
                    other => {
                        let val = expect_literal(other)?;
                        match val {
                            serde_json::Value::String(s) => Ok(Some(s)),
                            _ => Err(HypatiaError::Eval(
                                "$triple arguments must be strings or $*".to_string(),
                            )),
                        }
                    }
                }
            }).collect::<Result<Vec<_>>>()?;

            // Error: all wildcards is a no-op
            if patterns.iter().all(|p| p.is_none()) {
                return Err(HypatiaError::Eval(
                    "$triple requires at least one non-wildcard argument".to_string(),
                ));
            }

            // Optimization: if all 3 specified, use triple = ? (PK lookup)
            if patterns.iter().all(|p| p.is_some()) {
                let s = patterns[0].as_ref().unwrap();
                let p = patterns[1].as_ref().unwrap();
                let o = patterns[2].as_ref().unwrap();
                let key = crate::model::StatementKey::new(s, p, o);
                return Ok(OperatorResult::SqlCondition {
                    fragment: "triple = ?".to_string(),
                    params: vec![serde_json::Value::String(key.to_csv_key())],
                });
            }

            // Partial match: generate conditions on individual columns
            let columns = ["subject", "predicate", "object"];
            let mut fragments = Vec::new();
            let mut params = Vec::new();
            for (i, pattern) in patterns.iter().enumerate() {
                if let Some(value) = pattern {
                    fragments.push(format!("{} = ?", columns[i]));
                    params.push(serde_json::Value::String(value.clone()));
                }
            }
            if fragments.is_empty() {
                Ok(OperatorResult::SqlCondition {
                    fragment: "1=1".to_string(),
                    params: Vec::new(),
                })
            } else {
                Ok(OperatorResult::SqlCondition {
                    fragment: fragments.join(" AND "),
                    params,
                })
            }
        }
        _ => Err(HypatiaError::Eval(format!("unknown operator: {operator}"))),
    }
}

/// Handle comparison operators: $eq, $ne, $gt, $lt, $gte, $lte
fn comparison_op(
    op: &str,
    operands: &[AstNode],
    eval_fn: &dyn Fn(&AstNode) -> Result<OperatorResult>,
) -> Result<OperatorResult> {
    if operands.len() == 2 {
        // Two-argument form: ["$eq", "field", "value"]
        let field = expect_symbol(&operands[0])?;
        let value = expect_literal(&operands[1])?;
        let sql_field = resolve_field(&field);
        Ok(OperatorResult::SqlCondition {
            fragment: format!("{sql_field} {op} ?"),
            params: vec![value],
        })
    } else if operands.len() == 1 {
        // Single argument: the operand should already be a condition
        eval_fn(&operands[0])
    } else {
        Err(HypatiaError::Eval(format!(
            "comparison operator expects 1 or 2 arguments, got {}", operands.len()
        )))
    }
}

/// Extract a symbol name from an AST node.
fn expect_symbol(node: &AstNode) -> Result<String> {
    match node {
        AstNode::Symbol(s) => Ok(s.clone()),
        AstNode::Literal(serde_json::Value::String(s)) => Ok(s.clone()),
        _ => Err(HypatiaError::Eval(format!(
            "expected symbol or string, got {:?}", node
        ))),
    }
}

/// Extract a literal value from an AST node.
fn expect_literal(node: &AstNode) -> Result<serde_json::Value> {
    match node {
        AstNode::Literal(v) => Ok(v.clone()),
        AstNode::Symbol(s) => Ok(serde_json::Value::String(s.clone())),
        _ => Ok(ast_to_value(node)),
    }
}

/// Resolve a field name to its SQL column reference.
fn resolve_field(field: &str) -> String {
    let field = field.trim_start_matches('$');
    match field {
        "subject" | "predicate" | "object" | "triple" | "name" | "created_at" | "tr_start" | "tr_end" => {
            field.to_string()
        }
        // Assume it's a JSON content field
        _ => format!("json_extract_string(content, '$.{field}')"),
    }
}

/// Resolve a field name for LIKE operations (requires VARCHAR).
/// Timestamp columns are CAST to VARCHAR since DuckDB's LIKE doesn't accept TIMESTAMP.
fn resolve_field_like(field: &str) -> String {
    let field = field.trim_start_matches('$');
    match field {
        "created_at" | "tr_start" | "tr_end" => {
            format!("CAST({field} AS VARCHAR)")
        }
        "subject" | "predicate" | "object" | "triple" | "name" => {
            field.to_string()
        }
        _ => format!("json_extract_string(content, '$.{field}')"),
    }
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
        AstNode::Quote(inner) => ast_to_value(inner),
        AstNode::Operator { operator, operands, metadata } => {
            let mut arr = vec![serde_json::Value::String(operator.clone())];
            arr.extend(operands.iter().map(ast_to_value));
            if metadata.is_empty() {
                serde_json::Value::Array(arr)
            } else {
                // Merge with metadata
                let mut obj = metadata.clone();
                obj.insert(operator.clone(), serde_json::Value::Array(arr[1..].to_vec()));
                serde_json::Value::Object(obj)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn make_eval(ops: &[(&str, Vec<AstNode>)]) -> Box<dyn Fn(&AstNode) -> Result<OperatorResult>> {
        // Simple eval that matches operator patterns
        let pairs: Vec<(String, Vec<AstNode>)> = ops
            .iter()
            .map(|(op, args)| (op.to_string(), args.clone()))
            .collect();
        Box::new(move |node: &AstNode| {
            match node {
                AstNode::Operator { operator, operands, .. } => {
                    evaluate_operator(
                        operator,
                        operands,
                        &serde_json::Map::new(),
                        &|n| {
                            // Recursive eval for nested operators
                            match n {
                                AstNode::Operator { operator, operands, .. } => {
                                    evaluate_operator(operator, operands, &serde_json::Map::new(), &|_| {
                                        Err(HypatiaError::Eval("unexpected deep nesting".to_string()))
                                    })
                                }
                                _ => Err(HypatiaError::Eval("expected operator".to_string())),
                            }
                        },
                    )
                }
                _ => Err(HypatiaError::Eval("expected operator".to_string())),
            }
        })
    }

    #[test]
    fn eq_operator() {
        let eval = make_eval(&[]);
        let result = evaluate_operator(
            "$eq",
            &[AstNode::Symbol("$name".to_string()), AstNode::Literal(json!("Alice"))],
            &serde_json::Map::new(),
            &|_| Err(HypatiaError::Eval("should not recurse".to_string())),
        ).unwrap();
        match result {
            OperatorResult::SqlCondition { fragment, params } => {
                assert!(fragment.contains("="));
                assert_eq!(params.len(), 1);
            }
            _ => panic!("expected SqlCondition"),
        }
    }

    #[test]
    fn and_operator() {
        let result = evaluate_operator(
            "$and",
            &[
                AstNode::Operator {
                    operator: "$eq".to_string(),
                    operands: vec![AstNode::Symbol("$name".to_string()), AstNode::Literal(json!("test"))],
                    metadata: serde_json::Map::new(),
                },
                AstNode::Operator {
                    operator: "$gt".to_string(),
                    operands: vec![AstNode::Symbol("$age".to_string()), AstNode::Literal(json!(18))],
                    metadata: serde_json::Map::new(),
                },
            ],
            &serde_json::Map::new(),
            &|node: &AstNode| {
                match node {
                    AstNode::Operator { operator, operands, .. } => {
                        evaluate_operator(operator, operands, &serde_json::Map::new(), &|_| {
                            Err(HypatiaError::Eval("no deeper nesting".to_string()))
                        })
                    }
                    _ => Err(HypatiaError::Eval("expected operator".to_string())),
                }
            },
        ).unwrap();
        match result {
            OperatorResult::SqlCondition { fragment, params } => {
                assert!(fragment.contains("AND"));
                assert_eq!(params.len(), 2);
            }
            _ => panic!("expected SqlCondition"),
        }
    }

    #[test]
    fn search_operator() {
        let result = evaluate_operator(
            "$search",
            &[AstNode::Literal(json!("hello world"))],
            &serde_json::Map::new(),
            &|_| Err(HypatiaError::Eval("should not recurse".to_string())),
        ).unwrap();
        match result {
            OperatorResult::FtsQuery { query } => {
                assert_eq!(query, "hello world");
            }
            _ => panic!("expected FtsQuery"),
        }
    }

    #[test]
    fn contains_operator() {
        let result = evaluate_operator(
            "$contains",
            &[AstNode::Symbol("$tags".to_string()), AstNode::Literal(json!("rust"))],
            &serde_json::Map::new(),
            &|_| Err(HypatiaError::Eval("should not recurse".to_string())),
        ).unwrap();
        match result {
            OperatorResult::SqlCondition { fragment, params } => {
                assert!(fragment.contains("json_extract_string"));
                assert!(fragment.contains("LIKE"));
                assert_eq!(params[0], json!("%rust%"));
            }
            _ => panic!("expected SqlCondition"),
        }
    }

    #[test]
    fn not_operator() {
        let result = evaluate_operator(
            "$not",
            &[AstNode::Operator {
                operator: "$eq".to_string(),
                operands: vec![AstNode::Symbol("$name".to_string()), AstNode::Literal(json!("test"))],
                metadata: serde_json::Map::new(),
            }],
            &serde_json::Map::new(),
            &|node: &AstNode| {
                match node {
                    AstNode::Operator { operator, operands, .. } => {
                        evaluate_operator(operator, operands, &serde_json::Map::new(), &|_| {
                            Err(HypatiaError::Eval("no deeper".to_string()))
                        })
                    }
                    _ => Err(HypatiaError::Eval("expected operator".to_string())),
                }
            },
        ).unwrap();
        match result {
            OperatorResult::SqlCondition { fragment, .. } => {
                assert!(fragment.starts_with("NOT ("));
            }
            _ => panic!("expected SqlCondition"),
        }
    }

    #[test]
    fn like_operator() {
        let result = evaluate_operator(
            "$like",
            &[AstNode::Symbol("$name".to_string()), AstNode::Literal(json!("rust%"))],
            &serde_json::Map::new(),
            &|_| Err(HypatiaError::Eval("should not recurse".to_string())),
        ).unwrap();
        match result {
            OperatorResult::SqlCondition { fragment, params } => {
                assert!(fragment.contains("LIKE"));
                assert_eq!(params[0], json!("rust%"));
            }
            _ => panic!("expected SqlCondition"),
        }
    }

    #[test]
    fn like_operator_json_field() {
        let result = evaluate_operator(
            "$like",
            &[AstNode::Symbol("$data".to_string()), AstNode::Literal(json!("%language%"))],
            &serde_json::Map::new(),
            &|_| Err(HypatiaError::Eval("should not recurse".to_string())),
        ).unwrap();
        match result {
            OperatorResult::SqlCondition { fragment, params } => {
                assert!(fragment.contains("json_extract_string"));
                assert!(fragment.contains("LIKE"));
                assert_eq!(params[0], json!("%language%"));
            }
            _ => panic!("expected SqlCondition"),
        }
    }

    #[test]
    fn content_operator_single_field() {
        let mut map = serde_json::Map::new();
        map.insert("format".to_string(), json!("json"));
        let result = evaluate_operator(
            "$content",
            &[AstNode::Object(map)],
            &serde_json::Map::new(),
            &|_| Err(HypatiaError::Eval("should not recurse".to_string())),
        ).unwrap();
        match result {
            OperatorResult::SqlCondition { fragment, params } => {
                assert!(fragment.contains("json_extract_string(content, '$.format') = ?"));
                assert_eq!(params.len(), 1);
                assert_eq!(params[0], json!("json"));
            }
            _ => panic!("expected SqlCondition"),
        }
    }

    #[test]
    fn content_operator_multiple_fields() {
        let mut map = serde_json::Map::new();
        map.insert("format".to_string(), json!("markdown"));
        map.insert("data".to_string(), json!("hello"));
        let result = evaluate_operator(
            "$content",
            &[AstNode::Object(map)],
            &serde_json::Map::new(),
            &|_| Err(HypatiaError::Eval("should not recurse".to_string())),
        ).unwrap();
        match result {
            OperatorResult::SqlCondition { fragment, params } => {
                assert!(fragment.contains(" AND "));
                assert_eq!(params.len(), 2);
            }
            _ => panic!("expected SqlCondition"),
        }
    }

    #[test]
    fn content_operator_empty_object() {
        let result = evaluate_operator(
            "$content",
            &[AstNode::Object(serde_json::Map::new())],
            &serde_json::Map::new(),
            &|_| Err(HypatiaError::Eval("should not recurse".to_string())),
        ).unwrap();
        match result {
            OperatorResult::SqlCondition { fragment, params } => {
                assert_eq!(fragment, "1=1");
                assert!(params.is_empty());
            }
            _ => panic!("expected SqlCondition"),
        }
    }
}
