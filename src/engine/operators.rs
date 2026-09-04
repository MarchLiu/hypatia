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

/// Columns that exist on both `knowledge` and `statement`, so a bare
/// reference to one is ambiguous inside a join over both tables.
const AMBIGUOUS_COLUMNS: &[&str] = &["content", "created_at", "tr_start", "tr_end"];

/// Query-target context: which table (catalog) and primary-key column the
/// generated json_index fragments must correlate against.
///
/// `catalog`, `table` and `pk` are the only identifiers interpolated into
/// generated SQL. They are `&'static str` chosen by a closed match over
/// `QueryTarget`, so no caller input can reach them.
#[derive(Debug, Clone, Copy)]
pub struct OpContext {
    pub catalog: &'static str,
    pub table: &'static str,
    pub pk: &'static str,
    /// Qualify ambiguous column references with `table.` — set when the
    /// fragment will be spliced into a join over more than one table.
    pub qualify_columns: bool,
}

impl OpContext {
    pub fn for_target(target: crate::model::QueryTarget) -> Self {
        match target {
            crate::model::QueryTarget::Knowledge => Self {
                catalog: "knowledge",
                table: "knowledge",
                pk: "name",
                qualify_columns: false,
            },
            crate::model::QueryTarget::Statement => Self {
                catalog: "statement",
                table: "statement",
                pk: "triple",
                qualify_columns: false,
            },
        }
    }

    /// Same target, but emitting `table.`-qualified column references for
    /// splicing into a join (see `$not-summaried`). Qualifying here, at
    /// construction time, is what keeps the caller from having to rewrite
    /// finished SQL by textual substitution.
    pub fn qualified(self) -> Self {
        Self { qualify_columns: true, ..self }
    }

    /// Render a column reference, qualifying it when it would be ambiguous.
    fn column(&self, name: &str) -> String {
        if self.qualify_columns && AMBIGUOUS_COLUMNS.contains(&name) {
            format!("{}.{name}", self.table)
        } else {
            name.to_string()
        }
    }
}

/// Fields stored as arrays in Content: `$contains` on them routes to exact
/// membership over json_index (transitional auto-routing, see plan §3.2).
const ARRAY_FIELDS: &[&str] = &["tags", "scopes", "figures", "synonyms"];

/// Correlated membership predicate: the document's json_index has (path, token).
fn membership_fragment(ctx: &OpContext, path: &str, token: &str) -> (String, Vec<serde_json::Value>) {
    (
        format!(
            "EXISTS (SELECT 1 FROM docs d JOIN json_index j ON j.doc_id = d.id \
             WHERE d.catalog = '{}' AND d.key = {}.{} AND j.path = ? AND j.value = ?)",
            ctx.catalog, ctx.table, ctx.pk
        ),
        vec![
            serde_json::Value::String(path.to_string()),
            serde_json::Value::String(token.to_string()),
        ],
    )
}

/// Depth-first first-scalar-leaf of a JSON value, for @> recall narrowing.
fn first_leaf(v: &serde_json::Value, path: &str) -> Option<(String, String)> {
    match v {
        serde_json::Value::Object(m) => m
            .iter()
            .find_map(|(k, rv)| {
                let child = if path.is_empty() {
                    k.to_string()
                } else {
                    format!("{path}.{k}")
                };
                first_leaf(rv, &child)
            }),
        serde_json::Value::Array(e) => e.iter().find_map(|rv| first_leaf(rv, path)),
        other => crate::storage::json_index::scalar_token(other).map(|t| (path.to_string(), t)),
    }
}

/// Evaluate an operator AST node against its operands.
/// Returns the SQL contribution of this operator.
pub fn evaluate_operator(
    operator: &str,
    operands: &[AstNode],
    _metadata: &serde_json::Map<String, serde_json::Value>,
    ctx: &OpContext,
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
        "$eq" => comparison_op("=", operands, ctx, eval_fn),
        "$ne" => comparison_op("!=", operands, ctx, eval_fn),
        "$gt" => comparison_op(">", operands, ctx, eval_fn),
        "$lt" => comparison_op("<", operands, ctx, eval_fn),
        "$gte" => comparison_op(">=", operands, ctx, eval_fn),
        "$lte" => comparison_op("<=", operands, ctx, eval_fn),
        "$contains" => {
            if operands.len() != 2 {
                return Err(HypatiaError::Eval(
                    "$contains expects exactly two arguments (field, value)".to_string(),
                ));
            }
            let field = expect_symbol(&operands[0])?;
            let field = field.trim_start_matches('$');
            let value = expect_literal(&operands[1])?;
            let search_str = match &value {
                serde_json::Value::String(s) => s.clone(),
                other => other.to_string(),
            };
            // Transitional auto-routing (plan §3.2): array fields are exact
            // membership over json_index, not substring.
            if ARRAY_FIELDS.contains(&field) {
                let (fragment, params) = membership_fragment(ctx, field, &search_str);
                return Ok(OperatorResult::SqlCondition { fragment, params });
            }
            // $contains always reads a Content field, even when the name also
            // happens to be a column.
            let resolved = json_field(ctx, field)?;
            Ok(OperatorResult::SqlCondition {
                fragment: format!("{} LIKE ?", resolved.fragment),
                params: resolved
                    .with_values([serde_json::Value::String(format!("%{search_str}%"))]),
            })
        }
        "$has" => {
            // Exact membership (PG `?` / `?|`): ["$has", field, value].
            // Array value = any-of.
            if operands.len() != 2 {
                return Err(HypatiaError::Eval(
                    "$has expects exactly two arguments (field, value)".to_string(),
                ));
            }
            let field = expect_symbol(&operands[0])?;
            let value = expect_literal(&operands[1])?;
            let values: Vec<serde_json::Value> = match &value {
                serde_json::Value::Array(e) => e.clone(),
                other => vec![other.clone()],
            };
            let mut fragments = Vec::new();
            let mut params = Vec::new();
            for v in &values {
                let token = crate::storage::json_index::scalar_token(v).ok_or_else(|| {
                    HypatiaError::Eval("$has value must be a scalar or array of scalars".to_string())
                })?;
                let (f, mut p) = membership_fragment(ctx, &field, &token);
                params.append(&mut p);
                fragments.push(f);
            }
            Ok(OperatorResult::SqlCondition {
                fragment: format!("({})", fragments.join(" OR ")),
                params,
            })
        }
        "$json-contains" => {
            // Structural containment (PG `@>`): json_index recall + UDF recheck.
            if operands.len() != 1 {
                return Err(HypatiaError::Eval(
                    "$json-contains expects exactly one argument (a JSON value)".to_string(),
                ));
            }
            let rhs = expect_literal(&operands[0])?;
            let rhs_text = rhs.to_string();
            // NOTE: the indexed recall MUST precede the recheck UDF in the
            // conjunction — SQLite's planner mis-evaluates the correlated
            // EXISTS when the user-function conjunct comes first.
            let mut params = Vec::new();
            let mut recall = "1=1".to_string();
            if let Some((path, token)) = first_leaf(&rhs, "") {
                recall = format!(
                    "EXISTS (SELECT 1 FROM docs d JOIN json_index j ON j.doc_id = d.id \
                     WHERE d.catalog = ? AND d.key = {}.{} AND j.path = ? AND j.value = ?)",
                    ctx.table, ctx.pk
                );
                params.push(serde_json::Value::String(ctx.catalog.to_string()));
                params.push(serde_json::Value::String(path));
                params.push(serde_json::Value::String(token));
            }
            params.push(serde_json::Value::String(rhs_text));
            Ok(OperatorResult::SqlCondition {
                fragment: format!("{recall} AND json_contains({}, ?)", ctx.column("content")),
                params,
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
            let resolved = resolve_field(ctx, &field)?;
            Ok(OperatorResult::SqlCondition {
                fragment: format!("{} LIKE ?", resolved.fragment),
                params: resolved.with_values([serde_json::Value::String(pattern)]),
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
                let token = crate::storage::json_index::scalar_token(val)
                    .unwrap_or(str_val);
                let (f, mut p) = membership_fragment(ctx, key, &token);
                fragments.push(f);
                params.append(&mut p);
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
            let columns = ["head", "relation", "tail"];
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
    ctx: &OpContext,
    eval_fn: &dyn Fn(&AstNode) -> Result<OperatorResult>,
) -> Result<OperatorResult> {
    if operands.len() == 2 {
        // Two-argument form: ["$eq", "field", "value"]
        let field = expect_symbol(&operands[0])?;
        let value = expect_literal(&operands[1])?;
        let mut resolved = resolve_field(ctx, &field)?;
        // Ordering on JSON scalars compares numerically (plan decision #3):
        // json_extract returns JSON numbers already, but CAST makes intent
        // explicit; a numeric bound param keeps the comparison typed.
        let ordering = resolved.is_json && matches!(op, ">" | "<" | ">=" | "<=");
        if ordering {
            resolved.fragment = format!("CAST({} AS REAL)", resolved.fragment);
        }
        let value = if ordering && value.is_number() {
            serde_json::Value::Number(
                serde_json::Number::from_f64(value.as_f64().unwrap_or(0.0))
                    .unwrap_or(serde_json::Number::from(0)),
            )
        } else {
            value
        };
        Ok(OperatorResult::SqlCondition {
            fragment: format!("{} {op} ?", resolved.fragment),
            params: resolved.with_values([value]),
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

/// Field names that address a real table column rather than a Content
/// JSON path.
const COLUMN_FIELDS: &[&str] = &[
    "head", "relation", "tail", "triple", "name", "created_at", "tr_start", "tr_end",
];

/// A field reference rendered as SQL, plus the bindings its placeholders need.
#[derive(Debug)]
struct ResolvedField {
    /// SQL fragment. May contain `?` placeholders bound by `params`.
    fragment: String,
    /// True when the field resolved to a Content JSON path, not a column.
    is_json: bool,
    /// Bindings for `fragment`'s placeholders. These must be pushed ahead of
    /// the operator's own value parameters, because `fragment` is emitted to
    /// the left of the value's `?`.
    params: Vec<serde_json::Value>,
}

impl ResolvedField {
    /// Combine with the operator's value parameters, keeping placeholder order.
    fn with_values(self, values: impl IntoIterator<Item = serde_json::Value>) -> Vec<serde_json::Value> {
        let mut params = self.params;
        params.extend(values);
        params
    }
}

/// Validate a Content JSON path: `tags`, `meta.author`, `tags[0].name`.
///
/// The path is bound as a parameter rather than interpolated, so this check is
/// defense in depth. It exists so a malformed name fails loudly at the field
/// instead of producing a JSON path that silently matches nothing.
fn validate_field_path(field: &str) -> Result<()> {
    let invalid = || {
        HypatiaError::Validation(format!(
            "invalid field name {field:?}: expected a JSON path such as \
             \"tags\", \"meta.author\" or \"tags[0]\""
        ))
    };
    if field.is_empty() {
        return Err(invalid());
    }
    for segment in field.split('.') {
        let (name, mut subscripts) = match segment.find('[') {
            Some(i) => segment.split_at(i),
            None => (segment, ""),
        };
        if name.is_empty() || !name.chars().all(|c| c.is_alphanumeric() || c == '_' || c == '-') {
            return Err(invalid());
        }
        // Trailing array subscripts: `[0]`, `[0][1]`.
        while !subscripts.is_empty() {
            let Some(close) = subscripts.find(']') else {
                return Err(invalid());
            };
            let index = &subscripts[1..close];
            if index.is_empty() || !index.bytes().all(|b| b.is_ascii_digit()) {
                return Err(invalid());
            }
            subscripts = &subscripts[close + 1..];
            if !subscripts.is_empty() && !subscripts.starts_with('[') {
                return Err(invalid());
            }
        }
    }
    Ok(())
}

/// Address a Content JSON path, whatever the field is named.
///
/// The path is bound as a parameter — SQLite's `json_extract(content, ?)`
/// takes the path from a binding — so the field name never becomes SQL text.
fn json_field(ctx: &OpContext, field: &str) -> Result<ResolvedField> {
    let field = field.trim_start_matches('$');
    validate_field_path(field)?;
    Ok(ResolvedField {
        fragment: format!("json_extract({}, ?)", ctx.column("content")),
        is_json: true,
        params: vec![serde_json::Value::String(format!("$.{field}"))],
    })
}

/// Resolve a field name to a table column when it names one, and to a Content
/// JSON path otherwise.
fn resolve_field(ctx: &OpContext, field: &str) -> Result<ResolvedField> {
    let name = field.trim_start_matches('$');
    if COLUMN_FIELDS.contains(&name) {
        return Ok(ResolvedField {
            fragment: ctx.column(name),
            is_json: false,
            params: Vec::new(),
        });
    }
    json_field(ctx, field)
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
    use crate::model::QueryTarget;
    use serde_json::json;

    fn kctx() -> OpContext {
        OpContext::for_target(QueryTarget::Knowledge)
    }

    #[test]
    fn eq_operator() {
        let result = evaluate_operator(
            "$eq",
            &[AstNode::Symbol("$name".to_string()), AstNode::Literal(json!("Alice"))],
            &serde_json::Map::new(),
            &kctx(),
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
            &kctx(),
            &|node: &AstNode| {
                match node {
                    AstNode::Operator { operator, operands, .. } => {
                        evaluate_operator(operator, operands, &serde_json::Map::new(), &kctx(), &|_| {
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
                // `name` is a column (value only); `age` is a JSON field, so it
                // binds its path ahead of its value.
                assert_eq!(params, vec![json!("test"), json!("$.age"), json!(18.0)]);
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
            &kctx(),
            &|_| Err(HypatiaError::Eval("should not recurse".to_string())),
        ).unwrap();
        match result {
            OperatorResult::FtsQuery { query } => {
                assert_eq!(query, "hello world");
            }
            _ => panic!("expected FtsQuery"),
        }
    }

    /// `$tags` and `tags` name the same field: the leading `$` marks a symbol,
    /// it is not part of the name. `$contains` used to keep it, so the symbol
    /// form built the path `$.$tags` and never matched — and never routed to
    /// the array-membership branch either.
    #[test]
    fn contains_operator_array_field() {
        for field in ["$tags", "tags"] {
            let result = evaluate_operator(
                "$contains",
                &[AstNode::Symbol(field.to_string()), AstNode::Literal(json!("rust"))],
                &serde_json::Map::new(),
                &kctx(),
                &|_| Err(HypatiaError::Eval("should not recurse".to_string())),
            ).unwrap();
            match result {
                OperatorResult::SqlCondition { fragment, params } => {
                    assert!(fragment.contains("json_index"), "field {field}");
                    assert_eq!(params, vec![json!("tags"), json!("rust")], "field {field}");
                }
                _ => panic!("expected SqlCondition"),
            }
        }
    }

    #[test]
    fn contains_operator_scalar_field() {
        let result = evaluate_operator(
            "$contains",
            &[AstNode::Symbol("$data".to_string()), AstNode::Literal(json!("rust"))],
            &serde_json::Map::new(),
            &kctx(),
            &|_| Err(HypatiaError::Eval("should not recurse".to_string())),
        ).unwrap();
        match result {
            OperatorResult::SqlCondition { fragment, params } => {
                assert_eq!(fragment, "json_extract(content, ?) LIKE ?");
                assert_eq!(params, vec![json!("$.data"), json!("%rust%")]);
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
            &kctx(),
            &|node: &AstNode| {
                match node {
                    AstNode::Operator { operator, operands, .. } => {
                        evaluate_operator(operator, operands, &serde_json::Map::new(), &kctx(), &|_| {
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
            &kctx(),
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
            &kctx(),
            &|_| Err(HypatiaError::Eval("should not recurse".to_string())),
        ).unwrap();
        match result {
            OperatorResult::SqlCondition { fragment, params } => {
                assert_eq!(fragment, "json_extract(content, ?) LIKE ?");
                assert_eq!(params, vec![json!("$.data"), json!("%language%")]);
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
            &kctx(),
            &|_| Err(HypatiaError::Eval("should not recurse".to_string())),
        ).unwrap();
        match result {
            OperatorResult::SqlCondition { fragment, params } => {
                assert!(fragment.contains("EXISTS (SELECT 1 FROM docs d JOIN json_index j"));
                assert!(fragment.contains("knowledge.name"));
                assert_eq!(params.len(), 2); // path + value
                assert_eq!(params[0], json!("format"));
                assert_eq!(params[1], json!("json"));
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
            &kctx(),
            &|_| Err(HypatiaError::Eval("should not recurse".to_string())),
        ).unwrap();
        match result {
            OperatorResult::SqlCondition { fragment, params } => {
                assert!(fragment.contains(" AND "));
                assert_eq!(params.len(), 4); // 2 keys × (path + value)
            }
            _ => panic!("expected SqlCondition"),
        }
    }

    // ── Field-name injection ─────────────────────────────────────────
    //
    // Field names reach the engine straight from user JSE. They used to be
    // interpolated into the SQL string as `json_extract(content, '$.{field}')`,
    // so a single quote closed the literal and the rest of the name became
    // live SQL. They are now bound as a `json_extract(content, ?)` path
    // parameter, and validated on top of that.

    /// Payloads that escaped the JSON-path literal under the old `format!`.
    const INJECTION_PAYLOADS: &[&str] = &[
        "a'b",
        "a') OR 1=1 --",
        "x') OR 1=1 OR json_extract(content, '$.x",
        "a') UNION SELECT name, content, created_at FROM knowledge --",
        "a'); DROP TABLE knowledge; --",
        "a\"b",
        "",
        "a..b",
        "a b",
    ];

    fn field_operators() -> Vec<(&'static str, serde_json::Value)> {
        vec![
            ("$eq", json!("x")),
            ("$ne", json!("x")),
            ("$gt", json!(1)),
            ("$lt", json!(1)),
            ("$gte", json!(1)),
            ("$lte", json!(1)),
            ("$like", json!("x%")),
            ("$contains", json!("x")),
        ]
    }

    #[test]
    fn field_name_injection_is_rejected() {
        for (op, value) in field_operators() {
            for payload in INJECTION_PAYLOADS {
                let result = evaluate_operator(
                    op,
                    &[
                        AstNode::Symbol(format!("${payload}")),
                        AstNode::Literal(value.clone()),
                    ],
                    &serde_json::Map::new(),
                    &kctx(),
                    &|_| Err(HypatiaError::Eval("should not recurse".to_string())),
                );
                match result {
                    Err(HypatiaError::Validation(_)) => {}
                    other => panic!(
                        "{op} did not reject injected field name {payload:?}: {other:?}"
                    ),
                }
            }
        }
    }

    #[test]
    fn legitimate_field_names_still_resolve() {
        for field in [
            "data", "tags", "format", "meta.author", "a.b.c", "tags[0]",
            "rows[0][1]", "with_underscore", "with-dash", "字段", "f1",
        ] {
            resolve_field(&kctx(), field)
                .unwrap_or_else(|e| panic!("field {field:?} rejected: {e}"));
        }
    }

    #[test]
    fn column_fields_resolve_to_columns_not_json() {
        for field in COLUMN_FIELDS {
            let resolved = resolve_field(&kctx(), field).unwrap();
            assert!(!resolved.is_json, "field {field}");
            assert_eq!(resolved.fragment, **field);
            assert!(resolved.params.is_empty(), "field {field}");
        }
    }

    /// The path never becomes SQL text, so it cannot terminate the literal
    /// even if validation were bypassed.
    #[test]
    fn json_path_is_bound_not_interpolated() {
        let resolved = resolve_field(&kctx(), "$meta.author").unwrap();
        assert_eq!(resolved.fragment, "json_extract(content, ?)");
        assert_eq!(resolved.params, vec![json!("$.meta.author")]);
        assert!(!resolved.fragment.contains('\''));
    }

    #[test]
    fn ordering_comparison_casts_and_still_binds_the_path() {
        let result = evaluate_operator(
            "$gt",
            &[AstNode::Symbol("$age".to_string()), AstNode::Literal(json!(18))],
            &serde_json::Map::new(),
            &kctx(),
            &|_| Err(HypatiaError::Eval("should not recurse".to_string())),
        ).unwrap();
        match result {
            OperatorResult::SqlCondition { fragment, params } => {
                assert_eq!(fragment, "CAST(json_extract(content, ?) AS REAL) > ?");
                assert_eq!(params, vec![json!("$.age"), json!(18.0)]);
            }
            _ => panic!("expected SqlCondition"),
        }
    }

    /// `expect_symbol` also accepts a bare string, so the string form is the
    /// same entry point and must be validated identically.
    #[test]
    fn string_literal_field_names_are_validated_too() {
        let result = evaluate_operator(
            "$eq",
            &[
                AstNode::Literal(json!("a') OR 1=1 --")),
                AstNode::Literal(json!("x")),
            ],
            &serde_json::Map::new(),
            &kctx(),
            &|_| Err(HypatiaError::Eval("should not recurse".to_string())),
        );
        assert!(matches!(result, Err(HypatiaError::Validation(_))));
    }

    #[test]
    fn validation_error_names_the_offending_field() {
        let err = resolve_field(&kctx(), "a'b").unwrap_err();
        assert!(matches!(err, HypatiaError::Validation(_)));
        assert!(err.to_string().contains("invalid field name"), "{err}");
    }

    #[test]
    fn content_operator_empty_object() {
        let result = evaluate_operator(
            "$content",
            &[AstNode::Object(serde_json::Map::new())],
            &serde_json::Map::new(),
            &kctx(),
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
