//! PostgreSQL JSE compiler. Bindings are allocated while walking the AST, never
//! by rewriting SQLite SQL. Install schema_functions when creating a PG shelf.
use super::ast::AstNode;
use super::evaluator::{ast_to_value, extract_query_opts, query_opts_to_search_opts};
use super::operators::validate_field_path;
use crate::error::{HypatiaError, Result};
use crate::model::{QueryResult, QueryTarget, StatementKey};
use crate::storage::{Storage, json_index::scalar_token};
use serde_json::{Map, Value};

/// Quote an identifier, including embedded quotes. No search_path assumptions.
pub fn quote_identifier(name: &str) -> String {
    format!("\"{}\"", name.replace('"', "\"\""))
}

struct Compiler<'a> {
    schema: String,
    target: QueryTarget,
    store: &'a dyn Storage,
    params: Vec<Value>,
}

impl Compiler<'_> {
    fn bind(&mut self, value: Value, ty: &str) -> String {
        self.params.push(value);
        format!("${}::{}", self.params.len(), ty)
    }
    fn text(&mut self, value: impl Into<String>) -> String {
        self.bind(Value::String(value.into()), "text")
    }
    fn function(&self, name: &str) -> String {
        format!("{}.{}", self.schema, name)
    }
    fn table(&self, target: QueryTarget) -> String {
        format!("{}.{}", self.schema, quote_identifier(target.table_name()))
    }
    fn membership(&mut self, path: &str, token: &str) -> String {
        let path = self.text(path);
        let token = self.text(token);
        format!("COALESCE((q.tokens -> {path}) ? {token}, FALSE)")
    }
    // Return JSONB for JSON paths, and typed columns for fixed identifiers.
    fn field(&mut self, field: &str, content_only: bool) -> Result<(String, bool)> {
        let field = field.trim_start_matches('$');
        if !content_only
            && matches!(
                field,
                "name"
                    | "triple"
                    | "head"
                    | "relation"
                    | "tail"
                    | "created_at"
                    | "tr_start"
                    | "tr_end"
            )
        {
            return Ok((format!("q.{field}"), false));
        }
        validate_field_path(field)?;
        let (root, path) = if let Some(path) = field.strip_prefix("data.") {
            ("q.payload", path)
        } else if field.starts_with("data[") {
            ("q.payload", &field[4..])
        } else {
            ("q.content", field)
        };
        // Bound path segments preserve indexes and Unicode without constructing
        // a PostgreSQL array literal (or needing an array driver binding).
        let segments: Vec<_> = path
            .split(['.', '[', ']'])
            .filter(|s| !s.is_empty())
            .map(|s| self.text(s))
            .collect();
        Ok((
            format!("({root} #> ARRAY[{}]::text[])", segments.join(", ")),
            true,
        ))
    }
    fn scalar_text(&self, json: &str) -> String {
        // JSON1 extracts booleans as integers and null as SQL NULL.
        format!(
            "(CASE jsonb_typeof({json}) WHEN 'boolean' THEN CASE WHEN {json} = 'true'::jsonb THEN '1' ELSE '0' END WHEN 'object' THEN {}({json}) WHEN 'array' THEN {}({json}) ELSE {json} #>> '{{}}' END)",
            self.function("jse_compact"),
            self.function("jse_compact")
        )
    }
    fn condition(&mut self, node: &AstNode) -> Result<String> {
        let AstNode::Operator {
            operator, operands, ..
        } = node
        else {
            return Err(HypatiaError::Eval("expected SQL condition".into()));
        };
        let arity = |n| {
            if operands.len() == n {
                Ok(())
            } else {
                Err(HypatiaError::Eval(format!(
                    "{operator} expects {n} arguments"
                )))
            }
        };
        match operator.as_str() {
            "$knowledge" | "$statement" => {
                if operands.len() == 1 {
                    self.condition(&operands[0])
                } else {
                    Ok("TRUE".into())
                }
            }
            "$and" | "$or" => {
                let parts = operands
                    .iter()
                    .map(|n| self.condition(n))
                    .collect::<Result<Vec<_>>>()?;
                if parts.is_empty() {
                    return Ok("TRUE".into());
                }
                Ok(format!(
                    "({})",
                    parts.join(if operator == "$and" { " AND " } else { " OR " })
                ))
            }
            "$not" => {
                arity(1)?;
                Ok(format!("NOT ({})", self.condition(&operands[0])?))
            }
            "$eq" | "$ne" | "$gt" | "$lt" | "$gte" | "$lte" => {
                if operands.len() == 1 {
                    return self.condition(&operands[0]);
                }
                arity(2)?;
                let field = symbol(&operands[0])?;
                let value = ast_to_value(&operands[1]);
                let (mut lhs, json) = self.field(&field, false)?;
                let op = match operator.as_str() {
                    "$eq" => "=",
                    "$ne" => "!=",
                    "$gt" => ">",
                    "$lt" => "<",
                    "$gte" => ">=",
                    _ => "<=",
                };
                let ordering = json && !matches!(op, "=" | "!=");
                let rhs = if ordering {
                    lhs = format!(
                        "{}({})",
                        self.function("jse_numeric"),
                        self.scalar_text(&lhs)
                    );
                    // REAL affinity converts well-formed numeric strings. Other
                    // strings sort after every number under SQLite's type order.
                    let numeric = value.as_f64().or_else(|| {
                        value.as_str().and_then(|s| {
                            let s = s.trim();
                            if s.chars().all(|c| {
                                c.is_ascii_digit() || matches!(c, '+' | '-' | '.' | 'e' | 'E')
                            }) {
                                s.parse::<f64>().ok()
                            } else {
                                None
                            }
                        })
                    });
                    if let Some(n) = numeric {
                        if n.is_finite() {
                            self.bind(serde_json::json!(n), "double precision")
                        } else {
                            let p = self
                                .text(value.as_str().expect("only numeric strings may overflow"));
                            format!("{}({p})", self.function("jse_numeric"))
                        }
                    } else if value.is_null() {
                        self.bind(Value::Null, "double precision")
                    } else if let Some(b) = value.as_bool() {
                        self.bind(
                            serde_json::json!(if b { 1.0 } else { 0.0 }),
                            "double precision",
                        )
                    } else {
                        return Ok(format!(
                            "(CASE WHEN {lhs} IS NULL THEN NULL ELSE {} END)",
                            matches!(op, "<" | "<=")
                        ));
                    }
                } else if json {
                    if value.is_null() {
                        return Ok(format!("({lhs} = NULL::jsonb)"));
                    }
                    if value.is_number() || value.is_boolean() {
                        // Preserve numeric/bool equality while refusing string
                        // coercion: JSON1 has no column affinity for equality.
                        let text = self.scalar_text(&lhs);
                        // Equality must retain integer precision beyond 2^53.
                        // Only a JSON number/bool enters the numeric cast.
                        let p = self.text(if let Some(b) = value.as_bool() {
                            if b { "1".into() } else { "0".into() }
                        } else {
                            value.to_string()
                        });
                        return Ok(format!(
                            "(CASE WHEN {lhs} IS NULL OR {lhs} = 'null'::jsonb THEN NULL WHEN jsonb_typeof({lhs}) IN ('number', 'boolean') THEN ({text})::numeric {op} {p}::numeric ELSE {} END)",
                            op == "!="
                        ));
                    } else {
                        // A non-string JSON scalar never equals a text binding.
                        let text = self.scalar_text(&lhs);
                        let p = self.text(
                            value
                                .as_str()
                                .map(str::to_owned)
                                .unwrap_or_else(|| value.to_string()),
                        );
                        return Ok(format!(
                            "(CASE WHEN {lhs} IS NULL OR {lhs} = 'null'::jsonb THEN NULL WHEN jsonb_typeof({lhs}) IN ('number', 'boolean') THEN {} ELSE {text} {op} {p} END)",
                            op == "!="
                        ));
                    }
                } else {
                    // Real columns have TEXT affinity in SQLite. In particular,
                    // a bound bool is stored as integer 0/1 before that affinity.
                    let value = match value {
                        Value::Null | Value::String(_) => value,
                        Value::Bool(b) => Value::String(if b { "1" } else { "0" }.into()),
                        other => Value::String(other.to_string()),
                    };
                    let p = self.bind(value, "text");
                    if matches!(
                        field.trim_start_matches('$'),
                        "created_at" | "tr_start" | "tr_end"
                    ) {
                        format!("{p}::timestamp")
                    } else {
                        p
                    }
                };
                Ok(format!("{lhs} {op} {rhs}"))
            }
            "$contains" | "$like" => {
                arity(2)?;
                let field = symbol(&operands[0])?;
                let v = ast_to_value(&operands[1]);
                let text = v
                    .as_str()
                    .map(str::to_owned)
                    .unwrap_or_else(|| v.to_string());
                if operator == "$contains"
                    && matches!(
                        field.trim_start_matches('$'),
                        "tags" | "scopes" | "figures" | "synonyms"
                    )
                {
                    return Ok(self.membership(field.trim_start_matches('$'), &text));
                }
                let (lhs, json) = self.field(&field, operator == "$contains")?;
                let lhs = if json {
                    self.scalar_text(&lhs)
                } else {
                    format!("({lhs})::text")
                };
                let p = self.text(if operator == "$contains" {
                    format!("%{text}%")
                } else {
                    text
                });
                // SQLite LIKE folds ASCII only; PostgreSQL ILIKE folds according
                // to locale. Disable PG's default backslash escape as well.
                Ok(format!(
                    "translate({lhs}, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz') LIKE translate({p}, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz') ESCAPE ''"
                ))
            }
            "$has" => {
                arity(2)?;
                let field = symbol(&operands[0])?;
                let v = ast_to_value(&operands[1]);
                let values = if let Value::Array(a) = v { a } else { vec![v] };
                let mut parts = Vec::new();
                for v in values {
                    let token = scalar_token(&v).ok_or_else(|| {
                        HypatiaError::Eval("$has value must be a scalar or array of scalars".into())
                    })?;
                    parts.push(self.membership(&field, &token));
                }
                Ok(if parts.is_empty() {
                    "FALSE".into()
                } else {
                    format!("({})", parts.join(" OR "))
                })
            }
            "$content" => {
                arity(1)?;
                let AstNode::Object(map) = &operands[0] else {
                    return Err(HypatiaError::Eval("$content expects a JSON object".into()));
                };
                let parts: Vec<_> = map
                    .iter()
                    .map(|(k, v)| {
                        self.membership(k, &scalar_token(v).unwrap_or_else(|| v.to_string()))
                    })
                    .collect();
                Ok(if parts.is_empty() {
                    "TRUE".into()
                } else {
                    format!("({})", parts.join(" AND "))
                })
            }
            "$json-contains" => {
                arity(1)?;
                let p = self.text(ast_to_value(&operands[0]).to_string());
                // @> provides a GIN-eligible necessary condition. The recursive
                // recheck disallows PG's array-contains-scalar special case.
                Ok(format!(
                    "(q.content @> {p}::jsonb AND {}(q.content, {p}::jsonb))",
                    self.function("jse_json_contains")
                ))
            }
            "$triple" => {
                arity(3)?;
                let parts = operands.iter().map(symbol).collect::<Result<Vec<_>>>()?;
                if parts.iter().all(|s| s == "$*") {
                    return Err(HypatiaError::Eval(
                        "$triple requires at least one non-wildcard argument".into(),
                    ));
                }
                if parts.iter().all(|s| s != "$*") {
                    let key = StatementKey::new(&parts[0], &parts[1], &parts[2]);
                    return Ok(format!("q.triple = {}", self.text(key.to_csv_key())));
                }
                let mut conditions = Vec::new();
                for (col, v) in ["head", "relation", "tail"].iter().zip(parts) {
                    if v != "$*" {
                        conditions.push(format!("q.{col} = {}", self.text(v)));
                    }
                }
                Ok(format!("({})", conditions.join(" AND ")))
            }
            _ => Err(HypatiaError::Eval(format!(
                "{operator} is not a SQL condition"
            ))),
        }
    }
    fn operand(
        &mut self,
        node: &AstNode,
        metadata: &Map<String, Value>,
        not_summaried: bool,
    ) -> Result<Option<String>> {
        let AstNode::Operator {
            operator, operands, ..
        } = node
        else {
            return match node {
                AstNode::Quote(_) | AstNode::Literal(_) => Ok(None),
                _ => Err(HypatiaError::Eval(
                    "unexpected node in condition context".into(),
                )),
            };
        };
        let opts = query_opts_to_search_opts(&extract_query_opts(metadata), self.target);
        let (result, key) = match operator.as_str() {
            "$search" | "$similar" => {
                let n = operands.first().ok_or_else(|| {
                    HypatiaError::Eval(format!("{operator} expects a query argument"))
                })?;
                let v = ast_to_value(n);
                let text = v
                    .as_str()
                    .map(str::to_owned)
                    .unwrap_or_else(|| v.to_string());
                if operator == "$search" {
                    (self.store.execute_search(&text, &opts)?, "key")
                } else {
                    (
                        self.store.execute_similar(&text, &opts, self.target)?,
                        self.target.key_column(),
                    )
                }
            }
            "$k-hop" => {
                if not_summaried || self.target != QueryTarget::Statement {
                    return Err(HypatiaError::Eval(
                        "$k-hop is only valid inside $statement".into(),
                    ));
                }
                let ctx = super::operators::OpContext::for_target(self.target);
                let r = super::operators::evaluate_operator(
                    operator,
                    operands,
                    metadata,
                    &ctx,
                    &|_| Err(HypatiaError::Eval("unexpected condition".into())),
                )?;
                let super::operators::OperatorResult::KHop {
                    subject,
                    predicate,
                    depth,
                } = r
                else {
                    unreachable!()
                };
                (
                    self.store
                        .execute_khop(&subject, predicate.as_deref(), depth)?,
                    "triple",
                )
            }
            _ => return self.condition(node).map(Some),
        };
        let mut keys = Vec::new();
        for row in result.rows {
            if let Some(k) = row.get(key).and_then(Value::as_str) {
                keys.push(self.text(k));
            }
        }
        Ok(Some(if keys.is_empty() {
            "FALSE".into()
        } else {
            format!("q.{} IN ({})", self.target.key_column(), keys.join(", "))
        }))
    }
}

fn symbol(node: &AstNode) -> Result<String> {
    match node {
        AstNode::Symbol(s) | AstNode::Literal(Value::String(s)) => Ok(s.clone()),
        _ => Err(HypatiaError::Eval("expected symbol or string".into())),
    }
}

pub(super) fn execute(ast: &AstNode, store: &dyn Storage) -> Result<QueryResult> {
    let AstNode::Operator {
        operator,
        operands,
        metadata,
    } = ast
    else {
        return Err(HypatiaError::Eval(
            "top-level expression must be a query operator".into(),
        ));
    };
    let target = match operator.as_str() {
        "$knowledge" | "$not-summaried" => QueryTarget::Knowledge,
        "$statement" => QueryTarget::Statement,
        _ => {
            return Err(HypatiaError::Eval(format!(
                "invalid top-level operator {operator}"
            )));
        }
    };
    let schema = store
        .sql_schema()
        .filter(|s| !s.is_empty() && !s.contains('\0'))
        .ok_or_else(|| {
            HypatiaError::Config("PostgreSQL storage requires an explicit SQL schema".into())
        })?;
    let mut c = Compiler {
        schema: quote_identifier(schema),
        target,
        store,
        params: Vec::new(),
    };
    let mut conditions = Vec::new();
    let not_summaried = operator == "$not-summaried";
    let rest = if not_summaried {
        let tag = operands
            .first()
            .ok_or_else(|| HypatiaError::Eval("$not-summaried expects a tag argument".into()))?;
        conditions.push(c.membership("tags", &symbol(tag)?));
        conditions.push(format!(
            "NOT EXISTS (SELECT 1 FROM {} AS s WHERE s.tail = q.name AND s.relation = 'summary')",
            c.table(QueryTarget::Statement)
        ));
        &operands[1..]
    } else {
        operands.as_slice()
    };
    for n in rest {
        if let Some(condition) = c.operand(n, metadata, not_summaried)? {
            conditions.push(condition);
        }
    }
    let select = match target {
        QueryTarget::Knowledge => "q.name, q.content, q.created_at",
        QueryTarget::Statement => {
            "q.triple, q.head, q.relation, q.tail, q.content, q.created_at, q.tr_start, q.tr_end"
        }
    };
    let mut sql = format!("SELECT {select} FROM {} AS q", c.table(target));
    if !conditions.is_empty() {
        sql.push_str(&format!(" WHERE {}", conditions.join(" AND ")));
    }
    sql.push_str(if not_summaried {
        " ORDER BY q.created_at ASC"
    } else {
        " ORDER BY q.created_at DESC"
    });
    let opts = extract_query_opts(metadata);
    let limit = c.bind(Value::from(opts.limit), "bigint");
    let offset = c.bind(Value::from(opts.offset), "bigint");
    // SQLite LIMIT -1 means unlimited, and negative OFFSET means zero.
    sql.push_str(&format!(
        " LIMIT CASE WHEN {limit} < 0 THEN NULL ELSE {limit} END OFFSET GREATEST({offset}, 0)"
    ));
    store.execute_query(target, &sql, c.params)
}

/// Canonical membership tokens derived before JSONB normalizes numeric text.
/// Store this value in each row's tokens JSONB column, in the same transaction
/// as content and payload. The existing postings walk defines path depth,
/// opaque containers, null tokens, data parsing, and numeric spelling.
pub fn membership_tokens(content_json: &str) -> Value {
    let mut paths = Map::new();
    for posting in crate::storage::json_index::content_postings(content_json) {
        if let Some(token) = posting.value {
            let values = paths
                .entry(posting.path)
                .or_insert_with(|| Value::Array(Vec::new()));
            let values = values.as_array_mut().expect("token map contains arrays");
            let value = Value::String(token);
            if !values.contains(&value) {
                values.push(value);
            }
        }
    }
    Value::Object(paths)
}

/// Shelf-local compatibility helpers. All are immutable and use only their
/// arguments; content/payload stay JSONB and may retain native GIN indexes.
/// The store owns applying this DDL transactionally with its schema migration.
pub fn schema_functions(schema: &str) -> String {
    let s = quote_identifier(schema);
    // A schema name is data even inside a dollar-quoted SQL function body.
    let mut tag = "$jse$".to_string();
    while schema.contains(&tag) {
        tag.insert(1, 'x');
    }
    format!(
        r#"
CREATE OR REPLACE FUNCTION {s}.jse_compact(v jsonb) RETURNS text
LANGUAGE plpgsql IMMUTABLE STRICT AS {tag}
DECLARE result text;
BEGIN
  CASE jsonb_typeof(v)
    WHEN 'object' THEN
      SELECT '{{' || COALESCE(string_agg(to_jsonb(key)::text || ':' || {s}.jse_compact(value), ',' ORDER BY key COLLATE "C"), '') || '}}'
      INTO result FROM jsonb_each(v);
    WHEN 'array' THEN
      SELECT '[' || COALESCE(string_agg({s}.jse_compact(value), ',' ORDER BY ord), '') || ']'
      INTO result FROM jsonb_array_elements(v) WITH ORDINALITY AS a(value, ord);
    ELSE result := v::text;
  END CASE;
  RETURN result;
END {tag};

CREATE OR REPLACE FUNCTION {s}.jse_numeric(v text) RETURNS double precision
LANGUAGE plpgsql IMMUTABLE STRICT AS {tag}
DECLARE prefix text;
BEGIN
  prefix := substring(v FROM '^[[:space:]]*([+-]?([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?)');
  IF prefix IS NULL THEN RETURN 0; END IF;
  BEGIN
    RETURN prefix::double precision;
  EXCEPTION WHEN numeric_value_out_of_range THEN
    -- SQLite REAL conversion saturates overflow and underflows to zero.
    BEGIN
      IF abs(prefix::numeric) < 1 THEN RETURN 0; END IF;
    EXCEPTION WHEN numeric_value_out_of_range THEN
      IF prefix ~ '[eE]-[0-9]+$' THEN RETURN 0; END IF;
    END;
    IF ltrim(prefix) LIKE '-%' THEN RETURN '-Infinity'::double precision; END IF;
    RETURN 'Infinity'::double precision;
  END;
END {tag};

CREATE OR REPLACE FUNCTION {s}.jse_json_contains(lhs jsonb, rhs jsonb) RETURNS boolean
LANGUAGE plpgsql IMMUTABLE AS {tag}
DECLARE item record;
BEGIN
  IF lhs IS NULL OR rhs IS NULL THEN RETURN false; END IF;
  IF jsonb_typeof(lhs) <> jsonb_typeof(rhs) THEN RETURN false; END IF;
  IF jsonb_typeof(rhs) = 'object' THEN
    FOR item IN SELECT key, value FROM jsonb_each(rhs) LOOP
      IF NOT {s}.jse_json_contains(lhs -> item.key, item.value) THEN RETURN false; END IF;
    END LOOP;
    RETURN true;
  ELSIF jsonb_typeof(rhs) = 'array' THEN
    FOR item IN SELECT value FROM jsonb_array_elements(rhs) LOOP
      IF NOT EXISTS (SELECT 1 FROM jsonb_array_elements(lhs) AS a(value)
                     WHERE {s}.jse_json_contains(a.value, item.value)) THEN RETURN false; END IF;
    END LOOP;
    RETURN true;
  END IF;
  RETURN lhs = rhs;
END {tag};

"#
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::Evaluator;
    use crate::model::SearchOpts;
    use serde_json::json;
    use std::cell::RefCell;

    #[derive(Default)]
    struct RecordingStore {
        query: RefCell<(String, Vec<Value>)>,
        candidates: Vec<Map<String, Value>>,
        schema: Option<String>,
    }
    impl Storage for RecordingStore {
        fn sql_dialect(&self) -> super::super::SqlDialect {
            super::super::SqlDialect::Postgres
        }
        fn sql_schema(&self) -> Option<&str> {
            Some(self.schema.as_deref().unwrap_or("shelf\"quoted"))
        }
        fn execute_query(
            &self,
            _: QueryTarget,
            sql: &str,
            params: Vec<Value>,
        ) -> Result<QueryResult> {
            *self.query.borrow_mut() = (sql.into(), params);
            Ok(QueryResult::new(vec![]))
        }
        fn execute_search(&self, _: &str, _: &SearchOpts) -> Result<QueryResult> {
            Ok(QueryResult::new(self.candidates.clone()))
        }
        fn execute_similar(&self, _: &str, _: &SearchOpts, _: QueryTarget) -> Result<QueryResult> {
            Ok(QueryResult::new(self.candidates.clone()))
        }
        fn execute_khop(&self, _: &str, _: Option<&str>, _: i64) -> Result<QueryResult> {
            Ok(QueryResult::new(self.candidates.clone()))
        }
    }

    #[test]
    fn nested_bindings_payload_paths_and_injection_are_separate() {
        let store = RecordingStore::default();
        let poison = "x' OR TRUE -- ? $1";
        Evaluator::execute(
            &json!([
                "$knowledge",
                [
                    "$and",
                    ["$eq", "name", poison],
                    [
                        "$or",
                        ["$gt", "data.score", 2],
                        ["$has", "tags", [true, null, "a"]]
                    ]
                ]
            ]),
            &store,
        )
        .unwrap();
        let q = store.query.borrow();
        assert!(q.0.contains("FROM \"shelf\"\"quoted\".\"knowledge\" AS q"));
        assert!(q.0.contains("q.payload #> ARRAY[$2::text]::text[]"));
        assert!(q.0.contains("jse_numeric("));
        assert!(!q.0.contains(poison));
        assert_eq!(
            q.1,
            vec![
                json!(poison),
                json!("score"),
                json!(2.0),
                json!("tags"),
                json!("true"),
                json!("tags"),
                json!(""),
                json!("tags"),
                json!("a"),
                json!(100),
                json!(0)
            ]
        );
        assert!(q.0.contains("$10::bigint"));
        assert!(q.0.contains("$11::bigint"));
        assert!(!q.0.contains("json_extract"));
    }

    #[test]
    fn candidate_keys_use_each_operator_contract_and_statement_pk() {
        let row = json!({"key":"fts", "triple":"graph", "name":"vector"})
            .as_object()
            .unwrap()
            .clone();
        let store = RecordingStore {
            candidates: vec![row],
            ..Default::default()
        };
        Evaluator::execute(
            &json!([
                "$statement",
                ["$search", "x"],
                ["$similar", "x"],
                ["$k-hop", "a", "r", 2]
            ]),
            &store,
        )
        .unwrap();
        let q = store.query.borrow();
        assert_eq!(&q.1[..3], &[json!("fts"), json!("graph"), json!("graph")]);
        assert!(q.0.contains("q.triple IN ($1::text)"));
        assert!(q.0.contains("q.triple IN ($2::text)"));
        assert!(q.0.contains("q.triple IN ($3::text)"));
        assert!(q.0.contains("ORDER BY q.created_at DESC"));
        drop(q);
        Evaluator::execute(&json!(["$knowledge", ["$similar", "x"]]), &store).unwrap();
        assert_eq!(store.query.borrow().1[0], json!("vector"));
    }

    #[test]
    fn not_summaried_qualifies_both_tables_and_keeps_binding_order() {
        let store = RecordingStore::default();
        Evaluator::execute(
            &json!([
                "$not-summaried",
                "message",
                ["$eq", "format", "text"],
                ["$has", "tags", "rust"]
            ]),
            &store,
        )
        .unwrap();
        let q = store.query.borrow();
        assert!(q.0.contains("NOT EXISTS (SELECT 1 FROM \"shelf\"\"quoted\".\"statement\" AS s"));
        assert!(q.0.contains("s.tail = q.name AND s.relation = 'summary'"));
        assert!(q.0.contains("ORDER BY q.created_at ASC"));
        assert_eq!(
            q.1,
            vec![
                json!("tags"),
                json!("message"),
                json!("format"),
                json!("text"),
                json!("tags"),
                json!("rust"),
                json!(100),
                json!(0)
            ]
        );
    }

    #[test]
    fn exact_integer_equality_and_candidate_pagination() {
        let store = RecordingStore::default();
        Evaluator::execute(&json!({"$statement":[["$triple","a,b","r","c"],["$gte","tr_start","2025-01-01"],["$eq","data.id",9007199254740993u64],["$search","none"]],"limit":-1,"offset":-2}),&store).unwrap();
        let q = store.query.borrow();
        assert_eq!(
            q.1,
            vec![
                json!("\"a,b\",r,c"),
                json!("2025-01-01"),
                json!("id"),
                json!("9007199254740993"),
                json!(-1),
                json!(-2)
            ]
        );
        assert!(q.0.contains("q.tr_start >= $2::text::timestamp"));
        assert!(q.0.contains("$4::text::numeric"));
        assert!(q.0.contains("AND FALSE"));
        assert!(q.0.contains("LIMIT CASE WHEN $5::bigint < 0 THEN NULL ELSE $5::bigint END OFFSET GREATEST($6::bigint, 0)"));
    }

    #[test]
    fn containment_is_native_prefilter_plus_strict_recheck() {
        let store = RecordingStore::default();
        Evaluator::execute(
            &json!(["$knowledge", ["$json-contains", {"tags":"rust"}], ["$has", "tags", []]]),
            &store,
        )
        .unwrap();
        let q = store.query.borrow();
        assert!(q.0.contains("q.content @> $1::text::jsonb"));
        assert!(q.0.contains("jse_json_contains(q.content, $1::text::jsonb)"));
        assert!(q.0.contains("AND FALSE"));
        assert_eq!(q.1[0], json!(r#"{"tags":"rust"}"#));
    }

    #[test]
    fn path_validation_and_like_escape_contract() {
        let store = RecordingStore::default();
        assert!(
            Evaluator::execute(
                &json!([
                    "$knowledge",
                    ["$eq", "data.x');DROP TABLE knowledge;--", "x"]
                ]),
                &store
            )
            .is_err()
        );
        Evaluator::execute(&json!(["$knowledge", ["$like", "data", "A\\_%"]]), &store).unwrap();
        let q = store.query.borrow();
        assert!(q.0.contains("translate("));
        assert!(q.0.contains("ESCAPE ''"));
        assert_eq!(q.1[1], json!("A\\_%"));
    }

    #[test]
    fn membership_preserves_canonical_scientific_tokens_and_opaque_values() {
        let content = json!({"tags":[1e-7,true,null,""], "data":r#"{"small":1e-7,"decimal":1.0,"opaque":{"v":1e-7}}"#});
        let tokens = membership_tokens(&content.to_string());
        assert_eq!(tokens["tags"], json!(["1e-7", "true", ""]));
        assert_eq!(tokens["data.small"], json!(["1e-7"]));
        assert_eq!(tokens["data.decimal"], json!(["1.0"]));
        assert_eq!(tokens["data.opaque"], json!([r#"{"v":1e-7}"#]));
        assert!(tokens.get("data").is_none());
        let store = RecordingStore::default();
        Evaluator::execute(
            &json!([
                "$knowledge",
                ["$has", "data.small", 1e-7],
                ["$not", ["$has", "missing", ""]]
            ]),
            &store,
        )
        .unwrap();
        let q = store.query.borrow();
        assert!(q.0.contains("COALESCE((q.tokens -> $1::text) ? $2::text, FALSE)"));
        assert!(q.0.contains("NOT ((COALESCE((q.tokens -> $3::text) ? $4::text, FALSE)))"));
        assert_eq!(q.1[1], json!("1e-7"));
    }

    #[test]
    fn schema_dollar_quoting_cannot_end_function_body() {
        let ddl = schema_functions("s$jse$\"; DROP SCHEMA public; --");
        assert!(ddl.contains("AS $xjse$"));
        assert!(ddl.contains("\"s$jse$\"\"; DROP SCHEMA public; --\".jse_compact"));
    }

    /// Run against an explicitly configured disposable PostgreSQL database.
    /// Every DDL/data change is rolled back, including the unique shelf schema.
    #[cfg(feature = "postgres-backend")]
    #[test]
    fn postgres_helpers_match_sqlite_numeric_tokens_and_containment() {
        let Ok(url) = std::env::var("HYPATIA_TEST_POSTGRES_URL") else {
            return;
        };
        let mut client = postgres::Client::connect(&url, postgres::NoTls).unwrap();
        let mut tx = client.transaction().unwrap();
        let schema = format!("jse_test_{}", std::process::id());
        let s = quote_identifier(&schema);
        tx.batch_execute(&format!("CREATE SCHEMA {s}; {}", schema_functions(&schema)))
            .unwrap();
        let sqlite = rusqlite::Connection::open_in_memory().unwrap();
        for text in [
            "garbage",
            "12.5tail",
            "  -2e3ignored",
            ".25",
            "1e",
            "true",
            "1e400",
            "1e-400",
            "NaN",
            "",
        ] {
            let expected: f64 = sqlite
                .query_row("SELECT CAST(? AS REAL)", [text], |r| r.get(0))
                .unwrap();
            let actual: f64 = tx
                .query_one(&format!("SELECT {s}.jse_numeric($1)"), &[&text])
                .unwrap()
                .get(0);
            assert_eq!(actual, expected, "numeric {text:?}");
        }
        let null: Option<f64> = tx
            .query_one(&format!("SELECT {s}.jse_numeric(NULL)"), &[])
            .unwrap()
            .get(0);
        assert_eq!(null, None);
        let content = json!({"tags":["rust", 1, true, null], "data":r#"{"score":3,"flags":[true,null],"nested":{"a":1}}"#, "synonyms":{"head":["x"]}});
        let tokens = membership_tokens(&content.to_string());
        let postings = crate::storage::json_index::content_postings(&content.to_string());
        for (path, token) in [
            ("tags", "rust"),
            ("tags", "1"),
            ("tags", "true"),
            ("tags", ""),
            ("tags", "false"),
            ("data.score", "3"),
            ("data.flags", "true"),
            ("data.flags", ""),
            ("data.nested", r#"{"a":1}"#),
            ("synonyms.head", "x"),
            ("data", content["data"].as_str().unwrap()),
        ] {
            let expected = postings
                .iter()
                .any(|p| p.path == path && p.value.as_deref() == Some(token));
            let actual: bool = tx
                .query_one(
                    "SELECT COALESCE(($1::jsonb -> $2::text) ? $3::text, FALSE)",
                    &[&tokens, &path, &token],
                )
                .unwrap()
                .get(0);
            assert_eq!(actual, expected, "membership {path} {token}");
        }
        for (lhs, rhs) in [
            (json!([1]), json!(1)),
            (json!({"x":[1]}), json!({"x":1})),
            (json!([[1]]), json!([1])),
            (json!([1, 2]), json!([2])),
            (json!({"x":null}), json!({"x":null})),
            (json!({}), json!({"x":null})),
            (json!(true), json!(1)),
        ] {
            let expected = crate::storage::json_contains(&lhs, &rhs);
            let actual: bool = tx
                .query_one(
                    &format!("SELECT {s}.jse_json_contains($1,$2)"),
                    &[&lhs, &rhs],
                )
                .unwrap()
                .get(0);
            assert_eq!(actual, expected, "containment {lhs} {rhs}");
        }
        tx.batch_execute(&format!("CREATE TABLE {s}.knowledge(name text PRIMARY KEY, content jsonb, payload jsonb, tokens jsonb, created_at timestamp); CREATE TABLE {s}.statement(head text, relation text, tail text);")).unwrap();
        for (name, score) in [
            ("number", json!(3)),
            ("numeric_text", json!("3")),
            ("invalid", json!("oops")),
            ("null", Value::Null),
        ] {
            let payload = json!({"n":4,"small":1e-7,"id":if name == "number" {9007199254740993u64} else {9007199254740992u64}});
            let content =
                json!({"score":score,"tags":["message",true,null],"data":payload.to_string()});
            tx.execute(
                &format!("INSERT INTO {s}.knowledge VALUES($1,$2,$3,$4,'2025-01-01'::timestamp)"),
                &[
                    &name,
                    &content,
                    &payload,
                    &membership_tokens(&content.to_string()),
                ],
            )
            .unwrap();
        }
        tx.batch_execute(&format!(
            "INSERT INTO {s}.statement VALUES('summary','summary','number')"
        ))
        .unwrap();
        let store = RecordingStore {
            schema: Some(schema),
            ..Default::default()
        };
        for (expr, mut expected) in [
            (
                json!(["$knowledge", ["$eq", "data.id", 9007199254740993u64]]),
                vec!["number"],
            ),
            (
                json!(["$knowledge", ["$eq", "score", "3"]]),
                vec!["numeric_text"],
            ),
            (
                json!(["$knowledge", ["$ne", "score", "3"]]),
                vec!["invalid", "number"],
            ),
            (
                json!(["$knowledge", ["$lt", "score", "NaN"]]),
                vec!["invalid", "number", "numeric_text"],
            ),
            (
                json!(["$knowledge", ["$lt", "score", "1e400"]]),
                vec!["invalid", "number", "numeric_text"],
            ),
            (json!(["$knowledge", ["$ne", "missing", 3]]), vec![]),
            (
                json!(["$knowledge", ["$not", ["$has", "missing", ""]]]),
                vec!["invalid", "null", "number", "numeric_text"],
            ),
            (
                json!(["$knowledge", ["$has", "data.small", 1e-7]]),
                vec!["invalid", "null", "number", "numeric_text"],
            ),
            (
                json!(["$knowledge", ["$gt", "score", 2]]),
                vec!["number", "numeric_text"],
            ),
            (json!(["$knowledge", ["$eq", "score", 3]]), vec!["number"]),
            (
                json!(["$knowledge", ["$ne", "score", 3]]),
                vec!["invalid", "numeric_text"],
            ),
            (json!(["$knowledge", ["$lt", "score", 1]]), vec!["invalid"]),
            (json!(["$knowledge", ["$eq", "score", null]]), vec![]),
            (
                json!(["$knowledge", ["$gt", "data.n", 3], ["$has", "tags", true]]),
                vec!["invalid", "null", "number", "numeric_text"],
            ),
            (
                json!(["$knowledge",["$json-contains",{"tags":"message"}]]),
                vec![],
            ),
            (
                json!(["$not-summaried", "message", ["$has", "tags", null]]),
                vec!["invalid", "null", "numeric_text"],
            ),
        ] {
            Evaluator::execute(&expr, &store).unwrap();
            let q = store.query.borrow();
            let prepared = tx.prepare(&q.0).unwrap();
            let params: Vec<Box<dyn postgres::types::ToSql + Sync>> =
                q.1.iter()
                    .zip(prepared.params())
                    .map(|(v, t)| -> Box<dyn postgres::types::ToSql + Sync> {
                        match *t {
                            postgres::types::Type::INT8 => Box::new(v.as_i64()),
                            postgres::types::Type::FLOAT8 => Box::new(v.as_f64()),
                            postgres::types::Type::TEXT => Box::new(v.as_str().map(str::to_owned)),
                            _ => panic!("unexpected parameter type {t}"),
                        }
                    })
                    .collect();
            let refs: Vec<_> = params.iter().map(|p| &**p).collect();
            let mut names: Vec<String> = tx
                .query(&prepared, &refs)
                .unwrap()
                .iter()
                .map(|r| r.get(0))
                .collect();
            names.sort();
            expected.sort();
            assert_eq!(names, expected, "compiled query {expr}");
        }
        tx.rollback().unwrap();
    }
}
