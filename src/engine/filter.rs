//! A JSE condition compiled into a WHERE fragment over one table, for a query the
//! evaluator does not build itself. The vector search behind `similar` ranks by
//! distance, so it has to apply a filter before it ranks rather than after: filtering
//! the top hits afterwards returns fewer than asked, and no caller can tell how many
//! to over-fetch.
use super::SqlDialect;
use super::ast::AstNode;
use super::operators::{OpContext, OperatorResult, evaluate_operator};
use super::parser::Parser;
use crate::error::{HypatiaError, Result};
use crate::model::QueryTarget;
use crate::storage::Storage;
use serde_json::{Value, json};

/// A WHERE fragment and the values its placeholders bind, in the shelf's SQL dialect.
/// A SQLite fragment uses `?` and names the target table itself. A PostgreSQL fragment
/// uses `$1`…`$n` over the alias `q`, so the caller's own bindings start at `$n+1`.
#[derive(Debug, Clone)]
pub struct SqlFilter {
    pub fragment: String,
    pub params: Vec<Value>,
}

/// Compiles `condition`, such as `["$not", ["$contains", "tags", "message"]]`, for
/// `target` on `store`'s dialect. Only a filter compiles: a query (`$knowledge`, …),
/// `$search`, `$similar`, `$k-hop` or a bare value is refused rather than ignored.
pub fn compile(condition: &Value, target: QueryTarget, store: &dyn Storage) -> Result<SqlFilter> {
    let ast = Parser::parse(condition)?;
    let AstNode::Operator {
        operator,
        operands,
        metadata,
    } = &ast
    else {
        return Err(HypatiaError::Eval(format!(
            "a filter must be a JSE condition such as [\"$contains\", \"tags\", \"x\"], got {condition}"
        )));
    };
    refuse_non_filters(&ast)?;
    match store.sql_dialect() {
        SqlDialect::Sqlite => {
            let ctx = OpContext::for_target(target);
            match evaluate_operator(operator, operands, metadata, &ctx, &|n| {
                sqlite_condition(n, &ctx)
            })? {
                OperatorResult::SqlCondition { fragment, params } => {
                    Ok(SqlFilter { fragment, params })
                }
                _ => Err(HypatiaError::Eval(format!(
                    "{operator} is not a filter; use a condition such as $contains, $eq or $not"
                ))),
            }
        }
        SqlDialect::Postgres => super::postgres::compile_filter(&ast, target, store),
    }
}

/// Refuses, at any depth, what the compilers accept but a filter cannot hold: a nested
/// query compiles to "every row", and options such as `limit` are dropped, so either
/// would widen the result without a word.
fn refuse_non_filters(node: &AstNode) -> Result<()> {
    let AstNode::Operator {
        operator,
        operands,
        metadata,
    } = node
    else {
        return Ok(());
    };
    if matches!(
        operator.as_str(),
        "$knowledge" | "$statement" | "$not-summaried"
    ) {
        return Err(HypatiaError::Eval(format!(
            "a filter is a condition, not a query: remove the {operator}"
        )));
    }
    if let Some(option) = metadata.keys().next() {
        return Err(HypatiaError::Eval(format!(
            "a filter takes no options, got \"{option}\" on {operator}"
        )));
    }
    operands.iter().try_for_each(refuse_non_filters)
}

/// A filter that names a field the target's table lacks, such as `name` on statements.
pub(crate) fn unknown_field(table: &str, detail: &str) -> HypatiaError {
    HypatiaError::Validation(format!(
        "the filter names a field {table} entries do not have ({detail}); \
         it applies to every target searched, so search only the one it fits"
    ))
}

/// A nested condition on SQLite. Only an operator carries a filter, so anything else
/// is an error instead of a value that would quietly drop out of the WHERE clause.
fn sqlite_condition(node: &AstNode, ctx: &OpContext) -> Result<OperatorResult> {
    match node {
        AstNode::Operator {
            operator,
            operands,
            metadata,
        } => evaluate_operator(operator, operands, metadata, ctx, &|n| {
            sqlite_condition(n, ctx)
        }),
        _ => Err(HypatiaError::Eval(format!(
            "unexpected node in condition context: {node:?}"
        ))),
    }
}

/// The filter `similar` applies for its options: an entry passes when it carries at
/// least one of `tags` (if any are given), none of `exclude_tags`, and satisfies
/// `condition`. `None` when nothing narrows the search.
pub fn similar_filter(
    tags: &[String],
    exclude_tags: &[String],
    condition: Option<Value>,
) -> Option<Value> {
    let any_tag = |names: &[String]| {
        let mut each: Vec<Value> = names
            .iter()
            .map(|t| json!(["$contains", "tags", literal(t)]))
            .collect();
        if each.len() == 1 {
            each.pop().unwrap()
        } else {
            each.insert(0, json!("$or"));
            Value::Array(each)
        }
    };
    let mut parts = Vec::new();
    if !tags.is_empty() {
        parts.push(any_tag(tags));
    }
    if !exclude_tags.is_empty() {
        parts.push(json!(["$not", any_tag(exclude_tags)]));
    }
    parts.extend(condition);
    match parts.len() {
        0 => None,
        1 => parts.pop(),
        _ => {
            parts.insert(0, json!("$and"));
            Some(Value::Array(parts))
        }
    }
}

/// `s` as a JSE string literal. The parser reads a leading `$` as a field reference and
/// unescapes `$$` to `$`, so a value that starts with `$` gets one more.
fn literal(s: &str) -> String {
    if s.starts_with('$') {
        format!("${s}")
    } else {
        s.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{QueryResult, SearchOpts};

    /// Reports a dialect and nothing else: compiling never reaches the store.
    struct Dialect(SqlDialect);
    impl Storage for Dialect {
        fn sql_dialect(&self) -> SqlDialect {
            self.0
        }
        fn sql_schema(&self) -> Option<&str> {
            Some("shelf")
        }
        fn execute_query(&self, _: QueryTarget, _: &str, _: Vec<Value>) -> Result<QueryResult> {
            unreachable!("compiling a filter runs no query")
        }
        fn execute_search(&self, _: &str, _: &SearchOpts) -> Result<QueryResult> {
            unreachable!("compiling a filter runs no search")
        }
        fn execute_similar(&self, _: &str, _: &SearchOpts, _: QueryTarget) -> Result<QueryResult> {
            unreachable!("compiling a filter runs no search")
        }
        fn execute_khop(&self, _: &str, _: Option<&str>, _: i64) -> Result<QueryResult> {
            unreachable!("compiling a filter runs no traversal")
        }
    }

    #[test]
    fn tag_options_become_one_condition() {
        let t = |v: &[&str]| v.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        assert_eq!(similar_filter(&[], &[], None), None);
        assert_eq!(
            similar_filter(&t(&["rule"]), &[], None),
            Some(json!(["$contains", "tags", "rule"]))
        );
        assert_eq!(
            similar_filter(&[], &t(&["message", "summary"]), None),
            Some(json!([
                "$not",
                [
                    "$or",
                    ["$contains", "tags", "message"],
                    ["$contains", "tags", "summary"]
                ]
            ]))
        );
        let condition = json!(["$like", "name", "wu-%"]);
        assert_eq!(
            similar_filter(&t(&["a"]), &t(&["b"]), Some(condition.clone())),
            Some(json!([
                "$and",
                ["$contains", "tags", "a"],
                ["$not", ["$contains", "tags", "b"]],
                condition
            ]))
        );
        assert_eq!(
            similar_filter(&[], &[], Some(condition.clone())),
            Some(condition)
        );
    }

    #[test]
    fn a_tag_that_starts_with_a_dollar_stays_the_same_value() {
        for tag in ["$pinned", "$$x", "$"] {
            let filter = similar_filter(&[tag.to_string()], &[], None).unwrap();
            for dialect in [SqlDialect::Sqlite, SqlDialect::Postgres] {
                let compiled = compile(&filter, QueryTarget::Knowledge, &Dialect(dialect)).unwrap();
                assert_eq!(compiled.params, [json!("tags"), json!(tag)], "{dialect:?}");
            }
        }
    }

    #[test]
    fn both_dialects_compile_a_filter_over_the_target_table() {
        let condition = json!(["$not", ["$contains", "tags", "message"]]);
        let sqlite = compile(
            &condition,
            QueryTarget::Knowledge,
            &Dialect(SqlDialect::Sqlite),
        )
        .unwrap();
        assert!(
            sqlite.fragment.starts_with("NOT (EXISTS"),
            "{}",
            sqlite.fragment
        );
        assert!(
            sqlite.fragment.contains("knowledge.name"),
            "{}",
            sqlite.fragment
        );
        assert_eq!(sqlite.params, [json!("tags"), json!("message")]);

        let pg = compile(
            &condition,
            QueryTarget::Statement,
            &Dialect(SqlDialect::Postgres),
        )
        .unwrap();
        assert!(pg.fragment.contains("q.tokens"), "{}", pg.fragment);
        assert!(pg.fragment.contains("$1") && pg.fragment.contains("$2"));
        assert_eq!(pg.params, [json!("tags"), json!("message")]);
    }

    #[test]
    fn only_a_condition_is_a_filter() {
        let refused = |condition: &Value, dialect| {
            compile(condition, QueryTarget::Knowledge, &Dialect(dialect))
                .expect_err(&format!("{condition} on {dialect:?}"))
                .to_string()
        };
        // Refused before a dialect is chosen, so both say the same.
        for (condition, expected) in [
            (json!("message"), "must be a JSE condition"),
            (json!(42), "must be a JSE condition"),
            (json!(["$quote", "x"]), "must be a JSE condition"),
            (
                json!(["$knowledge", ["$contains", "tags", "x"]]),
                "remove the $knowledge",
            ),
            // A nested query compiles to "every row" in both compilers.
            (
                json!([
                    "$and",
                    [
                        "$knowledge",
                        ["$contains", "tags", "rule"],
                        ["$contains", "tags", "nope"]
                    ]
                ]),
                "remove the $knowledge",
            ),
            (json!(["$not", ["$statement"]]), "remove the $statement"),
            (
                json!({"$contains": ["tags", "rule"], "limit": 1}),
                "takes no options, got \"limit\"",
            ),
        ] {
            for dialect in [SqlDialect::Sqlite, SqlDialect::Postgres] {
                let err = refused(&condition, dialect);
                assert!(err.contains(expected), "{condition} on {dialect:?}: {err}");
            }
        }
        // Refused by each dialect's compiler, in its own words.
        for (condition, sqlite) in [
            (json!(["$similar", "rust"]), "not a filter"),
            (json!(["$search", "rust"]), "not a filter"),
            (
                json!(["$not", ["$search", "rust"]]),
                "expects SQL condition",
            ),
            (
                json!(["$and", ["$eq", "name", "a"], "stray"]),
                "unexpected node in condition context",
            ),
        ] {
            let err = refused(&condition, SqlDialect::Sqlite);
            assert!(err.contains(sqlite), "{condition}: {err}");
            refused(&condition, SqlDialect::Postgres);
        }
    }
}
