use serde_json::{Map, Value};

use super::ast::AstNode;
use crate::error::{HypatiaError, Result};

/// Recognized Hypatia JSE operators. A bare `"$name"` listed here parses as a
/// literal operator name, any other as a field reference; the condition-context
/// error relies on that to tell a flattened call from a stray field. It must
/// name exactly the operators the backends dispatch on (checked by
/// `whitelist_matches_the_operators_the_backends_dispatch`).
pub(super) const OPERATORS: &[&str] = &[
    "$knowledge",
    "$statement",
    "$and",
    "$or",
    "$not",
    "$search",
    "$similar",
    "$k-hop",
    "$not-summaried",
    "$gte",
    "$lte",
    "$gt",
    "$lt",
    "$eq",
    "$ne",
    "$like",
    "$contains",
    "$has",
    "$json-contains",
    "$content",
    "$triple",
    "$quote",
];

pub struct Parser;

impl Parser {
    /// Parse a JSON value into an AstNode.
    pub fn parse(value: &Value) -> Result<AstNode> {
        match value {
            Value::Null | Value::Bool(_) | Value::Number(_) => Ok(AstNode::Literal(value.clone())),
            Value::String(s) => {
                if s == "$$" {
                    Ok(AstNode::Literal(Value::String("$".to_string())))
                } else if s.starts_with("$$") {
                    // Escaped symbol: $$name → literal "$name"
                    Ok(AstNode::Literal(Value::String(s[1..].to_string())))
                } else if s.starts_with('$') && !OPERATORS.contains(&s.as_str()) {
                    // Non-operator symbol → field reference
                    Ok(AstNode::Symbol(s.clone()))
                } else {
                    Ok(AstNode::Literal(value.clone()))
                }
            }
            Value::Array(arr) => {
                if arr.is_empty() {
                    return Ok(AstNode::Array(Vec::new()));
                }
                // Check if first element is an operator string
                if let Some(Value::String(first)) = arr
                    .first()
                    .filter(|v| matches!(v, Value::String(s) if s.starts_with('$')))
                {
                    let operator = first.clone();
                    if operator == "$quote" {
                        // Quote: parse inner but wrap in Quote node
                        if arr.len() != 2 {
                            return Err(HypatiaError::Parse(
                                "$quote expects exactly one argument".to_string(),
                            ));
                        }
                        let inner = Self::parse(&arr[1])?;
                        return Ok(AstNode::Quote(Box::new(inner)));
                    }
                    // Regular operator call: [operator, arg1, arg2, ...]
                    let operands: Vec<AstNode> = arr[1..]
                        .iter()
                        .map(Self::parse)
                        .collect::<Result<Vec<_>>>()?;
                    return Ok(AstNode::Operator {
                        operator,
                        operands,
                        metadata: Map::new(),
                    });
                }
                // Plain array: parse each element
                let nodes: Vec<AstNode> =
                    arr.iter().map(Self::parse).collect::<Result<Vec<_>>>()?;
                Ok(AstNode::Array(nodes))
            }
            Value::Object(obj) => {
                // Find $ keys
                let dollar_keys: Vec<&String> = obj.keys().filter(|k| k.starts_with('$')).collect();

                match dollar_keys.len() {
                    0 => {
                        // Plain data object
                        Ok(AstNode::Object(obj.clone()))
                    }
                    1 => {
                        // Operator in object form: {"$op": value, "meta": ...}
                        let operator = dollar_keys[0].clone();
                        let op_value = &obj[&operator];
                        let mut metadata = obj.clone();
                        metadata.remove(&operator);

                        if operator == "$quote" {
                            let inner = Self::parse(op_value)?;
                            return Ok(AstNode::Quote(Box::new(inner)));
                        }

                        // The operator's value can be:
                        // - An array of operands: {"$and": [cond1, cond2]}
                        // - A single value: {"$eq": "value"}
                        let operands = match op_value {
                            Value::Array(arr) => {
                                arr.iter().map(Self::parse).collect::<Result<Vec<_>>>()?
                            }
                            _ => vec![Self::parse(op_value)?],
                        };

                        Ok(AstNode::Operator {
                            operator,
                            operands,
                            metadata,
                        })
                    }
                    _ => Err(HypatiaError::Parse(format!(
                        "object has multiple $ keys: {:?}",
                        dollar_keys
                    ))),
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn parse_literal_number() {
        let ast = Parser::parse(&json!(42)).unwrap();
        assert!(matches!(ast, AstNode::Literal(v) if v == json!(42)));
    }

    #[test]
    fn parse_literal_string() {
        let ast = Parser::parse(&json!("hello")).unwrap();
        assert!(matches!(ast, AstNode::Literal(v) if v == json!("hello")));
    }

    #[test]
    fn parse_symbol() {
        let ast = Parser::parse(&json!("$myField")).unwrap();
        assert!(matches!(ast, AstNode::Symbol(s) if s == "$myField"));
    }

    #[test]
    fn parse_escaped_symbol() {
        let ast = Parser::parse(&json!("$$name")).unwrap();
        assert!(matches!(ast, AstNode::Literal(v) if v == json!("$name")));
    }

    #[test]
    fn parse_operator_array_form() {
        let ast = Parser::parse(&json!(["$and", true, false])).unwrap();
        match ast {
            AstNode::Operator {
                operator, operands, ..
            } => {
                assert_eq!(operator, "$and");
                assert_eq!(operands.len(), 2);
            }
            _ => panic!("expected Operator node"),
        }
    }

    #[test]
    fn parse_operator_object_form() {
        let ast = Parser::parse(&json!({"$eq": "value"})).unwrap();
        match ast {
            AstNode::Operator {
                operator, operands, ..
            } => {
                assert_eq!(operator, "$eq");
                assert_eq!(operands.len(), 1);
            }
            _ => panic!("expected Operator node"),
        }
    }

    #[test]
    fn parse_operator_with_metadata() {
        let ast = Parser::parse(&json!({"$search": "query text", "catalog": "knowledge"})).unwrap();
        match ast {
            AstNode::Operator {
                operator, metadata, ..
            } => {
                assert_eq!(operator, "$search");
                assert_eq!(metadata["catalog"], json!("knowledge"));
            }
            _ => panic!("expected Operator node"),
        }
    }

    #[test]
    fn parse_quote_array() {
        let ast = Parser::parse(&json!(["$quote", {"$and": [1, 2]}])).unwrap();
        assert!(matches!(ast, AstNode::Quote(_)));
    }

    #[test]
    fn parse_plain_array() {
        let ast = Parser::parse(&json!([1, 2, 3])).unwrap();
        assert!(matches!(ast, AstNode::Array(nodes) if nodes.len() == 3));
    }

    #[test]
    fn parse_plain_object() {
        let ast = Parser::parse(&json!({"key": "value"})).unwrap();
        assert!(matches!(ast, AstNode::Object(_)));
    }

    #[test]
    fn parse_nested_operators() {
        let ast = Parser::parse(&json!([
            "$and",
            ["$eq", "name", "Alice"],
            ["$gt", "age", 18]
        ]))
        .unwrap();
        match ast {
            AstNode::Operator {
                operator, operands, ..
            } => {
                assert_eq!(operator, "$and");
                assert_eq!(operands.len(), 2);
                // Each operand should be an Operator node
                assert!(matches!(&operands[0], AstNode::Operator { .. }));
                assert!(matches!(&operands[1], AstNode::Operator { .. }));
            }
            _ => panic!("expected Operator node"),
        }
    }

    #[test]
    fn parse_knowledge_operator() {
        let ast = Parser::parse(&json!(["$knowledge", ["$eq", "name", "test"]])).unwrap();
        match ast {
            AstNode::Operator {
                operator, operands, ..
            } => {
                assert_eq!(operator, "$knowledge");
                assert_eq!(operands.len(), 1);
            }
            _ => panic!("expected Operator node"),
        }
    }

    /// Names handled by `"$a" | "$b" =>` match arms in `source`, including
    /// or-patterns rustfmt wrapped onto `| "$c"` continuation lines.
    fn dispatched_operators(source: &str) -> std::collections::BTreeSet<&str> {
        let mut dispatched = std::collections::BTreeSet::new();
        let mut pending = Vec::new();
        for line in source.lines() {
            let line = line.trim();
            let line = line.strip_prefix('|').map_or(line, str::trim_start);
            let (pattern, is_arm) = match line.split_once(" =>") {
                Some((pattern, _)) => (pattern, true),
                None => (line, false),
            };
            let names: Option<Vec<_>> = pattern
                .split(" | ")
                .map(|p| {
                    let name = p.strip_prefix('"')?.strip_suffix('"')?;
                    let bare = name.strip_prefix('$')?;
                    (!bare.is_empty() && bare.chars().all(|c| c.is_ascii_lowercase() || c == '-'))
                        .then_some(name)
                })
                .collect();
            match names {
                Some(names) => {
                    pending.extend(names);
                    if is_arm {
                        dispatched.extend(pending.drain(..));
                    }
                }
                None => pending.clear(),
            }
        }
        dispatched
    }

    #[test]
    fn dispatched_operators_reads_only_match_arm_patterns() {
        let source = r#"
            match operator {
                "$a" | "$b" => x,
                "$c" | "$d"
                | "$e" => y,
                s if s == "$f" => z,
                AstNode::Symbol(s) if s == "$*" => w,
                _ => json!([
                    "$g",
                    "$h"
                ]),
            }
        "#;
        let found: Vec<_> = dispatched_operators(source).into_iter().collect();
        assert_eq!(found, ["$a", "$b", "$c", "$d", "$e"]);
    }

    /// #28: $has, $json-contains and $triple were implemented by both
    /// backends but missing from the whitelist, so a flattened call to one
    /// got no hint. Hold the whitelist to exactly the names the dispatchers
    /// handle, so a new operator cannot be left out again.
    #[test]
    fn whitelist_matches_the_operators_the_backends_dispatch() {
        let mut dispatched = std::collections::BTreeSet::new();
        for source in [
            include_str!("evaluator.rs"),
            include_str!("operators.rs"),
            include_str!("postgres.rs"),
        ] {
            dispatched.extend(dispatched_operators(source));
        }
        // The parser itself turns both forms of $quote into `AstNode::Quote`,
        // whether or not a backend keeps an arm for it.
        dispatched.insert("$quote");
        let listed: std::collections::BTreeSet<_> = OPERATORS.iter().copied().collect();
        assert_eq!(listed, dispatched);
    }
}
