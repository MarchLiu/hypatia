use serde_json::{Map, Value};

use crate::error::{HypatiaError, Result};
use super::ast::AstNode;

/// Recognized Hypatia JSE operators.
const OPERATORS: &[&str] = &[
    "$knowledge", "$statement",
    "$and", "$or", "$not",
    "$search", "$similar", "$k-hop",
    "$gte", "$lte", "$gt", "$lt",
    "$eq", "$ne",
    "$like", "$contains", "$content",
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
                if let Some(Value::String(first)) = arr.first() {
                    if first.starts_with('$') {
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
                            .map(|v| Self::parse(v))
                            .collect::<Result<Vec<_>>>()?;
                        return Ok(AstNode::Operator {
                            operator,
                            operands,
                            metadata: Map::new(),
                        });
                    }
                }
                // Plain array: parse each element
                let nodes: Vec<AstNode> = arr.iter().map(|v| Self::parse(v)).collect::<Result<Vec<_>>>()?;
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
                                arr.iter().map(|v| Self::parse(v)).collect::<Result<Vec<_>>>()?
                            }
                            _ => vec![Self::parse(op_value)?],
                        };

                        Ok(AstNode::Operator {
                            operator,
                            operands,
                            metadata,
                        })
                    }
                    _ => {
                        Err(HypatiaError::Parse(
                            format!("object has multiple $ keys: {:?}", dollar_keys),
                        ))
                    }
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
            AstNode::Operator { operator, operands, .. } => {
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
            AstNode::Operator { operator, operands, .. } => {
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
            AstNode::Operator { operator, metadata, .. } => {
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
        let ast = Parser::parse(&json!(["$and", ["$eq", "name", "Alice"], ["$gt", "age", 18]])).unwrap();
        match ast {
            AstNode::Operator { operator, operands, .. } => {
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
            AstNode::Operator { operator, operands, .. } => {
                assert_eq!(operator, "$knowledge");
                assert_eq!(operands.len(), 1);
            }
            _ => panic!("expected Operator node"),
        }
    }
}
