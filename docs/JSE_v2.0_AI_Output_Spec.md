# JSE v2.0 AI Output Specification

*(AST Architecture + Static Scoping)*

---

## 1. Purpose

This specification defines JSE v2.0, which introduces **AST-based evaluation** and **static scoping** while maintaining full backward compatibility with v1.0's structural syntax.

Key changes from v1.0:
- **AST Architecture**: Expressions are parsed into an Abstract Syntax Tree before evaluation
- **Static Scoping**: Closures capture their definition environment, not their call environment
- **Scope Chaining**: Environments form parent-child chains for symbol resolution
- **First-class Functions**: Lambda expressions with proper lexical closure support

---

## 2. Architecture Overview

JSE v2.0 implements a **two-phase execution model**:

```
JSON Value → Parser → AST Node → Environment → Result
```

### Components

| Component | Responsibility |
|-----------|-----------------|
| **Parser** | Converts JSON values into AST nodes |
| **AST Nodes** | Immutable tree structures with evaluation logic |
| **Environment** | Manages symbol bindings with scope chaining |
| **Engine** | Orchestrates parsing and evaluation |

### Two-Environment Pattern

For static scoping, v2.0 distinguishes between:

- **Construction Environment** (`get_env()`): Where the AST node was created
- **Evaluation Environment** (`apply(env)`): Where the AST node is executed

This separation enables proper closures: lambdas capture their definition environment.

---

## 3. Environment Model

### Scope Chain

```javascript
Environment
├── parent: Environment | null
├── bindings: Map<string, Value>
└── functors: Map<string, Functor>
```

**Symbol Resolution**: When resolving a symbol:
1. Check current environment's `bindings`
2. If not found, recursively search `parent` chain
3. If chain exhausted, return "Symbol not found" error

### Registration

- `register(name, value)`: Adds new binding (errors if exists)
- `set(name, value)`: Sets or overwrites binding
- `load(functors)`: Bulk-loads a functor module

### Module System

Functors are organized into modules:

| Module | Purpose | Operators |
|--------|---------|-----------|
| **builtin** | Core operators | `$quote`, `$eq`, `$cond`, `$head`, `$tail`, `$atom?`, `$cons` |
| **utils** | Utility functions | `$not`, `$list?`, `$map?`, `$null?`, `$get`, `$set`, `$del`, `$conj`, `$and`, `$or` |
| **lisp** | Lisp enhancements | `$def`, `$defn`, `$lambda` |


---

## 4. AST Node Types

### 4.1 LiteralNode

Represents primitive values: numbers, strings, booleans, null.

```json
42
"hello"
true
null
```

### 4.2 SymbolNode

Represents variable references that are resolved at evaluation time.

```json
"$myVariable"
```

### 4.3 ArrayNode

Represents function calls or literal arrays.

**Function Call** (first element is a symbol):
```json
["$add", 1, 2]
```

**Literal Array** (first element is NOT a symbol):
```json
[1, 2, 3]
["$literalValue", "not an operator"]
```

### 4.4 ObjectNode

Represents key-value mappings.

```json
{ "key": "value", "number": 42 }
```

### 4.5 ObjectExpressionNode

Represents operator expressions in object form.

```json
{
  "$pattern": ["$*", "author of", "$*"],
  "confidence": 0.95
}
```

**Rules**:
- Exactly one key must start with `$` (the operator)
- Other keys are metadata (preserved in result)
- Metadata does not affect evaluation

### 4.6 QuoteNode

Special node that prevents evaluation of its content.

```json
["$quote", ["$unevaluated", "expression"]]
```

### 4.7 LambdaNode

Represents anonymous functions with closure support.

```json
{
  "$lambda": {
    "__lambda__": true,
    "params": ["x", "y"],
    "body": ["$add", "$x", "$y"]
  }
}
```

---

## 5. Operator Reference

### 5.1 Builtin Operators

| Operator | Description | Example |
|----------|-------------|---------|
| `$quote` | Return expression unevaluated | `["$quote", 42]` → `42` |
| `$eq` | Equality comparison | `["$eq", 1, 1]` → `true` |
| `$cond` | Conditional branching | `["$cond", true, "a", "b"]` → `"a"` |
| `$head` | First element of array | `["$head", [1,2,3]]` → `1` |
| `$tail` | Rest of array | `["$tail", [1,2,3]]` → `[2,3]` |
| `$atom?` | Is not array/object? | `["$atom?", 42]` → `true` |
| `$cons` | Prepend to array | `["$cons", 1, [2,3]]` → `[1,2,3]` |

### 5.2 Utils Operators

| Operator | Description | Example |
|----------|-------------|---------|
| `$not` | Logical NOT | `["$not", true]` → `false` |
| `$and` | Logical AND | `["$and", true, false]` → `false` |
| `$or` | Logical OR | `["$or", false, true]` → `true` |
| `$list?` | Is array? | `["$list?", [1,2]]` → `true` |
| `$map?` | Is object? | `["$map?", {}]` → `true` |
| `$null?` | Is null? | `["$null?", null]` → `true` |
| `$get` | Get object/array value | `["$get", obj, "key"]` |
| `$set` | Set object/array value | `["$set", obj, "key", val]` |
| `$del` | Delete object/array value | `["$del", obj, "key"]` |
| `$conj` | Append to array | `["$conj", arr, elem]` |

### 5.3 Lisp Operators

| Operator | Description | Example |
|----------|-------------|---------|
| `$def` | Define variable | `["$def", "x", 42]` |
| `$defn` | Define function | `["$defn", "add", ["x", "y"], ["$add", "$x", "$y"]]` |
| `$lambda` | Create closure | See LambdaNode above |

### 5.4 SQL Operators

The SQL module is not a utility module; it is only used to demonstrate that functors with the same name may have completely different behaviors in different scopes.


---

## 6. Backward Compatibility

### v1.0 Compatibility

All valid v1.0 JSE expressions remain valid in v2.0:

```json
// v1.0 Array Form - Still works
["$and", true, false]

// v1.0 Object Form - Still works
{
  "$if": ["$gt", 5, 3],
  "confidence": 0.98
}

// v1.0 Quote - Still works
["$quote", { "$raw": "data" }]
```

### New Capabilities

v2.0 adds:

```json
// First-class functions with closures
{
  "$def": "makeAdder",
  "$lambda": {
    "params": ["x"],
    "body": {
      "$lambda": {
        "params": ["y"],
        "body": ["$add", "$x", "$y"]
      }
    }
  }
}

// Static scoping preserves definition environment
{
  "$defn": "counter",
  [],
  ["$def", "count", 0,
   "$lambda", {
     "params": [],
     "body": ["$set", "count", ["$add", "$count", 1]]
   }]
}
```

---

## 7. AI Output Requirements (v2.0)

When generating JSE v2.0, an AI system MUST:

**Structural Rules** (from v1.0):
- Produce syntactically valid JSON
- Respect Symbol rules (`$` prefix, `$$` escape)
- Use `$quote` to prevent evaluation
- Never produce multiple Symbol keys in one object

**New in v2.0**:
- Understand that expressions form an AST, not direct evaluation
- Use `$lambda` for first-class functions
- Use `$def`/`$defn` for variable/function definitions
- Leverage static scoping for closures

---

## 8. Operator Whitelisting

Implementations MAY:
- Restrict available operators
- Add custom operators via `env.load()`
- Implement different operator sets per use case

**Recommended minimal set**: `$quote`, `$eq`, `$cond`

**Recommended safe set**: builtin + utils (no `$def`, `$defn`, `$eval`)

**Full power set**: builtin + utils + lisp 

---

## 9. Execution Model

### Phase 1: Parsing

```python
parser = Parser(env)
ast_node = parser.parse(json_value)
```

### Phase 2: Evaluation

```python
result = ast_node.apply(env)
```

### Error Handling

| Error Type | Description |
|------------|-------------|
| `SymbolNotFound` | Symbol not in scope chain |
| `TypeError` | Type mismatch in operation |
| `ArityError` | Wrong number of arguments |
| `EvalError` | General evaluation failure |

---

## 10. Migration from v1.0

### For Implementers

1. **Add Parser**: Parse JSON into AST nodes
2. **Implement Scope Chain**: Add parent field to Environment
3. **Add Lambda Support**: Implement closures with captured environments
4. **Maintain Compatibility**: Keep v1.0 syntax working

### For Users

No changes required! All v1.0 expressions continue to work.

To use new features:
- Use `$def` to define variables
- Use `$defn` to define functions
- Use `$lambda` to create closures

---

## 11. Example: Complete Program

```json
{
  "$defn": "factorial",
  ["n"],
  ["$cond",
    ["$eq", "$n", 0],
    1,
    ["$mul", "$n", ["$factorial", ["$add", "$n", -1]]]
  ],

  "$def": "result",
  ["$factorial", 5],

  "$get", "$$result", "value"
}
```

This demonstrates:
- Function definition with `$defn`
- Recursive function calls
- Static scoping
- Variable binding with `$def`
- Escaped symbol with `$$`

---

## 12. Compliance Checklist

A JSE v2.0 implementation MUST:

- [ ] Parse JSON into AST nodes before evaluation
- [ ] Implement environment with scope chaining
- [ ] Support static scoping for lambdas
- [ ] Maintain v1.0 structural syntax compatibility
- [ ] Provide the three-phase modules (builtin, utils, lisp)
- [ ] Support the four core node types (Literal, Symbol, Array, Object)
- [ ] Implement ObjectExpression metadata support
- [ ] Provide proper error types

---

## 13. Summary of Changes from v1.0

| Feature | v1.0 | v2.0 |
|---------|------|------|
| **Evaluation** | Direct interpretation | AST-based parsing + evaluation |
| **Scoping** | Not specified | Static scoping with closure support |
| **Functions** | Not specified | First-class with `$lambda` |
| **Variables** | Not specified | `$def` and `$defn` operators |
| **Environment** | Single namespace | Parent-child scope chain |
| **Extensibility** | Limited | Module-based functor loading |

---

# Appendix A: Type Definitions (TypeScript)

```typescript
type Value =
  | null
  | boolean
  | number
  | string
  | Value[]
  | { [key: string]: Value };

interface Environment {
  parent: Environment | null;
  bindings: Map<string, Value>;
  functors: Map<string, Functor>;
}

interface Functor {
  (env: Environment, args: Value[]): Result<Value, Error>;
}

interface AstNode {
  apply(env: Environment): Result<Value, AstError>;
  getEnv(): Environment;
  toJSON(): Value;
}
```

---

# Appendix B: Evaluation Semantics

Given expression `E` and environment `ENV`:

1. **Parse**: `AST = Parser.parse(ENV, E)`
2. **Evaluate**: `RESULT = AST.apply(ENV)`
3. **Return**: `RESULT`

For closures (lambda):
- `DEF_ENV = AST.getEnv()` (environment where lambda was defined)
- `CALL_ENV = ENV` (environment where lambda is called)
- Body evaluation uses `DEF_ENV` for symbol resolution (static scoping)

---

# Appendix C: Quick Reference

| Concept | Syntax | Result |
|---------|--------|--------|
| Literal | `42`, `"x"`, `true` | Self |
| Symbol | `"$x"` | Resolved value |
| Array call | `["op", args]` | Operator result |
| Array literal | `[non-symbol, ...]` | Array |
| Object expr | `{"$op": val}` | Operator result |
| Object data | `{"key": val}` | Object |
| Quote | `["$quote", x]` | Unevaluated x |
| Lambda | `["$lambda", params, body]` | Closure |
| Def | `["$def", "name", val]` | Binding |
| Defn | `["$defn", "name", params, body]` | Function |
| Escape | `"$$name"` | Literal `"$name"` |
