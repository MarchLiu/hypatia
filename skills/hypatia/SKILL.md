---
name: hypatia
description: "Interact with the Hypatia AI memory system using natural language. Translate user requests into hypatia CLI commands for knowledge CRUD, statement (RDF triple) management, JSE queries, full-text search, and shelf management. Trigger when: user mentions memories, knowledge bases, knowledge graphs, triples, statements, relationships, shelves, or asks to store, recall, remember, record, save, find, or search information in hypatia. Also trigger when user wants to create or explore relationships between concepts, query existing knowledge, or manage shelves. Examples: 'remember that Rust is a systems language', 'find everything about Alice', 'record that Alice knows Bob', 'search for programming', 'show all knowledge', 'list shelves'."
user-invocable: true
allowed-tools: Bash, Read, Grep, Glob
argument-hint: <natural-language instruction>
---

# Hypatia Query Skill

You are operating the Hypatia CLI — an AI-oriented memory management system. Translate the user's natural language request into the appropriate `hypatia` CLI command and execute it via Bash.

## Binary Location

First check which binary is available:

1. `hypatia` — if installed on PATH
2. `./target/debug/hypatia` — debug build in current project
3. `./target/release/hypatia` — release build in current project

Use the first one found. All examples below use `hypatia` for brevity.

## Shelf Management

| User says | Command |
|---|---|
| "list shelves" / "show connected shelves" | `hypatia list` |
| "connect shelf at PATH" / "open data at PATH" | `hypatia connect <path> [-n <name>]` |
| "disconnect shelf NAME" / "close shelf NAME" | `hypatia disconnect <name>` |
| "export shelf NAME to DEST" | `hypatia export <name> <dest>` |

## Archive Files

Archive files (images, PDFs, data, etc.) are stored in the shelf's `archives/` directory and referenced via `archive://` paths.

| User says | Command |
|---|---|
| "store this image" / "add figure fig.png" | `hypatia archive-store <file> -n <dest_path>` |
| "get archive fig.png" / "show image path" | `hypatia archive-get <name>` |
| "copy archive to /tmp" | `hypatia archive-get <name> -o /tmp/output.png` |
| "list archives" / "show all files" | `hypatia archive-list` |

`archive-store` automatically creates a knowledge entry with file metadata (size, MIME type) and an `is_a archive` statement. Archive files can be queried via JSE:

```bash
hypatia query '["$knowledge", ["$contains", "tags", "archive"]]'
hypatia query '["$knowledge", ["$content", {"mime_type": "image/png"}]]'
```

To reference an archive when creating knowledge:

```bash
hypatia knowledge-create "Euclid Prop 1" -d "equilateral triangle" --figures "archive://euclid/fig1.png"
```

### Scopes

Use `--scopes` to assign project or global scope to knowledge and statements. Empty string `""` means global. Comma-separated.

```bash
# Project-scoped only
hypatia knowledge-create "API convention" -d "REST endpoints use kebab-case" --tags "rule" --scopes "my-project"

# Global scope (empty string)
hypatia knowledge-create "prefer immutable" -d "always create new objects" --tags "rule" --scopes ""

# Both project and global
hypatia knowledge-create "no mock DB" -d "never mock database in tests" --tags "taboo" --scopes "my-project,"
```

## Knowledge CRUD

Knowledge entries are independent information points with a name, content, and tags.

### Create

```
hypatia knowledge-create <name> -d "<data>" -t "<tag1,tag2>" --figures "archive://path/to/file"
```

| User says | Command |
|---|---|
| "remember Rust as a systems programming language" | `hypatia knowledge-create "Rust" -d "systems programming language" -t "language,compiled"` |
| "save knowledge about Go with tags language and compiled" | `hypatia knowledge-create "Go" -d "" -t "language,compiled"` |
| "store that Python is a scripting language, tag it as dynamic" | `hypatia knowledge-create "Python" -d "scripting language" -t "dynamic"` |

### Read

```
hypatia knowledge-get <name>
```

| User says | Command |
|---|---|
| "show me knowledge about Rust" / "get Rust entry" | `hypatia knowledge-get "Rust"` |

### Delete

```
hypatia knowledge-delete <name>
```

| User says | Command |
|---|---|
| "delete knowledge Rust" / "remove the Rust entry" | `hypatia knowledge-delete "Rust"` |

## Statement Creation — Proactive Graph Building

Statements are RDF-style triples: `(subject, predicate, object)`. They form the edges of the knowledge graph.

### Key Principle: Always Enrich with Relationships

When the user asks to store or remember information, **always create statements alongside knowledge entries** to build graph connectivity. The goal is a rich, traversable knowledge graph, not isolated data points.

**Pattern**: After creating a knowledge entry, identify entities mentioned in the content and create `$triple` relationships between them. At minimum, create one `is_a` statement for every new knowledge entry.

```
hypatia statement-create <subject> <predicate> <object> -d "<data>"
```

### Triple Extraction Patterns

| User says | subject | predicate | object |
|---|---|---|---|
| "record that Alice knows Bob" | Alice | knows | Bob |
| "X is a Y" / "X is an Y" | X | is_a | Y |
| "X belongs to Y" / "X is part of Y" | X | belongs_to | Y |
| "X works for Y" | X | works_for | Y |
| "X is related to Y" | X | related_to | Y |
| "X depends on Y" | X | depends_on | Y |
| "X uses Y" | X | uses | Y |
| "X contains Y" | X | contains | Y |
| "X created Y" | X | created_by | Y (reversed) |

Normalize predicates to `snake_case`. Common predicates: `is_a`, `knows`, `related_to`, `works_for`, `belongs_to`, `depends_on`, `uses`, `contains`, `created_by`.

### Proactive Creation Examples

When the user says **"remember that Rust is a systems programming language"**, execute BOTH:

```bash
hypatia knowledge-create "Rust" -d "systems programming language" -t "language,compiled"
hypatia statement-create "Rust" "is_a" "systems programming language"
```

When the user says **"remember that PostgreSQL is a relational database used by the API"**, execute:

```bash
hypatia knowledge-create "PostgreSQL" -d "relational database" -t "database,relational"
hypatia statement-create "PostgreSQL" "is_a" "relational database"
hypatia statement-create "API" "depends_on" "PostgreSQL" -d "primary data store"
```

When the user says **"note that Alice is a senior engineer on the Backend team"**, execute:

```bash
hypatia knowledge-create "Alice" -d "senior engineer on Backend team" -t "person,engineer"
hypatia statement-create "Alice" "is_a" "senior engineer"
hypatia statement-create "Alice" "works_for" "Backend team"
```

### Auto-linking Rules

When creating knowledge entries, automatically extract and create statements for:
1. **Category**: `"X" is_a "<category>"` — every entity has a type
2. **Dependencies**: if the content mentions tools, frameworks, or systems X depends on → `X depends_on Y`
3. **Relationships**: if the content mentions other entities → link them with appropriate predicates
4. **Containment**: if X is part of Y → `X belongs_to Y`

## Search Strategy — Graph-First Retrieval

When searching for information, prefer precise graph operators over broad FTS search. Use the following decision tree:

### Search Decision Tree

```
1. User mentions a specific entity name?
   → knowledge-get (direct lookup)

2. User asks about relationships involving a known entity?
   → $triple operator (fastest, indexed)

3. User asks about a type of relationship (e.g., "who works for X")?
   → $triple with wildcard on entity positions

4. User wants to explore graph neighborhood (e.g., "who can X reach in 2 hops")?
   → $k-hop operator (graph traversal)

5. User wants semantic/similarity search (e.g., "find similar concepts")?
   → hypatia similar (vector search)

6. User wants to match a pattern (e.g., "names starting with X")?
   → $like operator

7. User wants to filter by content attributes (e.g., "markdown entries", "entries with specific format")?
   → $content operator

8. User asks about a specific entity's relationships?
   → $triple + knowledge-get combined

9. Broad/ambiguous query?
   → $search (FTS fallback)
```

### Prefer $triple over $search for entity queries

When the query involves a known entity (person, tool, concept), use `$triple` for precise, indexed lookups instead of FTS:

| Instead of | Use |
|---|---|
| `["$statement", ["$search", "Alice"]]` | `["$statement", ["$triple", "Alice", "$*", "$*"]]` |
| `["$statement", ["$search", "manages"]]` | `["$statement", ["$triple", "$*", "manages", "$*"]]` |
| `["$statement", ["$contains", "triple", "Alice"]]` | `["$statement", ["$triple", "Alice", "$*", "$*"]]` |

### Combined Queries

For rich retrieval, combine graph traversal with content filtering:

```bash
# Find all relationships involving Alice, created after a date
hypatia query '["$statement", ["$and", ["$triple", "Alice", "$*", "$*"], ["$gt", "created_at", "2025-01-01"]]]'

# Find all knowledge entries in markdown format about a topic
hypatia query '["$knowledge", ["$and", ["$content", {"format": "markdown"}], ["$search", "database"]]]'

# Find all people (entities that are_a "engineer")
hypatia query '["$statement", ["$triple", "$*", "is_a", "engineer"]]'
```

## Full-text Search

Search is the fallback for broad or ambiguous queries. It covers both knowledge and statements.

```
hypatia search <query> [-c <catalog>] [--limit N] [--offset N]
```

- `-c knowledge` — search only knowledge entries
- `-c statement` — search only statements
- No `-c` — search everything

### Examples

| User says | Command |
|---|---|
| "search for programming" / "find everything about programming" | `hypatia search "programming"` |
| "search knowledge for rust" | `hypatia search "rust" -c knowledge` |
| "what do you know about databases?" | `hypatia search "databases"` |

## Vector Similarity Search

Semantic search using embedding vectors. Finds entries with similar meaning, even when keywords don't match.

```
hypatia similar <query> [--limit N]
```

Requires an embedding model configured in `shelf.toml` (default: BAAI/bge-m3).

### Examples

| User says | Command |
|---|---|
| "find similar to distributed systems" | `hypatia similar "distributed systems"` |
| "semantic search for memory management" | `hypatia similar "memory management" --limit 5` |

## JSE Query Translation

JSE (JSON Search Expression) enables precise queries against the knowledge or statement tables.

```
hypatia query '<JSE-JSON>' [-s <shelf>]
```

### Query Structure

The top-level operator is always `$knowledge` or `$statement`:

```json
["$knowledge", condition1, condition2, ...]
["$statement", condition1, condition2, ...]
```

No conditions means "return all":

```json
["$knowledge"]
```

### Operator Reference

| Operator | Purpose | Syntax |
|---|---|---|
| `$eq` | Equals | `["$eq", "field", "value"]` |
| `$ne` | Not equals | `["$ne", "field", "value"]` |
| `$gt` | Greater than | `["$gt", "field", "value"]` |
| `$lt` | Less than | `["$lt", "field", "value"]` |
| `$gte` | Greater than or equal | `["$gte", "field", "value"]` |
| `$lte` | Less than or equal | `["$lte", "field", "value"]` |
| `$like` | SQL LIKE pattern match | `["$like", "field", "pattern"]` |
| `$contains` | Substring match in content JSON | `["$contains", "field", "value"]` |
| `$content` | Match content JSON key-value pairs | `["$content", {"key": "value"}]` |
| `$search` | Full-text search (FTS) | `["$search", "query text"]` |
| `$and` | Logical AND | `["$and", cond1, cond2, ...]` |
| `$or` | Logical OR | `["$or", cond1, cond2, ...]` |
| `$not` | Logical NOT | `["$not", condition]` |
| `$triple` | Triple position match | `["$triple", "Alice", "$*", "Bob"]` |
| `$k-hop` | K-hop graph traversal (statement only) | `["$k-hop", "subject", "predicate", depth]` |

### Critical Syntax Rules

1. **`$and` and `$or` take multiple operands directly**, NOT a nested array:
   - CORRECT: `["$and", ["$eq", "name", "rust"], ["$contains", "tags", "systems"]]`
   - WRONG: `["$and", [["$eq", "name", "rust"], ["$contains", "tags", "systems"]]]`

2. **`$search` must be inside `$knowledge` or `$statement`**, never top-level. When inside `$knowledge`, it searches FTS with catalog=knowledge. When inside `$statement`, it searches with catalog=statement.

3. **Field names** like `"name"`, `"triple"` are used as plain strings. For content JSON sub-fields (e.g., tags), `$contains` uses `json_extract_string` automatically.

4. **Available fields for knowledge**: `name`, `created_at`, plus any content JSON field via `$contains`.

5. **Available fields for statement**: `triple` (CSV-formatted subject,predicate,object), `subject`, `predicate`, `object`, `created_at`, `tr_start`, `tr_end`, plus content JSON fields via `$contains`. For position-based triple matching, prefer `$triple` over `$contains`.

### `$triple` Operator

The `$triple` operator provides position-based matching on statement triples. Each argument corresponds to subject, predicate, or object. Use `"$*"` as a wildcard to match any value.

```
["$triple", <subject_pattern>, <predicate_pattern>, <object_pattern>]
```

| Pattern | Meaning |
|---------|---------|
| `"Alice"` | Exact match |
| `"$*"` | Wildcard — match any value |

**Behavior**:
- All 3 specified (no wildcards): uses `triple = ?` (primary key lookup, fastest)
- Partial match: generates conditions on individual columns (`subject = ? AND object = ?`, etc.)
- At least one non-wildcard required — all wildcards is an error
- Arguments must be exactly 3

| User says | JSE | Command |
|---|---|---|
| "find all relationships where Alice is the subject" | `["$statement", ["$triple", "Alice", "$*", "$*"]]` | `hypatia query '["$statement", ["$triple", "Alice", "$*", "$*"]]'` |
| "find all knows relationships" | `["$statement", ["$triple", "$*", "knows", "$*"]]` | `hypatia query '["$statement", ["$triple", "$*", "knows", "$*"]]'` |
| "find everything related to Bob as object" | `["$statement", ["$triple", "$*", "$*", "Bob"]]` | `hypatia query '["$statement", ["$triple", "$*", "$*", "Bob"]]'` |
| "find the exact Alice knows Bob triple" | `["$statement", ["$triple", "Alice", "knows", "Bob"]]` | `hypatia query '["$statement", ["$triple", "Alice", "knows", "Bob"]]'` |
| "Alice's relationships with Bob, combined with date filter" | `["$statement", ["$and", ["$triple", "Alice", "$*", "Bob"], ["$gt", "created_at", "2025-01-01"]]]` | `hypatia query '["$statement", ["$and", ["$triple", "Alice", "$*", "Bob"], ["$gt", "created_at", "2025-01-01"]]]'` |

### `$like` Operator

SQL LIKE pattern matching with user-defined wildcards (`%` = any chars, `_` = single char).

```
["$like", "field", "pattern"]
```

| User says | JSE | Command |
|---|---|---|
| "find entries whose name starts with Rust" | `["$knowledge", ["$like", "name", "Rust%"]]` | `hypatia query '["$knowledge", ["$like", "name", "Rust%"]]'` |
| "find entries created in January 2025" | `["$knowledge", ["$like", "created_at", "2025-01-%"]]` | `hypatia query '["$knowledge", ["$like", "created_at", "2025-01-%"]]'` |
| "find statements with subject matching pattern" | `["$statement", ["$like", "subject", "Alice%"]]` | `hypatia query '["$statement", ["$like", "subject", "Alice%"]]'` |

### `$content` Operator

Match key-value pairs inside the `content` JSON column. Checks exact string equality for each specified key.

```
["$content", {"key1": "value1", "key2": "value2"}]
```

| User says | JSE | Command |
|---|---|---|
| "find all markdown-format entries" | `["$knowledge", ["$content", {"format": "markdown"}]]` | `hypatia query '["$knowledge", ["$content", {"format": "markdown"}]]'` |
| "find json-format statements" | `["$statement", ["$content", {"format": "json"}]]` | `hypatia query '["$statement", ["$content", {"format": "json"}]]'` |
| "find entries with specific data and format" | `["$knowledge", ["$content", {"format": "markdown", "data": "hello"}]]` | `hypatia query '["$knowledge", ["$content", {"format": "markdown", "data": "hello"}]]'` |

### `$k-hop` Operator

K-hop graph traversal on statement triples using recursive CTE. Explores entity neighborhoods and relationship paths.

```
["$k-hop", "subject", "predicate", depth]
```

- `"subject"` — starting entity (required)
- `"predicate"` — filter by predicate, or use `"$*"` for any
- `depth` — number of hops (positive integer)

| User says | JSE | Command |
|---|---|---|
| "who can Alice reach in 2 hops?" | `["$statement", ["$k-hop", "Alice", "$*", 2]]` | `hypatia query '["$statement", ["$k-hop", "Alice", "$*", 2]]'` |
| "what's reachable from Alice via knows in 3 hops?" | `["$statement", ["$k-hop", "Alice", "knows", 3]]` | `hypatia query '["$statement", ["$k-hop", "Alice", "knows", 3]]'` |
| "explore the graph around PostgreSQL" | `["$statement", ["$k-hop", "PostgreSQL", "$*", 2]]` | `hypatia query '["$statement", ["$k-hop", "PostgreSQL", "$*", 2]]'` |

### Natural Language to JSE Examples

| User says | JSE | Command |
|---|---|---|
| "find knowledge named rust" | `["$knowledge", ["$eq", "name", "rust"]]` | `hypatia query '["$knowledge", ["$eq", "name", "rust"]]'` |
| "find all relationships involving Alice" | `["$statement", ["$triple", "Alice", "$*", "$*"]]` | `hypatia query '["$statement", ["$triple", "Alice", "$*", "$*"]]'` |
| "who does Alice know?" | `["$statement", ["$triple", "Alice", "knows", "$*"]]` | `hypatia query '["$statement", ["$triple", "Alice", "knows", "$*"]]'` |
| "what type of things relate to Bob?" | `["$statement", ["$triple", "$*", "$*", "Bob"]]` | `hypatia query '["$statement", ["$triple", "$*", "$*", "Bob"]]'` |
| "find knowledge named rust that contains 'systems' in tags" | `["$knowledge", ["$and", ["$eq", "name", "rust"], ["$contains", "tags", "systems"]]]` | `hypatia query '["$knowledge", ["$and", ["$eq", "name", "rust"], ["$contains", "tags", "systems"]]]'` |
| "find knowledge NOT named rust" | `["$knowledge", ["$not", ["$eq", "name", "rust"]]]` | `hypatia query '["$knowledge", ["$not", ["$eq", "name", "rust"]]]'` |
| "find all is_a relationships" | `["$statement", ["$triple", "$*", "is_a", "$*"]]` | `hypatia query '["$statement", ["$triple", "$*", "is_a", "$*"]]'` |
| "find entries whose name starts with 'Al'" | `["$knowledge", ["$like", "name", "Al%"]]` | `hypatia query '["$knowledge", ["$like", "name", "Al%"]]'` |
| "show all knowledge" | `["$knowledge"]` | `hypatia query '["$knowledge"]'` |
| "find knowledge created after 2025-01-01" | `["$knowledge", ["$gt", "created_at", "2025-01-01"]]` | `hypatia query '["$knowledge", ["$gt", "created_at", "2025-01-01"]]'` |
| "find knowledge containing 'language' in data" | `["$knowledge", ["$contains", "data", "language"]]` | `hypatia query '["$knowledge", ["$contains", "data", "language"]]'` |

### Options (limit, offset, catalog)

Use object form for the top-level operator to pass options:

```json
{"$knowledge": [["$search", "rust"]], "limit": 10, "offset": 0}
```

Command: `hypatia query '{"$knowledge": [["$search", "rust"]], "limit": 10}'`

## Disambiguation Rules

When the user's request is ambiguous, follow this priority:

1. **Exact name mentioned** ("get Rust", "show Alice") → `knowledge-get` for direct lookup
2. **Explicit type** ("find knowledge about X" / "find statements about X") → JSE query with the corresponding top-level operator
3. **Relationship language** ("who does Alice know?", "what is Rust related to?", "relationships of X") → JSE `$statement` query with `$triple` operator
4. **Graph exploration** ("who can Alice reach?", "what's connected to X", "neighborhood of X") → JSE with `$k-hop` operator
5. **Semantic/similarity** ("find similar to X", "things like X", "semantically related") → `hypatia similar` (vector search)
6. **Pattern matching** ("names starting with X", "entries from January") → JSE with `$like` operator
7. **Content filtering** ("markdown entries", "entries with format X") → JSE with `$content` operator
8. **Broad/ambiguous** ("find everything about X", "what do you know about X") → `hypatia search` (covers both knowledge and statements)
9. **Create/store/remember** → knowledge-create + statement-create (always create relationships alongside knowledge entries)

## Shell Escaping

Always wrap JSE JSON in **single quotes** to prevent shell interpretation of `$`, `"`, `!`, etc.:

```bash
hypatia query '["$knowledge", ["$eq", "name", "rust"]]'
```

If the content contains single quotes, escape them:

```bash
hypatia query '["$knowledge", ["$eq", "name", "Alice'\''s project"]]'
```

## Output Format

- **Knowledge entries**: `{"name": "...", "content": {"format": "...", "data": "...", "tags": [...]}, "created_at": "..."}`
- **Statements**: `{"triple": "subject,predicate,object", "subject": "...", "predicate": "...", "object": "...", "content": {...}, "created_at": "...", "tr_start": "...", "tr_end": "..."}`
- **Search results**: `{"id": N, "catalog": "...", "key": "...", "content": "...", "rank": N.N}`
- **Empty results**: prints "No results found."

## Important Notes

- There is no `statement-get` CLI command. Use JSE queries (`$statement` with conditions) to find statements. `statement-delete` is available for deletion.
- The `-s` / `--shelf` flag defaults to `"default"` for all commands.
- The REPL mode (`hypatia repl`) is interactive and should NOT be used from this skill. Always use one-shot CLI commands.
- Content (`-d`) is stored as a string in the content JSON's `data` field.
- Tags (`-t`) are comma-separated strings.
- **Always create statements alongside knowledge entries** to maintain graph connectivity. Every knowledge entry should have at least one corresponding `is_a` statement.
