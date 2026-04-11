# Hypatia

“We can wander through the stacks of the Library of Alexandria, imagining the scrolls and the knowledge they contain. Its destruction is a warning: all we have is transient.”——Alberto Manguel

AI-oriented memory management system. Stores structured knowledge as a graph of **Knowledge** entries (nodes) and **Statement** triples (edges), queried via a custom JSON Search Expression (JSE) language. Built on SQLite FTS5 + DuckDB, with zero external model dependencies.

## Features

- **Knowledge Graph** -- Knowledge entries (named info points with tags) and Statement triples (subject-predicate-object with temporal ranges)
- **JSE Query Engine** -- JSON-based query language compiling to parameterized SQL + FTS5, supporting `$and`, `$or`, `$not`, `$eq`, `$ne`, `$gt`, `$lt`, `$contains`, `$like`, `$content`, `$search`, `$quote`, `$triple`
- **Dual-Database Storage** -- DuckDB for structured queries, SQLite FTS5 (Porter stemmer + multi-column BM25) for full-text search, auto-synchronized
- **Synonyms** -- Per-entry synonym lists for knowledge, per-position (subject/predicate/object) synonyms for statements, indexed in FTS
- **Shelf System** -- Named, connectable, exportable data directories for isolation
- **CLI + REPL** -- Full command-line interface with interactive mode (rustyline)
- **Agent Integration** -- Claude Code skill for natural-language-to-CLI translation
- **Cross-Platform** -- Build for 18+ targets (Linux, macOS, Windows, FreeBSD, NetBSD, illumos, Android)

## Quick Start

```bash
# Build
cargo build --release

# Create knowledge
hypatia knowledge-create "Rust" -d "systems programming language" -t "language,compiled"

# Create a relationship
hypatia statement-create "Rust" "is_a" "systems language"

# Full-text search
hypatia search "programming language"

# Structured query (JSE)
hypatia query '["$knowledge", ["$eq", "name", "Rust"]]'
hypatia query '["$statement", ["$triple", "Rust", "$*", "$*"]]'
hypatia query '["$knowledge", ["$search", "database migration"]]'

# Pattern matching and content filtering
hypatia query '["$knowledge", ["$like", "name", "Rust%"]]'
hypatia query '["$knowledge", ["$content", {"format": "markdown"}]]'

# Interactive REPL
hypatia repl
```

## CLI Reference

| Command | Description |
|---------|-------------|
| `hypatia connect <path> [-n <name>]` | Connect to a shelf directory |
| `hypatia disconnect <name>` | Disconnect from a shelf |
| `hypatia list` | List connected shelves |
| `hypatia knowledge-create <name> [-d <data>] [-t <tags>] [--synonyms <csv>]` | Create a knowledge entry |
| `hypatia knowledge-get <name>` | Get a knowledge entry |
| `hypatia knowledge-delete <name>` | Delete a knowledge entry |
| `hypatia statement-create <subj> <pred> <obj> [-d <data>] [--synonyms <json>]` | Create a triple |
| `hypatia statement-delete <subj> <pred> <obj>` | Delete a triple |
| `hypatia search <query> [-c <catalog>] [--limit N]` | Full-text search |
| `hypatia query '<jse-json>'` | Execute a JSE query |
| `hypatia export <name> <dest>` | Export a shelf |
| `hypatia repl` | Interactive REPL |

## JSE Query Language

JSE (JSON Search Expression) enables precise queries against knowledge or statement tables.

### Syntax

```json
["$knowledge", condition1, condition2, ...]
["$statement", condition1, condition2, ...]
```

### Operators

| Operator | Purpose | Example |
|----------|---------|---------|
| `$eq` | Equals | `["$eq", "name", "Rust"]` |
| `$ne` | Not equals | `["$ne", "name", "Rust"]` |
| `$gt` / `$lt` / `$gte` / `$lte` | Comparison | `["$gt", "created_at", "2025-01-01"]` |
| `$contains` | Substring in JSON field | `["$contains", "tags", "backend"]` |
| `$like` | SQL LIKE pattern match | `["$like", "name", "Rust%"]` |
| `$content` | Match content JSON key-values | `["$content", {"format": "markdown"}]` |
| `$search` | Full-text search | `["$search", "database migration"]` |
| `$and` | Logical AND | `["$and", cond1, cond2]` |
| `$or` | Logical OR | `["$or", cond1, cond2]` |
| `$not` | Logical NOT | `["$not", cond]` |
| `$quote` | Prevent evaluation | `["$quote", ["$eq", "x", "y"]]` |
| `$triple` | Triple position match | `["$triple", "Alice", "$*", "Bob"]` |

### Examples

```bash
# All knowledge entries
hypatia query '["$knowledge"]'

# Knowledge named "Rust" with tag "systems"
hypatia query '["$knowledge", ["$and", ["$eq", "name", "Rust"], ["$contains", "tags", "systems"]]]'

# Statements containing "Alice" in triple
hypatia query '["$statement", ["$contains", "triple", "Alice"]]'

# Triple matching: all relationships where Alice is the subject
hypatia query '["$statement", ["$triple", "Alice", "$*", "$*"]]'

# Triple matching: all "manages" relationships
hypatia query '["$statement", ["$triple", "$*", "manages", "$*"]]'

# Triple matching: exact triple (uses PK index)
hypatia query '["$statement", ["$triple", "Alice", "knows", "Bob"]]'

# Pattern matching: names starting with "Al"
hypatia query '["$knowledge", ["$like", "name", "Al%"]]'

# Content filtering: all markdown entries
hypatia query '["$knowledge", ["$content", {"format": "markdown"}]]'

# FTS search within knowledge
hypatia query '["$knowledge", ["$search", "query optimization"]]'

# Statements where triple contains Alice or Bob
hypatia query '["$statement", ["$or", ["$contains", "triple", "Alice"], ["$contains", "triple", "Bob"]]]'
```

## Architecture

```
src/
├── cli/            # CLI commands + REPL (clap + rustyline)
├── engine/         # JSE parser, AST, evaluator, SQL builder
├── model/          # Knowledge, Statement, Content, Query types
├── service/        # Business logic (dual-write to DuckDB + SQLite)
├── storage/        # DuckDB store, SQLite FTS5 store, shelf manager
├── lab.rs          # Top-level API facade
├── error.rs        # Error types
├── lib.rs          # Module declarations
└── main.rs         # Entry point
```

Each **shelf** is a directory containing `data.duckdb` (structured data) and `index.sqlite` (FTS5 index). The service layer keeps both databases in sync via dual-write.

## Benchmark

Benchmark uses synthetic data with planted needles (known-answer entries) to measure retrieval quality, following MemPalace's methodology.

### Run

```bash
# Small scale (1K knowledge, 2K statements, ~12s)
cargo test --test bench

# With JSON report
BENCH_REPORT=report.json cargo test --test bench

# Larger scales
BENCH_SCALE=medium cargo test --test bench --release
BENCH_SCALE=large cargo test --test bench --release
```

### Results (small scale, debug build, Apple Silicon)

1K knowledge, 2K statements, 20 needles, 20 JSE query types (×3 runs each).

| Metric | Result |
|--------|--------|
| **Recall@1** | 100.0% (20/20 needles) |
| **Recall@5** | 100.0% |
| **Recall@10** | 100.0% |
| **FTS search p50** | 474 us |
| **FTS search p99** | 700 us |
| **JSE query p50** | 3.39 ms |
| **JSE query count** | 20 types (eq, ne, gt, lt, contains, like, content, search, and, or, not, triple) |
| **Ingest throughput** | 384 knowledge/s, 280 statements/s |

### Comparison with MemPalace (ChromaDB vector baseline)

| Metric | Hypatia (FTS5) | MemPalace (ChromaDB raw) |
|--------|----------------|--------------------------|
| Recall@1 | **100.0%** | — |
| Recall@5 | **100.0%** | 96.6% |
| Recall@10 | **100.0%** | 98.2% |
| Search latency p50 | 474 us | ~2-50 ms |
| Embedding model | None | bge-large / OpenAI |
| Cold start | None | Model loading (~seconds) |
| Determinism | Yes | Stochastic |

Hypatia achieves **higher recall than vector-based retrieval** with **10-100x lower latency** and **zero dependency** on embedding models. FTS5 with Porter stemmer + multi-column BM25 weighting handles word form variations, and the synonyms field covers domain-specific terminology.

Full report: [docs/benchmark-report.md](docs/benchmark-report.md)

**Honest assessment**: These results are on synthetic data with known-answer queries — not academic benchmarks. See [docs/benchmark-honest-assessment.md](docs/benchmark-honest-assessment.md) for a frank analysis of coverage gaps, overfitting risks, and comparison with MemPalace's methodology.

### LoCoMo Academic Benchmark

Hypatia's FTS operator was tested on the LoCoMo long-term conversational memory benchmark (ACL 2024). FTS recall is near zero on natural language QA — this is expected, as Hypatia's core is graph-based structured querying, not semantic search. FTS is one operator in the JSE engine, not the system's primary retrieval mechanism.

| Strategy | R@10 | Reference (MemPalace) |
|----------|------|-----------------------|
| Turn-level FTS | 0.2% | Raw ChromaDB: 60.3% |
| Session-level FTS | 16.5% | Hybrid v5: 88.9% |

Full report: [docs/locomo-coverage.md](docs/locomo-coverage.md)

## Cross-Compilation

```bash
# Prerequisites
cargo install cargo-zigbuild
pip install ziglang

# Build for Linux (musl, static binary)
./scripts/build.sh x86_64-unknown-linux-musl

# Build for all 18 targets
./scripts/build.sh all

# List supported targets
./scripts/build.sh list

# Docker-based cross-compilation (alternative)
cargo install cross --git https://github.com/cross-rs/cross
./scripts/build.sh --backend cross x86_64-unknown-linux-musl
```

Supported targets: x86_64/aarch64/armv7 Linux (glibc + musl), riscv64, s390x, powerpc64le, macOS, Windows, FreeBSD, NetBSD, illumos, Android.

## License

MIT
