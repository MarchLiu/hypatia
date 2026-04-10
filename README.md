# Hypatia

AI-oriented memory management system. Stores structured knowledge as a graph of **Knowledge** entries (nodes) and **Statement** triples (edges), queried via a custom JSON Search Expression (JSE) language. Built on SQLite FTS5 + DuckDB, with zero external model dependencies.

## Features

- **Knowledge Graph** -- Knowledge entries (named info points with tags) and Statement triples (subject-predicate-object with temporal ranges)
- **JSE Query Engine** -- JSON-based query language compiling to parameterized SQL + FTS5, supporting `$and`, `$or`, `$not`, `$eq`, `$ne`, `$gt`, `$lt`, `$contains`, `$search`, `$quote`
- **Dual-Database Storage** -- DuckDB for structured queries, SQLite FTS5 for full-text search, auto-synchronized
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
hypatia query '["$statement", ["$eq", "subject", "Rust"]]'
hypatia query '["$knowledge", ["$search", "database migration"]]'

# Interactive REPL
hypatia repl
```

## CLI Reference

| Command | Description |
|---------|-------------|
| `hypatia connect <path> [-n <name>]` | Connect to a shelf directory |
| `hypatia disconnect <name>` | Disconnect from a shelf |
| `hypatia list` | List connected shelves |
| `hypatia knowledge-create <name> [-d <data>] [-t <tags>]` | Create a knowledge entry |
| `hypatia knowledge-get <name>` | Get a knowledge entry |
| `hypatia knowledge-delete <name>` | Delete a knowledge entry |
| `hypatia statement-create <subj> <pred> <obj> [-d <data>]` | Create a triple |
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
| `$search` | Full-text search | `["$search", "database migration"]` |
| `$and` | Logical AND | `["$and", cond1, cond2]` |
| `$or` | Logical OR | `["$or", cond1, cond2]` |
| `$not` | Logical NOT | `["$not", cond]` |
| `$quote` | Prevent evaluation | `["$quote", ["$eq", "x", "y"]]` |

### Examples

```bash
# All knowledge entries
hypatia query '["$knowledge"]'

# Knowledge named "Rust" with tag "systems"
hypatia query '["$knowledge", ["$and", ["$eq", "name", "Rust"], ["$contains", "tags", "systems"]]]'

# Statements about Alice
hypatia query '["$statement", ["$eq", "subject", "Alice"]]'

# FTS search within knowledge
hypatia query '["$knowledge", ["$search", "query optimization"]]'

# Statements where subject is Alice or Bob
hypatia query '["$statement", ["$or", ["$eq", "subject", "Alice"], ["$eq", "subject", "Bob"]]]'
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

| Metric | Result |
|--------|--------|
| **Recall@1** | 95.0% (19/20 needles) |
| **Recall@5** | 95.0% |
| **Recall@10** | 95.0% |
| **FTS search p50** | 382 us |
| **FTS search p99** | 825 us |
| **JSE query p50** | 2.99 ms |
| **Ingest throughput** | 369 knowledge/s, 281 statements/s |

### Comparison with MemPalace (ChromaDB vector baseline)

| Metric | Hypatia (FTS5) | MemPalace (ChromaDB raw) |
|--------|----------------|--------------------------|
| Recall@5 | 95.0% | 96.6% |
| Recall@10 | 95.0% | 98.2% |
| Search latency p50 | 382 us | ~2-50 ms |
| Embedding model | None | bge-large / OpenAI |
| Cold start | None | Model loading (~seconds) |
| Determinism | Yes | Stochastic |

Hypatia achieves comparable recall to vector-based retrieval with **10-100x lower latency** and **zero dependency** on embedding models. The trade-off is that FTS cannot handle semantic synonyms or paraphrase matching.

Full report: [docs/benchmark-report.md](docs/benchmark-report.md)

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

Private project. All rights reserved.
