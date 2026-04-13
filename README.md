# Hypatia

"We can wander through the stacks of the Library of Alexandria, imagining the scrolls and the knowledge they contain. Its destruction is a warning: all we have is transient."——Alberto Manguel

AI-oriented memory management system. Stores structured knowledge as a graph of **Knowledge** entries (nodes) and **Statement** triples (edges), queried via a custom JSON Search Expression (JSE) language. Built on SQLite FTS5 + DuckDB, with configurable embedding models (local ONNX or remote API) for semantic vector search.

## Features

- **Knowledge Graph** -- Knowledge entries (named info points with tags) and Statement triples (subject-predicate-object with temporal ranges)
- **JSE Query Engine** -- JSON-based query language compiling to parameterized SQL + FTS5, supporting `$and`, `$or`, `$not`, `$eq`, `$ne`, `$gt`, `$lt`, `$contains`, `$like`, `$content`, `$search`, `$quote`, `$triple`, `$k-hop`
- **Dual-Database Storage** -- DuckDB for structured queries + vector search, SQLite FTS5 (Porter stemmer + multi-column BM25) for full-text search, auto-synchronized
- **Configurable Vector Search** -- Local ONNX models (BGE-M3 default) or remote API (OpenAI-compatible) for semantic similarity search via DuckDB cosine distance
- **Synonyms** -- Per-entry synonym lists for knowledge, per-position (subject/predicate/object) synonyms for statements, indexed in FTS
- **Shelf System** -- Named, connectable, exportable data directories for isolation
- **CLI + REPL** -- Full command-line interface with interactive mode (rustyline)
- **Agent Integration** -- Claude Code skill for natural-language-to-CLI translation
- **Cross-Platform** -- Build for 18+ targets (Linux, macOS, Windows, FreeBSD, NetBSD, illumos, Android)

## Quick Start

```bash
# Build
cargo build --release

# Download embedding model (BGE-M3, recommended)
mkdir -p ~/.hypatia/default
hf download BAAI/bge-m3 --local-dir /tmp/bge-m3
cp /tmp/bge-m3/onnx/model.onnx ~/.hypatia/default/embedding_model.onnx
cp /tmp/bge-m3/onnx/model.onnx_data ~/.hypatia/default/model.onnx_data
cp /tmp/bge-m3/onnx/tokenizer.json ~/.hypatia/default/tokenizer.json

# Create knowledge
hypatia knowledge-create "Rust" -d "systems programming language" -t "language,compiled"

# Create a relationship
hypatia statement-create "Rust" "is_a" "systems language"

# Full-text search
hypatia search "programming language"

# Vector similarity search (requires embedding model)
hypatia backfill    # generate embeddings for existing entries
hypatia similar "programming language"  # semantic search

# Structured query (JSE)
hypatia query '["$knowledge", ["$eq", "name", "Rust"]]'
hypatia query '["$statement", ["$triple", "Rust", "$*", "$*"]]'
hypatia query '["$knowledge", ["$search", "database migration"]]'

# Interactive REPL
hypatia repl
```

## Embedding Models

Hypatia supports multiple embedding backends, configured via `shelf.toml` in the shelf directory (e.g., `~/.hypatia/default/shelf.toml`).

### Default: BAAI/bge-m3 (Local ONNX)

No configuration needed — place model files in the shelf directory and Hypatia auto-detects them.

```bash
# Download from HuggingFace
hf download BAAI/bge-m3 --local-dir /tmp/bge-m3
cp /tmp/bge-m3/onnx/model.onnx ~/.hypatia/default/embedding_model.onnx
cp /tmp/bge-m3/onnx/model.onnx_data ~/.hypatia/default/model.onnx_data
cp /tmp/bge-m3/onnx/tokenizer.json ~/.hypatia/default/tokenizer.json
```

- Dimensions: 1024
- Max sequence length: 8192
- Multilingual (100+ languages including Chinese and English)

### Switching to Other Local Models

Create or edit `~/.hypatia/default/shelf.toml`:

```toml
[embedding]
provider = "local"
dimensions = 1536
max_seq_length = 8192
```

#### Alibaba-NLP/gte-Qwen2-1.5B-instruct

```bash
hf download Alibaba-NLP/gte-Qwen2-1.5B-instruct --local-dir /tmp/gte-qwen2
# Export ONNX first (requires optimum-cli or transformers)
# Then place files in shelf directory
```

- Dimensions: 1536
- Max sequence length: 8192
- Strong multilingual performance

#### google/embedding-gemma-300M (EmbeddingGemma)

```bash
hf download google/embedding-gemma-300M --local-dir /tmp/embedding-gemma
# Export ONNX and place files
```

- Dimensions: 768
- Max sequence length: 8192
- Lightweight, fast inference

#### jinaai/jina-embeddings-v5-text-small

```bash
hf download jinaai/jina-embeddings-v5-text-small-text-matching \
  onnx/model.onnx onnx/model.onnx_data tokenizer.json \
  --local-dir /tmp/jina-v5-small
cp /tmp/jina-v5-small/onnx/model.onnx ~/.hypatia/default/embedding_model.onnx
cp /tmp/jina-v5-small/onnx/model.onnx_data ~/.hypatia/default/model.onnx_data
cp /tmp/jina-v5-small/tokenizer.json ~/.hypatia/default/tokenizer.json
```

shelf.toml:
```toml
[embedding]
provider = "local"
dimensions = 1024
max_seq_length = 32768
pooling = "last_token"
```

- Dimensions: 1024
- Max sequence length: 32768
- 677M params, last-token pooling

#### jinaai/jina-embeddings-v5-text-nano

```bash
hf download jinaai/jina-embeddings-v5-text-nano-text-matching \
  onnx/model.onnx onnx/model.onnx_data tokenizer.json \
  --local-dir /tmp/jina-v5-nano
cp /tmp/jina-v5-nano/onnx/model.onnx ~/.hypatia/default/embedding_model.onnx
cp /tmp/jina-v5-nano/onnx/model.onnx_data ~/.hypatia/default/model.onnx_data
cp /tmp/jina-v5-nano/tokenizer.json ~/.hypatia/default/tokenizer.json
```

shelf.toml:
```toml
[embedding]
provider = "local"
dimensions = 768
max_seq_length = 8192
pooling = "last_token"
```

- Dimensions: 768
- Max sequence length: 8192
- 239M params, last-token pooling, fastest inference

### Remote API (OpenAI-Compatible)

For cloud-based embeddings without local model files:

```toml
[embedding]
provider = "remote"
api_url = "https://api.openai.com/v1/embeddings"
api_key_env = "OPENAI_API_KEY"
api_model = "text-embedding-3-small"
dimensions = 1536
```

Works with any OpenAI-compatible API:
- OpenAI (`text-embedding-3-small`, `text-embedding-3-large`)
- Azure OpenAI
- Local servers (Ollama, LM Studio, vLLM)
- Other providers ( Voyage AI, Cohere, Jina)

Set the API key as an environment variable:
```bash
export OPENAI_API_KEY="sk-..."
```

### Configuration Reference

| Field | Default | Description |
|-------|---------|-------------|
| `provider` | `"local"` | `"local"` for ONNX, `"remote"` for HTTP API |
| `dimensions` | `1024` | Embedding vector dimensions |
| `max_seq_length` | `8192` | Max tokenizer sequence length (local only) |
| `pooling` | `"mean"` | Pooling strategy: `"mean"`, `"cls"`, or `"last_token"` (local only) |
| `model_path` | `embedding_model.onnx` | ONNX model path, relative to shelf dir (local only) |
| `tokenizer_path` | `tokenizer.json` | Tokenizer path, relative to shelf dir (local only) |
| `api_url` | OpenAI URL | API endpoint URL (remote only) |
| `api_key_env` | `OPENAI_API_KEY` | Environment variable name for API key (remote only) |
| `api_model` | `text-embedding-3-small` | Model name sent to API (remote only) |

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
| `hypatia similar <query> [--limit N]` | Vector similarity search |
| `hypatia backfill` | Generate embeddings for all entries |
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
| `$k-hop` | K-hop graph traversal | `["$k-hop", "Alice", "$*", 2]` |

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

# K-hop: all entities reachable from Alice in 2 hops
hypatia query '["$statement", ["$k-hop", "Alice", "$*", 2]]'

# K-hop: follow "knows" edges from Alice, 3 hops deep
hypatia query '["$statement", ["$k-hop", "Alice", "knows", 3]]'
```

## Architecture

```
src/
├── cli/            # CLI commands + REPL (clap + rustyline)
├── embedding/      # Embedding providers (ONNX local + remote API)
├── engine/         # JSE parser, AST, evaluator, SQL builder
├── model/          # Knowledge, Statement, Content, Query types
├── service/        # Business logic (dual-write to DuckDB + SQLite)
├── storage/        # DuckDB store, SQLite FTS5 store, shelf manager
├── lab.rs          # Top-level API facade
├── error.rs        # Error types
├── lib.rs          # Module declarations
└── main.rs         # Entry point
```

Each **shelf** is a directory containing `data.duckdb` (structured data), `index.sqlite` (FTS5 index), optional `shelf.toml` (embedding config), and embedding model files. The service layer keeps both databases in sync via dual-write.

## Benchmark

### LoCoMo Academic Benchmark

Tested on the LoCoMo long-term conversational memory benchmark (ACL 2024, 10 conversations, 1,540 non-adversarial QA pairs, 6,426 entries). Measures whether the correct evidence passage is found in top-K results.

#### Embedding Model Comparison

| Model | Params | Dims | Pooling | R@1 | R@5 | R@10 | Vec p50 |
|-------|--------|------|---------|-----|-----|------|---------|
| FTS (BM25, baseline) | — | — | — | 0.2% | 0.2% | 0.2% | 815 ms |
| Jina v5 text-nano | 239M | 768 | last_token | 8.7% | 19.6% | 25.1% | 25 ms |
| gte-multilingual-base | 305M | 768 | mean | 13.6% | 31.9% | 42.5% | 25 ms |
| gte-Qwen2-1.5B-instruct | 1.5B | 1536 | mean | 17.7% | 43.6% | 54.9% | 105 ms |
| Jina v5 text-small | 677M | 1024 | last_token | 24.9% | 50.7% | 60.4% | 54 ms |
| **BAAI/bge-m3** | **568M** | **1024** | **mean** | **38.6%** | **65.7%** | **75.2%** | **43 ms** |

#### BGE-M3 by Category (default model)

| Category | N | FTS R@10 | Vector R@10 | Improvement |
|----------|---|----------|-------------|-------------|
| Single-hop | 841 | 0.1% | **76.1%** | +76.0pp |
| Multi-hop | 282 | 0.4% | **75.5%** | +75.2pp |
| Temporal | 321 | 0.0% | **80.4%** | +80.4pp |
| Open-domain | 96 | 1.0% | **49.0%** | +47.9pp |
| **Overall** | **1540** | **0.2%** | **75.2%** | **+75.0pp** |

#### Comparison with MemPalace (ChromaDB)

| Metric | Hypatia (BGE-M3) | MemPalace ChromaDB | MemPalace Hybrid v5 |
|--------|-------------------|--------------------|---------------------|
| R@10 | 75.2% | 60.3% | 88.9% |
| Vec latency p50 | 43 ms | ~2-50 ms | ~2-50 ms |
| Ingest time | ~9 min | — | — |

**Note**: LoCoMo is designed as a semantic challenge, which is why BM25 FTS recall is near zero. Vector search closes this gap effectively. BGE-M3 (568M, 1024d) achieves the highest recall (R@10=75.2%) with good latency (43ms) and strong multilingual coverage (100+ languages) — the best default choice. Jina v5 text-small (677M) offers mid-range quality (R@10=60.4%) at 54ms. Notably, larger models don't always win: gte-Qwen2 (1.5B) and gte-multilingual-base (305M, ModernBERT) both underperform BGE-M3 on this benchmark.

#### Run the Benchmark

```bash
# Download test data
curl -sL https://huggingface.co/datasets/Percena/locomo-mc10/resolve/main/raw/locomo10.json -o locomo10.json

# Run (requires embedding model in ~/.hypatia/default/)
LOCOMO_DATA=locomo10.json LOCOMO_RESULTS=results.jsonl \
  cargo test --test locomo --release -- --nocapture
```

### Synthetic Benchmark

Benchmark uses synthetic data with planted needles (known-answer entries) to measure retrieval quality, following MemPalace's methodology.

#### Run

```bash
# Small scale (1K knowledge, 2K statements, ~12s)
cargo test --test bench

# With JSON report
BENCH_REPORT=report.json cargo test --test bench

# Larger scales
BENCH_SCALE=medium cargo test --test bench --release
BENCH_SCALE=large cargo test --test bench --release
```

#### Results (small scale, debug build, Apple Silicon)

1K knowledge, 2K statements, 20 needles, 20 JSE query types (x3 runs each).

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

Full report: [docs/benchmark-report.md](docs/benchmark-report.md)

**Honest assessment**: These results are on synthetic data with known-answer queries -- not academic benchmarks. See [docs/benchmark-honest-assessment.md](docs/benchmark-honest-assessment.md) for a frank analysis.

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
