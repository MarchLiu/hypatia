# Hypatia 架构重构方案

> 状态：已定稿（4 项待定点已拍板） · 2026-08
> 背景：duckdb+sqlite 双库 → 单源真相 SQLite + 外置 usearch 向量索引 + 内部 JSON 倒排 + HRT 三元组 + 无痛迁移

## 0. 目标与总览

```
旧：  data.duckdb(数据+向量+json)  +  index.sqlite(FTS)          —— 双源真相、SPO
新：  hypatia.sqlite(唯一源真相 + json倒排 + FTS)  +  vectors/*.usearch(可重建缓存)  —— HRT
```

核心原则：

- **SQLite = 唯一源真相**：数据、`json_index` 倒排、FTS5、向量 BLOB 列全在库内，WAL + busy_timeout 保证多 Agent 无冲突协作。
- **usearch = 派生缓存**：向量索引「可重建」，不参与源真相；坏了/陈旧从 BLOB 列重算。
- **三元组语义对齐图惯例**：`(subject, predicate, object)` → `(head, relation, tail)`。
- **保留复杂 JSON 访问能力**：`format=json` 时 `data` 内嵌 JSON 解包进倒排，`@>` 结构包含用 `json_contains` UDF recheck 支撑。

## 1. 数据模型：HRT 三元组

### 1.1 Rust 结构

```rust
pub struct StatementKey { pub head: String, pub relation: String, pub tail: String }
// new(head, relation, tail); to_csv_key() = "head,relation,tail"; from_csv()/csv_split() 沿用
```

### 1.2 `statement` 表

```sql
CREATE TABLE statement (
    triple     TEXT PRIMARY KEY,   -- 'head,relation,tail'（CSV，沿用 csv_escape 规则）
    head       TEXT NOT NULL,
    relation   TEXT NOT NULL,
    tail       TEXT NOT NULL,
    content    TEXT NOT NULL,
    embedding  BLOB,               -- f32 小端，源真相
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%d %H:%M:%f','now')),
    tr_start   TEXT,
    tr_end     TEXT
);
CREATE INDEX idx_stmt_head     ON statement(head);
CREATE INDEX idx_stmt_relation ON statement(relation);
CREATE INDEX idx_stmt_tail     ON statement(tail);
```

### 1.3 重命名影响面（subject→head / predicate→relation / object→tail）

| 位置 | 改动 |
|---|---|
| `model/statement.rs` | `StatementKey` 三字段、`Statement` |
| `storage/*` | `statement` 列名 + 索引名 + 全部 SQL（含 k-hop CTE） |
| `engine/operators.rs` | `resolve_field` 的列名表；`$triple` 的 columns 数组 |
| `engine/evaluator.rs` | `$k-hop`（`h.object=s.subject`→`h.tail=s.head`、`s.predicate`→`s.relation`）；`$not-summaried`（`predicate='summary'`→`relation='summary'`、`statement.object`→`statement.tail`） |
| `service/statement.rs` | 字段引用 |
| `cli/commands.rs` | `StatementCreate/Delete` 参数与输出 |
| `model/content.rs` | `Synonyms::Positional` 的 key：`"head"/"relation"/"tail"` |
| `skills/`、`docs/` | SKILL.md 的 `$triple`/statement-create 示例；design.md/memory.md 的 schema 与「Predicates→Relations」表 |

> 关系**值**（`is_a`、`knows`、`belongTo`、`summary`…）不变，变的是**列名/概念词**。

## 2. 存储架构：单 SQLite + 外置 usearch

### 2.1 目录形态

```
shelf/
├── hypatia.sqlite              # 唯一源真相
├── vectors/
│   ├── knowledge.usearch       # 可重建缓存
│   └── statement.usearch
├── archives/
└── shelf.toml                  # 嵌入模型配置（不变）
```

### 2.2 `hypatia.sqlite` 完整 schema

```sql
CREATE TABLE meta(k TEXT PRIMARY KEY, v TEXT NOT NULL);
INSERT INTO meta VALUES('schema_version','2');

CREATE TABLE knowledge(
    name TEXT PRIMARY KEY, content TEXT NOT NULL,
    embedding BLOB, created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%d %H:%M:%f','now'))
);

-- statement 表见 §1.2

-- 统一文档锚点：doc_id = usearch key = json_index.doc_id = FTS content_rowid
-- 【定稿】不保留 content 副本：搜索结果 JOIN knowledge/statement 现取
CREATE TABLE docs(
    id INTEGER PRIMARY KEY,            -- rowid
    catalog TEXT NOT NULL,             -- 'knowledge' | 'statement'
    key TEXT NOT NULL,                 -- name / triple CSV
    fts_key TEXT DEFAULT '', fts_data TEXT DEFAULT '',
    fts_tags TEXT DEFAULT '', fts_synonyms TEXT DEFAULT '',
    UNIQUE(catalog, key)
);

CREATE TABLE json_index(
    doc_id INTEGER NOT NULL, path TEXT NOT NULL, kind TEXT NOT NULL,
    value TEXT, array_index INTEGER,
    PRIMARY KEY(doc_id, path, array_index, value)
) WITHOUT ROWID;
CREATE INDEX idx_json_path_value ON json_index(path, value, doc_id);

CREATE VIRTUAL TABLE docs_fts USING fts5(
    fts_key, fts_data, fts_tags, fts_synonyms,
    content='docs', content_rowid='id', tokenize='porter unicode61'
);
```

连接参数（每次 open 设置）：

```rust
PRAGMA journal_mode = WAL;
PRAGMA busy_timeout = 5000;
PRAGMA synchronous = NORMAL;
PRAGMA foreign_keys = ON;
```

**`$search` 路径变化**（因 docs 去掉 content）：FTS 命中 `docs.id` 后 JOIN 源表取 content：

```sql
SELECT d.catalog, d.key, COALESCE(k.content, s.content) AS content, bm25(docs_fts,10.0,1.0,5.0,3.0) AS rank
FROM docs d JOIN docs_fts f ON d.id = f.rowid
LEFT JOIN knowledge k ON d.catalog='knowledge' AND k.name = d.key
LEFT JOIN statement s ON d.catalog='statement' AND s.triple = d.key
WHERE docs_fts MATCH ?1 [AND d.catalog = ?2]
ORDER BY rank LIMIT ?3 OFFSET ?4;
```

## 3. JSON 倒排索引（内部 path-tree）

写入时用 serde_json 遍历 `Content`（含解包，见 §3.1），叶子拍成 `(doc_id, path, kind, value, array_index)` 行；删除按 `doc_id` 清行。

### 3.1 【定稿】解包 `format=json` 的 `data` 内嵌 JSON

- content 顶层键（tags/scopes/figures/synonyms/format…）照常入倒排；
- 当 `format == "json"` 时，将 `data` 字符串再 parse 一次，其内部键以 **`data.filename` / `data.size_bytes` / `data.mime_type`** 这类**带前缀路径**追加进 `json_index`（一层即可，不递归深层）；
- 这使 `$content {"mime_type":"image/png"}`、`$has data.filename "…"` 等 SKILL.md 里承诺过的查询**真正可用**，也是保留「复杂 JSON 访问能力」的落点。

### 3.2 算子映射

| 算子 | 实现 | 命中索引 |
|---|---|---|
| `$has`（成员 ≈ `?`/`?\|`） | `doc_id IN (SELECT doc_id FROM json_index WHERE path=? AND value=?)` | ✅ 纯索引 |
| `$content`（键值相等） | 每个 key 一条 path+value 谓词求交 | ✅ 纯索引 |
| `$eq`/`$ne` | 同上（`$ne`=NOT EXISTS） | ✅ 纯索引 |
| `$gt/$lt/$gte/$lte` | 【定稿】`CAST(json_extract(content,'$.f') AS INTEGER/REAL) <op> ?`（数值化，修复历史字符串比较 bug） | ⚠️ JSON1 |
| `$contains`（子串） | `json_extract(content,'$.f') LIKE '%v%'` | ❌ JSON1 |
| `$like` | `json_extract(content,'$.f') LIKE ?` | ❌ JSON1 |
| `$triple` | `head/relation/tail = ?` | ✅ 列索引 |
| `$k-hop` | 递归 CTE（HRT） | ✅ 列索引 |
| `$search` | FTS5 | ✅ |
| `$similar` | usearch | ✅ |
| `$not-summaried` | anti-join + json_index tag 过滤 | ✅ |
| **`@>` 结构包含（本方案内实现）** | 倒排召回 + `json_contains(content,?)` UDF recheck | ✅ 召回+recheck |

### 3.3 `json_contains` UDF（因定稿 #4 而进入本方案范围）

- Rust 实现 PG `@>` / jq `contains()` 语义（递归 object/array/string，数组 multiset），`jq` 仅作测试基准；
- `conn.create_scalar_function("json_contains", 2, SQLITE_UTF8|DETERMINISTIC, …)`，per-connection 注册；
- 用于 `@>` 深层/数组场景的 recheck（`?`/`?|`/`?&`/`#>` 与浅层 `@>` 均纯索引、零 recheck）。

## 4. 向量索引：usearch

### 4.1 `VectorIndex` trait

```rust
trait VectorIndex {
    fn insert(&mut self, doc_id: u64, vector: &[f32]) -> Result<()>;
    fn delete(&mut self, doc_id: u64) -> Result<()>;
    fn search(&self, query: &[f32], k: usize) -> Result<Vec<(u64, f32)>>;
    fn save(&self, path: &Path) -> Result<()>;
    fn load(path: &Path, dim: usize) -> Result<Self>;
}
// 实现：UsearchIndex（本方案）；未来 Vec1Index / PgvectorIndex / DiskANN 挂同一 trait
```

### 4.2 usearch 集成

- crate `usearch` v2（Apache-2.0）；`IndexOptions{dimensions, metric: Cos}`；key = `docs.id`(u64)；
- 写路径：BLOB 入 SQLite（事务）→ 更新内存 usearch → 定期 `save()` 快照到 `vectors/*.usearch`；
- 读路径：`$similar` 走 usearch（HNSW，无需训练、增量友好）；
- 一致性：**可重建缓存**——`search` 前对账「索引条目数 == embedding BLOB 行数」，不等即提示/触发 `backfill` 重建。

### 4.3 【定稿】并发模型：单写多读

- **写索引单写者**：进程级文件锁（`vectors/.lock`）；同一时刻仅一个进程更新并 `save()` 索引；
- **读多路**：其余进程/Agent `Index::view()` mmap 只读，不阻塞、不冲突；
- **SQLite 侧照常 WAL 多写**（源真相无单写限制）——两个 Agent 并发 `knowledge-create` 都成功入 SQLite，索引快照由持锁者统一收口；未进快照的向量在下次对账/重建时补齐（最终一致，语义检索可容忍）；
- 此模型对「多 Agent 同时访问」反而是优势：读永不阻塞、写永不冲突、源真相零撕裂。

## 5. FTS：保留 FTS5

jieba 预分词 + porter、BM25 权重（key=10/tags=5/synonyms=3/data=1）不变；content 表换为 `docs`（无 content 副本，见 §2.2）。

## 6. 版本迁移工具

- **触发**：目录存在 `data.duckdb` 且无 `hypatia.sqlite`（或 `schema_version < 2`）→ 自动迁移；另提供 `hypatia migrate <shelf>`（含 `--dry-run` 预检：行数/维度/坏 JSON）。
- **流程**（单事务）：读 duckdb（SPO）→ 写 sqlite（HRT：`subject→head/predicate→relation/object→tail`，triple CSV 重编码，`FLOAT[N]`→BLOB）→ 建 docs → 重建 json_index（含解包）→ 重建 docs_fts → 写 `schema_version=2` → 由 BLOB 建 `vectors/*.usearch` → 旧文件改名 `.bak`。
- **幂等**：以 `schema_version` 为闸，达标即跳过；临时文件 + 原子 rename，失败不动原文件。

## 7. 分阶段实施

| 阶段 | 内容 | 验收 |
|---|---|---|
| **P1 存储核心 + 迁移** | 新 sqlite_store（HRT + docs + FTS5 + WAL）·`OpenShelf` 单连接 · 迁移工具 | 旧 shelf 打开即迁移；CRUD/FTS/查询全绿 |
| **P2 json 倒排 + 算子** | json_index 填充（含迁移数据 backfill）+ 解包 data.json + 算子重写 + `json_contains` UDF（含 `@>`） | `$has/$content/@>` 走索引、`$gt` 数值化，测试覆盖 |
| **P3 usearch 向量** | VectorIndex trait + UsearchIndex + `$similar` + 单写多读锁 + backfill 对账 | `$similar` 可用、可重建、多 Agent 并发验证 |
| **P4 收尾** | Cargo 去 duckdb · SKILL.md/docs HRT 化 · tests/REPL/CLI 回归 | `cargo test` 全绿、文档一致 |

> **实施偏差（2026-08-28）**：① `duckdb` crate 依赖保留至 P4 —— `migrate.rs` 需要它读取旧 `data.duckdb`，所有 shelf 迁移完成后才移除；② `json_index` 表在 P1 建表但**不填充**，P2 增加 backfill 为已迁移 shelf 补建倒排；③ P1 的向量检索以 **Rust 暴力 kNN**（`embedding` BLOB 列）占位，P3 换 usearch。

## 8. 已拍板决策记录

| # | 决策 |
|---|---|
| 1 | `docs` **不保留** content 副本，搜索 JOIN 源表现取 |
| 2 | 向量索引**单写多读**（文件锁 + mmap 只读）；SQLite 侧 WAL 多写不受限 |
| 3 | `$gt/$lt` **数值化比较**（CAST），修复历史字符串比较 |
| 4 | **解包** `format=json` 的 `data` 内嵌 JSON 入倒排（保留复杂 JSON 访问能力，`@>`+`json_contains` 进入本方案范围） |
