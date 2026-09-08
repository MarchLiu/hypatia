# 可配置 PostgreSQL + pgvector 后端需求分析

## 1. 结论与已确认范围

可以在保留现有后端的前提下，增加由配置选择的完整 PostgreSQL + pgvector 后端。用户已确认：PostgreSQL 接管知识、三元组、结构化查询、全文检索和向量存储；不是仅把向量从 SQLite 分离到 PostgreSQL。

建议提供两个完整实现：

- `sqlite`：现有 SQLite + FTS5 + JSON 路径索引 + usearch，仍为缺省后端。
- `pgvector`：PostgreSQL + JSONB + PostgreSQL 全文检索 + pgvector。此名称代表完整数据库后端，pgvector 本身只是 PostgreSQL 的向量扩展。

沿用每个 shelf 的 `shelf.toml`，新增 `[storage]`。一个 shelf 对应一个后端，同一进程可连接使用不同后端的多个 shelf。没有新配置时，旧 shelf 按原方式打开。显式选择 PostgreSQL 后，连接或配置失败必须报错，不能静默回退 SQLite。

完整后端接入需要覆盖读写接口、存储生命周期、迁移及检索能力。JSE 的结构化条件可以直接对等编译为 PostgreSQL JSONB 查询，保留现有 Parser/AST 并抽取 SQL 方言层即可；少数 SQLite 边界行为在方言层局部适配。主要工作量集中在完整存储接口、导出与迁移、写入一致性，以及全文检索等能力的适配。本文只交付分析与设计建议，没有实现后端或执行数据库变更。

## 2. 当前实现及证据

以下事实基于当前工作区源代码，而非历史设计文档。链接行号是本次检查时的位置。

| 事实 | 源码依据 | 对方案的影响 |
| --- | --- | --- |
| 默认使用 rusqlite 与 usearch；DuckDB 只在 `legacy-migration` feature 中启用 | [Cargo.toml](../Cargo.toml) | 应保留 SQLite 默认构建和旧数据迁移路径 |
| `Storage` 只有 SQL 查询、全文检索、语义检索、图遍历四类读方法 | [storage/mod.rs](../src/storage/mod.rs#L38) | 不能仅新增 `impl Storage for PgStore` 就覆盖所有命令 |
| `OpenShelf.store` 固定为 `SqliteStore`，还直接持有向量文件索引 | [shelf_manager.rs](../src/storage/shelf_manager.rs#L10) | 后端选择需进入 shelf 打开流程，usearch 生命周期需封装到本地后端 |
| CRUD 服务直接调用 `shelf.store`，再单独生成和写入向量 | [knowledge.rs](../src/service/knowledge.rs#L14)、[statement.rs](../src/service/statement.rs#L16) | 需统一写接口，并定义内容与向量的一致性 |
| Lab 也直接读取 store，批量补向量调用 SQLite 风格方法 | [lab.rs](../src/lab.rs#L68)、[lab.rs](../src/lab.rs#L239) | 适配范围包括业务入口与维护流程 |
| `knowledge` / `statement` 存 JSON 文本和 embedding BLOB；`docs`、FTS5、`json_index` 为辅助结构 | [sqlite_store.rs](../src/storage/sqlite_store.rs#L27) | PostgreSQL 可保留业务模型，辅助索引无需照搬 |
| `SqlBuilder` 使用 `?` 参数；operators 生成 JSON1 / json_index SQL；evaluator 还直接拼接查询 | [sql_builder.rs](../src/engine/sql_builder.rs#L3)、[operators.rs](../src/engine/operators.rs#L90)、[evaluator.rs](../src/engine/evaluator.rs#L192) | 方言依赖不只在 SqlBuilder |
| 配置目前只解析 `[embedding]`，文件读取或 TOML 解析失败会使用默认值 | [config.rs](../src/embedding/config.rs#L488) | 新存储配置必须使用返回 Result 的严格加载器 |
| registry 保存 shelf 名称到本地目录的映射；默认 shelf 固定来自 home 下的 `.hypatia/default` | [shelf_registry.rs](../src/storage/shelf_registry.rs)、[shelf_manager.rs](../src/storage/shelf_manager.rs#L320) | 可继续以本地目录承载远端 shelf 的连接配置 |
| 导出直接复制 SQLite 文件、vectors 和 archives | [shelf_manager.rs](../src/storage/shelf_manager.rs#L453) | PostgreSQL 必须新增逻辑导出路径 |

现有语义检索优先 usearch，失败或无命中时会走 SQLite BLOB 的 Rust 暴力 kNN。这里的回退属于同一后端内部的索引策略，不应扩展为 PostgreSQL 失败后切换数据源。

## 3. 配置与启动设计

下面全部是建议新增的配置，当前版本尚不支持。

保持默认后端：

```toml
[storage]
backend = "sqlite"
```

选择 PostgreSQL：

```toml
[storage]
backend = "pgvector"

[storage.postgres]
url_env = "HYPATIA_POSTGRES_URL"
schema = "hypatia_team_memory"
connect_timeout_seconds = 5
statement_timeout_ms = 30000

[storage.vector]
index = "hnsw"
metric = "cosine"

[embedding]
provider = "local"
model = "BAAI/bge-m3"
dimensions = 1024
```

数据库连接串通过指定环境变量读取，密码不写入仓库或导出文件；TLS 参数由连接配置及驱动实际支持的 TLS 实现共同处理，不能只写一个 `sslmode` 就认为已经实现证书验证。embedding provider 与存储后端独立，远程 embedding 也可以配合本地 SQLite，反之亦然。

建议统一加载成 `ShelfSettings { storage, embedding }`，保留现有路径模型的职责，不继续让 embedding 模块拥有完整 shelf 配置的解析入口。规则如下：

1. 没有 `shelf.toml` 或没有 `[storage]` 时，选择 SQLite。默认 shelf 的配置位置为 `~/.hypatia/default/shelf.toml`；其他 shelf 在各自目录。
2. 已存在的文件不可读、TOML 非法、backend 未知、PostgreSQL 必填项缺失均报配置错误。存储配置拼写错误不能变成默认本地写入。明确记录：非法旧配置以前可能被忽略，严格加载后会暴露错误。
3. 首期仅在连接 shelf 时加载配置。修改文件后通过重新连接或重启生效，不做运行中的自动切换。
4. 先解析和验证配置，再创建对应后端资源。PostgreSQL shelf 不创建或迁移 `hypatia.sqlite`，也不初始化 `vectors/`。
5. 连接时验证扩展、schema 版本、embedding 模型标识、维度和距离类型。缺扩展或元数据不匹配要明确提示处理方法。
6. PostgreSQL 不可用时，默认 shelf 启动失败应可诊断；非默认 shelf 可保留当前恢复失败后标记未连接的模式，但对其执行命令必须明确失败。

建议首期按 shelf 配置，不再引入全局存储默认值和 CLI 覆盖优先级。现有 registry 仍保存目录，不保存连接密码。`connect <path>` 仍可使用：远端 shelf 的这个 path 表示本地配置及附件目录。

## 4. 存储抽象与模块边界

建议保留 `OpenShelf` 作为 shelf 上下文，把具体后端状态移入实现内部：

```text
CLI / REPL / Lab
        |
CRUD Services + JSE Parser / Evaluator / SQL Dialect
        |
OpenShelf { settings, embedder, backend }
        |
        +-- LocalBackend { SqliteStore, VectorIndexes }
        |       SQLite + FTS5 + json_index + usearch
        |
        +-- PostgresBackend { client, schema, metadata }
                PostgreSQL + JSONB + tsvector + pgvector
```

面向领域操作定义对象安全的后端接口，具体方法签名在实现时根据当前模型收敛：

| 接口组 | 应覆盖的操作 |
| --- | --- |
| 数据读写 | knowledge / statement 的创建、读取、更新、删除，原有重复键和不存在错误语义 |
| 查询 | 执行对应方言的参数化查询、全文检索、接收向量的相似度检索、k-hop、未归纳知识查询 |
| embedding | 分页枚举缺失或过期向量、带内容版本条件写回、统计状态 |
| 生命周期 | 检查后端能力和版本、初始化或升级 schema、导出、重建派生索引、显式 flush |

`OpenShelf` 或服务层继续负责调用 embedding provider；后端只接收向量及关联元数据，数据库连接内部不执行模型推理。usearch 的增删、快照及重建全部留在 LocalBackend 内部。PG 的向量索引由数据库维护，不能要求它模拟本地 `vector_upsert` / `save_vector_indexes` 文件操作。

推荐 `Box<dyn ShelfBackend>` 作为边界，必要时拆分读写与维护 trait；只有两个实现时，内部用 enum 分派也可行。关键是业务层不再依赖公开的 `SqliteStore` 或 `rusqlite::Connection`。已有基于 `Storage` 的 evaluator mock 可分阶段适配。若已有外部 Rust 用户依赖 `OpenShelf.store`，应将其列为库 API 兼容性变化，不能因为 CLI 不变就认为完全无破坏。

当前项目以同步调用为主，首期优先评估同步 `postgres` 驱动及 `pgvector` Rust 类型适配。可在 PG 实现内部封装客户端可变借用，避免为新增后端把 CLI 全面异步化。连接池依据实际并发需求引入；目前每个 shelf 的连接恢复会使连接数随 shelf 数增长，应设超时并考虑后续惰性连接。

建议使用可选 Cargo feature `postgres-backend`：默认构建不拉取 PG 依赖；发布支持版包含该 feature 后，用户仅改配置即可选择两种后端。未编译该能力却配置 `pgvector` 时，明确报“当前二进制不支持”。需同时交付 feature 构建与不带 feature 的回归验证。

## 5. JSE 直接对等编译为 JSONB 查询

JSE 的结构化查询可以直接映射到 PostgreSQL SQL 与 JSONB 原生操作符。推荐保留现有 Parser、AST 和 evaluator 的查询流程，在 operators、SqlBuilder 及特殊查询生成处抽取方言接口，分别生成 SQLite 和 PostgreSQL 的参数化 SQL。当前两个后端的需求不以新增 QueryPlan 或另一套类型化条件树为前提；未来出现查询优化需求时再评估。

### 5.1 常见查询的直接映射

| JSE 查询意图 | PostgreSQL 生成形式 |
| --- | --- |
| 普通列相等或大小比较 | `name = $1`、`created_at >= $1` |
| JSON 字符串字段读取 | `content ->> 'format'` |
| JSON 嵌套路径读取 | `content #> $1::text[]`；文本值使用 `#>>` |
| 字符串数组包含元素 | `(content -> 'tags') ? $1::text` |
| 字符串数组包含任一元素 | `(content -> 'tags') ?\| $1::text[]` |
| JSON 结构包含 | `content @> $1::jsonb` |
| 与、或、非 | `AND`、`OR`、`NOT` |

表中的数组存在操作符针对字符串元素；其他标量类型按 JSE 的类型契约生成相应表达式。动态 JSON 路径使用参数绑定，静态列名由已验证的字段映射生成。

例如“tags 包含 rust，且 format 为 markdown”，可以生成以下查询（示例省略 schema 限定）：

```sql
SELECT name, content, created_at
FROM knowledge
WHERE (content -> 'tags') ? $1::text
  AND content ->> 'format' = $2::text
ORDER BY created_at DESC
LIMIT $3 OFFSET $4;
```

参数依次为 rust、markdown、limit 和 offset。PostgreSQL 直接使用 JSONB 查询，无需复制 SQLite 的 json_index 辅助表。生成 SQL 时由方言层分配占位符和参数类型，不对已生成的 SQLite SQL 做文本替换。

### 5.2 局部兼容事项

以下是现有 SQLite 行为和数据表示的局部边界，适合在方言层处理并用对照用例验证，不构成引入新查询架构的理由：

- 参数：SQLite 的 `?` 与 PG 的 `$1`、`$2`；类型由编译器指定，尤其 JSONB、时间、数值和 LIMIT/OFFSET。不能盲目替换问号，PG 本身也存在 `?` JSONB 运算符。
- JSON 路径：JSON1 与 JSONB 提取方式不同；字段路径仍应验证并作为值绑定，schema 等标识符要严格验证及正确引用。
- 数值比较：当前 SQLite `CAST(... AS REAL)` 对非法文本的行为与 PG 强制类型转换不同。需针对数字、数字字符串、非数字、缺失、null 定义兼容规则及测试，不能让查询因一条非数值内容整批失败。
- `$contains`、`$has`：现有 membership 使用字符串 token，会模糊部分 JSON 类型；PG 的 JSONB 运算符有自身类型规则，不能直接假设相同。LIKE 的大小写、通配符和转义也要逐项测试。
- `$json-contains`：优先利用 JSONB 包含查询，但当前 Rust `json_contains` 与 PG `@>` 并非所有边界都相同，例如数组包含标量。首期用对照用例决定兼容 SQL 或显式限制，不能依据代码注释宣称完全等价。
- `Content.data` 是字符串，即使 `format=json` 也不是内嵌对象。PG JSONB 保存整个 Content 后，`data.foo` 仍需受控解析内部字符串，不能直接写成普通嵌套对象访问。可维护经验证的派生 JSONB payload，保持原始 data 字符串不变。
- `$not-summaried` 的关联查询目前写在 evaluator，需一并纳入方言层或后端实现，不能只改通用 SqlBuilder。
- `$k-hop` 使用递归 CTE；SQLite 允许的 `GROUP BY triple` 加裸列选择不能直接用于 PG。用最小深度聚合后关联原表或窗口函数返回结果，保留关系过滤、去重和深度边界；对环路及高分支图设置上限或执行超时。

此外，当前 JSE 的 `$similar` / `$search` 是先取候选 key，再做普通查询并按 `created_at DESC` 排序。这与“过滤后取相似度 top-k”并不相同，也不会天然保留距离排序。首期按既有 JSE 行为做兼容验收；若需要过滤下推、混合检索或距离排序，作为明确的查询语义升级单独设计，避免 PG 后端悄悄改变同一个表达式的含义。

## 6. PostgreSQL 数据与索引设计

### 6.1 shelf 隔离

首期推荐一 shelf 一 schema，配置中显式给出 schema，数据库可共享。业务主键因此无需全面增加 `shelf_id`，不同 shelf 可以使用不同向量维度。每个 schema 保存版本与 embedding 元数据，两个客户端只有显式使用同一数据库/schema 才共享数据。不要直接以本地路径或可随意变更的 shelf 显示名推导远端身份。

schema 是命名与数据组织边界，不自动等同安全隔离。多用户部署须搭配数据库角色和 schema 权限。大量 shelf 的集中托管可后续评估共享表加 shelf_id、分区及 RLS，首期不引入该复杂度。所有 SQL 使用明确 schema 限定，避免连接复用时 search_path 导致串库。

### 6.2 表结构

保留 `knowledge(name, content, created_at)` 和 `statement(triple, head, relation, tail, content, created_at, tr_start, tr_end)` 的逻辑模型和键编码。建议 content 为 JSONB，embedding 为允许空值的 `vector(n)`，业务表带稳定内部文档 ID 以支持现有全文检索的 id 字段。两个 catalog 中的 id 不必视为全局唯一，对外身份使用 catalog + key。

为并发和补向量增加内容版本或哈希、embedding 对应版本以及模型元数据。表内 vector 可为空，使无模型时的普通 CRUD 仍可用。FTS 所需 tsvector 及字段权重在写事务中维护；PG 不必复制 SQLite 的 `docs_fts`、`json_index` 结构。

时间方面，当前 Rust 使用 `NaiveDateTime`，SQLite 用文本表示时间。首期可使用 `timestamp without time zone` 并显式约定系统时间为 UTC，以保持模型及序列化契约；不要直接切成 TIMESTAMPTZ 却遗漏 Rust 类型和输入时区的调整。

建议索引：业务主键和创建时间 B-tree；三元组 head/relation/tail 及经查询验证有用的组合索引；JSONB GIN；tsvector GIN；各业务表的 embedding HNSW。不是每个索引都要无条件建立，应通过真实查询计划和写入成本确认。

### 6.3 向量检索

示意 SQL（假定 schema、表、扩展均已初始化，维度为 1024；不是完整 migration）：

```sql
CREATE INDEX knowledge_embedding_hnsw
ON hypatia_team_memory.knowledge
USING hnsw (embedding vector_cosine_ops);

SELECT name, content, embedding <=> $1::vector AS distance
FROM hypatia_team_memory.knowledge
WHERE embedding IS NOT NULL
ORDER BY embedding <=> $1::vector
LIMIT $2;
```

沿用余弦距离，距离越小越相似。查询直接按距离运算符升序排序并限制数量，避免用变换后的相似度表达式排序妨碍索引匹配。HNSW 是近似检索，小规模数据可使用精确扫描；比较结果时用精确距离作为基线。

维度来自 embedding 配置，在 schema 元数据与列类型中验证，避免再配置一个相互矛盾的维度值。还要记录模型身份：相同维度的不同模型不能混用。模型或维度改变必须走显式重嵌入与切换流程，不能仅改 TOML 后继续写入。拒绝 NaN、Infinity、错误维度，并明确零向量处理策略。

pgvector 的可存储维度与特定类型 ANN 索引支持的维度上限不同。对默认 1024 维方案进行支持验证；远端大维度模型必须按选定扩展版本检查限制，不可直接承诺所有 embedding 配置都能建立 HNSW。halfvec、降维或其他索引策略涉及精度与行为变化，不能自动启用。

如果后续把过滤条件下推到 ANN 查询，需要测试选择性过滤造成不足 k 条的问题，结合候选扩张、支持版本的 iterative scan 或精确查询解决，并用 EXPLAIN 验证执行路径。

## 7. 全文检索与返回格式

现有 FTS5 使用 porter/unicode61，中文通过 Jieba 预分词；key/data/tags/synonyms 的 BM25 权重为 10/1/5/3。PG 可复用 `Content.fts_fields` 和中文分词，再生成带权 tsvector 与 GIN 索引。

首期建议先以 `simple` 文本配置验证中文及中英混合召回；英语词干、分词和权重需专项对照。PostgreSQL 原生 ts_rank/ts_rank_cd 不等于 FTS5 BM25，不能承诺排名或分数逐项一致。若要延续 rank 越小越优的接口，可返回 PG 评分的负值并按升序排列，但这只统一方向，不构成跨后端分数校准。

需要验收空查询、标点、中文、多词 AND/OR、key/tags/synonyms、catalog 过滤和分页。不要把 FTS5 的清洗函数原封不动当作 PG tsquery 编译器。

返回格式建立逐命令契约：CRUD/JSE 的 Content、全文检索 content、相似检索的 name/triple/distance、时间字段和错误类型。当前 ANN 与暴力 fallback 对 content 存在对象/字符串差异，应记录并通过公共转换层有计划地统一，避免把已有不一致误当 PG 特性。`QueryResult.total_count` 当前为本次返回行数，不应未经说明改为全量匹配数。

## 8. 事务、一致性与故障行为

现有内容写入后才生成向量，不能假设当前业务已经拥有“内容和 embedding 一次原子提交”。完整 PG 后端可以避免跨数据库双写，但仍要处理模型调用与事务的边界。

建议内容与其 FTS 派生数据在一个事务中提交；更新内容时，同时使旧向量失效并标记待补。模型推理放在事务之外，生成后通过内容版本或哈希进行条件更新，只有内容仍匹配时才安装向量，防止慢请求覆盖新内容。删除内容及所有数据库派生记录在同一事务完成。

无模型时保留普通 CRUD 能力，并使语义检索报模型不可用；模型网络错误、向量写回失败或维度不符需要返回可诊断状态。明确区分“正文已经保存但向量待补”与“正文未保存”，避免调用方盲目重试创建。索引维护与补向量应可重入，不能用进程退出时的 Drop 承担 PG 提交。

schema 迁移应有版本表、事务边界及并发迁移锁；扩展由具备权限的管理员预先安装，应用连接阶段检查。数据库事务失败要回滚；连接断开后的写重试必须考虑提交结果未知的情况，不能一律自动重试非幂等操作。

## 9. 切换、迁移、导出与附件

修改 backend 只改变连接目标，不迁移数据。原有 SQLite 文件不会自动出现在 PG；切回 SQLite 也只能看到本地原有内容，不包含切换后在 PG 新增的数据。文档和 CLI 提示必须明确这一点。

建议增加后端中立的逻辑迁移/导出能力，下面是流程建议，不是现有命令：

1. 对源 shelf 建立一致性快照或暂停写入，保存可恢复备份。SQLite 导出应使用 backup API 或正确的快照流程，当前直接复制文件不能直接作为并发/WAL 场景的一致性保证。
2. 初始化独立目标 schema，导入业务键、Content、时间、三元组、有效向量和模型元数据，不复制 usearch 文件。旧数据缺少可信模型身份时，不猜测模型，选择重新生成向量或由迁移配置明确指定并验证。
3. 分批导入并记录进度；先导数据再建大索引。明确重复键冲突策略，首次迁移推荐目标为空，避免隐式覆盖。
4. 校验行数、主键集合、规范化内容校验和、时间、向量维度及附件引用，再运行查询对照。
5. 校验通过后才修改源目录的连接配置；保留原库作为切换点备份。切换后的回退若需保留新写入，应先反向迁移，单纯改回配置不是无损回滚。

PostgreSQL 导出可采用一致性事务下的版本化逻辑包：manifest、knowledge/statement 记录、向量与模型信息、附件。连接密码不进入包；恢复时由操作者提供目标连接配置。现有 SQLite 导出格式需要继续支持，不能让旧 shelf 无法导入。

现有 figures 引用 `archive://` 本地文件。本文建议首期 PostgreSQL 只接管数据库内容，`archives/` 仍归 shelf 本地目录管理，导出时随包复制。多客户端共享 PG 并不会自动共享附件；跨机器使用需部署共享附件目录，或后续独立引入对象存储。这是明确的首期范围建议，并非用户已经确认的附件托管要求。

## 10. 实施拆分与验收

| 阶段 | 主要产出 | 完成条件 |
| --- | --- | --- |
| 1. 固定契约与配置 | 统一 ShelfSettings，缺省兼容，明确查询/返回/错误语义 | 旧合法 shelf 配置可继续运行；非法 storage 配置报错 |
| 2. 封装本地实现 | ShelfBackend、LocalBackend，清理 service/Lab/manager 的具体依赖 | 本地 CRUD、查询、向量、导出回归通过 |
| 3. 增加 SQL 方言层 | 保留 Parser/AST，直接生成 JSONB 查询，适配特殊查询 | 常见操作映射与局部边界对照通过；PG 不接收 SQLite SQL |
| 4. PostgreSQL 实现 | schema migration、CRUD、JSONB、FTS、图查询、pgvector | 在真实 PG + 扩展实例上通过后端契约测试 |
| 5. 运维与迁移 | 导出导入、补向量、配置切换说明、诊断和发布构建 | 完整迁移及切回演练、默认构建与支持版均可用 |

主要工作重点为完整读写接口、生命周期与迁移、写入一致性及全文检索适配；JSE 到 JSONB 的直接转换、驱动连接与 HNSW SQL 是相对集中的工作。没有目标数据规模、并发指标和测试环境数据，不给出吞吐、延迟或开发天数承诺。

最低验收集：

- 不配置 storage 时，全套现有本地功能继续通过；未启用 PG feature 时不要求数据库服务。
- 同一进程连接一个本地 shelf 和两个 PG schema，相同业务键不串数据。
- CRUD、三元组及时间、JSON 类型边界、`$not-summaried`、含环 k-hop 使用共享用例验证。
- 使用固定测试向量验证距离、维度和模型不匹配、空库、更新与删除；ANN 用召回指标及距离误差，不能要求近似结果集合始终完全相同。
- 中文/英文全文检索对照召回与排序方向，排名差异明确记录。
- 测试错误配置、缺环境变量、缺扩展、权限不足、连接超时、事务失败及并发更新时旧向量不得重新生效。
- 检查内容导出导入、向量补全、附件引用、切换及包含切换后新数据的回退流程。
- 对代表性数据执行查询计划与性能测量，再决定 HNSW 参数、索引组合及连接池规模。

本次仅做代码审查和方案分析，未运行 PG 集成测试或性能测试。在线资料检索因当前搜索工具缺少 API key 未成功，因此不宣称已核实最新 pgvector 版本、依赖 feature 名称或维度限制；实施时固定 PostgreSQL/pgvector/驱动版本并实测。可查阅的官方资料入口为 [pgvector](https://github.com/pgvector/pgvector)、[pgvector-rust](https://github.com/pgvector/pgvector-rust)、[PostgreSQL JSON 类型](https://www.postgresql.org/docs/current/datatype-json.html) 与 [全文检索](https://www.postgresql.org/docs/current/textsearch.html)。

## 11. 决策状态

已确认：完整 PostgreSQL + pgvector 后端，并保留现有本地后端。

本文建议作为实现起点的默认决策：每 shelf 配置、每 shelf 一个 PG schema、本地附件保留、同步驱动、可选编译 feature、配置切换不自动搬数据、保留 Parser/AST 并通过方言层直接生成 JSONB 查询、保留 JSE 现有候选筛选语义。它们足以形成可实施方案；若目标是大规模多租户托管、严格复刻 BM25 排名或跨机器自动共享附件，需要在实现前调整对应设计与验收范围。
