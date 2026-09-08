# PostgreSQL + pgvector 后端

Hypatia 可按 shelf 选择完整数据库后端。SQLite + FTS5 + usearch 仍为默认；`pgvector` 使用 PostgreSQL 管理 knowledge、statement、结构化查询、全文检索和向量。附件始终保存在 shelf 本地 `archives/` 目录。

## 构建与测试

```sh
cargo build --release                         # 仅本地后端
cargo build --release --features postgres-backend
bash scripts/build.sh --features postgres-backend
bash scripts/test-pgvector.sh
```

测试脚本要求本地已有 `pgvector/pgvector:pg17-trixie` 镜像，按镜像 ID 启动隔离容器和随机本机端口，安装扩展，执行默认与 PG 构建测试，退出时删除容器。它不使用机器上已有的 PostgreSQL 服务。实际验收镜像为 PostgreSQL 17.11 / pgvector 0.8.6；Rust 依赖固定在 Cargo.lock（postgres 0.19.14、pgvector 0.4.2、postgres-native-tls 0.5.3）。

不带 feature 的二进制遇到 PG 配置会报“不支持，使用 --features postgres-backend 重建”，不会创建 SQLite 作为替代。交叉编译启用 PG 时，还需目标平台的 native-tls 系统依赖；本次不宣称已验证全部交叉编译目标。

## Shelf 配置

在连接之前创建目录和 `shelf.toml`：

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

数据库管理员预先执行 `CREATE EXTENSION vector`。应用账户需要连接数据库、创建自己的 schema 及其中对象的权限；已有 schema 需相应读写及初始化检查权限。一个 shelf 对应一个显式配置的 schema。schema 名称只接受 1–63 字节 ASCII 字母、数字、下划线，首字符不能是数字，禁止 public、information_schema 和 pg_*。同一数据库/schema 才共享数据，schema 名称本身不提供用户权限隔离。

连接串有两种配置方式，`url_env` 和 `url` 必须且只能配置一个；两者同时出现、都未提供或值为空会报配置错误，不设置隐式优先级。原有 `url_env` 用法保持兼容。

可以通过 `url_env` 指定环境变量，也可以把上例的 `[storage.postgres]` 替换为直接配置：

```toml
[storage.postgres]
url = "postgresql://hypatia:your-password@db.example.com:5432/memory?sslmode=require"
schema = "hypatia_team_memory"
connect_timeout_seconds = 5
statement_timeout_ms = 30000
```

配置后执行：

```sh
hypatia connect /path/to/team-shelf --name team
hypatia knowledge-create "example" -d "正文" -s team
hypatia search "正文" -s team
hypatia backfill -s team
```

使用 `url` 时，连接串和密码保存在本地 shelf.toml；配置的 Debug 输出会隐藏 URL，连接解析错误不回显连接串，导出包不包含 shelf.toml 或连接凭据。远程 embedding 与数据库后端独立，仍支持已有的 `[embedding] provider="remote"`、`api_url`、`api_key_env`、`api_model`、`dimensions` 配置。

没有 shelf.toml 或没有 storage 配置时选 SQLite；默认 shelf 读取 `~/.hypatia/default/shelf.toml`。已有文件无法读取、TOML 非法、未知配置字段/后端或缺少必要字段均报错。以前被忽略的非法旧配置会因此暴露。仅在连接时加载配置，修改后重新连接或重启。PG shelf 不创建 hypatia.sqlite 或 vectors/，连接失败不会切换到本地。非默认 shelf 恢复失败会显示警告并保持未连接，操作该 shelf 会明确失败；默认 shelf 失败使启动失败。

## TLS 与模型身份

生产连接建议显式 `sslmode=require`。本实现的同步 postgres 驱动结合 native-tls：require 要求 TLS，并通过系统信任根校验证书链及主机名；默认 prefer 在服务端不支持 TLS 时允许明文，不应作为必须加密的部署配置。驱动不接受 libpq 的 verify-ca、verify-full 拼写或 sslrootcert 参数，这些配置会报错，不会被静默忽略。私有 CA 应通过操作系统信任配置管理。TLS 行为按驱动实现说明，本次 Docker 测试使用 sslmode=disable，没有运行证书轮换/私有 CA 部署测试。

PG 连接必须能识别向量模型：本地配置显式 model，或可读取本地模型和 tokenizer 的内容指纹；模型文件暂时不存在时，配置 model 仍可使普通 CRUD 工作。远程身份包含模型名和 endpoint 的哈希，不泄露连接凭据。本地未命名模型使用模型/tokenizer 内容指纹；命名模型按声明的模型标识、tokenizer 配置、pooling 和 max_seq_length 识别。操作者必须为不同模型版本使用不同 model 标识；不要用同一标识替换成不同模型。

模型、维度、距离或 PG 索引模式与已有 metadata 不一致时拒绝连接。改变模型/维度应逻辑导出后，以 --reembed 导入新配置的空 shelf/schema，再 backfill；修改 TOML 不是模型迁移。

向量只接受正确维度、有限值和非零范数。cosine 距离越小越相似。HNSW 的 vector 类型最多支持 2000 维；需要更高维度时必须显式设置 index="none"，使用精确扫描（存储上限 16000 维）。不自动采用 halfvec 或降维。

## 查询语义与已确认差异

- Parser/AST 保持不变，PG 方言直接生成参数化 SQL，schema、业务表和扩展对象均明确限定；不替换已有 SQLite SQL 中的问号。
- `$search` / `$similar` 仍先选候选键，再执行普通条件查询并按创建时间排序；不保证最终保持相关度/距离顺序，也不下推普通条件后重新选择 top-k。分页沿用现有 JSE 行为。
- PG 使用 simple 文本配置、现有 Jieba 预分词和 key/data/tags/synonyms 的相对权重 10/1/5/3。空格表示 AND，支持显式 AND/OR；PG ts_rank 返回负值以保持“越小越优”。PG 不复刻 Porter 词干或 FTS5 BM25，召回、排序和分数可不同。用户已接受这一差异。
- **用户已选择保留 SQLite 旧行为**：PG 的 data.foo / data[index] 比较使用受控解析后的 payload；非法内层 JSON 得到 NULL。SQLite 普通比较仍对 Content.data 字符串执行原 JSON1 路径访问，data.foo 可能匹配不到。SQLite 已有 `$has` 的 data.foo token 查询保持原样。
- `$has` 使用字面字段名，例如 `["$has","tags","rust"]`，不是 `$tags`。PG 写事务维护派生 tokens JSONB，使字符串/数字/布尔/null 和旧索引递归预算的 membership 语义一致，包括科学计数法 token。它是行内派生数据，不是复制一张 SQLite json_index 表。
- `$json-contains` 使用原生 JSONB 包含预筛选加严格递归复核，避免把 PG 数组包含标量的额外规则带入原契约。数值比较对非法文本受控转换，不因一条无效内容让整批查询失败。
- PG k-hop 去重有环图的边/深度状态，最大深度 100，并受 statement_timeout 限制。
- CRUD/JSE Content 保持对象；全文检索 content 保持 JSON 字符串；语义检索 content 现在统一为 JSON 对象，消除本地 ANN 与旧暴力 fallback 的不一致。时间仍为 UTC 约定下的 timestamp without time zone / NaiveDateTime。PG 仅接受微秒精度及普通时间，拒绝会丢失纳秒或闰秒编码的写入/导入，而不静默截断；total_count 仍是返回行数。

## 正文提交与 backfill

正文、FTS 和 JSON 派生数据在事务内提交。更新正文同时使旧向量失效。模型推理在事务外执行，生成后按内容版本条件写回；更新或删除再创建产生新版本，旧工作不能覆盖新正文。

正文保存成功但模型不可用、推理失败或向量写回失败时，CRUD 仍返回已保存的记录，同时在 stderr 提示“content saved; embedding pending”及 backfill。不要把这种状态当作正文创建失败而重复创建。

`hypatia backfill -s team` 分页补缺失向量，可重复执行。失败的条目在下次运行重试；并发改变的条目拒绝安装旧向量。PG 索引由数据库维护，backfill 不保存或重建 usearch 文件。

旧 SQLite 向量没有可信模型元数据时，不推断其模型，也不与新模型混用。普通 CRUD 和逻辑导出仍可用，语义检索/向量安装提示需要显式重建。模型可用后运行：

```sh
hypatia backfill -s old-local --reembed
```

此选项先显式清空派生向量并使在途任务的版本失效，再用已配置模型重新生成；正文不变。失败后可用普通 backfill 续补。已知模型身份改变仍建议使用新的空 shelf/schema，保留切换备份。

## 导出、迁移与回退

`hypatia export <shelf> <empty-directory>` 生成：

- manifest.json：格式版本、行数和 snapshot/附件 SHA-256 清单；
- snapshot.json：knowledge/statement、完整 Content、时间、有效向量及模型元数据；
- archives/：本地附件；
- 本地源额外保留 hypatia.sqlite，兼容旧目录导出格式。SQLite 使用 backup API，逻辑快照来自同一备份；PG 使用 repeatable-read 一致性只读事务。

输出先写临时目录，完成校验后一次 rename 发布。导出目标必须为空且在源 shelf 之外。数据库快照不等于附件文件系统快照；导出期间请暂停附件修改。导出不携带 shelf.toml、连接串、密码或本地模型文件，目标连接和 embedding 配置由操作者提供。

迁移示例：

```sh
# 1. 源继续保留，先导出备份
hypatia export old-local /backups/to-pg

# 2. 创建 target 目录及本文 PG shelf.toml，使用独立空 schema
hypatia connect /path/to/pg-target --name pg-target

# 3. 已有可信同模型向量可直接保留
hypatia import /backups/to-pg -s pg-target

# 或明确丢弃向量，按目标模型补齐
hypatia import /backups/to-pg -s pg-target --reembed
hypatia backfill -s pg-target
```

上面两条 import 是二选一，目标必须为空，重复键不隐式覆盖。导入先验证清单、行数和附件引用，再事务写入数据，随后按规范化内容、主键、时间与向量逐记录验证。源导出无可信向量身份时自动省略向量并在 backfill 重建。旧的仅 hypatia.sqlite + archives/ 导出可导入，源数据库只读备份到临时目录后再读取，不原地迁移源备份。

当前逻辑包在进程内加载后用一个事务导入；失败回滚可从完整包重试，未提供超大包的跨事务断点续传。附件先复制到目标，冲突内容拒绝覆盖；若后续数据库事务失败，已复制的相同附件可保留并用于重试。数据库数据提交成功后维护/验证失败会明确报告，保留源包供检查。

验证通过后再修改原目录配置或让调用方使用新 shelf 名称。修改 backend 只切换目标，不搬运数据。要保留切换后在 PG 新增的数据进行回退，应从 PG 再 export，import 到新的空 SQLite shelf，再切换调用方。单纯改回 backend="sqlite" 只能看到原本地旧数据。

多客户端共享 PG 不会自动共享本地附件。跨机器时需自行共享 archives 目录，或分别部署随导出包复制的附件。

## 库 API 与内部生命周期

OpenShelf.store 和公开 vectors 已替换为 OpenShelf.backend、settings。ShelfBackend 将 LocalBackend 的 SqliteStore/usearch 文件封装在内部；调用方不能再通过 OpenShelf.store.conn() 访问 SQLite。SqliteStore 的 insert/update 返回内容版本 i64，原只忽略返回值的调用不受影响。直接依赖这些公开字段/精确函数签名的 Rust 用户需要适配；这属于库 API 变更。

本地 schema 升至版本 3，新增内容版本时钟和缓存失效触发器，打开拒绝未知未来版本。usearch 是可重建缓存，跨连接读操作用同一 SQLite 快照校验时钟并读取正文；缓存文件采用独立临时文件和版本清单，失效后刷新，缓存持久化失败可继续精确检索。PG 提交不依赖 Drop，也不自动重试提交状态未知的写操作。

性能验收使用确定性向量和真实查询计划。1024 维样例验证 HNSW 索引可用与 index=none 精确扫描；小样本强制禁用顺序扫描只证明访问路径可用，不构成实际延迟或吞吐承诺。应按实际数据规模重新测量，再调整连接池、HNSW 参数和索引组合。

## 本次验证记录

| 验证 | 结果 |
| --- | --- |
| 默认构建单元测试 | 170 通过 |
| 默认共享后端/迁移契约 | 5 通过 |
| PG feature 单元测试（含真实 SQL 对照） | 172 通过 |
| PG feature 共享后端及迁移往返 | 6 通过 |
| PG 连接脱敏及真实故障、并发与访问路径测试 | 8 通过 |
| 默认与 PG feature 的 locked release 构建 | 均通过 |
| 独立 Docker 验收脚本 | 通过，容器自动清理 |
| 默认依赖检查 | 无 PostgreSQL 特有依赖 |
| 仓库小型基准 | 通过 |

完整 cargo test 还会启动已有 LoCoMo 模型评测，本次在 300 秒限制后超时，未宣称完成该评测或后续 LongMemEval。常规验收使用上面的确定性测试；TLS 私有 CA 部署和所有交叉编译平台也未在本次实测。
