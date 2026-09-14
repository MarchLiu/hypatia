# 降低上手成本方案

> 状态：**方案，尚未实施**（2026-09-10 初稿；2026-09-11 并入评审结论）
> 背景：如何让用户更简便地把 hypatia 用起来 —— 上手漏斗诊断、embedding 生命周期重定位、分发与安装
> 姊妹篇：[Agent 接口分层](agent-interfaces.md)（CLI / MCP / skill 的分工）
> 范围：本文只交付诊断、方案与工作顺序，没有修改任何实现

## 1. 结论

一句话：**hypatia 在零下载、零配置下已经是可用的，但产品的每一处都在掩盖这个事实。**

实测（§2.1）：新建一个不含任何模型文件的 shelf，`knowledge-create`、`statement-create`、`search`（FTS）、`query`（JSE / 图遍历）**全部正常工作**，只有 `similar`（向量）不可用。三种检索手段里有两种立刻可用 —— 而 README 把 2.3 GB 的模型下载放在第 2 步，每次写入还打一行 warning。

由此得出四条：

- **上手的主要成本不是架构问题，是暴露问题。** 最贵的两处（README 只给了 `cargo build`、以为必须先下 2.3 GB）都不需要改设计就能消掉 —— 前者连产物都已经挂在 GitHub Releases 上了（§2.4）。
- **「embedding 默认延迟 + 写入阈值自动触发 backfill」这个已定决策，副作用比性能收益重要**：它让「零模型起步 → 之后启用 → 已有内容自动补齐」这条升级路径成立且**回溯生效**。今天不成立。
- **这条升级路径今天在存储层就被堵住了。** model identity 在首次打开 shelf 时冻结，之后改 provider 会让整个 shelf 打不开，连 CRUD 都不行（§2.5）。必须先做任务 I，否则 D、E 的承诺都是假的。
- **延迟的开关应统一，flush 策略必须按 provider 分化**（§3.4）—— 因为本地与远程延迟的**理由**不同，理由决定了何时兑现。

## 2. 实测证据

以下事实基于本次在工作区的实际测量与源码检查，而非历史设计文档。行号是本次检查时的位置。测量环境：Apple Silicon，binary 自报 `hypatia 0.3.1`（构建于版本号改为 4.0.0 之前，其后 `src/` 无改动），`default` shelf 为 pgvector 后端，另用临时 SQLite shelf 交叉验证。

### 2.1 零模型下的可用性

新建空 shelf（无 `embedding_model.onnx` / `tokenizer.json`）：

| 操作 | 结果 |
|---|---|
| `knowledge-create` | ✅ 成功，附带一行 stderr warning |
| `statement-create` | ✅ 成功，附带一行 stderr warning |
| `search`（FTS） | ✅ 正常返回结果 |
| `query`（`$triple` 图遍历） | ✅ 正常返回结果 |
| `similar` | ❌ `model unavailable: no embedding provider configured` |

写入的 warning 来自 `src/storage/shelf_manager.rs:118-134`：

```
warning: knowledge/rust: content saved; embedding pending (model unavailable or content changed); run backfill
```

注意这是**尽力而为 + exit 0**。调用方无法从退出码发现向量没生成 —— 可观测性缺口见 §5.1（写入本身的 exit 0 应当保留）。

### 2.2 CLI 单次调用开销

| 操作 | 实测 | 是否触发 embedding |
|---|---|---|
| `knowledge-get` | 10–20 ms | 否 |
| `search`（FTS） | 10 ms | 否 |
| `query`（JSE） | 10 ms | 否 |
| `similar` | **820 ms**（冷启 2.75 s） | 是 |
| `knowledge-create` | **830 ms** | 是（`src/service/knowledge.rs:19` 同步 `embed_saved`） |
| `statement-create` | **830 ms** | 是（`src/service/statement.rs:32`） |

进程启动与打开数据库本身几乎免费（10 ms）。开销集中在一处：`model.onnx_data` 为 **2.27 GB**，每次需要向量的调用都重新加载。

对照 README 自己的基准（BGE-M3 `Vec p50 43 ms`，模型已在内存），**820 ms 中约 780 ms 是重复加载，占 95%**。

两种存储后端（pgvector / SQLite）测得数字一致 —— 瓶颈是模型加载，与存储层无关。

单次 `similar` 峰值 RSS **1.66 GB**。ONNX 是**惰性加载**的（`src/embedding/provider.rs:100` 的 `ensure_loaded()`），未触发向量操作前不付这笔内存。

### 2.3 backfill 与 `embed_batch`

`src/lab.rs:303-333`：backfill 从数据库**按 128 分页**取出待嵌入的行，但**逐条嵌入**：

```rust
let page = shelf_ref.backend.missing_embeddings(catalog, after.as_deref(), 128)?;
for (key, content, version) in page {
    match shelf_ref.embedder.embed(&content.embedding_text(&key)) { ... }
}
```

`embed_batch` 在 trait 上有默认实现（`src/embedding/provider.rs:17`），但**没有任何 provider 覆盖它，也没有任何调用方使用它**；默认实现本身就是 `texts.iter().map(|t| self.embed(t))`，即串行。

后果按 provider 分化：

- **本地 ONNX**：模型加载一次，之后 N 次单条前向。1000 条约 43 秒。可接受，但浪费 —— 批量张量能有数倍吞吐。
- **远程 API**：**N 次串行 HTTP 往返**。1000 条 × ~150 ms RTT ≈ **150 秒**。而 OpenAI 兼容的 `/v1/embeddings` 本就接受数组，分页用的 128 正是现成批大小 → 约 8 个请求 ≈ 1.2 秒。**差约 100×。**

**结论：把写侧的嵌入债推给 backfill，必须先让 backfill 收得下。对远程 provider 这是数量级问题，不是优化。**

### 2.4 上手漏斗

按用户实际卡住的顺序，而非技术分类：

| # | 步骤 | 现状 | 卡点 |
|---|---|---|---|
| 1 | 拿到 binary | README 只给了 `cargo build --release` —— **要求 Rust 工具链**；Releases 已有部分平台产物但 README 未提 | 🔴 最大 |
| 2 | 有个能用的 shelf | 需先知道 `hypatia connect`；无 `init` | 🟡 |
| 3 | 能写能查 | **已经做好**，被 warning 与 README 顺序掩盖 | 🟢 误伤 |
| 4 | 语义检索 | 2.3 GB + Python 的 `hf` + 三次 `cp` | 🔴 第二大 |
| 5 | 接到 Agent | 每家一套手工装法，无工具 | 🟠 |

补充事实：

- GitHub Release v4.0.0 已挂 `hypatia-aarch64-apple-darwin.gz` 与 `hypatia-x86_64-pc-windows-msvc.tar.gz`（后者由手动触发的 `.github/workflows/build-windows.yml` 上传），README 一个字没提。macOS 产物静态链接 ONNX Runtime（`otool -L` 只有系统库），单文件即可分发；Windows 包附带 `onnxruntime.dll`。缺 Intel Mac 与全部 Linux 产物。
- `scripts/` 下只有 `build.sh` 与 benchmark 脚本，**没有任何 skill 安装工具**，也没有安装脚本。`build.sh` 列了 21 个 target（`scripts/build.sh:32-67`），但 ort 的预编译 ONNX Runtime 只覆盖主流平台，且在 x86-64 上要求 **x86-64-v3**（AVX2），Linux 版依赖 **libc++**（ort 官方文档）—— musl、riscv、s390x 等 target 大概率编不过（未验证）。
- `hypatia model list | register | show` 已存在（`ModelCommands`），指向 `~/.hypatia/models/`，**但没有 `model install` / `download`** —— registry 概念有了，获取那一步仍要用户自己 `hf download` + `cp`。且实测 `model list` 显示的注册项指向 shelf 目录而非 `~/.hypatia/models/`，这条线只连了一半：`~/.hypatia/models/` 里的模型对没有 shelf.toml 的 shelf 不可见（§2.5 场景 2）。

### 2.5 升级路径实测

隔离 `HOME`（不触碰真实的 shelf 注册表），SQLite 后端，真实 BGE-M3 文件（软链接），binary 为主仓 `target/release/hypatia`（构建之后 `src/` 无改动）：

| # | 路径 | 结果 |
|---|---|---|
| 1 | 第 0 层 → 第 2 层：模型文件放进 shelf 目录 | ✅ `backfill` 补齐，`similar` 命中 |
| 2 | 第 0 层 → 第 2 层：模型放 `~/.hypatia/models/BAAI/bge-m3`（`model install` 的落点） | ❌ 没有 shelf.toml 的 shelf 只看 shelf 目录，`backfill` 报 `no embedding model found` |
| 3 | 第 1 层 → 第 2 层：先 remote，后改 local | ❌ shelf 整体打不开，**连 `knowledge-get` 都失败**；`backfill --reembed` 也跑不了（shelf 未连接），唯一出路是 export → 新 shelf → import |
| 4 | 第 0 天就写 `model = "BAAI/bge-m3"`，之后才装 | ✅ identity 不变，补齐成功；但未安装期间每条命令（含只读）都打 `model resolution failed` |
| 5 | PG 第 0 层（读码，未实测） | ❌ `src/storage/postgres_store.rs:117` 要求可信 identity：无模型文件且未配 `embedding.model` 的 PG shelf 建不起来 |

场景 3 之后，**任何 shelf 上的每一条命令**都在 stderr 打 `failed to restore shelf 's3'` —— 每次调用都会打开全部已注册 shelf（§5.3）。

根因：model identity 在**第一次以可信 identity 打开 shelf 时**就冻结（SQLite `src/storage/sqlite_store.rs:826-847`，PG `src/storage/postgres_store.rs:186-206`），而不是在第一条向量写入时；冻结之后任何配置变化都让 shelf 拒绝打开。PG 还有第二处缺口：`reset_embeddings` 在 PG 上只是 `clear_all_embeddings()`（`src/storage/backend.rs:336`），**不更新 `meta.embedding_model`** —— 即便 reembed 完，下次打开仍 mismatch。

结论：§3.2 的「回溯生效」目前只对场景 1、4 成立。场景 3、5 由任务 I 解决；场景 2 由任务 D 解决（把模型挂到目标 shelf 上，依赖 I）。见 §4。

### 2.6 远程 provider 的写路径重试

`src/embedding/provider.rs:362-430` 的 `request_with_retry`：单次超时 60 s、最多重试 3 次、指数退避 → 端点无响应时**一次写入最坏挂起约 4 分钟**。且只有 429 与传输错误会重试，**5xx 与其他 ≥400 一律直接失败**（`:402-411`），与 §3.4 设想的「瞬时 / 永久」分类不符。

这是 §3.3「同步嵌入把网络依赖放进写路径」的直接代价。

## 3. embedding 生命周期的重定位

### 3.1 写侧可延迟，读侧不可

`similar` 需要**当场**为查询文本生成向量，无法 backfill。所以「延迟」只适用于写侧。

这反而**锐化**了常驻进程的价值：写侧延迟之后，唯一还受模型加载折磨的地方只剩 `similar`。

| | 本地 ONNX | 远程 API |
|---|---|---|
| CLI 写入（延迟后） | ~15 ms | ~15 ms |
| CLI `similar` | 820 ms（95% 是模型加载） | 网络 RTT，无模型加载 |
| 常驻进程 `similar` | 首次 ~800 ms，之后 43 ms | RTT + 连接复用 |
| 常驻 RSS | 未用向量前几十 MB，之后 1.66 GB | 始终几十 MB |

惰性加载解决的是「启动时的 RAM」；延迟 embedding 解决的是「写操作意外把模型拉进内存」。**两者叠加，才让「一个只做 CRUD 的 Agent 永远不付这 1.66 GB」成立。**

（常驻进程的形态另议，见 [Agent 接口分层](agent-interfaces.md)。）

### 3.2 分层上手

延迟 + 自动 backfill 让下面这条路径成立，且**回溯生效**：

```
第 0 层（0 秒，0 下载）   装 binary → 立刻可用：FTS + 图 + JSE
第 1 层（1 分钟，0 下载） 配远程 API（一个 shelf.toml 块 + 一个环境变量）→ 向量可用
第 2 层（可选，一次 2.3 GB） hypatia model install BAAI/bge-m3 → 本地离线
```

关键性质：**从第 0 层升到第 1 或第 2 层时，此前写入的所有条目会自动补上向量。** 用户不会因为「一开始没配模型」而丢失任何东西。

今天不具备这个性质，原因有三个：

- 写入时嵌入一次，失败即永久 pending，除非用户自己想起来跑 `backfill`（任务 C 解决）。
- identity 在首次打开时冻结，切换 provider 后 shelf 打不开；PG 连第 0 层都建不起来（§2.5 场景 3、5，任务 I 解决）。
- `~/.hypatia/models/` 里的模型挂不到没有 shelf.toml 的 shelf 上（§2.5 场景 2，任务 D 解决）。

第 1 层与第 2 层是**二选一**，不是阶梯：两者之间切换是模型变更，已有向量不可比，必须 `backfill --reembed`。任务 I 之后，这个切换让 shelf **降级打开**（CRUD / FTS / JSE 照常，向量操作提示 reembed），而不是今天的整体拒开。

### 3.3 已定决策

- **embedding 默认延迟**（写入不同步嵌入），配合**写入阈值自动触发 backfill**。
- 理由不止性能：远程 provider 的存在本就让「写入时同步嵌入」成为坏设计 —— 它把一次本地 SQLite 写变成一次网络依赖（最坏挂起约 4 分钟，§2.6）。网络错误、限流、API key 缺失现在都发生在写路径，结果是 stderr 一行 warning 加一条永远没有向量的条目。延迟之后它们发生在 backfill 里：可重试、可退避、可观测、可批量。
- 向量本就是设计文档定义的「可重建的派生缓存」（见 `simplify-and-evolve.md`：「向量索引是唯一的例外，而它被有意设计为可重建的缓存」），因此延迟在哲学上是被背书的，不是妥协。
- **写入保持 exit 0**：写入的契约是「内容已提交」，向量缓存没生成不算写失败。自动 flush 永不改变触发它的那条命令的退出码。「欠账不可观测」在读侧与显式 backfill 上补（§5.1）。

### 3.4 本地与远程需要分情况吗

需要，但分化点不在开关，在 flush 策略。**延迟的理由不同，因此「何时兑现」必须不同：**

| | 延迟是为了什么 | 理由会消失吗 |
|---|---|---|
| 本地 ONNX | 不把 2.27 GB 拉进一个本来不需要它的进程（内存 + 启动） | **会** —— 模型一旦加载，同步嵌入只要 43 ms，延迟即失去意义 |
| 远程 API | 不让一次本地 SQLite 写入依赖网络（可靠性） | **不会** —— 与预热无关，每次写入都是独立的网络依赖 |

反直觉之处：直觉会说「本地便宜所以不用延迟，远程贵所以要延迟」，实际相反 —— 本地单次成本高（830 ms）但**可摊销**；远程单次成本低（~150 ms）但**不可摊销且会失败**。远程延迟的理由更硬。

**需要分情况的三处：**

1. **flush 触发策略**（第一版为代码常量，不进 shelf.toml）

   | | 本地 ONNX | 远程 API |
   |---|---|---|
   | 写入阈值（pending 条数） | 64 | 128（一批 = 一个请求） |
   | 读前补齐 | `similar` / `$similar` 前补齐（本就要加载模型） | 无 |
   | 时间上限 | 无（靠读前补齐） | 60 s，从 `pending_since` 起算 |

   - 本地阈值 64：越阈值那次写入 ≈ 780 + 43 × 64 ≈ 3.5 s，模型加载均摊 ≈ 12 ms / 次；再往上均摊收益递减，卡顿却线性增长。且必须 ≤ 单次 flush 上限 128，读前补齐才能一次清空欠账、写完即可搜到。（数字按 B 之前的单条推理估算，B 之后只会更低。）
   - 读前补齐按 `content_version` 倒序取一批，优先补最新写入 —— 欠账超过一批时（例如升级后的存量欠账），「写完即可搜到」仍对新条目成立。现有 `missing_embeddings` 按主键排序（`src/storage/sqlite_store.rs:784`），做不到这一点。
   - `pending_since` 是**当前这笔欠账最早一条的产生时间**，不是上次 flush 的时间 —— 否则闲置之后的第一次写入会立刻 flush。`created_at` 也不能用：内容更新时触发器把向量置空，但不动 `created_at`（`src/storage/sqlite_store.rs:374-377`）。
   - 时间上限在**每条命令入口**检查，只针对该命令的目标 shelf；不带 shelf 的命令跳过。CLI 没有常驻进程，「到期」只能在下一次调用时兑现。
   - 60 s：T 之内 `similar` 会漏掉新条目，而 hypatia skill 在写入前用 `similar` 查重；代价只是有欠账时每分钟最多一次批量请求。

   所有自动 flush 的共同约束：

   - **每次最多一批（128 条）**。已有 shelf 可能攒着大量 pending，升级后第一次越过阈值不能把整笔欠账算在一次写入上。
   - **只写入向量，不复用 `backfill_vectors`**：后者末尾的 `rebuild_vector_indexes`（`src/lab.rs:335`）在 SQLite 上全量重建 `json_index` 与向量缓存（均为 O(N)），在 PG 上对 knowledge / statement / docs 加 EXCLUSIVE 锁（`src/storage/postgres_store.rs:692`）。SQLite 的向量缓存已被 `install_embedding` 标记失效，下次搜索时懒重建。
   - **快速失败**：短超时、不重试、不拆批；完整重试与拆批只留给显式的 `hypatia backfill`。
   - 不改变触发它的命令的退出码。

   落点：

   - 阈值检查放在 `OpenShelf`（替换 `embed_saved`），所有写入方 —— CLI、REPL、基准测试、将来的 MCP —— 都经过它。
   - 读前补齐需要 `&mut`，而 `execute_similar` 是 `&self`（`src/storage/mod.rs:77`），因此放在 Lab：`Lab::similar` 改为 `&mut self`，`Lab::query` 在 JSE 含 `$similar` 时先补齐。直接调用 `execute_similar` 的地方（LoCoMo / LongMemEval 基准）在查询前显式 flush。
   - 状态放在 `meta` 表的一个 JSON 键：`pending_since`、熔断状态、远程的可用批大小，按 `model_identity` 键控（改配置即自然重置）。两个后端的 `meta` 都是 k/v；PG 打开时只校验预期的键（`src/storage/postgres_store.rs:196`），多出的键无碍。
   - pending 计数：PG 已有 `WHERE embedding IS NULL` 部分索引（`src/storage/postgres_store.rs:235`）；SQLite 没有，每次写入都是一次全表 COUNT → 补同样的部分索引，代价降为 O(pending)。

2. **失败处理 —— 远程独有的状态机**
   - 本地：`is_available() == false` 是**确定性**失败（文件不在），重试无意义 → 静默积累欠账；提示放在 `similar` 的报错与 `init` 里，不需要「每个 shelf 提示一次」这类持久化状态。
   - 远程：必须区分**瞬时**（网络 / 429 / 5xx → 退避重试）与**永久**（API key 错、model 名错 → 提示一次后停止重试，避免每次 flush 都撞墙）。本地没有这个维度。现有重试把 5xx 当作永久失败（§2.6），需要修正。
   - **熔断状态必须持久化**：CLI 进程之间没有记忆。欠账越过阈值后若 API 返回 401，之后每一次写入都会再撞一次 —— 恰好把 C 要消除的网络依赖原样搬回写路径。
   - legacy shelf（有向量但无 identity）上 `install_embedding` 必然报错（`src/storage/backend.rs:306`），归为永久失败，走熔断。

3. **`embed_batch` 的实现**（待办 B 实为两份工作）
   - 本地：批量张量，一次 forward 多条 → 数倍吞吐。必须与单条推理**数值等价**（查询向量永远单条计算）：按行处理 padding 与池化，单测覆盖三种池化（余弦 ≥ 0.9999）；ONNX 导出若把 batch 维固定为 1，退回逐条。
   - 远程：数组请求 → ~100×，且按 token 计费时请求开销的分摊方式也不同。结果按响应中的 `index` 字段排序（现为按位置取，`src/embedding/provider.rs:439`）。批量请求收到 4xx 时**按错误码分流**：
     - 413（批过大）：对半拆批重试，直至单条。400 含义不明：可能拒的是请求本身（如不支持的参数），也可能只是某一条输入。同一次调用里服务器只要接受过任何请求，400 就视为输入的问题，照 413 拆批；在此之前先探测两条 —— 最短的一条，不行再试最长的一条（任一条都可能恰好是被拒的那条）—— 任一通过就拆批，两条都失败才判定请求本身被拒、整批判失败。拆批得到的可用批大小记入 `meta` 状态，之后的自动 flush 沿用；自动 flush 自己遇到 400 / 413 时只把记录的批大小减半并停下，下次触发按新大小发送。
     - 请求本身被拒（401 / 403 / 404、探测失败的 400）或重试耗尽时，同一次调用里剩下的块不再发送，直接判失败 —— 否则一次 backfill 会把同一个错误重复几十遍。
     - 429：退避；自动 flush 直接停下，留给下一次触发。
     - 401 / 403 / 404：永久失败，进熔断。

**不该分情况的是开关本身。** `embedding.defer` 应为一个默认（`true`）、一种语义：

- `EmbeddingProvider` trait 的存在就是让 provider 可互换；在**策略层**按 provider 类型分支会漏抽象。
- 两种 provider 下延迟都不会**错**，只是价值来源不同 —— 不存在「此情况下延迟是坏主意」的场景。
- 可预测性：用户更换 provider 时开关不变，无需重新理解一套配置。

配置落点：`defer` 放 `[embedding]` 顶层，**只在用户关闭延迟时才写入 shelf.toml** —— 生产环境只走严格解析（`ShelfSettings::load`，`deny_unknown_fields`），旧 binary 读到未知键会直接拒开 shelf。flush 常量第一版写死在代码里；`EmbeddingConfig` 已有 `local: LocalConfig` 与 `remote: RemoteConfig` 两个子配置（`src/embedding/config.rs:5-16`），将来开放配置时，差异化策略天然落在这两处。

### 3.5 默认延迟的影响范围

对外语义的变化只来自一处 —— `embed_saved` 不再同步嵌入 —— 但它改掉了「写完即可被 `similar` 找到」这条语义。

| | 现在（有模型） | 本地 | 远程 |
|---|---|---|---|
| 写入耗时 | ~830 ms | ~15 ms；越阈值的那次付一次 flush（约 3.5 s） | ~15 ms；越阈值的那次付一次批量请求 |
| 无模型时写入的 stderr | 每次一行 warning | 无 | 无 |
| 写后立刻 `similar` | 能找到 | 能找到（读前补齐，最新写入优先） | 找不到，直到满 128 条或 60 s |
| 更新条目后 `similar` | 立刻用新向量 | 能找到 | **从向量检索中消失**，直到 flush（触发器置空向量，`src/storage/sqlite_store.rs:376`） |

不受影响：import（本就不嵌入；全仓 embedder 调用点只有 backfill、`execute_similar`、`embed_saved` 三处）、delete、FTS、图遍历、不含 `$similar` 的 JSE、写入退出码。

下游：

- **LoCoMo / LongMemEval 基准**（`tests/locomo.rs`、`tests/longmemeval.rs`）经 `KnowledgeService::create` 写入后直接调 `OpenShelf::execute_similar`，全程无 backfill —— 查询前需显式 flush，否则向量侧指标悄悄下降。README 公布的基准数字出自这里。
- **Skill**：三份 SKILL.md 均未提 backfill；远程下「写完不立刻可搜」需补一句。
- **`dsh-hypatia-auto-memory`**：写入变快、warning 消失；它只看退出码（`runOk`），兼容。若配远程，其 `similar` 会漏掉 60 s 内的新条目。
- **agent-interfaces 的 H**：写工具返回的 `embedding` 状态几乎总是 `pending`，只有本次写入恰好触发 flush 并包含自己时才是 `done`。

## 4. 待办与依赖

```
A. README 顺序调整 + model resolution 降噪     零依赖 · 零风险
I. model identity 延迟绑定 + 不匹配降级          零依赖 · D、E 的前置（§2.5）
B. backfill 批量化 + provider 覆盖 embed_batch   零依赖 · C 的前置
   （本地=批量张量，远程=数组请求）               实为两份实现，见 §3.4
C. 延迟 embedding 默认化                        依赖 B
   C1 基础设施（默认仍同步）                     分化策略见 §3.4
   C2 翻转默认
D. hypatia model install <name>                依赖 I
E. hypatia init（含分层状态提示）                依赖 C、D、I
F. 预编译分发：curl 安装脚本                      零依赖 · 漏斗第一道门
```

依赖：C 要等 B（否则延迟出去的债 backfill 收不下）；D 要等 I（否则装好的模型挂不到已有 shelf 上）；E 要等 C、D、I —— 它提示的 `model install` 来自 D，「已有内容会自动补齐」的文案要 C 与 I 落地才成立。其余互相独立。

**工作顺序**：I + A → B → C1 → C2 → D → E → F，全部在本分支、同一个 PR 内完成。I 修的是现存缺陷（切换 provider 后 shelf 打不开）且是 D、E 的前置，所以放最前。agent-interfaces 的 H 依赖 C 定义的 `embedding` 状态类型，等本 PR 合并后再做。

**A 的具体内容**：

- README Quick Start 第一屏改为「从 Releases 下载 binary（或 `cargo build`）→ `knowledge-create` → `search` → `query`」，把 `hf download` 移到「可选：启用语义检索」一节，并明说不装也能用、装了之后已有内容会自动补向量。
- `model resolution failed`（`src/embedding/config.rs:555`）：模型只是尚未安装时，不应每条命令都喊（§2.5 场景 4）。
- 原计划的 `embed_saved` warning 降噪移入 C —— C 会重写 `embed_saved`，先在 A 里改是一次性工作。

**I 的具体内容**：

- identity 推迟到**第一条向量写入**时才冻结：没有向量时，配置变更没有东西可失效，应当免费。SQLite 在 `configure_embedding` 的 mismatch 分支里，向量行数为 0 就覆盖 metadata；PG 的打开期 `meta` 校验同理。
- identity 不匹配时**降级打开**而非拒开：CRUD / FTS / JSE 照常；`similar` / `$similar` 报错并指向 `backfill --reembed`；写路径与普通 backfill 停止嵌入（避免混入另一个模型的向量）；只允许 `backfill --reembed`，且它能在原 shelf 上运行。
- PG 补齐：`reset_embeddings` 更新 `meta.embedding_model`；维度变化需在向量全空时修改 `vector(dims)` 列类型并重建 HNSW 索引（SQLite 存 blob，无此问题）。
- PG 第 0 层（§2.5 场景 5）：没有可信 identity 时也允许建库 —— `meta` 先写占位 identity，之后第一个带可信 identity 的打开者（且尚无向量）改绑；`vector(dims)` 列按配置维度（默认 1024）建，维度与后来的模型不一致时走上一条的改列流程（`src/storage/postgres_store.rs:117`、`:228`）。
- **写入向量以存储的 identity 为条件**（SQLite 比对 `meta.embedding_metadata`，PG 比对 `meta.embedding_model`）：另一进程并发改绑后，按旧配置运行的进程写入变为 no-op，条目保持 pending，不会混入另一个模型的向量。PG 改绑先无锁检查，确认没有向量后才加锁复查，因此降级 shelf 的打开不阻塞写入。
- **没有可信 identity 的打开者采纳已存 identity，不改绑**：共享同一数据库的多台机器里，缺模型的那台不会把 identity（和列维度）来回翻转。PG 在没有任何向量时导出不携带 identity，占位 identity 不会阻碍导入。

**C 的具体内容**：

- **C1**（默认仍同步嵌入，行为基本不变，可独立验证）：pending 计数 + SQLite 部分索引、`meta` 状态键（`pending_since`、熔断、可用批大小）、install-only flush 路径、`hypatia backfill --status`、显式 `backfill` 在 errors > 0 时非零退出、`similar` 结果不完整时的 stderr 提示。
- **C2**（翻转默认）：`embed_saved` 改为阈值检查、Lab 读前补齐、每条命令入口的时间上限检查、基准测试查询前显式 flush、skill 文档补一句延迟语义。warning 按原因分流：没配模型 → 不说话；配了但失败 → 保留 warning；`content_version` 竞态 → 不说话。「语义检索未启用」的提示放到 `init` 与 `similar` 的报错里，那才是用户真正想要向量的地方。
  - C2 实现时定下的细节：
    - 自动 flush 的「快速失败」经 `EmbeddingProvider::try_embed_batch` 暴露：10 s 超时、不重试。远程先发整批；整批失败时按错误码分三类 —— Rejected（400 / 413 / 其他 4xx）、Refused（401 / 403 / 404、缺 key、响应不是 JSON）、Transient（429、5xx、网络，含读响应体时超时或断连）。Refused 与 Transient 立即停下；Rejected 在同一次 flush 里沿用 `backfill` 的探测与拆批（400、422 先探测，413 直接拆）找出被拒的输入，但不重试，最多 32 个请求、合计不超过 30 s（拒绝返回很快，从 128 条里隔离出一条约 16 个请求）。
    - 一个请求都没被接受：请求本身被拒，按失败类型熔断（Refused 为永久，其余为瞬时），不跳过任何条目。服务器接受了其他请求、却单独拒绝某条输入：这条输入记入 `skipped`（按 catalog、key、content_version，最多 32 条），自动 flush 跳过它直到内容变化。显式 `backfill` 不看 `skipped`，仍会尝试并报告原因；只要嵌入成功一条就清空 `skipped`。
    - 批大小只从 413 学习：拆批后被接受的最大块记为 `remote_batch`，但本批有条目单独失败时不学（可能正是那一条太大）。400 不再让批变小，因此不需要回长逻辑。
    - 请求成功但个别条目失败：provider 单独报错、或向量含非有限值 / 全零的条目记入 `skipped`，不熔断。向量维度与配置不符按永久失败熔断（配置错误）。一条都没写进去、且不是条目本身的原因：按瞬时失败熔断；多条输入一条都没写进去时，即使 provider 是逐条报错，也视为 provider 故障，不跳过。仍在欠账中的被跳过条目（编辑、删除或已嵌入的不算）加上本次单独失败的超过 32 条：同样熔断，而不是轮换着反复重试。`backfill --status` 以 `passed_over` 报告被跳过、仍在欠账中的条目数。存储错误让本批立即停下，不再逐条等锁。
    - 熔断退避：瞬时失败 60 s 起翻倍，上限 1 h。永久失败每小时仍重试一次：换 API key 不改变 model identity，状态本身感知不到修复。显式 `backfill` 只要嵌入成功一条（或没有失败），就解除熔断、清掉 `remote_batch` 与 `skipped`。`paused` 只在熔断未到重试时间时报告。
    - `pending_since` 只在为空时设置，且只计自动 flush 能还的欠账：只剩被跳过的条目时视为没有欠账，否则每条命令都会被判为超时。删掉最后一条欠账等方式留下的陈旧起点，由命令入口的 `settle_pending` 清掉。没有被跳过的条目时，入口只查有无欠账，阈值检查最多数到阈值，都不随欠账规模变慢。读写条目的 shelf 命令在入口做 settle，只对远程检查 60 s 上限；`backfill`、`import`、`export`、`disconnect`、`archive-get`、`archive-list` 与不带 shelf 的命令跳过。
    - `meta` 状态的读-改-写是原子的：SQLite 用 `BEGIN IMMEDIATE`，PG 用按键的 `pg_advisory_xact_lock`。
    - 读前补齐不看 `defer`：关闭延迟的 shelf 在 `similar` 前同样补上存量欠账。关闭延迟时，写入只在向量能写入时才嵌入。

**D 的具体内容**：

- **自带下载器**：`ureq` 已是依赖（带 TLS，远程 provider 在用），直接走 `https://huggingface.co/<repo>/resolve/main/<file>`，不需要 Python 的 `hf`。要求：Range 断点续传（2.27 GB）、按 HF LFS 的 sha256 校验、下载完成后原子 rename、stderr 输出进度。
- D 与 F 不耦合：三种分发方式都不会把 2.3 GB 的模型塞进包里。
- **挂到目标 shelf**：没有 shelf.toml 的 shelf 只看 shelf 目录（§2.5 场景 2），因此 D 要在目标 shelf 的 shelf.toml 写入 `model = "<name>"`。只改**没有向量**的 shelf，写后立刻以新配置打开一次完成改绑（依赖 I）；已有向量的 shelf 只打印切换命令与 reembed 提示。
- D 实现时定下的细节：
  - 命令为 `hypatia model install <Org/Name> [-s <shelf>] [--revision <rev>]`，落到 `~/.hypatia/models/<Org>/<Name>/`，保留仓库内的相对路径（`onnx/…`），现有的模型解析无需改动即可找到。名字只允许 `Org/Name` 形式，防止写出 models 目录；已用 `model register` 登记（软链接）的名字拒绝写入。
  - 先把 revision 解析成 commit（`/api/models/<repo>/revision/<rev>`，revision 整段百分号编码，返回的 sha 须为 40 位十六进制），所有文件都从这一个 commit 下载；目录名与写入 shelf.toml 的名字用 Hub 返回的 `id`（只接受大小写差异）。只列根目录与 `onnx/`，并跟随 `Link` 分页（只跟随同一 Hub 的链接）：取 `model.onnx`（优先 `onnx/`），同目录的 `model.onnx_data` / `model.onnx.data` 及编号分片 `model.onnx_data_<n>`，以及同目录的 `tokenizer.json`，没有才用根目录的 —— BAAI/bge-m3 根目录与 `onnx/` 的 tokenizer 并不相同。图文件最后下载，中断的安装不会被当成可用模型。没有 ONNX 导出的仓库直接报错。
  - 模型目录里写 `.hypatia-install.json`，记录 commit 与每个文件的版本（LFS 文件用 sha256，其他用 git oid），每落地一个文件更新一次。只有记录与 Hub 当前版本一致的文件才跳过；没有记录但已在的大文件按 sha256 校验通过后接受。文件换了版本时提示依赖该模型的 shelf 需要 `backfill --reembed`（identity 只含模型名，不含 commit）。
  - 下载写到 `<file>.<版本前 16 位>.part`，续传发 `Range`：206 且 `Content-Range` 起点正确才追加；服务器返回 200、起点不对或 416 都从头来。边写边算 sha256，LFS 文件比对 Hub 给出的 sha256（忽略大小写），小文件只比大小；校验不符删除 `.part` 并报错；`sync_all` 后原子 rename。请求带 `Accept-Encoding: identity`。连接 30 s、等响应头 60 s 超时，不设整体超时；卡住时中断重跑即可续传。
  - 同一模型目录用操作系统文件锁防止并发安装（进程退出即释放，Ctrl-C 后可直接重跑）；models 目录下沿途有软链接（例如 `model register` 登记的模型）一律拒绝写入。
  - 支持 `HF_ENDPOINT`（镜像）与 `HF_TOKEN`。token 在同主机重定向时保留（Hub 对小文件用同主机 307），跨主机（CDN）时丢弃。Hub 对不存在的仓库也回 401，因此没有 `X-Error-Code` 时提示「不存在，或 gated / private」；`GatedRepo` 时提示先在 Hub 上接受条款。
  - 下载前先确认目标 shelf 已连接。shelf.toml 用 `toml_edit` 原地修改：缺 `[embedding]` 时建独立的表（不是内联表），保留注释与 `[storage]` 等其他设置，清掉已被 `model` 取代的 `model_path` / `tokenizer_path`；写入沿用原文件权限（可能含数据库密码），`sync_all` 后 rename，软链接则写到它指向的文件。重新打开失败时恢复原文件，恢复也失败则两个错误一起报告。remote shelf 与已名为该模型的 shelf 不改；非 BAAI/bge-m3 的模型额外提示检查 `dimensions` / `pooling` / `max_seq_length`。
  - 配置里 `model` 形如 `Org/Name` 且未安装时，提示信息改为「run `hypatia model install <name>`」。

**E 的状态提示建议**（依赖 C、I 落地，否则最后一句不成立）：

```
✓ shelf 'default' 已就绪
✓ 全文检索、图遍历、JSE 查询可用
○ 语义检索未启用 —— 运行 `hypatia model install BAAI/bge-m3`（约 2.3 GB）
  或在 shelf.toml 配置远程 API。已有内容会在启用后自动补齐向量。
```

E 实现时定下的细节：

- 命令为 `hypatia init [<dir>] [-n <name>]`：不带目录时报告 default shelf（`Lab::new` 已确保它存在）；带目录时连接该目录，已以同一路径登记则直接复用（比较时跟随符号链接；命名与登记用输入的路径，只解析 `..`，与 `connect` 一致），同一目录换名字则报错，名字已登记给别的目录时也报错并提示 `-n`（连接会覆盖那条登记，即使那个 shelf 只是启动时没打开），因此重复执行不改变任何东西。已登记但启动时没打开的 shelf 会重新打开一次，报出真实的打开错误。输出为英文，与其余 CLI 一致。
- 状态分三层：shelf 就绪（路径与后端）；全文检索、图遍历、JSE 可用；语义检索。语义检索分三种：未启用时给出唯一的下一步（未配置模型 → `model install BAAI/bge-m3 -s <shelf>` 或配置远程 API；配置了未安装的 `Org/Name` → 安装该模型；远程缺 key → 说出环境变量名），并说明已有内容会自动补齐；模型存在但写不进向量（模型变更、legacy 向量）→ 给出 reembed 提示；可用 → 说明用的是哪个模型，有欠账时说明条数及暂停原因或 `backfill` 命令。
- 「语义检索未启用」的同一段提示也用于 `similar` / `$similar` 的报错（`OpenShelf::semantic_search_off`），替代原来「找不到哪个模型文件」的报错；`similar` 与 `backfill` 都先检查它，再检查模型变更（重新 embed 也需要可用的模型）。提示里的命令一律带 `-s <shelf>`：`model install` 与 `backfill --reembed` 不带 `-s` 时作用于 default shelf，照着非 default shelf 的提示执行会改错 shelf。`init` 不跑命令入口的欠账检查，不加载模型，不连网，所以「可用」只表示已配置好，写作 "set up with"。

**F 的具体内容**：

- curl 安装脚本：探测平台 → 从 GitHub Releases 下载对应产物 → 放入 `~/.local/bin`。curl 下载不带 quarantine 属性，避开 Gatekeeper 对未签名 macOS 二进制的拦截；脚本内检查 AVX2（x86-64）与 libc++（Linux）。
- 现有产物格式不统一（`.gz` 与 `.tar.gz`），脚本需兼容，或在 workflow 里统一。补齐 Intel Mac 与 Linux（glibc x86_64 / aarch64）产物需要一个 release matrix workflow。发布动作由维护者执行，本分支只提供脚本与 workflow 文件。

**关于顺序**：F 是漏斗第一道门 —— 受众是「想给自己的 Agent 加记忆」的人，多数不装 Rust 工具链。它排在最后不是因为不重要，而是完整覆盖依赖维护者配合发布；在此之前，A 把 Releases 链接放上 README 第一屏，已能覆盖 macOS arm64 与 Windows x64 用户。

## 5. 顺带发现的缺陷

以下是检查过程中发现、与本方案相关但可独立修复的问题。

### 5.1 embedding 欠账不可观测

`src/storage/shelf_manager.rs:118-134` 的 `embed_saved` 尽力而为，失败往 stderr 打一行 warning 并 **exit 0**。调用方只看退出码，无法发现向量没生成。结果是条目写进去了、`similar` 找不到它、而且没人知道。

**写入的 exit 0 应当保留。** 写入的契约是「内容已提交」，向量是可重建的派生缓存。下游（如 `dsh-hypatia-auto-memory` 的 `runOk`）遇非零退出会重试，而 `statement-create` 对重复三元组报错、不幂等（见姊妹篇 §5.3）—— 一次误报的失败会连锁成一次真失败。

真正缺的是可观测性。延迟成默认之后 pending 从异常变成常态，信号应落在：

- **`similar` / `$similar`**：有欠账或熔断打开时，stderr 提示「N 条尚未嵌入，结果可能不完整」。结果不完整而没有提示，才是用户会被坑的地方。
- **显式 `hypatia backfill`**：它唯一的职责就是嵌入，但现在 errors > 0 也 exit 0（`src/cli/commands.rs:428-432`）。这里才应非零退出。
- **`hypatia backfill --status`**：可查询的欠账视图。
- **MCP 写工具**：结构化的 `embedding` 状态字段（姊妹篇 §3.1）。

### 5.2 `backfill` 的不可用提示对远程 provider 有误导

`src/lab.rs:268` 的 `is_available()` 检查失败时，错误信息只提本地路径（"place embedding_model.onnx and tokenizer.json in the shelf directory"），即使该 shelf 配置的是 remote provider。远程不可用（API key 未设、endpoint 不通）会得到误导性提示。

### 5.3 每次调用都打开全部已注册 shelf

`src/storage/shelf_manager.rs:240-254` 的 `restore_registered`（在 `:234` 调用）在每次 CLI 调用时打开所有已注册 shelf。后果：一个 shelf 配置出错，所有 shelf 上的每条命令都打一行 `failed to restore shelf`（§2.5）；注册了 PG shelf 时，操作 SQLite shelf 也要连一次 PG。建议改为按需打开命令的目标 shelf。与本方案独立，单独立项。

## 6. 兼容性

数据层无破坏：不需要迁移，导出格式不变，新旧 binary 双向可打开 shelf —— 前提是守住下表的约束。

| 改动 | 旧 binary 打开新 shelf | 约束 |
|---|---|---|
| SQLite 部分索引 | ✅ | **不升 `schema_version`**：旧版只认 1 / 2 / 3，其他一律拒开（`src/storage/sqlite_store.rs:309-316`）；用 `CREATE INDEX IF NOT EXISTS` |
| `meta` 新键 | ✅ | SQLite 只按键名读取；PG 只校验预期的键 |
| I：无向量时改绑 identity | ✅ | identity 的计算方式不变，同配置算出同一 identity |
| `embedding.defer` | ❌ | 只在用户关闭延迟时写入 |
| D 写入 `model =` | ⚠️ | 只改无向量的 shelf，写后立即以新 binary 打开完成改绑 |
| 导出 `Manifest` | — | `deny_unknown_fields`：不加字段，flush 状态不导出 |

行为契约变化（需在 PR 描述中列出）：

- 默认延迟：远程下写后 60 s / 128 条内 `similar` 找不到新条目，更新过的条目在下次 flush 前从向量检索中消失；本地写后即可搜到。
- 写入耗时：平时从 ~830 ms 降到 ~15 ms，但本地越过阈值的那次写入约 3.5 s。
- 显式 `backfill` 在 errors > 0 时非零退出。
- stderr 文案变化（warning 按原因分流、`similar` 不完整提示）。仓内调用方不受影响：DSH 插件只看退出码，skill 不引用这些文案。
- identity 不匹配从「整个 shelf 打不开」放宽为「降级打开」；PG 可在零模型下建库。
- 以占位 identity 新建的 PG shelf，旧版 binary 打不开（旧版要求可信 identity），直到新版以真实模型打开一次完成改绑。

`Lab::similar` 改为 `&mut self` 不构成对外破坏：crate 未发布到 crates.io，只影响仓内 REPL 与测试。

B 的兼容陷阱见 §3.4 第 3 点：批量与单条数值等价、固定 batch 维时退回逐条、按 `index` 排序、4xx 按错误码分流。
