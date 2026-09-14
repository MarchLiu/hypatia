# Agent 接口分层：CLI、MCP 与 Skill

> 状态：**实施中**（2026-09-10 方案；2026-09-11 修订：对齐姊妹篇的 embedding 决策，补 H 的运行时选型、§5.3 的修法与实施顺序；2026-09-14 修订：姊妹篇已合并入上游 main，按上游实际接口重写 §3.1，新增 §3.5 多会话与多进程，记录 H v1 的决定）
> 背景：Agent 通过什么与 hypatia 对话 —— 三层分工、MCP 的范围与边界、skill 的不可替代性，以及明确不做的事
> 姊妹篇：[降低上手成本方案](onboarding-plan.md)（上手漏斗与 embedding 生命周期）
> 范围：分析与设计建议；实施进度见 §5 的实施顺序

## 1. 结论

**MCP 与 skill 不是替代关系，是接口层与判断层的上下游。**

- MCP 能把接口层从散文变成 schema，让已知的漂移（§5.1）**变得不可能发生**，而不是被修好。
- 判断层**必须**留在 skill 里 —— 这是机制上的必然（§3.2），不是偏好：MCP prompts 需要用户显式调用，而记忆协议要求自动触发。
- MCP 值得做，但在上手漏斗里排在最后：一个连 binary 都装不上的用户，永远走不到配置 MCP 那一步。
- **不做**通用 server 模式（长连接 / 连接池 / 多线程）—— 收益错配且被后端矩阵放大（§4.1）。
- 近期最值得做的不是 G 或 H，而是 **§5.3 的 statement-create 幂等**：skill、DSH 插件、将来的 MCP 三条路都经过同一个 `insert_statement`，今天没有一条有安全的重试语义。
- 延迟 embedding（姊妹篇任务 C）落地后，H 的写侧延迟收益已被 CLI 拿走；多会话下向量缓存总是冷的，模型又改为用完即释放，读侧常驻收益也不复存在。**H 的价值只剩接口本身**：schema 化，以及结构化的欠账与错误（§3.3、§3.5）。

## 2. 三层模型

| 层 | 载体 | 内容 | 可机器校验 |
|---|---|---|---|
| 契约 | JSE 规范 | 算子语义 | 是（已有测试覆盖） |
| **接口** | **CLI / MCP tools** | 有哪些操作、参数类型、返回结构 | **是（schema）** |
| **判断** | **skill** | 何时调用、怎么组合、如何裁决 | 否（散文，靠 eval） |

`skills/hypatia/SKILL.md`（569 行）实际是混着的，粗估接口层约六成、判断层约四成：

- **接口层（MCP 可吸收）**：Binary Location、Shelf / Archive / CRUD 的命令形式、JSE 算子参考、Critical Syntax Rules、Options、Output Format。其中 `Shell Escaping` 与 `Sandboxed / Read-Only Direct Access (known workaround)` 两整节是纯粹的「接口是字符串」税，MCP 之后整节消失。
- **判断层（MCP 吸收不了）**：`Statement Creation — Proactive Graph Building` 与 `Auto-linking Rules`（「宁冗勿漏关系」的操作化）、`Thinking Aloud Protocol`、**`Search Strategy — Graph-First Retrieval` 与 `Search Decision Tree`**（`memory-webway.md` 那句「图 + 向量 + 全文的组合构成完备的知识发现能力」落到 Agent 身上的具体形态）、`Disambiguation Rules`、`Scopes` 约定。

`skills/hypatia-memory/SKILL.md`（501 行）几乎全是判断层，MCP 一行都吸收不了。
`skills/hypatia-dream/SKILL.md`（245 行）介于两者之间，CLI 调用部分可吸收，整理策略保留。

## 3. MCP

### 3.1 范围与边界

若要做，建议卡死为：

- 作为 **`hypatia mcp` 子命令**，不是独立 binary —— 单一安装、单一版本，任何 Agent 配一行 `command: hypatia, args: [mcp]`。
- **运行时：同步、手写 stdio JSON-RPC，不引入 tokio。** 代码库是全同步的（`ureq`、同步 `postgres`、`rusqlite`，零 `async fn`）；官方 Rust SDK `rmcp` 基于 tokio，引入它意味着给同步代码库塞一个 runtime，且每次调 `Lab` 都要 `spawn_blocking` 包一层。只做 tools 与 resources 时协议面很小：`initialize`、`notifications/initialized`、`ping`、`tools/list`、`tools/call`、`resources/list`、`resources/templates/list`、`resources/read`。stdin 逐行读、逐条处理的同步循环即可，与 §4.1「不引入 server 基础设施」一致。协议版本（2026-09 核实）：采用带 `initialize` 握手的版本，支持 2025-11-25、2025-06-18、2025-03-26、2024-11-05，客户端请求其中之一就原样返回，否则返回 2025-11-25；批量请求自 2025-06-18 起已移除，一律拒绝；resource 不存在返回 -32002；参数校验错误按 2025-11-25 归为 tool 的 `isError`，让模型能自行纠正。2026-07-28 版取消了握手、改为每个请求自带版本，而 Claude Code 与 Codex 仍用握手，暂不实现。
- **tools 只覆盖数据面**，贴着现有子命令，**零「智能记忆」端点**（理由见 §4.2）：`query`、`search`、`similar`、`session_current`、`knowledge_create` / `knowledge_get` / `knowledge_update` / `knowledge_delete`、`statement_create` / `statement_delete`、`archive_store` / `archive_get` / `archive_list`、`list_shelves`、`shelf_status`、`backfill`（限批，见下）。`shelf_status` 与状态 resource 返回同一份文档：模型能否自主读取 resource 因宿主而异，工具则总能调用。
- **管理面留在 CLI，不做成 tool**（2026-09-14 定）：`connect` / `disconnect` / `init` 会写 `~/.hypatia/shelves.json`，而长驻进程写回的是启动时读到的旧表，会冲掉其他会话期间的注册（§3.5）；`model install` 是 GB 级下载，会长时间占住同步循环；`export` / `import` 是整库文件操作。这些是人在终端里做一次的事，不是 agent 在对话中反复调用的事。
- **配置只在启动时读一次**：shelf 注册表、`shelf.toml` 与模型可用性在 server 启动时确定，`model install`、`connect`、`init` 之后要重启 server；启动时下发给宿主的说明里已写明。
- **archive 名必须是 archives 目录内的相对路径**：`Lab` 层只接受普通路径段，拒绝 `.`、`..`、绝对路径与盘符前缀（与导出校验 archive 引用的规则一致），并按真实路径复查，挡住 archives 目录内已有符号链接的出逃；CLI 一并受益。此前名字不经检查就拼接，经 MCP 等于把任意写路径交给模型输出。`file` 参数仍可指向用户可读的任意本地文件。
- **暴露 `knowledge_update`**（2026-09-14 定）：CLI 同时新增 `knowledge-update`，两者共用 `Lab::patch_knowledge` 的字段合并语义：未给出的字段保持原值，给出的字段替换，空值清空该字段；内容无变化时不写库，版本号与向量都不动。先读后写之间若有其他会话改了同一条目，写回的是整条内容，那次改动会丢失，即使改的是本次没有给出的字段；窗口只有两条 SQL、不含模型调用，同一条目被并发更新也少见，所以不做版本比对。
- **resources** 用 resource template（`hypatia://{shelf}/status`、`hypatia://{shelf}/knowledge/{name}`、`hypatia://{shelf}/statement/{head}/{relation}/{tail}`，每段百分号编码）；`resources/list` 只枚举 shelf 与 shelf 状态，不枚举条目 —— 大库会把列表撑爆。
- 薄薄盖在 `src/lab.rs` 的 facade 上，不复制业务逻辑。`src/cli/repl.rs` 已证明 `Lab` 可跨命令长期持有。
- **本地模型用完即释放**（2026-09-14 定）。加载本就是惰性的，只有 flush 与 `similar` 会触发；v1 在每个 tool call 与 resource 读取结束时释放本地模型，不做空闲计时。实测每个进程加载 bge-m3 后约占 1.5 GB 私有内存，进程之间不共享（§3.5），常驻 N 个会话就是 N 份。释放后的实测（bge-m3，macOS，2026-09-14）：调用中 physical footprint 峰值 1.5 GB；释放后空闲时约 50 MB 到 540 MB，并非归零，因为分配器留住了部分已释放的内存供下次加载复用；连续 12 次调用后 RSS 稳定在约 870 MB，不再增长，没有泄漏。代价是每次需要本地模型的调用都要重新加载，实测每次约 1.7 到 1.9 s；远程 provider 不受影响。实现上给 `EmbeddingProvider` 加一个默认空实现的释放方法，本地 provider 回到待加载状态，由 MCP 层在调用结束时对各 shelf 调用；CLI 进程本就用完即退，无需调用。provider 仍是 shelf 作用域，因为配置在每个 shelf 各自的 `shelf.toml`，一个 server 会挂多个 shelf。
- 写工具**不阻塞在 embedding 上**，返回 **shelf 级的欠账快照**，而不是每次写入的 `done | pending | failed`。姊妹篇 §3.5 已指出：延迟默认化后，逐条状态几乎永远是 `pending`，没有信息量。快照直接序列化上游的 `Lab::embedding_debt`，字段示意：

  ```json
  { "name": "wu-2026-09-10-xxx", "created": true,
    "embedding": { "pending_knowledge": 12, "pending_statement": 5, "passed_over": 0,
                   "pending_since": "…", "blocked": null,
                   "paused": { "permanent": true, "reason": "remote provider: 401 unauthorized",
                               "retry_after": null } } }
  ```

  `blocked` 说明此刻为什么写不了向量：模型不可用、向量属于另一个模型、没有可识别的模型身份，或向量早于身份追踪。`paused` 说明自动补齐因失败暂停，并标明是否要改配置。`passed_over` 是自动补齐会跳过、只有显式 backfill 才会重试的条目数。这是 MCP 相对 CLI 的**真增量**：CLI 侧只能靠 stderr 与 `backfill --status` 看见同一份状态。
- **shelf 状态 resource** 直接序列化上游的 `Lab::shelf_status`：名字、路径、是否 PG、embedder、`semantic_search_off`（语义检索为何关闭、如何打开）、`attention`（有模型却写不了向量的原因与处置）、欠账快照。它就是 `hypatia init` 报告的那份状态，MCP 不另起字段。
- **错误走结构化的 tool 错误**，不照搬 CLI 的「exit 0 加一行文本」：条目不存在、语义检索关闭（上游 `execute_similar` 返回 `ModelUnavailable`，消息里带下一步）都作为 tool 错误返回，agent 不需要匹配措辞。§5.1 的漂移 #5 在 MCP 侧由此消除。
- **`backfill` tool 限批**：每次调用最多补一批（默认 64 条，上限 512），返回本批结果与剩余欠账，由调用方决定是否继续。`Lab::backfill_vectors` 一次补完整个 shelf、没有批量参数，本地模型下可能跑几分钟，期间同步循环无法响应同一会话的其他请求，写事务也会让别的会话等锁（§3.5）。需要在 `Lab` 的 impl 末尾新增一个限批方法；显式 backfill 的完整重试语义（姊妹篇 §3.3）仍在 Lab 内，MCP 不另写。`--reembed` 属于管理面，不暴露。默认值取 64，是因为 Codex 的工具调用默认 60 s 超时。向量被锁定（模型不可用、身份不匹配等）时直接报错说明原因，不返回全零。批次按从新到旧取待补条目，若最新一批总是失败会挡住更早的欠账；v1 不做游标分页，一批一条也没补上时，工具描述要求 agent 停下，改用终端里按键分页的完整 `hypatia backfill`。
- **入口先还逾期欠账**：写入阈值检查在 `OpenShelf` 内，`similar` 之前的补齐在 `Lab` 内，走 `Lab` 即命中；但远程时间上限不在 `Lab` 内部，而是由 CLI 在命令入口显式调用 `Lab::flush_if_overdue`。每个针对 shelf 的 tool call 入口同样调用一次，尽力而为、忽略其错误，与 CLI 和 REPL 一致。
- **依赖已满足**：姊妹篇已合并为上游 main 的 f9b897d，本分支已 rebase 其上，上文引用的接口均已在代码中核实。

### 3.2 MCP prompts 承载不了记忆协议

MCP 三个 primitive 的控制权归属不同：

- **tools**：模型自主调用，工具定义常驻上下文
- **resources**：应用 / 用户选择
- **prompts**：**用户显式调用**（多数客户端表现为斜杠命令或模板选择器）

而 `hypatia-memory` 要求「每 5 轮扫一次窗口」「session 启动时预加载 rules / taboos」「收敛时立刻梳理」—— **必须自动触发**。MCP prompts 天然做不到；skill 才有 description 匹配与 always-on 的调度语义。

**因此判断层必须留在 skill 里。** 这是本文最硬的一条结论。

### 3.3 主要收益

不是「少装一个东西」，而是三条：

1. 接口层从散文变成 schema，**漂移变得不可能**。§5.1 列出的五处漂移中，后三处将不可能再发生。
2. skill 变短，判断层信噪比提高，`Search Decision Tree` 这类真正有价值的内容被遵守的概率上升。
3. skill 不再需要写 Shell Escaping、Binary Location、沙箱 workaround 这类宿主细节，跨 Agent 分叉压力下降（不会归零）。

**不再是收益的**：性能。写侧延迟已由姊妹篇拿走，实测写入从约 830 ms 降到约 15 ms。读侧模型常驻原本是剩下的一项，但多会话并发写入时向量缓存总是冷的，模型又改为用完即释放（§3.5），这一项也放弃了，换来空闲会话不再常驻整份模型：实测空闲占用从约 1.5 GB 降到 0.5 GB 左右。

### 3.4 安装：更统一，但不更普适

| | Skill | MCP |
|---|---|---|
| 装的动作 | 复制目录 / 插件注册 | 一条 `mcp add` 或一个 JSON 块 |
| **跨 Agent 一致性** | **每家不同** | **一种形状到处适用** |
| **覆盖面** | **任何能读文件 + 跑 bash 的 Agent** | 仅支持 MCP 的宿主 |
| 装完的行为 | 取决于宿主怎么调度 | 一致 |
| 生效 | 通常即时 | 通常要重启客户端 |

**MCP 的安装更统一，skill 的安装更普适。**

证据在仓内：`skills/hypatia-memory/SKILL.md` 为 **Codex Hooks** 和 **OpenCode** 各写了一整节集成说明；`dsh-hypatia-auto-memory` 又自带一份 80 行的 DSH 专用变体，且需靠 provider 检查防止与主仓那份互相覆盖。**一份内容分叉成四种装法**，成本随 Agent 数量线性增长。

但当前安装负担里有相当一部分不是「skill 这个形态」的固有成本，而是**没做工具**的成本 —— `scripts/` 下没有任何 skill 安装工具，装 skill 至今是手工复制。

对「Agent 中立」（见 `memory-webway.md`）的取舍：MCP 更统一但覆盖面窄，skill 覆盖全但每家不同。一个把 Agent 中立写进设计目标的项目不能只看统一性 —— **两者应当共存，而不是替代**。

### 3.5 多会话与多进程

stdio 本身不是并发瓶颈：宿主为每条 stdio 连接拉起一个 server 进程，Claude Code 与 Codex CLI 都是每个会话一个。所以「多个会话同时记忆」就是多个 `hypatia mcp` 进程同时读写同一个 shelf。以下结论出自代码核对与 2026-09-14 的实测（本机 bge-m3，12 物理核，48 GB 内存）。

| 方面 | 结论 | 对 H 的处置 |
|---|---|---|
| 单连接内的并发请求 | JSON-RPC 允许多个请求同时在途、按 id 配对。同步循环逐条处理，协议允许，结果正确；慢调用会挡住后面的调用，取消通知打断不了正在执行的调用 | backfill 限批；GB 级下载不做成 tool |
| SQLite | WAL，`busy_timeout` 5 s，多进程读写安全：写串行，读不阻塞。某进程持写锁超过 5 s 时，其他会话的写入会失败 | 同上，避免长写事务 |
| 本地向量缓存 | 按 `meta` 里的内容时钟编版本，缓存文件名带时钟，保存用唯一临时文件加改名，多进程下正确。但任何写入都会推进时钟，其他进程下次检索要加载或全量重建缓存 | 不把读侧常驻当收益 |
| 嵌入模型内存 | 每个进程各自加载。实测单进程峰值私有内存 1513 MB，三进程并发时每个 1512 MB，进程间不共享；同一进程内不同 shelf 也各有一份 | 用完即释放；实测释放后空闲 footprint 约 0.05 到 0.54 GB，并非归零 |
| 推理线程 | 会话未设线程数，ONNX Runtime 默认按物理核数开线程池。三进程各写一条时耗时与单进程相当（约 1.9 s 对 1.86 s）；多进程同时批量推理会超额占核，这一点是推断，未实测 | 后续可限线程数，不在 H 内 |
| shelf 注册表 | 只在进程启动时读一次。connect / disconnect 把启动时的旧表整份、非原子地写回，会冲掉其他会话期间的注册，并发写还可能写出损坏的文件 | connect / disconnect / init 不做成 tool |
| 延迟 embedding | 两个进程可能同时补同一批，向量写入按内容版本做比对，结果正确，只是重复计算 | 无需处置 |

真正需要跨进程共享一份模型时，现成的办法是把远程 provider 指向本机的 Ollama，它本身就是共享的 embedding 服务。hypatia 自己做 embedding 守护进程，就是 §4.1 否决的 server 模式。

**协议层的竞态与传输无关，今天用 CLI 就存在。** hypatia-memory 的摘要级联按项目查询未摘要的消息，而不是按会话。同一项目里的两个会话会拿到同一批消息，各自花几十秒生成摘要再写回，同一批消息被摘要两次。语义抽取的「先搜再建」同理。修法不在 MCP：让级联按会话分区，跨会话的合并交给 hypatia-dream。这是 skill 的单独修复，不在 H 范围内。

## 4. 明确不做

### 4.1 通用 server 模式（长连接 / 连接池 / 多线程）

- **收益错配**：server 的实际价值只有「嵌入器常驻」一件事。实测进程启动与开库只要 10 ms，连接池省不出东西；操作天然串行，多线程用不上。而「嵌入器常驻」用延迟 embedding 能拿到写侧的大部分（姊妹篇 §3，§3.5 的实测已证实），用 MCP stdio 能顺带拿到读侧。
- **成本被后端矩阵放大**：PG 后端已是原生实现（`src/storage/postgres_store.rs` 零处 `json_index`，走 GIN / tsvector / 递归 CTE / pgvector），与 SQLite 侧差异显著且预期继续拉大；`ShelfBackend` trait 有 38 个方法。再加一个「进程内 vs 常驻」的轴，就是 2×2 且两轴都在动。
- **唯一真正的设计约束是「CLI 是唯一接口」**，而这条约束在 PG 后端上也被遵守（可选后端藏在同一接口后面）。MCP 遵守它的方式是作为 `hypatia mcp` 子命令，而非独立服务。

> 注：早期讨论曾引用 `memory-webway.md`「hypatia 强调的是可以部署在每个电脑本地，而那个内部系统是一个基于 PostgreSQL 的服务」来论证「server 的事不归 hypatia」。作者已明确表示两个系统分叉后不会再长成一样、做 PG 支持没有顾虑，因此**该论据作废**。上面三条不依赖它。

**何时该重估**：如果负载从「本机若干 Agent 串行小批量读写」变成「并发的、需要共享状态的、跨进程协调的」（例如需要读-判-写的原子性来做知识版本仲裁），那时常驻就不是加速器而是必需品。但届时第一选择可能仍是 PG 的事务，而非 hypatia 的守护进程。

### 4.2 「智能记忆」MCP 端点

有了 MCP，暴露 `remember(text)` / `recall(query)` 这类端点会很有诱惑力。**这正是 `dsh-hypatia-auto-memory` 犯的错** —— 把判断从有完整上下文的主 Agent，移到只看得见转录文本的旁路。

`memory-nolinear.md` 的整套设计假设判断在 Agent 一侧：话题何时收敛、这条是 extends 还是 supersedes、哪段是纠错链。**MCP tools 要薄，贴着图原语。**

## 5. 待办与顺带发现的缺陷

实施顺序（2026-09-11 定）：

```
0. 本文修订                                    已完成
1. statement-create 幂等（§5.3）               已完成
2. skill 文本修正：漂移 #1、#2（§5.1）          已完成
3. G. hypatia skill install --agent ...         v1 已实现
4. H. hypatia mcp 子命令                        v1 已实现
```

排序理由：1 是所有集成路径共同经过的写路径缺陷，且成本几十行；2 必须先于 G，不能把已知错误的文本编进 binary；G 是本分支的名义交付物，但主要惠及新用户；H 绝对价值最大；原先被姊妹篇阻塞，现已解除，但其性能收益已不复存在（§3.3）。

**H v1 的决定（2026-09-14）**：

- 管理面不做成 tool，其中 connect / disconnect 明确不暴露（§3.1、§3.5）。
- 本地模型每次 tool call 用完即释放，不做空闲计时（§3.1）。
- backfill 限批（§3.1）。
- 暴露 `knowledge_update`；CLI 的 `knowledge-update` 与 MCP 工具均已实施（§3.1）。
- 不在 H 内：hypatia-memory 摘要级联按会话分区，ONNX 线程数限制，同一进程内多个 shelf 共享模型（§3.5）。

**G v1 的范围**：

- 三份 skill 用 `include_str!` 内置进 binary —— 「零依赖」指装 skill 不需要 repo checkout，skill 版本等于 binary 版本。漂移 #1、#2 这类错误从此必须在编译前修掉。
- `--agent claude|codex|opencode`，写入各宿主的用户级 skill 目录；`--dir` 可装到任意目录。原计划「以 hypatia-memory 的集成章节为准」不成立：那几节讲的是 hooks 与插件，不是 skill 目录。各宿主目录按其官方文档与源码核实（2026-09）：claude → `~/.claude/skills`（不认未文档化的 `CLAUDE_CONFIG_DIR`）；codex → `~/.agents/skills`（`$CODEX_HOME/skills` 已废弃，仅兼容读取）；opencode → `~/.config/opencode/skills` 或 `$OPENCODE_CONFIG_DIR/skills`。OpenCode 还会读前两处，同名 skill 只留一份；Codex 仍兼容读取废弃的 `$CODEX_HOME/skills`。`install` 与 `status` 对这些位置里「存在但非最新」的同名副本逐个告警，不依赖宿主的扫描顺序。
- `hypatia skill status`：对比已装副本与内置版本，让「装过但过时」可见。状态分五种：missing、current、outdated（hypatia 装的旧版，可直接升级）、modified（装后被改过）、unmanaged（内容不同且非 hypatia 所装）。后两种不加 `--force` 不覆盖，且 `install` 以非零退出。
- 安装记录按内容哈希存在 `~/.hypatia/skill-installs.json`，不在宿主目录里放标记文件，免得宿主把它当资源加载。
- `skill` 子命令在 `Lab::new()` 之前分发，装 skill 不打开、也不创建任何 shelf。
- **已定：不做 `--agent dsh`**。`dsh-hypatia` 插件自己把内置 skill 注册为 runtime skill，优先于用户级 `~/.dsh/skills`，往那里装的副本会被遮蔽。`dsh-hypatia-auto-memory` 的 DSH 变体属于插件自身，也不归 G。
- **G 只装 skill 文件，不装 hooks**。hypatia-memory 靠宿主 hooks 触发，装完后 CLI 会提示；hooks 仍由 `codex-integration/install.sh` 与 `opencode-integration/` 负责，它们依赖 repo checkout。把 hooks 也收进 CLI 是后续可选项。
- 顺带修正：`codex-integration/README.md` 声称 `install.sh` 会装 skills，脚本实际只装 hooks。
- 宿主适配层生成留 v2。

G 的进阶形态（v2）：**判断层单一源，宿主适配层由 CLI 生成**。现在主仓 `hypatia-memory` 与 DSH 变体是两份手写、会漂移的文档（两者对协议的实现差了整整一个分层 cascade）。若适配是生成的，这类漂移也会变成不可能。

### 5.1 skill 与 CLI 之间没有同步机制，已产生五处漂移

1. `hypatia-memory` skill 提到 `knowledge-update`，CLI **没有这个子命令**。
2. 通用 skill 把 `--scopes ""` 定义为全局，CLI 实际解析为「无 scope」；只有尾逗号才会写入空字符串全局标记。
3. `search` 用 `-c/--catalog`，`similar` 用 `-t/--target`（易混，下游为此写过 NOTE）。
4. 内容以 `-` 开头（markdown 列表）时必须用 `--data=<v>`，`-d <v>` 会被 clap 当作新 flag。
5. 「条目不存在」走 **exit 0 + stdout 文本**，需正则 `^Knowledge '.*' not found\.$` 才能与真实 JSON 区分 —— CLI 换一次措辞，下游即崩。

处置：

- 1–2：文本已修，`hypatia` 与 `hypatia-memory` 两份 skill 及其 `dsh-hypatia/skills/` 镜像同步。长期靠 G v2 的生成消除。
  - #1 改为「先删后建」：`knowledge-delete` 不级联 statement（两个后端都没有外键），实测 `belongTo` 边在重建后保留。代价是 session 节点的 `created_at` 重置，对运营型节点无害。
  - #2 实测：`--scopes ""` 存为无 scope，`["$has","scopes",""]` 查不到；`","` 存为 `[""]`，`"p,"` 存为 `["p",""]`。已改为尾逗号写法。原文会让 agent 写下的全局规则在全局查询里永久不可见。
  - 后续（2026-09-14）：CLI 已补 `knowledge-update`，hypatia-memory 改回用它替换 session 节点，保留 `created_at`，也不再有删与建之间的中断窗口。
- 3–4：MCP 的 schema 化之后不可能再发生；CLI 侧不动，改 flag 名是无收益的破坏。
- 5：**暂缓**。改 exit code 会让 DSH 插件的 `runOk` 把「不存在」当失败去重试，在 §5.3 修好之前会撞上不幂等的 `statement-create`；改措辞会让它的正则失效。MCP 会替 MCP 宿主解决；skill 宿主的收益不足以抵消破坏面。

### 5.2 写入时 embedding 欠账不可观测

姊妹篇合并之前，`OpenShelf::embed_saved` 尽力而为，失败往 stderr 打一行 warning 并 **exit 0**。调用方（如 `dsh-hypatia-auto-memory` 的 `runOk`）只看退出码，无法发现向量没生成。

姊妹篇 §3.3 已定：**写入保持 exit 0**。向量是可重建的缓存，没生成不算写失败；自动 flush 永不改变触发它的那条命令的退出码；下游按非零重试还会撞上 §5.3。因此这不是「静默失败」，而是「欠账不可观测」：CLI 侧的信号移到 `similar` 结果不完整时的 stderr 提示、`backfill --status`，以及显式 `backfill` 在 errors > 0 时的非零退出（姊妹篇 §5.1）。MCP 侧的对应物是 §3.1 的欠账快照 —— 同一份状态，结构化地随写入返回。

### 5.3 `statement-create` 对重复三元组报错而非幂等

`src/storage/sqlite_store.rs:560` 的 `INSERT INTO statement` 没有 `ON CONFLICT`，而主键是 `triple` 单列（`head,relation,tail` 的 CSV 键，`sqlite_store.rs:36-46`），因此重复三元组触发 UNIQUE 约束错误、非零退出。这使得「崩溃后重放补边」无法实现 —— 任何多步图写入（先建 knowledge 再建若干 statement）在中途失败后都无法安全重试。

这对任何 Agent 集成都是硬伤：`hypatia` skill 的 `Always Enrich with Relationships` 原则要求每个知识点都挂三元组，而这个写入序列目前没有安全的重试语义。skill 走 CLI、DSH 插件走 CLI、将来的 MCP 走 `Lab`，三条路都经过同一个 `insert_statement`。

**修法（已实施）**：

- 语义是 **if-not-exists，不是 upsert**：`statement` 表有 `content` 列，且 `StatementService::update` 已单独存在；同一三元组带不同 content 再次 create 应当 no-op 并报告已存在，而不是悄悄覆盖。
- SQLite：`INSERT ... ON CONFLICT(triple) DO NOTHING`；PG 侧 `insert_statement` 同改。existed 时跳过 FTS doc 与 postings 的重写。
- service 层用受影响行数区分 created / existed；CLI 输出 `already exists` 且 **exit 0**；MCP tool 返回 `created: false`。
- 两个后端各加一个重复插入的测试。
- 不动 schema、不动 export `Manifest`，老 binary 无影响；`insert_statement` 不在 onboarding 分支的改动范围内。
- 已存在时 CLI 输出 `Statement already exists: (<head>, <relation>, <tail>)`，与 `Created statement: ...` 区分。`hypatia-dream` skill 及其 `dsh-hypatia/skills/` 镜像已同步：替换流程中目标三元组已存在时，只有它的 metadata 与源一致（上一轮 create 之后中断）才继续删源；否则跳过替换，因为源的 metadata 并没有被带过去，删掉就丢了。
- 下游：DSH 插件 `runCreate` 的 `DUPLICATE` 分支对 statement 不再触发；插件所有调用点都丢弃返回值，行为不受影响。其 `cli-contract` 集成测试钉住了旧的 UNIQUE 报错，需随插件更新。
- PG 上每次重复创建会让 `content_versions` 序列跳一个号：`DEFAULT nextval` 在冲突检查之前求值。行级 `content_version` 不变；序列值只用作 CAS 令牌，跳号无害。
- `knowledge-create` 仍对重复名报错：knowledge 的名字只是标识，内容才是事实，重复创建可能意味着两份不同的内容撞名；statement 的三元组本身就是事实，重复即同一事实。

### 5.4 `content_version` 已存在但未暴露

`content_version` 在两个后端均已实现（`sqlite_store.rs` 11 处、`postgres_store.rs` 15 处，PG 侧还挂着 `nextval` 序列），但**完全没有暴露到 CLI 或 JSE**（`src/cli/`、`src/engine/`、`src/model/` 零引用）。

若将来要做知识版本 / current-version 过滤（记忆系统的「修正留痕，不悄悄覆盖」需要它），地基已在存储层，缺的是把它出到接口层 —— 比从零建便宜得多。

## 附：`dsh-hypatia-auto-memory` 插件的交叉结论

同一轮讨论也覆盖了该插件（DSH 侧的自动记忆流水线，在 `feat/dsh-hypatia-auto-memory` 分支上，未合并）。其结论属于插件仓，此处仅记录与本文的交叉点：

- 该插件实现的是记忆协议的**降级子集** —— 缺 log₁₆ 分层 cascade、语义 session 切分、`supersedes` 裁决、`belongTo` / `session-*` 节点。它对模型宣称已执行完整协议，会导致 Agent 停止自己做这些事，属于净损失，应先改文档。
- 处置结论是**在现有基础上改而非重写**，切割线为「保留边界层（CLI wrapper / content policy / config / 设置卡片 / fiber 组合，约 2000 行），重写记忆模型层（约 700–1000 行）」。
- 它是 §4.2「智能端点」反面教材的来源，也是 §5.1 五处漂移中多条的发现现场。
