# Agent 接口分层：CLI、MCP 与 Skill

> 状态：**方案，尚未实施**（2026-09-10）
> 背景：Agent 通过什么与 hypatia 对话 —— 三层分工、MCP 的范围与边界、skill 的不可替代性，以及明确不做的事
> 姊妹篇：[降低上手成本方案](onboarding-plan.md)（上手漏斗与 embedding 生命周期）
> 范围：本文只交付分析与设计建议，没有修改任何实现

## 1. 结论

**MCP 与 skill 不是替代关系，是接口层与判断层的上下游。**

- MCP 能把接口层从散文变成 schema，让已知的漂移（§5.1）**变得不可能发生**，而不是被修好。
- 判断层**必须**留在 skill 里 —— 这是机制上的必然（§3.2），不是偏好：MCP prompts 需要用户显式调用，而记忆协议要求自动触发。
- MCP 值得做，但在上手漏斗里排在最后：一个连 binary 都装不上的用户，永远走不到配置 MCP 那一步。
- **不做**通用 server 模式（长连接 / 连接池 / 多线程）—— 收益错配且被后端矩阵放大（§4.1）。

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
- **tools = 现有子命令 1:1**（knowledge / statement CRUD、search、similar、query、archive、shelf），**零「智能记忆」端点**（理由见 §4.2）。
- **resources** = shelf 与条目的 URI 映射（如 `hypatia://default/knowledge/<name>`）。
- 薄薄盖在 `src/lab.rs` 的 facade 上，不复制业务逻辑。`src/cli/repl.rs` 已证明 `Lab` 可跨命令长期持有。
- embedder **惰性 + 按 shelf 的空闲卸载**（仅本地 provider）—— 生命周期是 shelf 作用域而非 server 作用域，因为 provider 配置在每个 shelf 各自的 `shelf.toml`，一个 server 会挂多个 shelf。
- 写工具**不阻塞在 embedding 上**，并把 pending 变成一等返回值：

  ```json
  { "name": "wu-2026-09-10-xxx", "created": true,
    "embedding": "pending", "reason": "remote provider: 429 rate limited" }
  ```

  这是 MCP 相对 CLI 的**真增量**（修 §5.2 的静默失败），且只有在「embedding 默认延迟」落地后才自然 —— 见姊妹篇 §3.3。
- 新增 `backfill` tool 与一个 pending 计数的 resource，让 Agent 或 host 能看见欠账并触发对账。

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

## 4. 明确不做

### 4.1 通用 server 模式（长连接 / 连接池 / 多线程）

- **收益错配**：server 的实际价值只有「嵌入器常驻」一件事。实测进程启动与开库只要 10 ms，连接池省不出东西；操作天然串行，多线程用不上。而「嵌入器常驻」用延迟 embedding 能拿到写侧的大部分（姊妹篇 §3），用 MCP stdio 能顺带拿到读侧。
- **成本被后端矩阵放大**：PG 后端已是原生实现（`src/storage/postgres_store.rs` 零处 `json_index`，走 GIN / tsvector / 递归 CTE / pgvector），与 SQLite 侧差异显著且预期继续拉大；`ShelfBackend` trait 有 38 个方法。再加一个「进程内 vs 常驻」的轴，就是 2×2 且两轴都在动。
- **唯一真正的设计约束是「CLI 是唯一接口」**，而这条约束在 PG 后端上也被遵守（可选后端藏在同一接口后面）。MCP 遵守它的方式是作为 `hypatia mcp` 子命令，而非独立服务。

> 注：早期讨论曾引用 `memory-webway.md`「hypatia 强调的是可以部署在每个电脑本地，而那个内部系统是一个基于 PostgreSQL 的服务」来论证「server 的事不归 hypatia」。作者已明确表示两个系统分叉后不会再长成一样、做 PG 支持没有顾虑，因此**该论据作废**。上面三条不依赖它。

**何时该重估**：如果负载从「本机若干 Agent 串行小批量读写」变成「并发的、需要共享状态的、跨进程协调的」（例如需要读-判-写的原子性来做知识版本仲裁），那时常驻就不是加速器而是必需品。但届时第一选择可能仍是 PG 的事务，而非 hypatia 的守护进程。

### 4.2 「智能记忆」MCP 端点

有了 MCP，暴露 `remember(text)` / `recall(query)` 这类端点会很有诱惑力。**这正是 `dsh-hypatia-auto-memory` 犯的错** —— 把判断从有完整上下文的主 Agent，移到只看得见转录文本的旁路。

`memory-nolinear.md` 的整套设计假设判断在 Agent 一侧：话题何时收敛、这条是 extends 还是 supersedes、哪段是纠错链。**MCP tools 要薄，贴着图原语。**

## 5. 待办与顺带发现的缺陷

```
G. hypatia skill install --agent claude|codex|opencode|dsh    零依赖
H. hypatia mcp 子命令                                          建议最后
```

G 的进阶形态：**判断层单一源，宿主适配层由 CLI 生成**。现在主仓 `hypatia-memory` 与 DSH 变体是两份手写、会漂移的文档（两者对协议的实现差了整整一个分层 cascade）。若适配是生成的，这类漂移也会变成不可能。

### 5.1 skill 与 CLI 之间没有同步机制，已产生五处漂移

1. `hypatia-memory` skill 提到 `knowledge-update`，CLI **没有这个子命令**。
2. 通用 skill 把 `--scopes ""` 定义为全局，CLI 实际解析为「无 scope」；只有尾逗号才会写入空字符串全局标记。
3. `search` 用 `-c/--catalog`，`similar` 用 `-t/--target`（易混，下游为此写过 NOTE）。
4. 内容以 `-` 开头（markdown 列表）时必须用 `--data=<v>`，`-d <v>` 会被 clap 当作新 flag。
5. 「条目不存在」走 **exit 0 + stdout 文本**，需正则 `^Knowledge '.*' not found\.$` 才能与真实 JSON 区分 —— CLI 换一次措辞，下游即崩。

其中 3–5 在 MCP 的 schema 化之后不可能再发生；1–2 需要人工对齐或让 skill 从 clap 定义生成（见 G 的进阶形态）。

### 5.2 写入时 embedding 失败是静默的

`src/storage/shelf_manager.rs:118-134` 的 `embed_saved` 尽力而为，失败往 stderr 打一行 warning 并 **exit 0**。调用方（如 `dsh-hypatia-auto-memory` 的 `runOk`）只看退出码，无法发现向量没生成。§3.1 的结构化 `embedding` 状态字段是 MCP 侧的修法；CLI 侧的处置见姊妹篇 §5.1。

### 5.3 `statement-create` 对重复三元组报错而非幂等

`src/storage/sqlite_store.rs:560` 的 `INSERT INTO statement` 没有 `ON CONFLICT`，而主键是 `(head, relation, tail)`，因此重复三元组触发 UNIQUE 约束错误、非零退出。这使得「崩溃后重放补边」无法实现 —— 任何多步图写入（先建 knowledge 再建若干 statement）在中途失败后都无法安全重试。

这对任何 Agent 集成都是硬伤：`hypatia` skill 的 `Always Enrich with Relationships` 原则要求每个知识点都挂三元组，而这个写入序列目前没有安全的重试语义。建议加 upsert 或 `--if-not-exists`。

### 5.4 `content_version` 已存在但未暴露

`content_version` 在两个后端均已实现（`sqlite_store.rs` 11 处、`postgres_store.rs` 15 处，PG 侧还挂着 `nextval` 序列），但**完全没有暴露到 CLI 或 JSE**（`src/cli/`、`src/engine/`、`src/model/` 零引用）。

若将来要做知识版本 / current-version 过滤（记忆系统的「修正留痕，不悄悄覆盖」需要它），地基已在存储层，缺的是把它出到接口层 —— 比从零建便宜得多。

## 附：`dsh-hypatia-auto-memory` 插件的交叉结论

同一轮讨论也覆盖了该插件（DSH 侧的自动记忆流水线，尚未提交）。其结论属于插件仓，此处仅记录与本文的交叉点：

- 该插件实现的是记忆协议的**降级子集** —— 缺 log₁₆ 分层 cascade、语义 session 切分、`supersedes` 裁决、`belongTo` / `session-*` 节点。它对模型宣称已执行完整协议，会导致 Agent 停止自己做这些事，属于净损失，应先改文档。
- 处置结论是**在现有基础上改而非重写**，切割线为「保留边界层（CLI wrapper / content policy / config / 设置卡片 / fiber 组合，约 2000 行），重写记忆模型层（约 700–1000 行）」。
- 它是 §4.2「智能端点」反面教材的来源，也是 §5.1 五处漂移中多条的发现现场。
