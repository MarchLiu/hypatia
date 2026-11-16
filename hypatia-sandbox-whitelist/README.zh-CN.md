# dsh-hypatia-auto-memory

> 本文件是 [README.md](./README.md) 的中文版。若两者不一致，以英文版为准。

面向 [DSH](https://github.com/deepseek-ai/deepseek-harness) 的事件驱动
[Hypatia](https://github.com/MarchLiu/hypatia) 记忆：自动记录对话、后台巩固
（span 摘要、log₁₆(n) 归档级联、带裁决的工作单元抽取）——全部走**专用模型路由**，
主会话零模型工作。

写入是全自动的；**读取由 Agent 负责**，通过随包附带的 `hypatia-memory` 技能完成，
检索纪律则由同样随包附带的 `smart-memory` 预设承载。这种分工是有意为之——见下文
*Recall*。

**本插件取代 `dsh-hypatia`。** 后者把写入留给 Agent，实际上并没有发生：在编写本
插件的那个部署环境里，模型加载了协议，却在十五个会话中只有一个会话主动执行了
`hypatia` 命令——而且是被人直接要求才执行的。这里，协议改由 **DSH 原生事件**驱动，
插件通过自己的 `hypatia mcp` 进程（在 subprocess service 上）与 hypatia 通信
（CLI 加 argv 数组作为回退）——没有 shell、没有工具调用、没有沙箱提权、没有审批
弹窗。`dsh-hypatia` 里仍值得保留的东西都带过来了：对 Agent 自己的 bash `hypatia`
调用的自动批准，以及它的 `hypatia` / `hypatia-dream` 技能。二者装一个即可，不要
同时安装。

## 安装

前置条件：`hypatia` CLI 在 PATH 上（或把它的 basename 加进 `binaries`）。带 `mcp`
子命令的构建（hypatia #20）走 MCP；更旧的构建则走 CLI，首次调用会有一条警告。
另外两个后加的能力在二进制具备时启用、不具备时跳过：`--no-embed`（hypatia #26）
让日志层不进向量索引，`similar --exclude-tags`（#35）让日志层不进入工作单元候选
检索。首次调用会记录二进制提供了哪些（`hypatia over mcp: no-embed yes, similar
filters yes`）。

本插件还需要 **dsh ≥ 0.1.7**，而且这个下限由 dsh 自己强制执行，不靠本 README：
包里声明了
`peerDependencies["@deepseek-ai/dsh"]: ">=0.1.7-rc.1 || ^0.2.0-rc.1 || ^0.2.1-alpha.1"`
（同样的范围也写在 `dsh.engines.dsh` 里，供展示），dsh 在安装 bundle 时、以及每次启动时，
会把每个 `@deepseek-ai/dsh*` peer 与自身版本比对——运行时太旧会直接拒绝安装，而不是加载
一个面向它并不具备的 API 写成的插件；确实要接受风险时，用 `dsh plugin allow-version`
授予精确版本豁免。范围把每条已发布通道逐一写出来，而不是只写一个开放下界：node-semver
只在区间里存在同 `[major, minor, patch]` 的比较子时才接受预发布版，因此单写
`>=0.1.7-rc.1` 其实一个 `0.2.x` 候选版都匹配不到。

凡是运行中的 dsh 已经提供的 peer——`@deepseek-ai/dsh`、`@deepseek-ai/cordis`，以及
`dsh-llm` / `dsh-tools` / `dsh-storage-domain` 三个服务——都在 `peerDependenciesMeta`
里标为 `optional`，这样 npm 和 profile 安装都不会各自再拉一份：它们由 launcher 提供，
而"需要共享实例的包存在两份"正是插件用错那一份的来源。dsh 会忽略这个标记——它的检查只读
`peerDependencies`——所以上面的范围在安装时和启动时依旧生效。

```bash
# 从本地 checkout 安装（开发）
dsh plugin --profile web add link:/path/to/dsh-hypatia-auto-memory

# 发布后从 registry 安装
dsh plugin --profile web add dsh-hypatia-auto-memory
```

对其它 profile（desktop、dsh-tui）重复以上步骤。安装后重启 profile。

浏览器半——设置卡片与会话里的记忆标签页——预构建在 `lib/client.js` 中，随包发布。如果
你改了 `src/client/*`，在本目录运行 `npm install && npm run build` 重新生成它；当 DSH
checkout 里 `pnpm run dev:web` 处于运行状态时，client-plugin 的改动无需整页刷新即可
热更新。

**替换 `dsh-hypatia`：** 把它从 profile 移除并重启。

```bash
dsh plugin --profile web remove dsh-hypatia
```

两者注册了同名的技能，registry 保留先注册的那一个。本插件拒绝遮蔽另一个提供者
（改为告警），所以留着 `dsh-hypatia` 会把那份 Agent 驱动的协议原样交给 Agent——
而那正是本插件要修复的失败。如果它因其它原因必须留下，至少在其 bundle 行上设
`skills: false`。

磁盘上**同名技能**——`~/.agents/skills`、`~/.dsh/skills`、自定义目录，或项目的
`.dsh/skills` / `.agents/skills`——在每个 Agent 会话里同样会胜出。DSH 的技能
registry 按 scope 分层：最近一层的同名条目直接胜出，rank（project > plugin >
user）只在一层内打破平局。Agent 预设（`st`、`standard`、…）会在预设层挂载自己的
文件系统技能提供者，比本插件的全局注册更靠近 Agent。这包括 `hypatia skill install
--agent codex` 写到 `~/.agents/skills` 的规范 `hypatia-memory`：它是 Agent 驱动的
协议，需要 DSH 没有的宿主 hook。删掉这样一份拷贝，或把它移出 skills 目录；原地改
目录名不够，因为 DSH 是按 frontmatter 里的 `name` 来命名技能的。启动告警会指出
目录名。技能中心展示的是全局视图，所以它可能把本插件列为某技能的提供者，而会话
实际加载的是磁盘拷贝。

## Agent 预设：smart-memory（长程记忆）

本插件包内置一个 Agent 预设。`cordis.patch.yml` 插入一行 `preset-smart-memory`
声明 `@deepseek-ai/dsh-agent-preset`、`config.id: smart-memory`，因此是 profile 的
bundle 层把它送进 Agent 预设选择器，与自带的 `standard`、`ptc`、`minimal` 并列：
装上插件就有一个「记忆优先」的 Agent 可用，不必再手写 profile 补丁。

**它承载的是分工里「读」的那一半，仅此而已。** 这里的写入本来就是自动的——日志
记录、巩固、会话开场的规则种子。预设承载的是自动层做不到的纪律：**什么时候必须去
查**（制定计划前、每个里程碑、出现陌生内部名称或眼熟的 bug 时、任何破坏性操作之前
——commit、push、删除或覆盖、改配置、安装或升级、迁移、发布——用户提到历史时，以及
长程任务中的周期性回顾），检索质量要求（查询写成概念而非用户原话；想法用
`hypatia similar`，精确标识符用 `hypatia search`；始终排除运维层），以及唯一一条
后台流程无法强制执行的写入规则：用户给出的永久性规则、禁忌或纠正立即落库，且对已存在
条目用 `knowledge-update`，而不是删除后重建。

人设规定的是*什么时候*，`hypatia-memory` 技能承载的是*怎么做*——命令、参数，以及
shelf 与 scope 的规则。预设刻意不重述这套协议，也不挂载它：它只是提示词，不是第二个
提供者，所以在工作涉及记忆时加载该技能。

**工具集。** 即 `standard` 的花名册去掉 plan mode 组与 `present`（web profile 下
46 个工具）。它的子插件——`@deepseek-ai/dsh-persona`、各 `dsh-tool-*` 行、
`dsh-skill-filesystem`——来自 dsh 安装本身，**刻意不**作为本包的依赖；由 launcher
提供。

**这里不挂载插件自身的任何机制。** 预设不会启动第二个 collector、consolidator、
recall 监听或自动批准。那些属于上面那行 `hypatia-auto-memory`，且无论会话跑在哪个
预设下行为一致；因此用本预设的会话写记忆的方式与 `standard` 完全相同。

**怎么选。** 在新会话的预设选择器里选「长程记忆」，或在 profile 自己的补丁文件里把它
设为该界面的默认值：

```yaml
- id: agent-preset-registry
  name: '@deepseek-ai/dsh-agent-preset-registry'
  config:
    selectedDefault: smart-memory
```

**替换手写的那一份。** 如果你在本预设随包分发之前自己声明过它，请删掉那份拷贝。包内的
行刻意沿用 row id `preset-smart-memory`，而同名 row id 的两行会折叠为靠后的那一行
——profile 的补丁文件在所有 bundle 层**之后**应用，所以残留的拷贝会取胜并遮蔽包内
声明；插件管理器随后会把该行报成 `unaddressable`，因为同一个 id 出现了两个候选行。
另一种失败是两行**不同** row id 声明同一个 `config.id`：registry 会用
`Duplicate agent preset: smart-memory` 拒绝第二次注册。在原处留一段注释即可。

## 工作原理

```
session/event (emit)
  ├─ turn/end ──▶ durable queue (storageDomain tasks)
  │                 ├─ log-message  ──▶ hypatia msg-<sid>-<index>
  │                 ├─ consolidate  ──▶ ctx.llm.stream ──▶ sum-* (summary 1) + wu-*
  │                 └─ cascade      ──▶ $not-summaried ──▶ sum<N>-* (summary N)
  └─ session/title | compaction/summary ──▶ session-node ──▶ session-<sid> + belongTo
session/disposed ──▶ final flush + consolidation, thresholds waived (DSH 仍在运行时
                     关闭的会话)
collect fiber 拆除 ──▶ 对每个仍打开的会话做同样的收尾，并在队列停止前落盘；模型
                     调用留到下次启动完成（重启、配置变更重启，或崩溃：启动时的
                     consolidation backfill）
显式请求 ──▶ /hypatia-summarize（不过模型）| hypatia_summarize | 记忆标签页按钮
                     ──▶ 同一个 consolidate 任务，但跳过阈值
agent/created ──▶ rules/taboos inject()
```

- **传输**是一个私有的 `hypatia mcp` 进程：在 subprocess service 上通过 stdio 走
  JSON-RPC，一次一个请求（那个 server 本来就是这样处理的）。它不是 `dsh-mcp-client`
  条目，所以模型永远看不到这些工具，插件的调用也不经过任何工具策略、hook 或审批。
  走 CLI 时，每次调用都是一个打开 shelf 的进程；这里一批写入共享一个。内容以 JSON
  传输，而不是单个 argv 元素，所以不再因为要装进命令行而被截断到 96 KiB。tags 和
  scopes 按 CLI 的方式以逗号切分，所以无论哪种传输，条目存储结果一致。

  一个进程意味着同一时间只有一个调用，而 CLI 能并行到 `queue.concurrency` 那么多。
  一个调用要等正在飞行的那个——最坏情况是一个 `similar` 在重载嵌入模型，或一次写入
  在偿还逾期的嵌入债务——会话启动时的 rules/taboos 预载也在其中。

  进程在首次调用时启动，60 秒无调用后关闭。`hypatia mcp` 在语义调用后保留嵌入模型
  的分配（数百 MB），并在启动时读取一次 shelf registry。出于同样原因，一个
  `shelf '<name>' is not connected` 错误会重启它，这样期间已连接的 shelf 能在重试时
  被找到，而设置卡片的 shelf 列表始终来自 `hypatia list`。一个调用超过 30 秒期限会
  杀掉进程，正如 CLI 路径会杀掉命令一样；下一次调用会另起一个。

  `transport: cli` 恢复为每次调用一条 CLI 命令。一个提供不了可用 MCP 的二进制——
  没有 `mcp` 子命令、缺少本插件调用的某个工具、或握手返回错误——会让插件在 profile
  重载前一直用 CLI。其它任何启动失败只让那一次调用失败，下一次仍会再试 MCP：一个
  完全无法启动的二进制走 CLI 也会失败，而一次超时或崩溃的启动未必会重演。

- **Collector** 记录人类的 `user/message` 和 `assistant/message`，脱敏密钥、按消息
  发送时间把相对日期改写为绝对日期、对用户和助手文本都设上限，并排入幂等的日志
  任务。

  工具调用、thinking 和 reasoning 块是模型内部的，不会在内容里留下占位符。只输出
  这些内容（仅有工具调用、空的 reasoning 标记等）的助手轮次根本不产生 `msg-*` 条目
  ——没有任何值得记住的实质内容。插件来源的消息被跳过——避免反馈回路。默认完全不写
  工具调用台账；设 `hypatia-auto-memory.collector.toolLedger: true` 可为**有**
  文本/图片内容的助手消息记录一份精简台账（`1. \`read\` ×5 — ✅ 1.2s total`）。无论
  哪种模式，**工具输出本身永远不会存入**日志层。

- **被用户撤回的消息不会记入记忆。** `dsh-rewind-plugin`（`/rewind`）与两个
  `plugin:rollback` 插件都靠追加一条 surface 替换，把消息从模型视野里藏起来；而日志
  是 append-only 的，消息本身仍留在会话里。本插件读这条记录：被藏起来的消息不写，
  早先的 flush 已经写出的则从 shelf 上删掉。来源是一份白名单，只收语义确实是"用户丢
  弃了这个分支"的生产者——DSH 自己的压缩不在其中（它折叠的是用户保留的历史），编辑
  类插件也不在其中。两个边界：回退之前已经写出的摘要仍带着被撤回的原文（摘要从不重
  写），以及落在会话最后一次 flush 之后的回退完全看不到。见
  `docs/rewind-memory-consistency.md`。

- **日志层不嵌入。** `msg-*` 和 `session-*` 条目以及 `belongTo` / `summary` 边都以
  `embed: false` 写入（hypatia #26）：已存储、可全文检索、可被 JSE 找到，但不生成
  向量。协议设计了沿 `summary` 边从摘要精确下钻的召回方式，而不是按语义命中原始
  消息——嵌入后，一条原始消息会因为它攥着原话而排在被蒸馏出来的知识之前。在测量过
  的 shelf 上，这一层占条目的 85%、三元组的 77%，每条都要在对话热路径上跑一次前向。
  摘要和工作单元照旧嵌入。三元组没有 tags，所以 shelf 的 `embedding.skip_tags` 够不
  到这些边；插件每次写入都会说明这一点。在没这个 flag 的二进制上，传输会丢弃它，
  一切都照旧嵌入。

  span 在 `turn/end` 处切分，并且——对于超过 `flushWindowMs` 还在跑的 turn——在
  `step/end` 处切分，从不在任意时刻切。DSH 在**运行它所请求的工具之前**先追加
  `assistant/message`，等工具全部跑完才追加 `step/end`，所以一个 span 始终包含完整
  的 step，没有条目会缺少它自己的工具结果。DSH 压缩在 `compaction/prune` 之后重新
  追加的工具结果是旧结果的副本，不会被重复计数。

  条目按**稠密消息序号**命名，而不是 session-log seq：DSH 给每个流式 token delta 都
  分配一个 seq，导致 seq 编号产生稀疏键空间，破坏了 `$not-summaried` 的 FIFO 分批。
  序号从不可变日志前缀重新计数，所以重放能复现相同的名字。

- **Queue** 在调度之前就持久化：会话内有序、`queue.concurrency` 跨会话限流、带退避
  的重试、以及启动时的进度水位回填。一个运行中吸收新工作的任务会被重新调度而不是
  丢弃。Writer 都是 get-before-create，并且会**重申它的边**，所以条目和它的三元组
  之间若崩溃，重放即可修复。

- **Consolidator** 在 `turn/end` 阈值满足时运行，并在 `session/disposed` 无条件再
  跑一次——否则一个短于 `checkEveryTurns` 轮的任务会被记录却永远变不成知识。
  `turn/end` 携带原因：用户中断被视为任务边界并降低 token 门槛，而 blocked、failed
  或 truncated 的 turn 根本不是切记忆的地方。它的 transcript 与日志路径完全一致地
  脱敏并绝对化日期，超预算的 span 只巩固其**最旧**的条目，水位也只推进到那里，剩下
  的不会被打成已完成。

  抽取调用在**声明支持的路由上关闭推理**。抽取是朝固定 JSON 形状的机械变换，而在开启
  thinking 的路由上，推理与答案**共享同一个 `maxOutputTokens`**：实测中一条解析为
  `high` 的 `deepseek-flash` 路由在 2000 的预算里花掉约 800–1400 思考，然后带着写到
  一半的 JSON 撞上 `max-tokens`（9.3s，而同一条路由上同类成功调用要 25s）。路由无法
  被要求关闭（或 provider 忽略该控制）时，预算改为携带固定的思考余量；而**截断不再
  算永久失败**，而是重试——重试时只要 1 个工作单元而不是 `maxWorkUnitsPerRun`，因为
  正是工作单元让回复变长。Cascade 是例外：它保留 thinking（它是在压缩十六条摘要，不是
  在填一个形状），并用同一份余量来支付。

  路由上的 `reasoningEffort` 覆盖以上全部策略。把它设成该模型声明过的档位，这条路由上的
  每次调用都会带上它，预算余量也随之生效（任何非 `off` 的档位都与正文共用上限）。档位
  词表属于适配器自己（设置卡片只列出该模型声明的档位：`llm-deepseek` 上是
  `off`/`low`/`high`/`max`）；而**路由没有声明的值会被丢弃并记一条警告**，不会发出去——
  核心会在任何 provider I/O 之前拒绝未声明的档位，发出去等于把一处配置笔误变成反复重试
  后永久失败的任务。留空即保持上面的按用途策略。

- **Cascade** 把十六个未归档的 tier-N 摘要归档为一条 tier-(N+1) 条目，用 hypatia 自带
  的 `$not-summaried`（它对 `statement.tail` 做反连接并按 `created_at ASC` 排序，
  免费得到 FIFO 分批）。这就是协议的 log₁₆(n) 归档：每一层只压缩上一层已经蒸馏过的
  内容。

- **也可以显式要求立刻整合。** 三个面对应同一个 host 侧动作：斜杠命令
  **`/hypatia-summarize`**（在 composer 的 `/` 菜单里，由 UI 执行、**不过模型**）、工具
  **`hypatia_summarize`**（`hypatia-memory` 技能的执行路径，Agent 因此不会自己手写
  `sum-*` 条目），以及记忆标签页上的**「立即整理」按钮**——它就在报告「落后多少」的那
  一行上，正是读者需要它的时刻。一次请求按**触发那一刻**的水位切出
  `[lastConsolidatedSeq, min(seq, lastLoggedSeq))`，入队与自动触发**完全相同**的
  `consolidate` 任务：同一种 kind、同一个执行器、同一个幂等写手。`checkEveryTurns` 与
  `minNewTokens` 被刻意绕过——它们用来给进行中的对话定节奏，而显式请求本身就是「现在
  就切」的理由。实测中一个会话落后 151 个事件，而门槛只累计到 3000 中的 645，所以
  「等闸门开」不是一个应当被接受的答案。配置开关**不**绕过；而且请求只是一次切分，
  不是冻结：之后产生的事件照旧等下一次触发。

  「没有需要整理的内容」和「这个区间已经入队」会如实回报而不是再入队一次，所以问两遍
  不会产生两条摘要——何况写手按名字幂等，即使并发也不会。结清后按钮**禁用但保留在原
  位**：隐藏它会让用户每次落后时都要重新发现它，而那正是他们找它的时候。它走记忆标签
  页自己的路由族，其中第一条**写**路由（`POST /consolidate`）完整保留读路由的三条自
  认证——回环 socket、回环 `Host`、同源——并且只接受 `POST`，所以链接或 `<img>` 无法触
  发这次写入。三个面都不会在主会话里发起模型调用。

- **工作单元是裁决出来的，不是猜出来的。** 候选来自 `similar`（唯一会报告距离的
  检索），且排除操作层——在具备该能力的二进制上（hypatia #35）通过查询里的
  `--exclude-tags`；在没这个能力的二进制上，抓四倍的量再丢弃——再按距离上限过滤，
  然后由一次小型模型调用裁决 `duplicate | refines | extends | supersedes |
  contradicts | unrelated`。矛盾会保留**两条**条目并记录 `supersedes`——记忆系统绝不
  能悄悄忘记它曾经相信过什么。

- **Housekeeping** 每次启动对进度表跑一遍。它重置那些 session 在 shelf 里已无
  `msg-*` 条目的行——一个被清空或被替换的 shelf 否则会让该会话永久不被记录——并删除
  DSH 已不存在的会话的行和任务。重置的代价是一次重新记录和一次重新巩固，所以两遍
  都只在有正面证据时才行动：shelf 查询失败或空的会话列表都不做任何改动。

  第三遍巩固那些关闭时只能落盘、以及崩溃或 `kill -9` 完全没留下的内容。关闭中的会话
  所欠的东西本身活不下来：DSH 里 `session/disposed` 的观察者是 fire-and-forget，
  collect fiber 先于 session store 释放会话就被拆除，而队列随 fiber 一起消失——所以在
  那里 `enqueue` 是一次静默的空操作，一整轮的尾巴就这么丢了。两件事修好了它：collect
  fiber 的 disposer 现在会为每个仍打开的会话**落盘**两段范围（把收尾 effect 注册在队列
  自己的 disposer **之后**，因为 cordis 按注册逆序拆除 fiber 的 effects，于是它跑的时候
  队列还能写），而这一遍在下一次启动把活干完。关闭时只等最后的日志写入：一次 hypatia
  写入是毫秒级，而它可能触发的模型调用是几十秒，永远塞不进 DSH 5s 的关机预算。

  这一遍**刻意不看**节流实时触发器的 `consolidation.minNewTokens` 下限。对话还在进行时
  下限是对的——总还有下一轮——但启动时没有下一轮，在这里被跳过的尾部就是永久跳过：
  实测一个 live shelf 上 15 个会话带着 1–1950 的 pending token，没有一个被巩固过，9 个
  有消息的会话里 8 个 `lastConsolidatedSeq: 0`。成本改由
  `housekeeping.consolidationBackfillPerScope` 封顶——每次启动**每个 project scope**
  最多这么多个会话，尾巴最大的优先，于是积压分几次启动消化，而不是一次爆出无界的模型
  调用。consolidate 记录处于 `failed` 的范围无论多小都优先：`resumeKind` 不会重排
  failed 行，重新入队是它唯一的回头路。

  这些任务随后从 **storage** 读取事件，而不只是从活 store。`session/created`——唤醒
  被延迟任务的信号——只在 Agent 实际运行的地方触发，所以一个已结束的会话可能再也不
  会发它，而 subagent 会话（侧栏甚至不列出）几乎肯定不会：有一个就曾跨重启停留在
  `deferred` 状态，尽管它的 transcript 正开在屏幕上。`sessionPersistence.open(id, 'read')`
  返回同样的日志而不发布任何东西，执行器只用 `snapshotEvents`、
  `inheritedEventCount`、`seq` 和 `header`，所以一个不会再被重新打开的尾部仍能变成
  知识。没有 `sessionPersistence` 的组合行为照旧：任务等待。

- **Recall** 在会话启动时预载项目/全局的 rules 和 taboos。除此之外什么都不推送，
  除了一行在 shelf 不是 `default` 时指明 shelf 名字，好让 Agent 自己的 `hypatia`
  调用去插件写入的地方找。检索由 Agent 通过随包技能完成，这正是
  [`memory-nolinear.md`](https://github.com/MarchLiu/hypatia/blob/main/docs/memory-nolinear.md)
  为那些持有上下文、能调用工具的 Agent 所开出的方案；此前的每轮注入器还会因为从
  `agent/pre-step` 返回而不调用 `next()`，否决掉 DSH 自己的运行时上下文段。

- **Auto-approve** 只回答 Agent 自己的 bash `hypatia` 调用的审批请求，而且仅限这些：
  可执行词必须是一个可信的 basename（去掉 `KEY=value` 前缀之后），并且命令在
  **单引号之外**不能有管道、重定向、串联或命令替换——一个引号内满是 `|` 和 `>` 的 JSE
  参数仍然合格。双引号**不**能当作"已成字面量"：shell 会在双引号内展开 `$(…)` 和
  反引号，所以 `hypatia query "$(id)"` 和任何带组合的命令一样交给人类。其余都交给
  人类。插件自己的写入从不走这条路。

  Scope 是实测而非想当然：在 DSH 里，一个 bash 调用**只有**在沙箱拒绝、且模型带
  `sandbox_permissions` 重试时才会发起审批请求。检索（`query`、`search`、`list`、
  `knowledge-get`）在 `workspace-write` 沙箱内运行，无论有没有本插件都不会询问。这里
  回答的是 hypatia 的**写入**——`~/.hypatia` 在工作区之外，所以 `knowledge-create`
  之类会被拒绝、提权，然后每次用户说"记住这个"都会卡在弹窗上。

- **Skills** 随包附带：`hypatia-memory` 是本插件的变体（自动层写入、Agent 检索），
  外加 hypatia 的 `hypatia` CLI 参考（逐字节相同）和 `hypatia-dream`（打过补丁，见
  *已知限制*）的 vendored 拷贝——因为移除 `dsh-hypatia` 否则会把它们一起带走。另一个
  插件注册同名技能会被放过并报告，因为 DSH 保留先发生的运行时注册。磁盘上的拷贝会被
  登记上去但在 Agent 会话里仍胜出——预设会在更近的一层加载磁盘技能——所以会连同目录
  一起报告。

- **会话里的「记忆」标签页**是上述一切的读取面。它和 **对话**、**轨迹** 并排（一个
  `conversation.view` 注册项，id 为 `memory`，排序在轨迹之后），对当前会话显示三块
  内容。**状态**：记录水位、整合水位、待整合 token、`session-<id>` 节点是否存在，以及
  用尽重试次数的任务和拦住它的那条错误。**跨度摘要**：整合产出的
  `sum-<session>-<from>-<to>` 条目及其归档层级。**工作单元**：由这些摘要派生的 `wu-*`
  条目——它们通过 `derivedFrom` 边找到，因为名字是内容寻址的，不携带会话信息。正文用
  外壳自带的 `MarkdownText` 渲染，所以摘要读起来和 GUI 里其它地方的助手 Markdown
  一致。

  原始 `msg-*` 条目**故意不显示**：它是对话原文，读的人刚刚写过，而正确的读者是
  Agent。

  数据经由本插件注册在 DSH web 服务器上的、会话作用域的 HTTP 路由族
  `/api/dsh-hypatia-auto-memory/session` 到达浏览器：状态每次轮询都从内存中的状态表
  现算，shelf 内容只在可能变化时才读（打开时、切换会话时、手动刷新时，以及整合推进了
  该会话水位时），其间缓存 15 秒。标签页在屏幕上时每 5 秒轮询一次，响应是有界的：
  最新的 40 条摘要与 40 条工作单元，每条正文 4000 字符、整份响应 120000 字符。
  JSE 没有 `ORDER BY`，所以「最新」只能在读完所有候选之后判定；扫描触到上限时计数
  变成下界，标签页会把该次响应标为已截断，而不是让短列表无从解释。这个路由族里唯一的
  **写**路由 `POST /consolidate` 就是「立即整理」按钮调用的那个；它自己不写任何东西
  ——它入队的是与自动触发完全相同的整合任务（见上文「也可以显式要求立刻整合」）。
  路上否掉了两条通道。**session projection** 形态最
  合适，但驱动它的事件在本仓库之外写不出来：`Session.append` 没有任何途径设置信封的
  `ignorable` 标记，而持久化读取路径会拒绝一个带着未知事件类型、又没有该标记的会话
  日志——所以自定义事件会破坏会话重载。**设置命名空间**即便在还存在时也是根作用域的：
  宿主无法知道浏览器正在看哪个会话，只能把每个会话的数据都发出去。（dsh 0.1.7 已整体移除
  宿主侧的设置注册 API，下面的路由族现在是唯一通道。）

  这条路由不属于组合应用自己那批已鉴权的路由——插件注册的路由从来都不是——所以它像
  `dsh-hypatia-ui` 那样自鉴权：socket 必须是回环，`Host` 必须是回环名（`localhost`、
  `127.x`、`[::1]`），请求必须同源。本页面里的浏览器标签三条都满足；另一来源的页面
  即便同机也会在最后一条上失败，DNS 重绑定的页面会在 `Host` 检查上失败。
  `~/.dsh/storages/hypatia_auto_memory.json` 和 `hypatia_auto_memory_diag.json` 仍是
  持久记录；标签页是同一批事实的视图，不是它们的替代品。

## 首次激活的沙箱白名单

Hypatia CLI 把记忆存储写在 `~/.hypatia` 下。全新部署的沙箱策略默认不允许这个路径，
 hypatia 的每次写入都会停在授权弹窗上。插件在**首次激活**时会检查各 patch 层
（`~/.dsh/cordis.patch.yml` 与 `~/.dsh/profiles/*/cordis.patch.yml`）里是否已有把
`~/.hypatia` 加入白名单的 `sandbox-policy` 条目：

- **已含白名单** —— 什么都不做，永不弹窗。
- **未含** —— 下一个会话的第一次工具调用时，你会收到**一次真实的授权弹窗**（绝不自动代答），
  文案说明：批准将把 `~/.hypatia` 加入部署级沙箱白名单、对**所有会话**生效、以及如何撤销。
  批准后插件会把该路径拼进已携带 `sandbox-policy` 条目的那个 patch 层（若没有任何条目，
  则在 `~/.dsh/cordis.patch.yml` 追加完整条目）。改动在下次部署启动时生效；删除该行即撤销。
  拒绝只跳过本次激活，下次激活可能再次询问。

Windows 上直接跳过：本地沙箱的 ACL 后端不支持 `additionalWritableRoots`，写了也是死配置。
在插件配置里设置 **`sandboxWhitelistPrompt: false`** 可彻底关闭该询问（下次激活生效）。

## 配置

插件的 cordis `Config` 就是它的配置表单。dsh ≥ 0.1.7 把插件的 Config schema 投影到
侧边栏的**「插件」页**，卡片渲染在该 bundle 自己的页面上；如果某个部署的 profile
不渲染这个页面，同一张卡片会退到**「设置 → 内置插件」**里作为一个标签页。卡片落在
存在的那个位置上，并在两者之间迁移时绝不会同时出现（`src/client/plugin-card-seat.ts`）。
在 bundle 自己的页面上，卡片默认收起成一行标题——那一页已经写了插件名；退到「设置」
标签页时默认展开。收起不会丢掉暂存的编辑，但在点保存之前不写入任何东西，离开页面即丢弃。
下面每个字段保存即生效，无需重载 profile。配置以插件条目 `config:`
的形式存进 profile patch；0.1.7 之前的 `settings.yaml` 段落会在首次启动时被一次性导入。
所有字段可选，展示的是默认值。空白的巩固路由在保存时就被拒绝；重复的路由在运行时被丢弃
并记一条警告。

**`consolidation.models` 是优先级顺序，不是轮换。** 一个任务的第一次尝试走第一条路由，
之后每次重试降级到下一条，超出列表长度时复用最后一条。两个需要知道的后果：一个任务能触达
的备选链长度是 `min(queue.maxAttempts, models.length)`——默认 `maxAttempts: 3` 时第四条
及以后的路由永远不会被用到；而每降一级要等一个 `queue.retryDelayMs`（5 秒），因为降级发生
在下一次重试，而不是同一次尝试内部。这是对原先「不管偏好、把所有已选路由轮流用完」的替换；
改变偏好就重排列表，或直接用卡片上的 ↑/↓ 按钮。

**每条路由还可以设置思考程度。** `reasoningEffort` 取一个适配器私有的档位——设置卡片只
列出该模型声明的那些，`llm-deepseek` 上是 `off`/`low`/`high`/`max`。留空即保持上文
*Consolidator* 里那套按用途策略；设了它就会覆盖该路由上所有调用的策略，输出上限也随之
变大来支付这份思考。模型没有声明的档位会在调用时被丢弃并记一条警告，而不是发出去——核心
会在任何 provider I/O 之前拒绝未声明的档位。

```yaml
hypatia-auto-memory:
  enabled: true
  binaries: [hypatia]
  transport: mcp                 # 或 cli；没有 `mcp` 的二进制回退到 cli
  shelf: default                 # 所有条目写入/查找的地方；启动时读取（见下）
  autoApprove: true              # 批准 Agent 自己的纯 `hypatia …` bash 调用
  collector:
    enabled: true
    maxAssistantChars: 8000      # 每条消息在截断标记前的上限
    maxUserChars: 32000          # 用户输入或粘贴的内容，同上
    toolLedger: false            # 默认不写工具调用台账；true 才为有内容的助手消息记录
  consolidation:
    enabled: true
    # 优先级顺序：每次整合先用第一条路由，重试才降级到下一条（见上文说明）。
    models: []                   # 选择一条或多条 { provider, model } 路由
    # 每条路由还可以带 `reasoningEffort: <档位>`——卡片只列出该模型声明的档位
    # （deepseek 上是 off/low/high/max）。留空即保持按用途策略。
    maxInputTokens: 16000        # transcript 上限（chars/4 估算）
    maxOutputTokens: 2000
    timeoutMs: 120000
    checkEveryTurns: 5           # 两次接受任务之间的最小轮数
    minNewTokens: 3000           # 触发所需的最小未巩固 token 数
    maxWorkUnitsPerRun: 3
    adjudicate: true             # 判断新工作单元与邻近记忆的关系
    dedupMaxDistance: 0.45       # 值得裁决的候选的余弦距离上限
    dedupCandidates: 5
    cascade:
      enabled: true
      batchSize: 16              # 归档上一层之前，每层的条目数
  queue:
    concurrency: 1               # 并行会话数
    maxAttempts: 3               # 每个任务的重试次数；同时决定模型备选链能走多深
    retryDelayMs: 5000           # 也是换到下一条路由前的等待时间
    # 不是分批旋钮。span 在 turn/end 切分；对一个跑这么久的 turn，
    # 下一个 step/end 会把已完成的部分刷出。
    flushWindowMs: 120000
  recall:
    enabled: true
    preloadRulesTaboos: true
  housekeeping:
    reconcileOnStartup: true     # 重置 session 在 shelf 里已无 msg-* 的行
    pruneVanishedSessions: true  # 删除 DSH 已不存在的会话的行和任务
    backfillConsolidation: true  # 巩固会话内触发器从未到达的已记录尾部
    consolidationBackfillPerScope: 4  # 每次启动每个 scope 允许排入的会话数
```

bundle 行上的 cordis config 块只承载技能打包：

```yaml
- id: hypatia-auto-memory
  config:
    skills: true                 # 注册 hypatia-memory + hypatia + hypatia-dream
    skillsDir: /abs/path         # 覆盖打包的 skills/ 目录
```

`enabled`、`shelf` 和 `autoApprove` 在采集器启动时读取，所以保存其中任一项的改动都会
重启采集器（排空队列、重新打开状态）；其它开关原地生效。0.1.7 之前的 `settings.yaml`
也是靠这次重启生效的：dsh 在所有插件启动之后才导入它，而采集器本身会等到那一刻才启动，
所以基本不会先按默认值跑起来。

### 选择 shelf

`shelf` 指定所有条目写入、所有查找读取的 hypatia shelf——记录、巩固、级联、
rules/taboos 预载、启动清理。设置卡片把它做成 `hypatia list` 所报告内容的下拉框
——标签页打开时按需读取；成功的结果在宿主侧缓存 60 秒，失败的结果从不缓存——并标注
已注册但未连接的 shelf。

- **像 `enabled` 一样，保存即生效**：采集器会在新 shelf 上重启，日志会说明。
- **进度按 shelf 保存。** 水位和排队的任务按 shelf 存储，所以新选的 shelf 会让每个
  会话从零开始，而切回去会恰好从该 shelf 上次停下的地方继续——不会把已有的内容重新
  记录或重新巩固进某个 shelf。代价落在新 shelf 上：每个活会话从它的开头开始记录——
  其它会话下次活跃时也一样——并重新巩固，每个 span 一次模型调用。为另一个 shelf
  排队的工作原样等待，直到再次选中它，清理（修剪消失会话、失败任务）也只碰当前
  shelf 的行。`default` 保留该设置出现之前写的行。
- **Agent 会跟随。** 当 shelf 不是 `default` 时，会话种子会告诉 Agent 在自己的
  `hypatia` 命令上带 `--shelf <name>`；随包的 `hypatia-memory` 技能同样说明，
  `hypatia-dream` 在请求没指定时整理那个 shelf。
- **缺失的 shelf 在启动时记录，而非拒绝：** 写入它会像其它 hypatia 失败一样失败并
  重试，直到它被连接（`hypatia connect <dir> --name <name>`）。走 MCP 时，该失败会
  重启 server，所以 `connect` 之后的重试能到达那个 shelf。

列表通过与 Memory 标签页相同的路由族到达浏览器——`GET /api/dsh-hypatia-auto-memory/shelves`，
在启动时的对账与回填之前就挂载（`enabled: false` 时也挂载），所以 shelf 坏掉、需要换一个
时列表就在那里。（0.1.7 之前它走一个只读设置命名空间，该宿主侧 API 已被移除；
shelf 清单本来就不是配置，而设置域现在只投影插件的 Config 表单。）

## 运维清单

1. 一个 turn 结束后，条目在几秒内出现：
   `hypatia knowledge-get msg-<sessionId>-<n>`（`n` 从 0 开始数消息）。
   会话里的**记忆**标签页不用敲 CLI 就能回答同一个问题，并且能给出 CLI 看不到的东西：
   水位本身。
2. 水位在 state domain 里；失败的任务连同最后一次错误留在 `tasks` 表里供检查。这两者
   就是记忆标签页读的内容；换过 shelf 后要刷新它，因为它读的快照对应本次运行正在写入
   的那个 shelf。
3. 写入中途重启是安全的：已存储的消息被跳过（get-before-create）；未覆盖的范围从
   水位重新排队。
4. 检查路由告警：没选 `models` 的巩固只记录一次告警，其余时间静默。
5. 后台调用实际用了哪个模型，逐次记在
   `~/.dsh/storages/hypatia_auto_memory_diag.json`（`tables.model_calls`，按 `seq`
   倒序，上限 100 条）。设置卡片里的列表不是答案：`consolidation.models` 是优先级顺序，
   同一个任务的几行会沿它往下走——第一次尝试是队首那条，重试是降级后的下一条，这也正是
   确认「备选确实被用上了」的方式。`effort` 是这次调用**实际发出**的档位——路由拒绝了
   配置值时它就不是配置值；`outputTokens` / `reasoningTokens` 来自路由自己上报的用量，
   它们才说明输出上限是被正文吃掉的还是被推理吃掉的（路由不上报用量时两者都是 0）。
   `outcome` 为 `pending`（在飞行中，或进程中途死了）、
   `ok`（产出了可用结果）、`incomplete`（调用结束了但没产出——撞输出上限、被中止、回复
   解析不出来）或 `error`（调用本身抛错）；后两种都花掉了一次尝试且没有写入任何条目。
   `detail` 是 finish kind、固定标签或 provider 的错误消息，截断到 200 字符——绝不放消息
   正文，也绝不放模型输出。
6. `hypatia backfill --status` 报告 shelf 的嵌入债务。用当前的 hypatia，记录一个
   turn 不会增加债务；只有摘要和工作单元在下一次 flush 前处于待处理。`hypatia scope
   list --count` 显示在用的 scope——本项目 `msg-*` 条目所带的那个，就是会话种子读取
   的那个。
7. 卸载：`dsh plugin --profile web remove dsh-hypatia-auto-memory`——hypatia 条目
   本身留在 `~/.hypatia/`。

## 故障模式

| 症状 | 原因 / 处理 |
|---|---|
| 聊完没有条目 | `enabled: false`、缺少 `hypatia` 二进制、或 collect fiber 处于 PENDING（需要基础组合里的 `sessions`、`storageDomain`、`subprocess`）——查 profile 日志 |
| 重启后一整轮什么都没记录 | storage domain 因为某条已存记录不符合 schema 而拒绝打开。storage service 在读时校验、不在写时校验，所以一次坏写入只会在下次启动时暴露——而一条坏记录会拖垮整个 domain。在 profile 日志里找 `startup failed` / `does not match its schema`。缺 `error` 的任务行现在已默认化；其它坏行则停 DSH、从 `~/.dsh/storages/hypatia_auto_memory.json` 删除它再重启 |
| 「记忆」标签页始终读不到数据，或显示读取失败 | 页面不是通过**回环来源**访问的。该路由自鉴权——回环 socket、回环 `Host`、同源——所以从另一台设备读 DSH（LAN 地址，或绑定 `0.0.0.0`）会被按设计拒绝，隧道送来公网 `Host` 也一样。响应两半一起被拒，所以标签页报的是读取失败而不是部分数据。请用 `127.0.0.1` 或 `localhost` 打开 GUI |
| 条目只在 turn 结束后才出现 | 设计如此：span 在 `turn/end` 切分，或当 turn 跑得比 `flushWindowMs` 久时在第一个 `step/end` 切分，所以条目不会缺它自己的工具结果 |
| 有记录，没摘要 | `consolidation.models` 为空或无效——首次触发时告警一次 |
| 有摘要但没有 `sum2-*` | 该项目里未归档的 tier-1 摘要还不足 `cascade.batchSize` 条 |
| 工作单元没有关系 | shelf 没有嵌入模型（`similar` 失败），所有候选都超过了 `dedupMaxDistance`，或裁决调用本身没有产出裁决——`model_calls` 里表现为 `incomplete` / `max-tokens`，也就是开启了思考的路由把整个输出预算花在推理上、还没开始作答就撞了上限。现在裁决会先确认适配器声明了 `off`，再对该路由请求 `reasoningEffort: 'off'`；对无法接受它的路由则把上限提到 1024——而路由的 `reasoningEffort` 若指定了真实档位，拿到的就是这 1024 再加上思考余量。自 hypatia #19 起，本地模型在 `~/.hypatia/models/<org>/<name>`（或 Hugging Face 缓存）里找，不再挨着 shelf：一个以前能回答 `similar`、现在说 `is not installed` 的 shelf，需要 `hypatia model install <model>`，或 `hypatia model register <model> <dir>` 指向它已有的文件 |
| `similar` 仍返回 `msg-*` 行 | 它们是在本插件让日志层退出嵌入之前写的，或由一个没有 `--no-embed` 的 hypatia 写的。在 shelf 的 `shelf.toml` 里用 `embedding.skip_tags = ["message"]` 加一次 `hypatia backfill` 一次性收回知识向量（比该 key 旧的二进制会拒绝打开 shelf，所以先升级所有共享它的二进制）。不要把 `session` 加进那个列表：`skip_tags` 匹配任何带该 tag 的条目，而人们写的关于会话的知识也带它——在试过的 shelf 上就有一条这样的条目丢了向量。插件自己的 `session-*` 节点很少，现在已逐次写入退出。三元组没有 tags 也没有 update 命令，所以之前写的 `belongTo` / `summary` 边保留向量 |
| 任务处于 `deferred` 状态 | live store 和 persistence 都没能提供它的会话。不是错误：它不消耗尝试次数，一旦二者之一能提供就运行。通常 storage 立即回答——只有组合里没有 `sessionPersistence`，或它的读取失败（日志里找 `could not be read`）时才会持久化这个状态。被 deferred 的巩固任务会在下次启动时被 backfill 重新入队，所以它不必等会话被重新打开 |
| 任务处于 `failed` 状态 | 一次 hypatia 或模型错误在 `maxAttempts` 之后持久化了。失败的 `log-message` 记录在下次启动时被修剪，因为水位会重新推导它们的范围；其它类型保留供检查——删掉一条好让下一个触发器重新创建它 |
| 水位说已记录，但 shelf 里没条目 | shelf 在记录后被重置或切换。启动时处理：`housekeeping.reconcileOnStartup` 重置任何 session 已无 `msg-*` 的行，该会话下次活跃时从头重新记录——并重新巩固。shelf 查询失败则不动该行。一个所有消息都被故意删除的会话无法区分，会被重新记录；如果这要紧，关掉这个开关 |
| 警告 `hypatia mcp unavailable: …; using the hypatia CLI until the profile reloads` | 二进制早于 `hypatia mcp`（消息会引用它的 `unrecognized subcommand`）、缺本插件调用的某工具、或握手返回错误。一切继续走 CLI；升级 hypatia 并重载 profile 以使用 MCP |
| 每次调用都失败 `hypatia mcp exited …` 或 `timed out` | 二进制能启动但它的 MCP server 起不来——`binaries` 里一个改了 clap 退出码或措辞的包装脚本不会被识别为旧二进制。设 `transport: cli` |
| 改了 `shelf` 之后每次写入都失败 | shelf 没注册或没连接——启动时记录 `shelf "<name>" is not registered`。连接它（`hypatia connect <dir> --name <name>`）后重新保存设置（或重载 profile），或换一个 |
| Agent 在 `default` 里搜，而插件写在别处 | 一份磁盘上的 `hypatia-memory`（见下）替换了随包技能，或 recall 被禁用，所以没有东西告诉 Agent 用哪个 shelf |
| 手动乱改后出现重复 `msg-*` | 在 hypatia 里删掉该条目，并在 state domain 里把该会话的 `lastLoggedSeq` 调低——回填会重新创建它一次 |
| Agent 自己的 `hypatia` 写入仍要审批 | `autoApprove: false`、命令在单引号外有管道/重定向/串联（双引号内的 `$(…)`/反引号也算）、或首词不是 `binaries` 之一——按设计只回答纯调用。读取根本不会走到审批，所以那里没有要修的 |
| Agent 拿到了一份教它手动记录消息的记忆协议 | 另一个插件先注册了 `hypatia-memory`（profile 里还有 `dsh-hypatia`），或磁盘上有 `hypatia-memory`——通常是 `hypatia skill install --agent codex` 写的 `~/.agents/skills/hypatia-memory`。Agent 预设会在比插件更近的一层加载磁盘技能，所以它在会话里胜出，尽管技能中心可能列出本插件。删掉启动告警指出的那份拷贝 |
| 不知道实际用了哪个模型 | 列表是优先级顺序：一个任务的第一次尝试就是 `consolidation.models` 的队首，重试则是它降级到的那条。每次尝试都落在 `~/.dsh/storages/hypatia_auto_memory_diag.json`（`model_calls`），带 purpose、provider/model、**实际发出的 `effort`**、outcome、耗时以及路由自己上报的 token 拆分——把同一个任务的几行按序读下来，就能确认备选确实被用上了，而 `effort` 是确认配置的思考档位真的生效了的依据。usage ledger 答不了这个问题——它只折算 Agent 的 turn（`assistant/message`），看不见插件直连的 `llm.stream`。`pending` 行表示调用仍在进行中，或进程在调用中途死了：记录在调用前先落盘、调用结束后回填 |
| 配了 `reasoningEffort` 却像没生效 | 该路由没有声明这个档位，于是调用把它丢掉了，以免撞上核心的 `UNSUPPORTED_REASONING_EFFORT`。会有一条警告点名路由、值和该路由实际声明的档位，此后这些尝试的 `model_calls.effort` 是空的。卡片只提供模型声明的档位，所以这通常只发生在手改 profile、或把这条路由换成了别的模型之后 |
| 启动告警从不出现在终端 | `dsh web` 不挂日志导出器，所以插件任何级别的日志行都没地方去；控制台导出器的默认阈值也会丢掉警告（warn 是 2 级，高于 info 的 1 级）。临时挂一个 logger，如 `dsh-logbook` 或 `dsh-boot-doctor` 来读它们 |
| 项目 scope 看起来不对 | Scope = 会话 cwd 的 git 顶层目录的 basename（`git rev-parse --show-toplevel`）；git 找不到工作树或答不上（未安装、或某 repo 被判定不安全）时，则是 cwd 本身的 basename。一次慢于 3 秒的 `rev-parse` 会让那个会话留在 cwd 自己的名字上。两个同名 checkout 按设计共享 scope；linked worktree 按它自己的目录而非主 checkout 的目录定 scope；git 会解析符号链接，所以通过一个与目标不同名的链接打开的 checkout 会得到目标的名字。一个会被 hypatia 改写的名字会先归一化：位于 `/` 的会话定 scope 为 `/`，逗号变 `_`，去掉首尾空白。这些修复之前写的条目留在原处：子目录会话写在那个目录名下（`repo/src` 写在 `src` 下——git 根从未被读取）、位于 `/` 的会话无 scope、`a,b` 同时写在 `a` 和 `b` 下、`foo,` 写在 `foo` 和全局下。带空白的名字被去空白存储，这也正是查询现在所请求的。没有任何东西迁移它们；`knowledge-update --scopes` 可移动单条并保留其 `created_at`。`hypatia scope list --count`（hypatia #30）显示每种在用的拼写及其条目数 |

## 已知限制

刻意为之，依赖它们之前值得了解：

- **没有话题切换检测。**
  [`memory-nolinear.md`](https://github.com/MarchLiu/hypatia/blob/main/docs/memory-nolinear.md)
  把话题切换列为最强的会话切分信号，排在任务边界和时间间隔之上。检测它需要每个 turn
  一次模型调用，这与本插件的核心取舍（主会话零模型工作）相悖，所以未实现。任务边界
  （`turn/end` 原因）和会话关闭是被尊重的；一个覆盖三个无关话题的会话会被总结成一条。
- **摘要名是机械的。** 协议要求从内容里提取描述性名字。模型写的标题不能做 key——
  重试时可能措辞不同、悄悄分叉条目——所以 key 保持可推导（`sum-<session>-<from>-<to>`、
  `sum<N>-<digest>`），描述性标题放在正文的第一个标题里。
- **没有宿主摘要就没有 `session-<id>` 节点。** 协议禁止凭空捏造，DSH 也不发自己的
  session-summary 事件；节点在出现 `session/title` 或 `compaction/summary` 时由它们
  构建。
- **插件尚未使用 `knowledge-update`。** hypatia 在 #20 加入它（保留 `created_at`、
  丢弃旧向量并在下次 flush 时重新嵌入），但更旧的二进制没有，所以插件仍从不在原地
  编辑条目：第二个会话标题不会替换第一个，工作单元也从不重写。
- **`knowledge-update --data=""` 会清空正文，且没有回滚。** hypatia 给什么存什么，
  两种情况都打印 `Updated knowledge: <name>`——tags、scopes、`created_at` 全在，
  文字没了；纯空白 payload 是同一个事故（在 4.0.0 上实测）。插件自己从不调用这个命令
  （见上一条），但手写规则的 agent 会，而丢记忆的写法就是 `--data="$(cat …)"` 里的
  文件从未生成。因此这道防线放在指令层：`hypatia-memory` 技能与 `smart-memory`
  预设都要求先确认 payload 含非空白字符、再传、再用 `knowledge-get` 回读。
- **重复的 `knowledge-create` 靠错误文本识别。** hypatia 对知识条目没有 upsert，所以
  `hypatia-cli.js` 识别 `UNIQUE constraint failed:` / `duplicate key` 使重放幂等。
  `statement-create` 自 hypatia #20 起自身幂等（重复执行退出 0 并打印 `Statement
  already exists`）；更旧的二进制走同样的错误文本路径。匹配被限制在一个函数内，
  `test/integration` 接受两种 statement 行为。
- **已嵌入的仍是嵌入的。** 日志层在写入时退出向量索引，所以这个版本之前写的、或
  由一个没有 `--no-embed` 的 hypatia 写的条目和边保留它们的向量，会继续出现在
  `similar` 里。`skip_tags` + `backfill` 收回知识条目的（见*故障模式*）；边除了删除
  重建没有收回路径，插件不做这件事。
- **裁决和去重需要嵌入模型。** 在没有嵌入模型的 shelf 上，`similar` 直接失败，工作
  单元以无关系的方式存储——绝不会被丢弃。
- **启动回填读取活会话。** 一个已持久化但未加载的会话的缺口，在它下次加载时被补上。
- **级联在批次中途崩溃会移动分组边界。** 已连接的成员离开该层的未归档集合，所以
  重试会归档另一批。每样东西仍恰好归档一次；只是分组不再是第一次尝试想要的。
- **三个随包技能里有两个是 vendored 拷贝。** `hypatia` 和 `hypatia-dream` 从
  [hypatia 仓库](https://github.com/MarchLiu/hypatia) 的 `skills/` 拷贝而来，因为
  已发布的包够不到自身之外。没有任何东西刷新它们：上游原件变了就手工拷进来。
  `hypatia` 逐字节相同。`hypatia-dream` 带两处本地补丁，刷新时必须重新打：
  - **Shelf。** 请求里没给 shelf 时，它用会话种子指明的那个，只有种子没指明时才用
    `default`。上游总是回退到 `default`，那会去整理一个本插件可能没写的 shelf（见
    *选择 shelf*）。
  - **图读取排除操作关系。** 两个 statement 查询都排除了 `summary` 和 `belongTo`，
    技能本就视其为受保护、从不作为证据。本插件为每条已记录消息、span 摘要和归档层
    写它们，所以在测量的 shelf 上它们占 1258 条三元组里的 971 条。算进读取的 10000
    行上限里，它们会让技能在真正审阅的三元组远未达到上限前就停在截断检查上。

  它的 `evals/evals.json` 为这两处补丁新增 case 9 和 10。
- **顶层的 `enabled: false` 在采集器启动时读取**；保存改动会重启采集器，而各特性
  开关原地生效。

## 目录结构

```
dsh-hypatia-auto-memory/
├── package.json          # bundle + client manifest、依赖
├── cordis.patch.yml      # bundle 层：插件行 + smart-memory 预设
├── tsconfig.json         # 浏览器 TS/TSX 类型检查配置
├── tsconfig.build.json   # client bundle 的声明产物
├── tsdown.config.ts      # 浏览器 CJS 工厂构建
├── src/
│   ├── index.js          # fiber 组合（collect + 可选子模块）
│   ├── config.js         # cordis Config schema（volatile）、默认值、热更新
│   ├── shelf.js          # 每 shelf 表视图、`hypatia list` 解析
│   ├── state.js          # storageDomain 规格 + 进度助手
│   ├── collector.js      # session/event 过滤、台账、回填
│   ├── content-policy.js # 脱敏、日期、上限、slug（纯函数）
│   ├── queue.js          # 带重试/dispose 的持久化每会话队列
│   ├── hypatia-client.js # 传输切换：MCP、CLI 回退
│   ├── hypatia-mcp.js    # 私有 `hypatia mcp` 连接 + 结果映射
│   ├── hypatia-cli.js    # 仅 argv 的子进程包装
│   ├── writer.js         # 幂等 get-before-create 写入
│   ├── consolidator.js   # 阈值、prompt、llm.stream、校验
│   ├── cascade.js        # log₁₆(n) 分层摘要归档
│   ├── model-log.js      # 每次尝试实际用了哪个模型的有界记录
│   ├── memory-status.js  # 两张状态表按会话折叠（纯函数）
│   ├── memory-api.js     # 记忆标签页与 shelf 清单；含唯一的写路由
│   ├── on-demand.js      # /hypatia-summarize 与 hypatia_summarize 工具
│   ├── recall.js         # 会话启动时的 rules/taboos 预载
│   ├── auto-approve.js   # 批准 Agent 自己的纯 bash hypatia 调用
│   ├── skills.js         # 随包技能注册（从不遮蔽其它提供者）
│   ├── status.js         # 计数器 + 结构化日志
│   └── client/           # 设置卡片 + 记忆标签页
│       ├── index.tsx     # client 插件入口 + slot 注册
│       ├── SettingsCard.tsx
│       ├── MemoryView.tsx     # 记忆标签页主体
│       ├── memory-client.ts   # 它的请求与合并规则（纯函数）
│       ├── plugin-card-seat.ts # 卡片落座：官方「插件」页，否则设置标签页
│       ├── shelves.ts    # shelf 下拉选项（清单来自 /shelves 路由）
│       └── slot-contract.ts
├── lib/
│   └── client.js         # 构建好的浏览器工厂（提交此文件）
├── skills/
│   ├── hypatia-memory/   # 本插件变体（检索由 Agent 负责）
│   ├── hypatia/          # 从 hypatia 的 skills/ vendored
│   └── hypatia-dream/    # 从 hypatia 的 skills/ vendored，带本地补丁
├── scripts/
│   └── it-shelf.sh       # 手动 CLI 探测用的一次性 shelf
├── test/                 # node:test 单元测试（npm test）
│   └── integration/      # 针对真实 hypatia 的契约（npm run test:integration）
├── docs/
│   ├── jev-decision-model-research.md          # Jev/Laya 决策模型调研结论
│   └── jev-decision-model-research-scripts/    # 对应实验脚本
└── README.md
```
