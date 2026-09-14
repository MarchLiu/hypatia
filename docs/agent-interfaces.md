# Agent 接口：CLI、MCP 与 Skill

> 状态：已实施（2026-09）。本文记录 agent 与 hypatia 交互的分层、各接口的设计决定，以及已知限制。

## 1. 三层分工

| 层 | 载体 | 内容 |
|---|---|---|
| 契约 | JSE 规范 | 算子语义 |
| 接口 | CLI、MCP tools | 有哪些操作，参数与返回结构 |
| 判断 | skill | 何时调用、怎么组合、如何裁决 |

- MCP 把接口层从散文变成 schema。参数类型、返回结构与错误不再靠 skill 的文字描述，也就不会再与 CLI 漂移。
- 判断层必须留在 skill 里。MCP 的 prompts 要由用户显式调用，而记忆协议要求自动触发：每隔几轮扫描、会话开始时加载规则、话题收敛时整理。
- 两者共存。MCP 的配置方式在各宿主之间统一，但只覆盖支持 MCP 的宿主；skill 覆盖任何能读文件、跑命令的 agent。

## 2. CLI 的改动

**statement-create 幂等。** 重复的三元组不写入、不报错，以 0 退出并输出 `Statement already exists: (...)`。多步图写入（先建 knowledge，再建若干 statement）中途失败后可以直接重放。语义是 if-not-exists 而不是 upsert，已存储的内容不会被覆盖。knowledge-create 仍拒绝重名，因为同名可能是两份不同的内容。

**knowledge-update。** 只改给出的字段：未给出的保持原值，给出的替换，`""` 清空。`created_at` 与 format 保留。内容没有变化时不写库，版本号与向量都不动。先读后写之间不做版本比对：同一条目恰好在这两条 SQL 之间被并发修改时，那次修改会丢失。

**skill install / status。** 三份 skill 编译进 binary，逐字节写入宿主的用户级目录：

| `--agent` | 目录 |
|---|---|
| `claude` | `~/.claude/skills` |
| `codex` | `~/.agents/skills`（`~/.codex/skills` 已废弃） |
| `opencode` | `~/.config/opencode/skills` 或 `$OPENCODE_CONFIG_DIR/skills` |

- 安装记录按内容哈希存在 `~/.hypatia/skill-installs.json`。hypatia 装过、未被改动的旧版本直接升级；被改过或不是 hypatia 装的副本，不加 `--force` 不覆盖。
- 宿主还会读取的其他目录里若有过时的同名副本，逐个告警。
- 只装 skill 文件，不装 hooks。hypatia-memory 需要宿主的 hooks 才会触发。
- DSH 由 dsh-hypatia 插件自行注册内置 skill，不需要安装。

**archive 名。** 只接受 archives 目录内、由普通路径段组成的相对路径，与导出校验 archive 引用的规则一致，并按真实路径复查以挡住符号链接。此前名字直接拼接，`/Users/me/.zshrc` 或 `../../x` 会写到 shelf 之外。

## 3. MCP 服务：`hypatia mcp`

**运行方式。** 作为子命令而不是独立程序：单一安装、单一版本。同步循环逐行处理 stdio 上的 JSON-RPC，不引入 async runtime；stdout 只输出协议消息。

**协议。** 支持带 initialize 握手的 2025-11-25、2025-06-18、2025-03-26、2024-11-05。请求的版本受支持就原样返回，否则返回 2025-11-25。2026-07-28 版取消了握手，当前宿主仍在用握手，暂不实现。不支持批量请求。参数与业务错误作为 tool 结果的 `isError` 返回，让模型能自行纠正；未知方法、未知工具、resource 不存在分别返回 -32601、-32602、-32002。

**工具只覆盖数据面。** query、search、similar、session_current、knowledge 与 statement 的增删改查、archive 的存取列、list_shelves、shelf_status、backfill。

- 不做 `remember(text)` 这类「智能记忆」端点：判断需要完整的对话上下文，只能在 agent 一侧做。
- connect、disconnect、init、model install、export、import 与 re-embed 留在 CLI。长驻进程只在启动时读注册表，写回时会冲掉其他会话期间的注册；下载与整库操作也会长时间占住同步循环。

**欠账与状态。** 每个作用于 shelf 的工具先还逾期的 embedding 欠账，与每条 CLI 命令一致。写工具返回 shelf 级的欠账快照：待补数、跳过数、欠账开始时间、锁定与暂停的原因。延迟 embedding 下逐条状态几乎总是「待补」，没有信息量。shelf_status 工具与 `hypatia://{shelf}/status` resource 返回同一份文档，因为模型能否自主读取 resource 因宿主而异。

**backfill 限批。** 每次最多补 64 条，上限 512，避免超出宿主的工具调用超时（Codex 默认 60 s）。向量被锁定时报错并说明原因。按从新到旧取条目：若最新一批总是失败，会挡住更早的欠账，这时应改用终端里完整的 `hypatia backfill`。

**resources。** 列出 `hypatia://{shelf}/status`，另有 `hypatia://{shelf}/knowledge/{name}` 与 `hypatia://{shelf}/statement/{head}/{relation}/{tail}` 两个模板，路径段百分号编码。不列出条目，否则大库会把列表撑爆。

**配置只读一次。** 注册表、shelf.toml 与模型可用性在启动时确定，`model install`、`connect`、`init` 之后需要重启服务。

## 4. 多会话

宿主为每个会话启动一个 `hypatia mcp` 进程，多会话记忆就是多个进程读写同一个 shelf。

| 方面 | 现状 | 处置 |
|---|---|---|
| SQLite | WAL，`busy_timeout` 5 s，多进程读写安全；持写锁超过 5 s 会让其他会话的写入失败 | backfill 限批 |
| 向量缓存 | 按内容时钟编版本，多进程下正确；但任何写入都会让其他进程下次检索时重新加载或重建缓存 | 不把常驻当作性能收益 |
| 模型内存 | 每个进程各自加载，不共享；实测（bge-m3，macOS）调用中峰值 1.5 GB | 每次请求后释放：空闲时 50 到 540 MB，RSS 稳定在约 870 MB，无泄漏；每次语义调用重新加载约 1.8 s |
| 推理线程 | ONNX Runtime 默认按物理核数开线程池，多进程同时批量推理会超额占核（推断，未实测） | 未处理 |
| 注册表 | 启动时读一次，整份非原子写回 | 管理操作不做成 tool |

需要跨进程共享一份模型时，把远程 provider 指向本机的 Ollama 即可。hypatia 不做常驻的 embedding 守护进程：它的实际收益只有模型常驻，而 SQLite 与 PostgreSQL 两个后端的差异会再乘上「进程内还是常驻」这一轴。

协议层有一处与传输无关的竞态：hypatia-memory 的摘要级联按项目查询未摘要的消息，同一项目里的两个会话会重复摘要同一批消息。修法是让级联按会话分区，属于 skill 的后续修改。

## 5. 后续

- hypatia-memory 的摘要级联按会话分区。
- backfill 限批改为游标分页。
- 限制 ONNX 推理线程数；同一进程内多个 shelf 共享同一个模型。
- skill 的接口部分仍靠人工与 CLI 对齐，可以改为从 CLI 定义生成。
- `content_version` 已在两个后端实现，尚未暴露到 CLI 或 JSE，可以作为知识版本的基础。
