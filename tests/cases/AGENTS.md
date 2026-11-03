# 行为测试数据约定

本目录用数据集固定用户可观察的行为。新增一条问题时，先写期望和可复现的输入，再修改实现。目录名按被测行为命名，不按当前接入的 Agent 或编辑器命名。

## 目录与入口

```text
tests/cases/
  AGENTS.md
  run.py                         # 唯一的 Python 入口、执行器和报告生成器
  report_template.html           # 无外部依赖的报告模板
  report.html                    # 运行后生成，双击查看
  datasets/
    memory_recall/
      control.jsonl              # 证明测试装置能识别正常行为
      negative.jsonl             # 已知失败场景
```

执行 `python3 tests/cases/run.py` 会自动发现 `datasets/**/*.jsonl`，逐条运行并刷新 `report.html`。`--strict` 在存在未满足断言时返回 1；数据无效或正向对照失败时返回 2。用 `--output <path>` 可以另存报告。报告支持搜索、筛选、分页、详情和复制供 AI 排查的脱敏元数据。

## 新增数据

每行是一个独立 JSON 对象，至少包含：

| 字段 | 用途 |
| --- | --- |
| `id` | 全目录唯一、稳定的场景编号 |
| `intent` | 用户可理解的预期行为 |
| `suite` | 行为领域；当前为 `memory_recall` |
| `runner` | 执行路径；当前仅支持 `prompt_hook` |
| `role` | `control` 或 `negative` |
| `project` | 当前工作空间名 |
| `seed` | 运行前放入隔离知识库的记录数组；每项有 `name`、`data`，可有 `scopes`、`tags`、`embed` |
| `prompt` | 触发行为的用户输入 |
| `expect` | 至少一项可观察断言 |
| `diagnosis` | 可选的原因假设；失败时在报告中展示，不能代替源码验证 |
| `source` | 可选的仓库相对源码位置，例如 `codex-integration/hooks/hypatia_user_prompt_submit.sh#L65` |

`prompt_hook` 当前支持 `expect.recall_contains`、`expect.recall_excludes`、`expect.logged_embed` 和 `expect.logged_excludes`。前两项是记忆名称数组，`logged_embed` 是布尔值，`logged_excludes` 是不允许原样保存的字符串数组。不要写入未被执行器检查的断言；`run.py` 会拒绝未知断言和重复 `id`。

```json
{"id":"paraphrase_miss","intent":"换一种说法仍能召回旧决定","suite":"memory_recall","runner":"prompt_hook","role":"negative","project":"alpha","seed":[{"name":"decision-rust","data":"Use Rust for the backend service.","scopes":["alpha"],"tags":["memory"]}],"prompt":"Which language did we choose for the backend?","expect":{"recall_contains":["decision-rust"]}}
```

同一行为领域可以按问题类型拆成多个 `.jsonl` 文件；不要为每组数据另建 Python 脚本。需要新增执行路径或断言类型时，只修改根目录的 `run.py`，并补充本约定。每个执行路径至少保留一个可以通过的 `control`，防止测试装置失效时把所有用例误报为产品缺陷。

## 负样本设计

一条用例只隔离一个主要失败机制。优先覆盖同义改写、当前输入回显、跨项目串扰、过期事实、日志污染和凭据泄漏；后续扩展摄取、抽取、多跳推理、拒答与最终 Agent 答案。对照直接使用 Agent 的效果时，应加入同输入、同评价标准的无 Hypatia 基线，不把 hook 召回分数当作最终答案改善。

测试数据只能使用虚构项目和虚构凭据。报告会遮盖 `sk-...` 形式的测试令牌，但这不是通用秘密扫描器；写入数据集前仍须自行检查。不要读取或修改真实的 `~/.hypatia`、`~/.codex` 数据。当前 `prompt_hook` 使用真实 hook 和隔离的 SQLite FTS5 CLI 替身，结论只覆盖 hook 行为。
