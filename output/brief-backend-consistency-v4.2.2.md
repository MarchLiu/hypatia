# Brief: Hypatia 双后端一致性修复（#47 / #48 / #49 → v4.2.2）

> 编译自 claims.json @ `84e4183`（compilation.json status: ready）· 2026-11-01 · 面向工程师

## TL;DR

三个 open bug（#47、#48、#49）同属一族：**本地（SQLite）后端缺少入口校验/归一，而 pgvector 后端有，导致"参数错了"被伪装成"查不到"或行为分叉**。全部已在 v4.2.2 修复并在 `tests/backend_contract.rs` 锁死行为。#46（书架 Web 管理页）按维护者决策暂不做内置 UI，采用第三方 hypatia-archive 并已写入文档。

## 修复清单

### 1. FTS 检索入口校验（#49 → PR #51）

- **根因**：`Backend::search`（CLI 与 JSE 共用入口）无校验。`--limit=-1` 命中 SQLite `LIMIT -1` = 无限制的方言，静默返回全量书架；空/纯标点查询漏出 `fts5: syntax error`。pgvector 侧有内部校验，同一命令直接报错。
- **修复**：共享入口统一校验 `limit/offset >= 0`（文案与 pg/similar 对齐）；sanitize 后为空的查询返回 0 行、exit 0，与 pg 的 `websearch_to_tsquery` 行为一致。

### 2. catalog 参数校验（#47 → PR #50，社区贡献 maxshine）

- **根因**：`-c/--catalog` 任意取值在本地静默空结果，pg 报错，`--help` 未说明。
- **修复**：CLI 引入枚举 `Catalog`（`knowledge|statement|all`，默认 `all`）；`Lab::search` 共享层二次校验（覆盖 JSE）；移除 pg 内部旧守卫避免重复拦截。

### 3. 时间列比较语义（#48 → PR #52）

- **根因**：`created_at`/`tr_start`/`tr_end` 在 SQLite 是纯文本比较——`' ' < 'T'` 让 ISO `T` 分隔的查询把同一天全部判"更小"（少给行，最难察觉）；毫秒尾巴让 `$eq` 永不相等。pg 强转 `::timestamp`，接受 `'today'` 等日期词，非法值漏出 22007。
- **修复**（维护者拍板：时间归一 + 只认绝对时间戳）：新增 `engine::timecmp` 共享校验与归一（UTC 毫秒格式 `YYYY-MM-DD HH:MM:SS.SSS`）；SQLite 侧 `strftime('%Y-%m-%d %H:%M:%f', col)` 包列；pg 绑归一值再 `::timestamp`。裸日期与日期词两后端统一为 validation 错误。**原则：系统内部一律绝对时间戳，相对日期表达在写入/查询前转换。**

## 数据流（修复后）

```
CLI search ─┐
            ├─→ Lab::search（catalog 校验）──→ Backend::search（limit/offset 校验）
JSE $search ┘                                        │
                                    SQLite: strftime 归一列 vs 归一绑定值
                                    pg:     ::timestamp vs 归一绑定值
```

## 设计要点与经验

- **防分叉的正确位置是共享 contract 测试**（r006）：三个 bug 能长期存在，都因为 `tests/` 没有对应的双后端断言。语义修复必须同时补 contract 用例，本轮共新增 8 组双后端断言。
- **校验上移到共享入口**而不是各后端自查：一处校验，CLI/JSE/两后端同时生效。
- 少给行 > 报错更危险：时间过滤是最常用条件，"那段时间没数据"不会被怀疑是查询错了。

## 验证与发布

- `tests/backend_contract` 25/25（双后端）；`cargo test --lib` 除 2 个与仓库无关的 embedding 环境失败外全部通过；postgres feature 下 engine 78/78。
- **v4.2.2** 已发布（macOS ARM + Linux x64/ARM64，含 postgres-backend），release notes 附 issue/PR 引用。

## 遗留与建议下一步

1. **#46**：内置 UI 搁置，等更好的设计；现阶段以第三方 hypatia-archive 为准（已写入文档）。
2. **/calibrate**：v4.2.2 尚无真实使用反馈，建议观察一段时间后校准本轮"修复即预期"的判断。
3. contract 测试目前依赖本地双后端跑通，pg 路径在无 pgvector 环境下仅编译期覆盖——考虑在 CI 加 pgvector service。
