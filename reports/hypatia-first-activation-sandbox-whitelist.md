# 实施报告：dsh-hypatia 首次激活一键授权写入沙箱白名单

日期：2026-11-10　执行者：task agent（cybernetics round）

## 结论

功能已**完整实现并全部通过 16 项单元测试 + 真实配置 dry-run 验证**，代码暂存于
`/Users/mars/jobs/hypatia/hypatia-sandbox-whitelist/`。唯一未完成步骤是**写入
node_modules 的沙箱提权被拒绝**（本子代理权限固定、审批无应答方，fail closed），
因此安装需要人（或父代理）执行一条命令：

```bash
bash /Users/mars/jobs/hypatia/hypatia-sandbox-whitelist/install.sh
```

脚本幂等、先备份（`.pre-sbx-backup/`）、装完自动跑语法检查与全部测试，并提示重启 DSH。

## 设计决策

### 落点：启用的包，而非旧包
- 实际启用：`dsh-hypatia-auto-memory`（cordis.patch.yml `id: hypatia-auto-memory`，2.4.4）。
- 旧包 `dsh-hypatia`（0.2.5）仅在 node_modules 中存在、未被 profile 引用，**不同步修改**
  （README 明确该插件已取代 dsh-hypatia；给废弃包加功能只会制造混淆）。
- 源码仓库：本机未找到（mdfind 只有 node_modules 与 pnpm store 链接；package.json 指向
  github.com/tkliuxing/dsh-hypatia-auto-memory，非本机仓库）→ 按委托书预案**直接改安装产物**，
  重装会被覆盖（install.sh 可重复执行恢复/重装）。

### patch 写入层与形态
- 目标层：**已携带 `sandbox-policy` 条目的那个 patch 文件**（本机是
  `~/.dsh/profiles/web/cordis.patch.yml` 571-577 行那条）；无条目时追加完整条目到
  `~/.dsh/cordis.patch.yml`（含 `mode/workspaceRoot` 的 `!!js` 表达式原样重述）。
- **文本级拼插而非 YAML 重排**：整文件 dump 会丢注释、重排全部条目、可能破坏其他 `!!js` 行；
  拼插只新增 1 行（已有列表）或 2 行（无 `additionalWritableRoots` 键时按缩进插入键+项），
  文件其余字节原样保留。缩进从文件自身推导（顶层/嵌套 bundle 均可）。
- 幂等检测：解析所有候选 patch（js-yaml + cordis 同款 `tag:yaml.org,2002:js` 标量方言），
  收集 `sandbox-policy.additionalWritableRoots` 字面量，`path.resolve` 规范化比较（win32 折大小写）。

### 真实征批的机制
- approval 服务要求**开放 turn**（`approval.request()` 在 turn 外抛错，见
  dsh-user-approval lib/index.js `request()`）。插件激活在 turn 外，因此把提问挂为
  `tools/pre-execute` 的**一次性**钩子：首个工具调用时在 turn 内调 `approval.request({agent, toolName: 'hypatia/sandbox-whitelist', reason})`。
- 不带 callId，auto-approve 的插应答器不会代答 → waterfall 经 dsh-api-remotes 转发到客户端
  UI，真人作答。批准 → 直接写 patch（宿主侧 fs 写，无第二次审批——弹窗即知情批准，
  patch 行即持久可见、可撤销的凭证）。拒绝 → 本次激活不再问，下次激活可再问。
- 双语 reason 文案含：路径、部署级影响（ALL sessions / 所有会话）、撤销方法。

### 触发条件（全部满足才弹）
(a) 非 win32（dsh-sandbox-local ACL 后端不支持 additionalWritableRoots，跳过并 status.warn）；
(b) 组合 patch 检测不含 `~/.hypatia`；(c) 插件配置 `sandboxWhitelistPrompt !== false`
（新 volatile 布尔字段，默认 true，下次激活生效）。

## 改动文件清单

| 文件 | 改动 |
|---|---|
| `src/sandbox-whitelist.js` | 新增（~330 行，纯核心 + 编排，见下） |
| `src/index.js` | +1 import；auto-approve 挂载点后新增 `hypatia-auto-memory/sandbox-whitelist` 子插件（inject `approval`，try/catch 包裹） |
| `src/config.js` | `DEFAULTS.sandboxWhitelistPrompt: true` + schema `z.boolean().default(...).volatile()` |
| `package.json` | dependencies 加 `js-yaml: ^4.1.0`（运行时由 profile node_modules 解析，install.sh 兜底软链） |
| `README.md` / `README.zh-CN.md` | 新增 "First-activation sandbox whitelist / 首次激活的沙箱白名单" 一节 |
| `test/sandbox-whitelist.test.mjs` | 新增 16 项单元测试 |

## 测试证据（本机可复现）

1. `node --test test/*.test.mjs`（staging 与装后同文件）：**16 pass / 0 fail**。覆盖：
   路径规范化比较、真实 `!!js` patch 解析、三种拼插（已有列表 / 缺键 / 无条目追加）、
   多候选文件优先级、模拟 home 的端到端（批准→写入→二次激活静默跳过）、恰好问一次、
   拒绝不落盘、双语文案含撤销说明、无可写文件时如实抛错。
2. 真实 dry-run（只读）：`~/.dsh` 候选 6 个 profile patch 全部可解析；`~/.hypatia`
   已在白名单 → `would arm prompt: false`（幂等验证，本机行为不变）。
3. 真实文件拼插预览：对 `profiles/web/cordis.patch.yml` 生成 +29 字节（恰一行）。
4. `node --check` 全部改动源文件通过。

## 已知限制与风险

- **重装覆盖**：改的是安装产物，插件升级/重装会回退；建议用户把 staging 目录纳入自己的仓库，
  或向 tkliuxing/dsh-hypatia-auto-memory 上游提 PR。
- **win32**：整体跳过（sandbox-policy 行在那里本是死配置）。
- **拒绝后重问**：无持久化计数，每次激活（缺白名单时）都可能再问一次；`sandboxWhitelistPrompt: false` 可关。
- **js-yaml 为新增依赖**：profile node_modules 已有 4.3.2，满足 `^4.1.0`；独立部署需 `npm i`。
- 安装后被 DSH 加载需**重启部署**；本机现有白名单 → 装后首次启动也应静默（dry-run 已证）。
