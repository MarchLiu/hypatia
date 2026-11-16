# dsh-hypatia 首次安装能否隐式获得 additionalWritableRoots 白名单 — 源码调查报告

调查时间：2026-11-10；环境：DSH profile `web`（/Users/mars/.dsh/profiles/web），dsh-sandbox-policy 0.2.2-rc.1，dsh-hypatia-auto-memory 2.4.4。

## 1. additionalWritableRoots 的来源与合并机制

**唯一来源是 sandbox-policy 插件的部署级 Config（patch 配置）。**

- `@deepseek-ai/dsh-sandbox-policy/lib/index.js:114-122`：`SandboxPolicyService.Config = { mode, workspaceRoot, additionalWritableRoots: array(string).default([]) }`。
- `lib/index.js:79-83`：`resolveAdditionalWritableRoots` 只做校验（绝对路径）与去重冻结；`lib/index.js:131-135` 构造时一次性读入。
- `lib/index.js:166-174`：`resolve()` 把**同一份冻结列表**原样返回给每个会话（`...{additionalWritableRoots}`），**没有任何按插件、按会话、按 workspace 的追加来源**，也不存在多来源累加合并。
- patch 层合并语义是**整键替换、不深合并**：
  - `@deepseek-ai/dsh-base/cordis.patch.yml` 头注释（第 1-13 行）明确写道："A patch replaces the targeted row's whole `config` rather than merging into it"；
  - 官方文档字符串 `@deepseek-ai/dsh-app-boot/lib/index.js:3794`："config is replaced wholesale, not deep-merged"；
  - 实现：`@deepseek-ai/cordis-plugin-include/lib/index.js:218-268`（`applyPatches`，262-265 行 `target[key] = value`）。
  - 实测验证：用 `composeEntries(loadOptionalPatches(dsh-base), loadOptionalPatches(profile cordis.patch.yml))` 组合后，`sandbox-policy` 行的 config **只剩** `additionalWritableRoots`（本机的 `mode`/`workspaceRoot` 实际已被用户 patch 行整体覆盖掉，`workspaceRoot` 由 `lib/index.js:134` 的 `?? process.cwd()` 兜底；会话内 workspace-write 来自会话级 `sandbox/mode` 事件 / permission-presets 折叠，见 `@deepseek-ai/dsh-permission-presets/lib/index.js:88-89`）。
- 层顺序（`dsh-app-boot/lib/index.js:1906-1914`）：bundle 层 → profile 层（`profiles/web/cordis.patch.yml`）→ home 层（`$DSH_HOME/cordis.patch.yml`）→ `--patch` overlay，后者胜出。
- 相关约束：headless `--add-dir` 只是校验，不会注入 —— 要求 sandbox-policy 行**已经**包含这些 roots（`@deepseek-ai/dsh-headless/lib/index.js:315-319`，`lib/startup.js:96`）。Windows ACL 后端**完全不支持** additionalWritableRoots（`@deepseek-ai/dsh-sandbox-local/lib/index.js:360`）。

## 2. 插件 manifest 的权限声明能力

**不存在。** `@deepseek-ai/dsh-package-manifest/lib/types/types.d.ts`（`DshManifest`/`DshBundleManifest`/`DshClientManifest`）的全部字段为：`manifestVersion / bundle(patch, platforms, listedByDefault) / profile(bundles) / client(platform, inject, immediately, external)` + `engines`。没有 sandbox、writable、permission、capabilities 之类的声明字段；安装/组合管线（`dsh-app-boot` 的 `readBundlePatches`/`readProfilePatches`、`dsh-plugin-manager`）也不读取任何此类声明。**没有"插件声明权限 → 安装时一次性批准 → 自动生效"机制**；approval seam 的授权全部是 per-call 一次性（`@deepseek-ai/dsh-user-approval/lib/index.js:31,149`，只有 `allowed-once`）。

## 3. hypatia 插件现状声明

- `dsh-hypatia-auto-memory/package.json` → `dsh: { engines, bundle.patch: ./cordis.patch.yml, client }`；其 `cordis.patch.yml` 只 insert 两行：`hypatia-auto-memory` 插件行 + `preset-smart-memory` 预设行。**未声明任何可写路径。**
- 该包**自带等价的 auto-approve 机制**（替代 dsh-hypatia）：`src/auto-approve.js:141-165` —— `tools/pre-execute` 记住"纯 hypatia CLI"调用的 callId，`approval/request` 前插应答 `allowed-once`，从而 bash 沙箱升级提示被自动放行；`src/config.js:38` 默认 `autoApprove: true`。
- 旧包 `dsh-hypatia/src/index.js:134-165` 同机制。"纯"定义：basename 匹配 `hypatia`、无 shell 组合（`&&`、`;`、管道、重定向、`$()`；见 `src/index.js:63-132`）。

## 4. 结论

### a. 现状
**不能。** 其他用户首次安装 dsh-hypatia-auto-memory 后：
- bash 里"纯 hypatia CLI"调用的沙箱升级授权已被插件自动应答（隐式、一次性、逐调用）——这部分开箱即用；
- 但 `~/.hypatia` 不会进入 additionalWritableRoots：fs 工具写入、非纯 bash 命令、以及任何依赖"预先放行目录"的路径仍会走人工授权，除非用户像本机一样手改 profile patch。

### b. 可行路径（按推荐排序）

1. **（推荐，短期）激活时一次性"一键授权"写入 patch 行** —— 在 dsh-hypatia-auto-memory 激活/首跑时检测 `ctx.sandboxPolicy`（或组合配置）是否已含 `~/.hypatia`，若无则发起一次 approval 请求，用户批准后把 `sandbox-policy` 行写进用户 patch 层（`$DSH_HOME/cordis.patch.yml` 或 `profiles/<name>/cordis.patch.yml`）。改动点：`dsh-hypatia-auto-memory/src/index.js`（apply 流程，约 600-640 行附近已有 autoApprove 挂载点）。注意两点硬约束：(a) patch 是整键替换，写入时必须**完整重述**当前行的 config（mode/workspaceRoot+已有 roots），不能只写增量（可先 `dsh --dump-config` 或经 cordis include API 读取现值）；(b) 写配置文件本身需要一次用户批准，正好构成"知情批准"。
2. **（长期，上游）manifest 声明 + 安装时批准**：扩展 `DshManifest`（`@deepseek-ai/dsh-package-manifest/lib/types/types.d.ts`）增加如 `dsh.sandbox.additionalWritableRoots`；由 `@deepseek-ai/dsh-plugin-manager`（安装管线）+ `dsh-client-ui-plugin-manager`（安装 UI）在安装时弹一次性批准；批准结果落为用户 patch 层的 sandbox-policy 行（或由 `@deepseek-ai/dsh-sandbox-policy/lib/index.js` 的 Config/resolve 增加按插件来源并累加）。这是唯一能"装完即生效"的正路，但需要改上游 3 个包（均为 MIT，repo 路径 packages/…）。
3. **（不推荐）sandbox-policy 按插件信任自动放行**：让 sandbox-policy 读取已装插件的声明并静默累加 roots。改动点同上（sandbox-policy + plugin-manager），但把"隐式放行"扩大到部署级全局白名单——所有会话、所有工具共享——安全面最大，违背 DSH "approval 全部 allowed-once、无持久授权" 的现有设计（dsh-user-approval）。

### c. 安全权衡
- 现有 auto-approve（隐式）：只放行"纯 hypatia CLI"的单次 bash 调用，命令解析保守（无组合即仍弹窗），授权一次性、不落盘 —— 影响面小，可接受。
- patch 行写入（知情批准）：一次性明示、写入内容用户可见、可随时撤销（删行）；但 roots 是**部署级全局**，授权后所有会话/所有工具都能写 `~/.hypatia`，不止 hypatia —— 需在批准文案中说明。
- manifest 自动放行（不推荐形态）：供应链风险最大——安装任意插件即静默扩大文件系统写边界；若走上游实现，必须绑定"安装时人工批准"，绝不能静默。
- 兼容性提醒：Windows ACL 后端不支持 additionalWritableRoots（dsh-sandbox-local/lib/index.js:360），任何自动写入方案需在 win32 上跳过并提示。

## 关键证据索引
- sandbox-policy Config/resolve：`@deepseek-ai/dsh-sandbox-policy/lib/index.js:79-83,114-135,166-174`
- patch 整键替换：`@deepseek-ai/cordis-plugin-include/lib/index.js:218-268`；`@deepseek-ai/dsh-base/cordis.patch.yml:1-13,243-248`；`@deepseek-ai/dsh-app-boot/lib/index.js:3794`
- manifest schema：`@deepseek-ai/dsh-package-manifest/lib/types/types.d.ts`（DshManifest 全字段）
- hypatia auto-approve：`dsh-hypatia/src/index.js:63-165`；`dsh-hypatia-auto-memory/src/auto-approve.js:141-165`；`src/config.js:38`
- approval 只有 allowed-once：`@deepseek-ai/dsh-user-approval/lib/index.js:31,149`
- Windows 限制：`@deepseek-ai/dsh-sandbox-local/lib/index.js:360`
- headless add-dir 仅校验：`@deepseek-ai/dsh-headless/lib/index.js:315-319`
- 实测：composeEntries 组合本机两层 patch 后 sandbox-policy config 仅剩 additionalWritableRoots
