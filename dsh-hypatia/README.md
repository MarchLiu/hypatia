# dsh-hypatia

Hypatia AI memory integration for [DSH](https://github.com/deepseek-ai/deepseek-harness) (DeepSeek Harness).

安装本插件后，DSH 中的 hypatia 使用完全隐式：

- **内置 skills** —— 自动向每个会话注册 `hypatia` 与 `hypatia-memory` 技能（知识/三元组 CRUD、JSE 查询、语义搜索、自动记忆抽取协议），无需单独安装 skill
- **免授权弹窗** —— 纯 `hypatia` CLI 命令的沙箱提权自动获批；hypatia 把数据写在 `~/.hypatia/`（会话工作区之外），没有本插件时每次写入都会弹一次授权
- **权限边界不扩散** —— 只有 hypatia 被放行，其它任何操作维持原有的交互式授权与 `workspace-write` 沙箱不变

## 安装

前置条件：系统已安装 `hypatia` CLI（在 PATH 上，或配置 `binaries` 匹配其实际路径）。

```bash
# 从 npm registry（发布后）
dsh plugin --profile web add dsh-hypatia

# 或从本地 checkout（开发）
dsh plugin --profile web add link:/path/to/hypatia/dsh-hypatia
```

`dsh plugin` 会读取包内的 `dsh.bundle.patch` 声明，把本插件作为 bundle 层加入 profile；重启 `dsh web` 后生效。对其它 profile（desktop、dsh-tui）重复同样命令即可。

## 工作原理

### 自动授权（不放宽全局权限）

DSH 的授权授予是一次性的（`allowed-once`），且 `ApprovalRequest` 设计上不携带工具参数（只有 toolName / callId / reason），因此无法靠"记住选择"实现免弹窗。本插件用 callId 关联两个扩展点：

```
tools/pre-execute  ──看到完整 bash 参数──►  纯 hypatia 命令？记录 callId
approval/request   ──同一 call 内触发───►  callId 命中？回答 allowed-once
                                            否则 next() 交给人工弹窗
```

**"纯 hypatia 命令"的判定（安全关键）**：

- 可执行文件 basename 必须是 `hypatia`（支持绝对路径与 `KEY=VALUE` 环境变量前缀）
- 命令中不允许出现**引号外**的 shell 组合符：`&&` `;` `|` `>` `<` `` ` `` `$()` 换行
  —— `hypatia knowledge-create x && rm -rf ~` 这类搭车命令仍会弹窗
- 引号内负载（JSON 数据、JSE 查询 `'["$knowledge"]'`）按引号感知扫描，不会误伤

所有自动批准仍写入 `approval/asked` / `approval/decided` 审计日志。

### 内置 skills

插件启动时把 `skills/*/SKILL.md` 注册为 runtime skill（`ctx.skills.register`，provider 为 `dsh-hypatia`）。Runtime 条目优先于用户级 `~/.dsh/skills/` 同名条目，因此插件版本即为权威版本；`user-invocable: false` 的 frontmatter（hypatia-memory）会被正确映射为仅模型可调用。

## 配置

在 profile 的 `cordis.patch.yml` 中覆盖（所有字段可选）：

```yaml
- id: hypatia
  config:
    autoApprove: true       # 免授权总开关（默认 true）
    binaries: [hypatia]     # 信任的 CLI basename 列表
    skills: true            # 是否注册内置 skills（默认 true）
    skillsDir: /abs/path    # 覆盖内置 skills 目录
```

两个子能力是独立子插件（`dsh-hypatia/auto-approve` 只需 `approval` + `tools` 服务，`dsh-hypatia/skills` 只需 `skills` 服务），部署缺少其一不影响另一个。

## 目录结构

```
dsh-hypatia/
├── package.json          # dsh.bundle.patch 声明
├── cordis.patch.yml      # bundle 层：插入 id=hypatia 的插件行
├── src/index.js          # host 插件（零依赖 ESM）
└── skills/               # 与仓库 ../skills/ 同步的 SKILL.md 副本
    ├── hypatia/
    └── hypatia-memory/
```

**维护注意**：`skills/` 是仓库根 `skills/` 的发布副本，修改根目录 skill 后需同步（`cp -R ../skills/* skills/`）。

## 卸载

```bash
dsh plugin --profile web remove dsh-hypatia
```

卸载后 hypatia 恢复为普通工具：skills 消失（或回落到 `~/.dsh/skills/` 中的用户级副本），写操作恢复逐次弹窗授权。
