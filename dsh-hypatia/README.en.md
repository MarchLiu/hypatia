# dsh-hypatia

Hypatia AI memory integration for [DSH](https://github.com/deepseek-ai/deepseek-harness) (DeepSeek Harness). (Chinese README: [README.md](./README.md).)

With this plugin installed, hypatia usage in DSH is fully implicit:

- **Bundled skills** — the `hypatia` and `hypatia-memory` skills are registered into every session automatically; no separate skill installation needed.
- **No repeated approval prompts** — pure `hypatia` CLI sandbox escalations are approved automatically. Hypatia writes under `~/.hypatia/` (outside any session workspace); without this plugin every such write would prompt.
- **First-activation one-click whitelist** — if the deployment has not yet whitelisted `~/.hypatia`, the plugin asks ONCE (a real approval prompt) and, on approval, splices the directory into the deployment-level `sandbox-policy → additionalWritableRoots`. Already whitelisted deployments skip silently (idempotent).
- **Permission boundaries do not spread** — only hypatia is let through; every other operation keeps the normal interactive approval and `workspace-write` sandbox.

## How the sandbox whitelist works

1. On activation the plugin reads `~/.dsh/cordis.patch.yml` and each profile's `profiles/*/cordis.patch.yml`. If any `sandbox-policy` entry already declares `~/.hypatia` (normalized path comparison), it does nothing.
2. Otherwise it arms a one-shot hook on the first tool call of the session and shows a REAL, bilingual approval prompt (deployment-level impact and how to revoke are stated in the prompt). Nothing auto-answers it.
3. On approval the root is added by textual splice into the existing `additionalWritableRoots` list (or a complete `sandbox-policy` block is appended when none exists) — comments and `!!js` expressions in hand-maintained patch files survive byte-for-byte.
4. Windows is skipped (the local sandbox ACL backend has no `additionalWritableRoots`); a rejection stops asking for the rest of the session.

**To disable the prompt:** set `sandboxWhitelistPrompt: false` in the profile patch.

**To revoke the whitelist:** delete the `additionalWritableRoots` entry from the corresponding `cordis.patch.yml`; the durable patch line is the visible, revertible artifact of the approval. Changes take effect on next DSH start.

## Configuration

Override in the profile's `cordis.patch.yml` (all fields optional):

```yaml
- id: hypatia
  config:
    autoApprove: true             # auto-approve master switch (default true)
    binaries: [hypatia]           # trusted CLI basenames
    skills: true                  # register bundled skills (default true)
    skillsDir: /abs/path          # override the packaged skills directory
    sandboxWhitelistPrompt: true  # ask once to whitelist ~/.hypatia (default true; false disables the ask)
```

The capabilities are independent sub-plugins: `dsh-hypatia/auto-approve` needs `approval` + `tools`, `dsh-hypatia/skills` needs `skills`, and `dsh-hypatia/sandbox-whitelist` needs `approval` — a deployment missing one service does not affect the others.

## Install / uninstall

```bash
dsh plugin --profile web add dsh-hypatia          # from the registry
dsh plugin --profile web add link:/path/to/dsh-hypatia  # from a local checkout
dsh plugin --profile web remove dsh-hypatia       # uninstall
```

Takes effect after restarting `dsh web`.
