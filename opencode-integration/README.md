# OpenCode Integration

OpenCode plugin that logs every conversation turn to the hypatia
knowledge graph — user messages in full, assistant messages shaped by
intent, tool calls as a condensed ledger.

## What it records

| Content | Policy |
|---|---|
| User message | Full text (redacted, dates absolutized) |
| Assistant reply | Shaped by user intent: report → summary; discussion → markdown context; operation → duration + method + outcome |
| Tool / bash / MCP calls | What was called, duration, success/failure; on failure only a cleaned one-line error description — stack traces and native-crash address dumps are reduced to a brief label |

All saved content passes through a policy pipeline:
secret redaction (API keys, tokens, passwords, private keys) →
relative-to-absolute date conversion (`昨天` → `2026-09-07`, …).

## Install

```bash
mkdir -p ~/.opencode/hypatia-memory-plugin
cp index.ts package.json ~/.opencode/hypatia-memory-plugin/
```

Then register it in `~/.config/opencode/opencode.json`:

```json
{
  "plugin": ["file:/Users/<you>/.opencode/hypatia-memory-plugin"]
}
```

Restart OpenCode. The plugin requires the `hypatia` CLI on `PATH`
(`~/.local/bin/hypatia` by default).

State (turn counters, last user prompt) lives in
`~/.opencode/hypatia-memory/state.json`; extraction signals are written
to `~/.opencode/hypatia-memory/extract-needed`.
