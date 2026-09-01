# Codex Integration for Hypatia

Wires Hypatia's automatic memory system into Codex (CLI and desktop app)
through the native Codex hooks system. Works globally: every project that
uses Codex gets memory managed by Hypatia.

## Architecture

- `~/.codex/hooks.json` registers lifecycle hooks (`SessionStart`,
  `UserPromptSubmit`, `Stop`) pointing at bundled hook scripts.
- Hook scripts are thin, deterministic shells: they log turns into the
  Hypatia shelf via the `hypatia` CLI, retrieve relevant memories, and emit
  trigger signals (`TRIGGER:log` / `TRIGGER:extract` / `TRIGGER:immediate`).
- AI-heavy work (summarization, work-unit extraction) is performed by the
  Codex agent following the `hypatia-memory` skill.

## Layout

```
codex-integration/
├── hooks/            # hook scripts (installed to ~/.codex/hooks/)
├── hooks.json        # hooks registry (installed to ~/.codex/hooks.json)
├── install.sh        # installs hooks + skills into ~/.codex
└── README.md
```

## Install

```bash
./install.sh
```

Then restart Codex and trust the hooks (CLI: `/hooks`; desktop app:
Settings → Hooks). Requires the `hypatia` binary on PATH.

## Status

P1–P4 — real hooks implemented and verified end-to-end:

- `SessionStart` injects project/global rules and taboos.
- `UserPromptSubmit` logs user messages, recalls memories, and emits trigger
  signals (`TRIGGER:log` / `immediate` / `extract` / `summary`).
- `Stop` logs assistant messages.
- Verified with a live `codex exec` session: both user and assistant messages
  landed in the default Hypatia shelf with `tag=message` and project scope.

Remaining for production use: trust the hooks in the desktop app
(Settings → Hooks) or CLI (`/hooks`).
