#!/bin/bash
# Install the Hypatia Codex integration into the global Codex config.
#
# Usage: ./install.sh
#
# Installs:
#   ~/.codex/hooks.json           hook registry
#   ~/.codex/hooks/*.sh           hook scripts
#
# Requires a working `hypatia` CLI on PATH (or ~/.local/bin/hypatia).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$REPO_ROOT/codex-integration"
CODEX_HOME="${CODEX_HOME:-$HOME/.codex}"

mkdir -p "$CODEX_HOME/hooks"
cp "$SRC"/hooks/*.sh "$CODEX_HOME/hooks/"
chmod +x "$CODEX_HOME/hooks/"*.sh
cp "$SRC/hooks.json" "$CODEX_HOME/hooks.json"

echo "Installed hypatia hooks to $CODEX_HOME:"
ls -1 "$CODEX_HOME/hooks"
echo
echo "Next steps:"
echo "  1. Restart Codex (CLI or desktop app)."
echo "  2. Review and trust the hooks (CLI: /hooks · app: Settings → Hooks)."
echo "  3. Verify: 'hypatia search message --limit 5' after a conversation."
echo
echo "Optional: HYPATIA_BIN=/path/to/hypatia overrides binary discovery."
