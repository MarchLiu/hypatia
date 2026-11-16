#!/usr/bin/env bash
# Install the first-activation sandbox whitelist feature into the live
# dsh-hypatia-auto-memory package. Backs up every overwritten file to
# .pre-sbx-backup/ first. Run once, then restart DSH.
#
#   bash install.sh
set -euo pipefail

P="${1:-/Users/mars/.dsh/profiles/web/node_modules/dsh-hypatia-auto-memory}"
S="$(cd "$(dirname "$0")" && pwd)"

[ -f "$P/src/index.js" ] || { echo "plugin not found at $P" >&2; exit 1; }

mkdir -p "$P/.pre-sbx-backup/test"
for f in src/index.js src/config.js README.md README.zh-CN.md package.json; do
  cp "$P/$f" "$P/.pre-sbx-backup/$f"
done

cp "$S/src/index.js" "$S/src/config.js" "$S/src/sandbox-whitelist.js" "$P/src/"
mkdir -p "$P/test"
cp "$S/test/sandbox-whitelist.test.mjs" "$P/test/"
cp "$S/README.md" "$S/README.zh-CN.md" "$S/package.json" "$P/"

# js-yaml must resolve from the plugin; the profile node_modules already has it.
if [ ! -e "$P/node_modules/js-yaml" ]; then
  ln -s /Users/mars/.dsh/profiles/web/node_modules/js-yaml "$P/node_modules/js-yaml"
fi

cd "$P"
node --check src/index.js
node --check src/config.js
node --check src/sandbox-whitelist.js
node --test test/*.test.mjs | tail -6
echo "installed into $P (backup in $P/.pre-sbx-backup) — restart DSH to take effect"
