#!/bin/bash
# hypatia_lib.sh — shared helpers for the Hypatia Codex hooks.
#
# Design rules:
#   * Fail-open: every hook must exit 0 and never break a Codex session.
#   * Thin shells: hooks do deterministic work only (log, query, trigger
#     signals). AI-heavy work (summarization, work-unit extraction) is done
#     by the Codex agent following the hypatia-memory skill.
#   * State lives in ~/.codex/hypatia/ (logs + per-session turn counters).

HYPO_HOME="${HYPO_HOME:-$HOME/.codex/hypatia}"
HYPO_LOG="${HYPO_LOG:-$HYPO_HOME/hooks.log}"
HYPO_STATE="${HYPO_STATE:-$HYPO_HOME/sessions}"

ensure_dirs() {
  mkdir -p "$HYPO_HOME" "$HYPO_STATE" 2>/dev/null || true
}

hypo_log() {
  ensure_dirs
  printf '%s %s\n' "$(date '+%Y-%m-%dT%H:%M:%S%z')" "$*" >>"$HYPO_LOG" 2>/dev/null || true
}

# Locate the hypatia binary: $HYPATIA_BIN override, common paths, then PATH.
find_hypatia() {
  if [ -n "${HYPATIA_BIN:-}" ]; then
    printf '%s\n' "$HYPATIA_BIN"
    return 0
  fi
  for c in "$HOME/.local/bin/hypatia" /usr/local/bin/hypatia /opt/homebrew/bin/hypatia; do
    if [ -x "$c" ]; then
      printf '%s\n' "$c"
      return 0
    fi
  done
  command -v hypatia 2>/dev/null || printf 'hypatia\n'
}

# Run hypatia CLI; on failure log and return non-zero (caller fails open).
hyp() {
  local bin
  bin="$(find_hypatia)" || return 1
  ensure_dirs
  "$bin" "$@" 2>>"$HYPO_LOG"
}

# Project scope: git root basename, falling back to cwd basename.
project_scope() {
  local dir="$1"
  local root
  root="$(git -C "$dir" rev-parse --show-toplevel 2>/dev/null)" || root="$dir"
  basename "$root"
}

# Emit the Codex hook JSON output carrying additionalContext (stdin: context).
# Usage: emit_context <hookEventName> <<<"$context"
emit_context() {
  local event="$1"
  local context
  context="$(cat)"
  jq -n --arg ev "$event" --arg ctx "$context" \
    '{continue:true, hookSpecificOutput:{hookEventName:$ev, additionalContext:$ctx}}' 2>/dev/null \
    || printf '{"continue":true}\n'
}

# Emit a bare continue (side-effect-only hooks).
emit_continue() {
  printf '{"continue":true}\n'
}
