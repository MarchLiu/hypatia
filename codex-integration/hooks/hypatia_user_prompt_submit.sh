#!/bin/bash
# UserPromptSubmit hook:
#   1. Logs the user message into Hypatia (tag: message).
#   2. Recalls relevant memories and injects them as additionalContext.
#   3. Emits trigger signals (TRIGGER:log / immediate / extract / summary)
#      that the Codex agent follows via the hypatia-memory skill.
. "$(dirname "$0")/hypatia_lib.sh"

payload="$(cat)"
event="$(printf '%s' "$payload" | jq -r '.hook_event_name // "UserPromptSubmit"')"
session_id="$(printf '%s' "$payload" | jq -r '.session_id // empty')"
turn_id="$(printf '%s' "$payload" | jq -r '.turn_id // empty')"
prompt="$(printf '%s' "$payload" | jq -r '.prompt // empty')"
cwd="$(printf '%s' "$payload" | jq -r '.cwd // empty')"

if [ -z "$session_id" ] || [ -z "$turn_id" ]; then
  emit_continue
  exit 0
fi

project="$(project_scope "$cwd")"
msg_name="msg-${session_id}-${turn_id}"
ts="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"

# 1) Log the user message (idempotent: skip if already stored).
if ! hyp knowledge-get "$msg_name" 2>/dev/null | grep -q '"name"'; then
  stored_prompt="${prompt:0:20000}"
  content="## Role
user

## Timestamp
$ts

## Content
$stored_prompt"
  hyp knowledge-create "$msg_name" -d "$content" --tags "message" --scopes "$project" >/dev/null 2>&1 \
    && hypo_log "logged user msg $msg_name" \
    || hypo_log "FAILED to log user msg $msg_name"
fi

# 2) Turn counter (per session) for extraction cadence.
state_file="$HYPO_STATE/${session_id}.json"
turns=1
if [ -f "$state_file" ]; then
  prev="$(jq -r '.turns // 1' "$state_file" 2>/dev/null)"
  [ "$prev" -gt 0 ] 2>/dev/null && turns=$((prev + 1))
fi
printf '{"turns":%s}\n' "$turns" >"$state_file" 2>/dev/null

# 3) Trigger signals.
triggers="[hypatia-memory] TRIGGER:log"
case "$prompt" in
  *remember*|*记住*|*记忆*|*忘了*|*forget*|*忘记*|*忘掉*)
    triggers="$triggers, TRIGGER:immediate" ;;
esac
if [ $((turns % 5)) -eq 0 ]; then
  triggers="$triggers, TRIGGER:extract"
fi

unsummarized="$(hyp query "[\"\$not-summaried\", \"message\", [\"\$contains\", \"scopes\", \"$project\"]]" 2>/dev/null | jq -r 'length' 2>/dev/null)"
if [ "${unsummarized:-0}" -ge 16 ] 2>/dev/null; then
  triggers="$triggers, TRIGGER:summary"
fi

# 4) Recall: lightweight FTS over the recent prompt.
recall=""
search_q="$(printf '%s' "$prompt" | tr '\n' ' ' | cut -c1-300)"
if [ -n "$search_q" ]; then
  recall="$(hyp search "$search_q" --limit 5 -c knowledge 2>/dev/null | jq -r '.[] | "- \(.key): \(.content)"' 2>/dev/null)"
fi

context="## Hypatia Memory Instructions
Signals: $triggers

If TRIGGER:log — a message entry was already stored; no action needed.
If TRIGGER:immediate — the user explicitly asked to remember/forget: use the hypatia skill to complete the memory operation.
If TRIGGER:extract — every 5 turns: follow the hypatia-memory skill's Semantic Extraction Protocol for the last completed work unit.
If TRIGGER:summary — too many unsummarized messages: follow the summary cascade (level 2+) for scope \"$project\"."

if [ -n "$recall" ]; then
  context="$context

## Reference Information (from Hypatia)
$recall"
fi

emit_context "$event" <<<"$context"
exit 0
