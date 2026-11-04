#!/bin/bash
# UserPromptSubmit hook:
#   1. Recalls relevant memories and injects them as additionalContext.
#   2. Logs the user message into Hypatia (tag: message), redacted and not embedded.
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

# 1) Recall BEFORE logging, so the current message can never echo back as a
#    memory. Entries are additionally filtered: raw message logs (msg-*) are
#    never recalled, and entries carrying scopes must include this project
#    (scope-less entries are global and always eligible).
recall=""
flat_prompt="$(printf '%s' "$prompt" | tr '\n' ' ')"
if [ "${#flat_prompt}" -gt 300 ]; then
  # Long inputs: keep both ends so trailing keywords survive the budget.
  search_q="${flat_prompt:0:150} ${flat_prompt: -150}"
else
  search_q="$flat_prompt"
fi
if [ -n "$search_q" ]; then
  # Over-fetch (--limit 50) so scope/echo filtering cannot starve the window,
  # then keep the best 5 eligible hits.
  recall="$(hyp search "$search_q" --limit 50 -c knowledge 2>/dev/null | jq -r --arg p "$project" '
    def entry_scopes: ((.content | fromjson? | .scopes) // []);
    [ .[]
      | select((.key | startswith("msg-")) | not)
      | select((entry_scopes | length) == 0 or (entry_scopes | index($p)))
    ][0:5][] | "- \(.key): \(.content)"' 2>/dev/null)"
fi

# 2) Log the user message (idempotent: skip if already stored).
if ! hyp knowledge-get "$msg_name" 2>/dev/null | grep -q '"name"'; then
  # Redact credential-shaped tokens before anything reaches the shelf.
  stored_prompt="$(printf '%s' "${prompt:0:20000}" | sed -E 's/sk-[A-Za-z0-9_-]+/sk-***/g')"
  content="## Role
user

## Timestamp
$ts

## Content
$stored_prompt"
  # Raw turn logs stay out of the vector index; they are FTS/JSE searchable.
  hyp knowledge-create "$msg_name" -d "$content" --tags "message" --scopes "$project" --no-embed >/dev/null 2>&1 \
    && hypo_log "logged user msg $msg_name" \
    || hypo_log "FAILED to log user msg $msg_name"
fi

# 3) Turn counter (per session) for extraction cadence.
state_file="$HYPO_STATE/${session_id}.json"
turns=1
if [ -f "$state_file" ]; then
  prev="$(jq -r '.turns // 1' "$state_file" 2>/dev/null)"
  [ "$prev" -gt 0 ] 2>/dev/null && turns=$((prev + 1))
fi
printf '{"turns":%s}\n' "$turns" >"$state_file" 2>/dev/null

# 4) Trigger signals.
triggers="[hypatia-memory] TRIGGER:log"
case "$prompt" in
  *remember*|*记住*|*记忆*|*忘了*|*forget*|*忘记*|*忘掉*)
    triggers="$triggers, TRIGGER:immediate" ;;
esac
if [ $((turns % 5)) -eq 0 ]; then
  triggers="$triggers, TRIGGER:extract"
fi

unsummarised="$(hyp query "[\"\$not-summarised\", \"message\", [\"\$contains\", \"scopes\", \"$project\"]]" 2>/dev/null | jq -r 'length' 2>/dev/null)"
if [ "${unsummarised:-0}" -ge 16 ] 2>/dev/null; then
  triggers="$triggers, TRIGGER:summary"
fi

context="## Hypatia Memory Instructions
Signals: $triggers

If TRIGGER:log — a message entry was already stored; no action needed.
If TRIGGER:immediate — the user explicitly asked to remember/forget: use the hypatia skill to complete the memory operation.
If TRIGGER:extract — every 5 turns: follow the hypatia-memory skill's Semantic Extraction Protocol for the last completed work unit.
If TRIGGER:summary — too many unsummarised messages: follow the summary cascade (level 2+) for scope \"$project\"."

if [ -n "$recall" ]; then
  context="$context

## Reference Information (from Hypatia)
$recall"
fi

emit_context "$event" <<<"$context"
exit 0
