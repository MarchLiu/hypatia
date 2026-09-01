#!/bin/bash
# Stop hook — log the assistant message into Hypatia (side-effect only;
# Codex Stop hooks cannot inject context in this version).
. "$(dirname "$0")/hypatia_lib.sh"

payload="$(cat)"
session_id="$(printf '%s' "$payload" | jq -r '.session_id // empty')"
turn_id="$(printf '%s' "$payload" | jq -r '.turn_id // empty')"
last="$(printf '%s' "$payload" | jq -r '.last_assistant_message // empty')"
cwd="$(printf '%s' "$payload" | jq -r '.cwd // empty')"

if [ -z "$session_id" ] || [ -z "$turn_id" ] || [ -z "$last" ]; then
  emit_continue
  exit 0
fi

project="$(project_scope "$cwd")"
msg_name="msg-${session_id}-${turn_id}-a"
ts="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"

if ! hyp knowledge-get "$msg_name" 2>/dev/null | grep -q '"name"'; then
  stored="${last:0:20000}"
  content="## Role
assistant

## Timestamp
$ts

## Content
$stored"
  hyp knowledge-create "$msg_name" -d "$content" --tags "message" --scopes "$project" >/dev/null 2>&1 \
    && hypo_log "logged assistant msg $msg_name" \
    || hypo_log "FAILED to log assistant msg $msg_name"
fi

emit_continue
exit 0
