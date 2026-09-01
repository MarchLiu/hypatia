#!/bin/bash
# SessionStart hook — inject project rules/taboos into the session context.
. "$(dirname "$0")/hypatia_lib.sh"

payload="$(cat)"
event="$(printf '%s' "$payload" | jq -r '.hook_event_name // "SessionStart"')"
session_id="$(printf '%s' "$payload" | jq -r '.session_id // empty')"
cwd="$(printf '%s' "$payload" | jq -r '.cwd // empty')"

if [ -z "$cwd" ]; then
  emit_continue
  exit 0
fi

project="$(project_scope "$cwd")"
hypo_log "SessionStart session=$session_id project=$project"

rules="$(hyp query "[\"\$knowledge\", [\"\$contains\", \"tags\", \"rule\"], [\"\$or\", [\"\$contains\", \"scopes\", \"$project\"], [\"\$contains\", \"scopes\", \"\"]]]" 2>/dev/null | jq -r '.[].content.data // empty' 2>/dev/null)"
taboos="$(hyp query "[\"\$knowledge\", [\"\$contains\", \"tags\", \"taboo\"], [\"\$or\", [\"\$contains\", \"scopes\", \"$project\"], [\"\$contains\", \"scopes\", \"\"]]]" 2>/dev/null | jq -r '.[].content.data // empty' 2>/dev/null)"

context="## Hypatia Memory Context
project: $project
session: $session_id

Every message in this session is automatically stored in the Hypatia knowledge graph (tag: message, scope: $project)."

if [ -n "$rules" ]; then
  context="$context

### Project/global rules (follow these)
$rules"
fi

if [ -n "$taboos" ]; then
  context="$context

### Project/global taboos (avoid these)
$taboos"
fi

if [ -z "$rules" ] && [ -z "$taboos" ]; then
  context="$context

(No project rules or taboos stored yet.)"
fi

emit_context "$event" <<<"$context"
exit 0
