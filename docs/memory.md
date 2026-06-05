# Memory

Agent memory in Hypatia has two layers (see `skills/hypatia-memory/SKILL.md`):

## Layer 1: Conversation log (every message)

1. Every user and assistant message is stored as one `knowledge` entry, tag `message`. Name is auto-generated: `msg-<session>-<turn>`.
2. When a session summary is available (compaction, end-of-session digest), store it as `knowledge` with tag `session`.
3. Messages in a session link to the session: `message belongTo session`.
4. After each new message, check unsummarized messages using `$not-summaried` or `session-current`. Summary trigger conditions:
   - **Level 1:** When accumulated token count reaches `max_tokens × 0.9` (estimated as `chars/4`; depon model: GLM-5.1=200k, DeepSeek V4 Pro=1M).
   - **Level 2+:** When 16 unlinked items accumulate at the previous level.
   - **Emergency:** When context tokens reach `settings.max_token × 0.9` → force compression + new session.
5. Summaries use predicate `summary` (not `summarizes`). Tags: `["summary", "summary 1"]`, `["summary", "summary 2"]`, etc.
6. Summaries have meaningful names extracted from their content.
7. Batch order is **FIFO** (oldest unlinked items first).

## Layer 2: Semantic extraction (unchanged)

1. On explicit remember/forget mentions of hypatia → immediate extraction.
2. Otherwise every 5 turns, or session end / compact → work-unit extraction.
3. Extraction creates knowledge **and** relationship statements.
4. Forget also deletes related knowledge and statements.
5. Affirmative patterns → `rule`; negative patterns → `taboo`.
6. Every entry records `scopes` (project name; global rules use `""`).
7. New sessions load matching project + global rules and taboos.

## AI API Message Construction

```
[system_prompt, uncompressed_history, reference_info, user_input]
```

- **uncompressed_history:** All messages not yet summarized.
- **reference_info:** ≤5 entries from knowledge base, found by analyzing (not verbatim-searching) the user input via JSE queries.
- **user_input:** Latest user message, always last.

## Predicates

| Predicate | Subject | Object |
|-----------|---------|--------|
| `belongTo` | message | session |
| `summary` | summary entry | message or lower-level summary |
| `is_a`, `refines`, `extends`, `supersedes`, `derivedFrom` | semantic layer | (as before) |

## CLI Commands

| Command | Purpose |
|---------|---------|
| `hypatia session-current --scope <project>` | Get unsummarized messages for a scope |
| `hypatia query '["$not-summaried", "<tag>", ["$contains", "scopes", "<p>"]]'` | Find unlinked items at any summary level |
