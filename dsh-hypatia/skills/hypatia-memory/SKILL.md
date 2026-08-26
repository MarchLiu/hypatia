---
name: hypatia-memory
description: Automatic memory extraction and management for hypatia knowledge graph
user-invocable: false
allowed-tools: Bash, Read, Grep, Glob
---

# Hypatia Memory System

You are an automatic memory management system built on hypatia. Your job is to:

1. **Log every conversation turn** into the knowledge graph (messages, sessions, hierarchical summaries).
2. **Construct AI API messages** with system prompt + uncompressed history + reference info + latest user input.
3. **Extract semantic memories** (rules, taboos, work units) using the original extraction rules.

All layers run in the same hook invocations; conversation logging always runs first.

## Trigger Conditions

This skill is activated via hooks in `~/.claude/settings.json` (or Cursor equivalent):

| Hook Event | When | Output Signal | AI Response |
|---|---|---|---|
| `UserPromptSubmit` | Every user message | `TRIGGER:log` | Record user message + check summary cascade + optional semantic extract |
| `UserPromptSubmit` | Every user message (if remember/forget) | `TRIGGER:immediate` | Explicit remember/forget (semantic layer) |
| `UserPromptSubmit` | Every 5 turns | `TRIGGER:extract` | Scan for completed work units (semantic layer) |
| `Stop` / assistant turn hook | Session end or each assistant reply | `TRIGGER:log` | Record assistant message + check summary cascade |
| `Stop` | Session ending | `TRIGGER:session-end` | Record session summary if available + final semantic extract pass |

**On every `TRIGGER:log`:** always execute [Conversation Logging Protocol](#conversation-logging-protocol) first.

If the hook outputs nothing (no trigger), no action is needed.

## Codex Hooks Integration (Codex CLI / Desktop App)

The same protocol runs natively in Codex via `~/.codex/hooks.json` (see
`codex-integration/` in the Hypatia repo). The bundled scripts translate
Codex lifecycle events into the exact trigger signals above:

| Codex event | Hook script | What it does |
|---|---|---|
| `SessionStart` | `hypatia_session_start.sh` | Loads project/global rules + taboos and injects them as `additionalContext` |
| `UserPromptSubmit` | `hypatia_user_prompt_submit.sh` | Logs the user message, recalls relevant memories, emits `TRIGGER:log` + optional `TRIGGER:immediate` / `TRIGGER:extract` (every 5 turns) / `TRIGGER:summary` (≥16 unsummarized) |
| `Stop` | `hypatia_stop.sh` | Logs the assistant message (side-effect only; Codex Stop hooks cannot inject context) |

Installation: `./codex-integration/install.sh`, restart Codex, then review and
trust the hooks (CLI: `/hooks`; desktop app: Settings → Hooks).

The hook scripts are thin, deterministic shells: they write message entries
(`msg-<session_id>-<turn_id>`, tag `message`) and retrieval context, and emit
trigger signals. The AI-heavy steps in this document (summary synthesis,
work-unit extraction) are still performed by the agent following this skill —
the hooks never make model calls.

## Session Startup

When a new session begins, load relevant rules and taboos:

1. Determine the current project name from the working directory (use `basename` of the git root or CWD)
2. Run these queries to load rules and taboos for the current project and global scope:

```bash
# Load project-specific and global rules
hypatia query '["$knowledge", ["$contains", "tags", "rule"], ["$or", ["$contains", "scopes", "<PROJECT>"], ["$contains", "scopes", ""]]]'

# Load project-specific and global taboos
hypatia query '["$knowledge", ["$contains", "tags", "taboo"], ["$or", ["$contains", "scopes", "<PROJECT>"], ["$contains", "scopes", ""]]]'
```

3. Internalize these rules and taboos for the current session. Follow rules and avoid taboos in all interactions.

---

## Conversation Logging Protocol

This protocol runs on **every** user and assistant message (`TRIGGER:log`). It is independent of semantic work-unit extraction.

### Identifiers

Resolve from hook context when available; otherwise derive:

| Field | Source |
|---|---|
| `<PROJECT>` | `basename` of git root or CWD |
| `<SESSION_ID>` | Hook `session_id`, Cursor `conversation_id`, or stable hash of transcript path |
| `<TURN>` | Monotonic turn counter within session (increment per logged message) |
| `<ROLE>` | `user` or `assistant` |

### Step 1: Record the message

Every conversational turn becomes one knowledge entry.

```bash
hypatia knowledge-create "msg-<SESSION_ID>-<TURN>" \
  -d "## Role
<ROLE>

## Timestamp
<ISO-8601>

## Content
<full message text>" \
  --tags "message" \
  --scopes "<PROJECT>"
```

Rules:

- **One message → one knowledge entry.** Never batch multiple turns.
- Tag is `message` (no role tag; use content to determine role).
- Name is auto-generated: `msg-<SESSION_ID>-<TURN>`.
- Do not skip trivial messages (greetings, "ok", etc.) — the log layer is complete.
- Never store secrets (passwords, API keys, tokens) — redact before writing.

### Step 2: Record session knowledge (when summary available)

If the hook or environment provides a **session-level summary** (e.g. compaction summary, session title, or end-of-session digest):

```bash
hypatia knowledge-create "session-<SESSION_ID>" \
  -d "<session summary text>" \
  --tags "session" \
  --scopes "<PROJECT>"
```

- Create or update `session-<SESSION_ID>` when new summary text arrives (prefer `knowledge-update` if entry exists).
- If no session summary is available, skip this step — do not fabricate session summaries.

### Step 3: Link message to session

When both `msg-<SESSION_ID>-<TURN>` and `session-<SESSION_ID>` exist:

```bash
hypatia statement-create "msg-<SESSION_ID>-<TURN>" "belongTo" "session-<SESSION_ID>" \
  --scopes "<PROJECT>"
```

Predicate is exactly `belongTo` (message → session).

### Step 4: Hierarchical summary cascade

After writing each new message, run the cascade from level 1 upward.

**Constants:** `BATCH_SIZE = 16` (for L2+)

**Predicate:** All summary triples use predicate `summary`.

| Triple | Meaning |
|---|---|
| `<summary-name> summary <item-name>` | Summary condenses the item |

| Level | Tag | Triggers when | Summarizes |
|---|---|---|---|
| 1 | `["summary", "summary 1"]` | Token count ≥ `max_tokens × 0.9` | `message` entries |
| 2 | `["summary", "summary 2"]` | Count ≥ 16 unlinked L1 | `summary 1` entries |
| N | `["summary", "summary N"]` | Count ≥ 16 unlinked L(N-1) | `summary (N-1)` entries |

**Token-based L1 threshold:**
- Estimate tokens by character count / 4 (agent-side estimation).
- `max_tokens` depends on the model in use (e.g. GLM-5.1: 200k, DeepSeek V4 Pro: 1M).
- When the accumulated token count of unsummarized messages reaches 90% of the model's max_tokens, trigger L1 summary generation.

**Context compression trigger:**
- If context tokens reach `settings.max_token × 0.9`, also trigger summary generation and start a new session. This is independent of the L1 count/trigger — it's an emergency compression.

#### 4a. Check unsummarized items at level L

Use `$not-summaried` (native JSE operator with LEFT JOIN):

```bash
hypatia query '["$not-summaried", "<TAG>", ["$contains", "scopes", "<PROJECT>"]]'
```

Or use the shorthand:

```bash
hypatia session-current --scope <PROJECT>
```

| Level | `<TAG>` |
|---|---|
| 1 | `message` |
| 2 | `summary 1` |
| N | `summary (N-1)` |

Results are sorted **oldest first** (ASC). For L1: count tokens (estimate `chars/4`). For L2+: take first 16 if count ≥ 16.

#### 4b. Generate and store summary

When a batch is ready at level L:

1. **Synthesize** a concise summary from the items' content (not verbatim concatenation).
2. **Extract a name** for the summary from its content — a short, descriptive identifier.
3. **Create** summary knowledge:

```bash
hypatia knowledge-create "<extracted-summary-name>" \
  -d "<synthesized summary markdown>" \
  --tags "summary,summary <L>" \
  --scopes "<PROJECT>"
```

- Summary has a meaningful name extracted from the summarized content (e.g. "error-handling-refactor", "api-design-discussion").
- Tag format: `summary,summary <L>` where L is the level number.

4. **Link** summary to each source item:

```bash
hypatia statement-create "<summary-name>" "summary" "<item-name>" \
  --scopes "<PROJECT>"
```

Run one `statement-create` per item in the batch.

#### 4c. Repeat upward

After creating a level-L summary, re-run step 4a for level L+1 (the new summary may complete another batch at the next tier).

Stop when a level has **fewer than the required threshold** — do not partially summarize.

### Step 5: AI API Message Construction

When submitting a conversation to the AI API, construct the messages list as:

```
[system_prompt, uncompressed_messages..., reference_info, latest_user_input]
```

**System prompt:** Constructed using the existing logic (rules, taboos, project context).

**Uncompressed messages:** The current set of messages that have not been summarized. Query with:

```bash
hypatia query '["$not-summaried", "message", ["$contains", "scopes", "<PROJECT>"]]'
```

**Reference info:** Analyze the user's latest input — do NOT use it verbatim as a search query. Instead:

1. Identify key entities, concepts, and topics from the user's input.
2. Construct 1-3 JSE queries targeting these topics. Example strategies:
   - Search for related past messages: `["$not-summaried", "message", ["$contains", "scopes", "<PROJECT>"]]` + filter in reasoning
   - Full-text search: `["$knowledge", ["$search", "<derived keywords>"]]`
   - Vector similarity: `["$knowledge", ["$similar", "<conceptual query>"]]`
   - Statement graph exploration: `["$statement", ["$triple", "<entity>", "$*", "$*"]]`
3. Collect up to **5** relevant knowledge entries (from conversation history or existing knowledge base).
4. Format them as a reference message placed as the **second-to-last** message:

```
## Reference Information
The following relevant context was retrieved from the knowledge base:

1. <entry-name>: <summary or key content>
2. <entry-name>: <summary or key content>
...
```

**Latest user input:** Always the most recent user message, placed last.

**After receiving the response:** Save the assistant's response as a new message in the conversation history.

---

## Semantic Extraction Protocol (unchanged)

This layer extracts **insights** (rules, taboos, work units). It does not replace conversation logging.

### Phase 1: Assess Topic Continuity

When receiving `TRIGGER:extract`:

1. **Read the current user message** and the immediately preceding conversation (last ~5 exchanges)
2. **Determine if the current message starts a new topic** — is it unrelated to what was being discussed just before?
3. **Decision:**
   - **Topic changed** → the conversation segment BEFORE the current message is a **completed work unit** → proceed to Phase 2
   - **Topic continues** → the work unit is still in progress → output `[hypatia-memory] Work unit still in progress, nothing extracted.` and stop (logging still completed in Step 1)
   - **TRIGGER:immediate** → bypass topic detection, extract what user asked about directly → jump to Phase 4

For `TRIGGER:session-end`:

- Treat ALL conversation since last extraction as potentially containing completed work units
- Run a full pass: find all boundaries, extract each work unit

### Phase 2: Delimit the Work Unit

When a completed work unit is detected:

1. **Read backwards** from just before the current (topic-changing) message
2. **Find the boundary** — the first message that introduced this topic
3. **The work unit spans** from that boundary message to the last message before the current one

Skip short or insubstantial segments (greetings, single-line acknowledgments like "thanks" or "ok").

### Phase 3: Classify the Work Unit

| Pattern | Signature | Extraction Strategy |
|---------|-----------|---------------------|
| **One-shot correct** | Question → correct answer, no back-and-forth | Extract Q+A directly |
| **Correction chain** | Question → answer → user correction → fix → ... | Synthesize: initial Q + each correction + final answer |
| **Exploration** | Open-ended discussion without single "correct" answer | Extract key findings, decisions, rationale |
| **Bug fix** | Bug report → investigation → root cause → fix | Extract: symptoms, root cause, fix approach |
| **Design decision** | Tradeoff discussion → decision → rationale | Extract: options considered, decision, why |
| **Trivial** | Greeting, chitchat, simple factual lookup | **Skip** — not worth remembering |

### Phase 4: Synthesize the Memory

**For one-shot correct:**
```
Title: <topic-slug>
Content:
  ## Context
  <1 line summary>
  ## Solution
  <the answer or approach>
  ## Key Detail
  <non-obvious detail>
```

**For correction chains:**
```
Title: <topic-slug>
Content:
  ## Context
  ## Initial Attempt
  ## Why It Was Wrong
  ## Correct Approach
  ## Lesson
```

**Synthesis rules:**
- Capture the lesson, not the log.
- Be specific. "Use `Arc<Mutex<T>>`" is good. "Use proper synchronization" is useless.
- Include non-obvious details.
- Name things well.

### Phase 5: Selective Extraction

**What to include:** technical decisions, non-obvious solutions, error patterns, design patterns, user preferences, project conventions.

**What to discard:** full debug logs, temporary paths, verbose tool outputs, repetitive retries, "thank you"/"ok" exchanges.

### Phase 6: Store

```bash
hypatia knowledge-create "wu-<date>-<slug>" \
  -d "<synthesized content>" \
  --tags "memory,work-unit,<topic-tags>" \
  --scopes "<PROJECT>"

hypatia statement-create "wu-<date>-<slug>" "is_a" "work-unit" \
  --scopes "<PROJECT>"
```

Optionally link to conversation graph:

```bash
hypatia statement-create "wu-<date>-<slug>" "derivedFrom" "msg-<SESSION_ID>-<TURN>"
```

### Deduplication

Before storing, check for similar knowledge:

```bash
hypatia search "<keywords>" --limit 5 -c knowledge
```

- **Supersedes**: new contradicts old → create `supersedes` statement
- **Duplicates**: identical → skip
- **Extends**: adds to old → create `extends` statement

---

## Explicit Memory Operations (TRIGGER:immediate)

When the user explicitly asks to remember or forget:

### Remember / Store

1. Identify what to remember
2. Classify as `rule`, `taboo`, or general `memory`
3. Determine scopes
4. Create:
   ```bash
   hypatia knowledge-create "<name>" \
     -d "<content>" \
     --tags "memory,<type>" \
     --scopes "<PROJECT>,<optional-global>"
   ```
5. Create `is_a` statement and relationship statements

### Forget

1. Search: `hypatia search "<topic>" --limit 10`
2. Delete knowledge and related statements (including `message` / `summary` entries if full erasure)
3. Confirm to user

---

## Output Format

**For conversation logging:**
```
[hypatia-memory] Logged msg-abc-042. Cascade: +1 summary 1 (token threshold).
```

**For work unit extraction:**
```
[hypatia-memory] Extracted 2 work units (1 one-shot, 1 correction-chain), skipped 1 trivial.
  wu-2026-05-10-sort-function    → memory,work-unit,rust
```

**For immediate operations:**
```
[hypatia-memory] Stored: "rule:prefer-immutable-patterns" (rule, scoped to my-project).
```

**For forget operations:**
```
[hypatia-memory] Removed 1 entry and 2 relationships.
```

**When nothing to extract (semantic only):**
```
[hypatia-memory] Work unit still in progress, nothing extracted.
```

---

## Important Rules

1. **Never store sensitive information** — no passwords, API keys, tokens
2. **Logging is complete; semantic extraction is selective** — log every message; extract work units only when substantive
3. **Be conservative with work unit quality** — skip when unsure
4. **Be aggressive with extraction frequency** — check every 5 turns
5. **Synthesize summaries and memories, don't transcribe** — compress content
6. **Correction chains are gold** — the most valuable memories come from mistakes
7. **Use structured tags** — `message`, `session`, `summary <N>`, `memory`, `work-unit`, `rule`, `taboo`
8. **Don't interrupt the user** — memory operations are background tasks
9. **Prefer creating semantic memories when in doubt** — for work units only; always create message logs
10. **Tag and scope discipline** — every entry includes `--scopes "<PROJECT>"`; global rules use `""`

## Graph Schema Reference

```
session-<SESSION_ID>  (tags: session)
    ↑ belongTo
msg-<SESSION_ID>-<TURN>  (tags: message)

<summary-name>  (tags: summary, summary 1)
    ↓ summary (×batch)
msg-...

<summary-name>  (tags: summary, summary 2)
    ↓ summary (×16)
<summary-name>...  (tags: summary, summary 1)

wu-<date>-<slug>  (tags: memory, work-unit)  ← semantic layer, optional derivedFrom → msg-*
```
