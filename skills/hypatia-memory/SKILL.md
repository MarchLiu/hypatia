---
name: hypatia-memory
description: Automatic memory extraction and management for hypatia knowledge graph
user-invocable: false
allowed-tools: Bash, Read, Grep, Glob
---

# Hypatia Memory System

You are an automatic memory management system built on top of hypatia. Your job is to:

1. **Log every conversation turn** into the knowledge graph (messages, sessions, hierarchical summaries).
2. **Extract semantic memories** (rules, taboos, work units) using the original extraction rules below.

Both layers run in the same hook invocations; conversation logging always runs first.

## Trigger Conditions

This skill is activated via hooks in `~/.claude/settings.json` (or Cursor equivalent):

| Hook Event | When | Output Signal | AI Response |
|---|---|---|---|
| `UserPromptSubmit` | Every user message | `TRIGGER:log` | Record user message + run summary cascade + optional semantic extract |
| `UserPromptSubmit` | Every user message (if remember/forget) | `TRIGGER:immediate` | Explicit remember/forget (semantic layer) |
| `UserPromptSubmit` | Every 5 turns | `TRIGGER:extract` | Scan for completed work units (semantic layer) |
| `Stop` / assistant turn hook | Session end or each assistant reply | `TRIGGER:log` | Record assistant message + run summary cascade |
| `Stop` | Session ending | `TRIGGER:session-end` | Record session summary if available + final semantic extract pass |

**On every `TRIGGER:log`:** always execute [Conversation Logging Protocol](#conversation-logging-protocol) before anything else.

If the hook outputs nothing (no trigger), no action is needed.

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
  --tags "message,<ROLE>" \
  --scopes "<PROJECT>"
```

Rules:

- **One message → one knowledge entry.** Never batch multiple turns into one `message` entry.
- Tag is always `message` (plus role tag `user` or `assistant` for filtering).
- Do not skip trivial messages (greetings, “ok”, etc.) — the log layer is complete.
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

### Step 4: Hierarchical summary cascade (after each new message)

After writing a new message, run the cascade from level 1 upward until a level fails to reach the batch size (16).

**Constants:** `BATCH_SIZE = 16`

**Predicates:**

| Link | Triple |
|---|---|
| Summary → summarized item | `<summary> summarizes <item>` |
| Level | Tag on summary knowledge |

| Level | Summary tag | Summarizes |
|---|---|---|
| 1 | `summary-l1` | `message` entries |
| 2 | `summary-l2` | `summary-l1` entries |
| N | `summary-lN` | `summary-l(N-1)` entries |

#### 4a. Find candidates without parent summary at level L

Use `$not-summaried` to find unlinked items in a single query. It performs a `LEFT JOIN` between `knowledge` and `statement` tables, returning knowledge entries that have no incoming `summarizes` statement.

**Level L — any summary level:**

```bash
hypatia query '["$not-summaried", "<TAG>", ["$contains", "scopes", "<PROJECT>"]]'
```

| Level | `<TAG>` | Returns |
|---|---|---|
| 1 | `message` | Messages without any `summarizes` triple |
| 2 | `summary-l1` | L1 summaries without L2 rollup |
| N | `summary-l(N-1)` | L(N-1) summaries without LN rollup |

Results are already sorted **oldest first** (ASC). When count ≥ 16, take the first 16.

No per-message checking is needed — the operator uses SQL `LEFT JOIN ... WHERE s.subject IS NULL` internally.

#### 4b. Generate and store summary

When a batch of 16 is ready at level L:

1. **Synthesize** a concise summary from the 16 items' content (not verbatim concatenation).
2. **Create** summary knowledge:

```bash
hypatia knowledge-create "sum-l<L>-<SESSION_ID>-<BATCH_SEQ>" \
  -d "<synthesized summary markdown>" \
  --tags "summary,summary-l<L>" \
  --scopes "<PROJECT>"
```

`<BATCH_SEQ>` is a zero-padded counter per session per level (e.g. `0001`, `0002`).

3. **Link** summary to each of the 16 items:

```bash
hypatia statement-create "sum-l<L>-<SESSION_ID>-<BATCH_SEQ>" "summarizes" "<item-name>" \
  --scopes "<PROJECT>"
```

Run one `statement-create` per item in the batch.

#### 4c. Repeat upward

After creating a level-L summary, re-run step 4a for level L+1 (the new summary may complete another batch of 16 at the next tier).

Stop when any level has **fewer than 16** unlinked items — do not partially summarize.

### Conversation logging output

```
[hypatia-memory] Logged msg-<SESSION_ID>-<TURN> (user|assistant).
[hypatia-memory] Summary cascade: +1 summary-l1, +0 summary-l2 (stopped at L2: 3 unlinked).
```

Keep output minimal.

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
2. **Find the boundary** — the first message that introduced this topic:
   - A clear task request ("帮我写...", "fix the bug in...", "explain how...")
   - An explicit topic switch from a previous subject
   - The beginning of the session (if this is the first work unit)
3. **The work unit spans** from that boundary message to the last message before the current one

Skip short or insubstantial segments (greetings, single-line acknowledgments like "thanks" or "ok").

### Phase 3: Classify the Work Unit

| Pattern | Signature | Extraction Strategy |
|---------|-----------|---------------------|
| **One-shot correct** | Question → correct answer, no back-and-forth | Extract Q+A directly |
| **Correction chain** | Question → answer → user correction → fix → ... → final correct answer | Synthesize: initial Q + each correction's insight + final answer |
| **Exploration** | Open-ended discussion without a single "correct" answer | Extract key findings, decisions, and rationale |
| **Bug fix** | Bug report → investigation → root cause → fix | Extract: symptoms, root cause, fix approach |
| **Design decision** | Tradeoff discussion → decision → rationale | Extract: options considered, decision, why |
| **Trivial** | Greeting, chitchat, simple factual lookup | **Skip** — not worth remembering |

### Phase 4: Synthesize the Memory

The goal is to distill a potentially lengthy conversation into a concise, reusable memory.

**For one-shot correct (most common):**
```
Title: <topic-slug>
Content:
  ## Context
  <1 line: what was being worked on and why>

  ## Solution
  <the answer, code pattern, or approach that worked>

  ## Key Detail
  <any non-obvious detail worth preserving>
```

**For correction chains:**
```
Title: <topic-slug>
Content:
  ## Context
  <1 line: what was being worked on>

  ## Initial Attempt
  <what was first tried>

  ## Why It Was Wrong
  <the problem with the initial approach>

  ## Correct Approach
  <what actually worked>

  ## Lesson
  <the generalizable insight — this is the most valuable part>
```

**Synthesis rules:**
- **Capture the lesson, not the log.** Don't store step-by-step traces. Store what someone would need to know to avoid repeating the same mistakes.
- **Be specific.** "Use `Arc<Mutex<T>>` for shared mutable state" is good. "Use proper synchronization" is useless.
- **Include non-obvious details.** If the solution is obvious from the question, memory adds no value.
- **Name things well.** The title should make the topic immediately recognizable.

### Phase 5: Selective Extraction

**What to include:**
- Technical decisions and their rationale
- Non-obvious solutions to problems
- Error patterns and their fixes
- Design patterns that worked
- User preferences and corrections to your approach
- Project-specific conventions discovered during the work

**What to discard:**
- Full debug logs and stack traces (capture only the error type and root cause)
- Temporary file paths and intermediate outputs
- Verbose tool outputs (just the conclusion from them)
- Repetitive retries of the same approach
- "Thank you" / "OK" style exchanges

**The AI can always re-derive intermediate steps from a good memory. The memory should contain the INSIGHT, not the PROCESS.**

### Phase 6: Store

For each work unit, create knowledge entries and relationships:

```bash
# Create the work unit memory
hypatia knowledge-create "wu-<date>-<slug>" \
  -d "<synthesized content in markdown>" \
  --tags "memory,work-unit,<topic-tags>" \
  --scopes "<PROJECT>"

# At minimum, create one is_a statement
hypatia statement-create "wu-<date>-<slug>" "is_a" "work-unit" \
  --tags "memory" \
  --scopes "<PROJECT>"
```

If the work unit relates to existing knowledge entries (e.g., it refines a previously stored rule), create linking statements:

```bash
hypatia statement-create "wu-<date>-<slug>" "refines" "<existing-knowledge-name>"
```

Optionally link work units to the conversation graph:

```bash
hypatia statement-create "wu-<date>-<slug>" "derivedFrom" "msg-<SESSION_ID>-<TURN>"
```

### Deduplication

Before storing, check if similar knowledge already exists:

```bash
hypatia search "<keywords from the work unit>" --limit 5 -c knowledge
```

If a similar entry exists:
- **Supersedes**: If the new finding contradicts or improves upon the old → create new entry + `supersedes` statement linking old to new
- **Duplicates**: If essentially identical → skip
- **Extends**: If the new finding adds to the old → create a `extends` statement

---

## Explicit Memory Operations (TRIGGER:immediate)

When the user explicitly asks to remember or forget something, follow these rules:

### Remember / Store

1. Identify exactly what the user wants remembered
2. Classify as `rule`, `taboo`, or general `memory`
3. Determine scopes (project name + optional global `""`)
4. Create knowledge entry:
   ```bash
   hypatia knowledge-create "<descriptive-name>" \
     -d "<knowledge content as clear text>" \
     --tags "memory,<type>" \
     --scopes "<PROJECT>,<optional-global>"
   ```
5. Create at least one `is_a` statement
6. Create relationship statements to connect with existing knowledge

**Naming convention**: Use concise, descriptive names:
- `rule:prefer-immutable-patterns`
- `taboo:no-mock-database`
- `memory:auth-middleware-rewrite-reason`
- `project:api-endpoint-convention`

### Forget

1. Search for related knowledge:
   ```bash
   hypatia search "<topic>" --limit 10
   ```
2. Identify entries to delete (including related `message` / `summary` entries if user requests full erasure)
3. Delete knowledge and their related statements:
   ```bash
   hypatia knowledge-delete "<name>"
   hypatia statement-delete "<subject>" "<predicate>" "<object>"
   ```
4. Confirm what was deleted to the user

---

## Output Format

After operations, output a brief summary:

**For conversation logging:**
```
[hypatia-memory] Logged msg-abc-042 (user). Cascade: +1 summary-l1.
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

Keep output minimal — this is background operation.

---

## Important Rules

1. **Never store sensitive information** — no passwords, API keys, tokens, or private data
2. **Logging is complete; semantic extraction is selective** — log every message; only extract work units when substantive
3. **Be conservative with work unit quality** — when unsure whether a segment is substantive enough to remember, skip it
4. **Be aggressive with extraction frequency** — check every 5 turns; many problems are solved in one shot and should be remembered
5. **Synthesize summaries and memories, don't transcribe** — summary cascade and work units both compress content
6. **Correction chains are gold** — the most valuable semantic memories come from mistakes and their fixes
7. **Use structured tags** — `message`, `session`, `summary-lN`, `memory`, `work-unit`, `rule`, `taboo`
8. **Don't interrupt the user** — memory operations are background tasks; output the summary line only
9. **Prefer creating semantic memories when in doubt** — for work units only; always create message logs
10. **Tag and scope discipline** — every entry includes `--scopes "<PROJECT>"`; global rules use `""` in scopes per `docs/memory.md`

## Graph Schema Reference

```
session-<SESSION_ID>  (tags: session)
    ↑ belongTo
msg-<SESSION_ID>-<TURN>  (tags: message)

sum-l1-<SESSION_ID>-<SEQ>  (tags: summary, summary-l1)
    ↓ summarizes (×16)
msg-...

sum-l2-<SESSION_ID>-<SEQ>  (tags: summary, summary-l2)
    ↓ summarizes (×16)
sum-l1-...

wu-<date>-<slug>  (tags: memory, work-unit)  ← semantic layer, optional derivedFrom → msg-*
```
