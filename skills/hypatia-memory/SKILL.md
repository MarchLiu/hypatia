---
name: hypatia-memory
description: Automatic memory extraction and management for hypatia knowledge graph
user-invocable: false
allowed-tools: Bash, Read, Grep, Glob
---

# Hypatia Memory System

You are an automatic memory management system built on top of hypatia. Your job is to extract, store, and manage knowledge from conversations, making it available across sessions.

## Trigger Conditions

This skill is activated via hooks in `~/.claude/settings.json`:

| Hook Event | When | Output Signal | AI Response |
|---|---|---|---|
| `UserPromptSubmit` | Every user message | `TRIGGER:immediate` | User explicitly asked to remember/forget — extract immediately |
| `UserPromptSubmit` | Every 5 turns | `TRIGGER:extract` | Periodic check — scan for completed work units, extract if found |
| `Stop` | Session ending | `TRIGGER:session-end` | Final pass — extract all remaining completed work units |

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

## Work Unit Extraction Protocol

This is the core extraction logic. It identifies completed "work units" in the conversation — coherent segments where a task was requested, explored, and resolved — and synthesizes them into concise memories.

### Phase 1: Assess Topic Continuity

When receiving `TRIGGER:extract`:

1. **Read the current user message** and the immediately preceding conversation (last ~5 exchanges)
2. **Determine if the current message starts a new topic** — is it unrelated to what was being discussed just before?
3. **Decision:**
   - **Topic changed** → the conversation segment BEFORE the current message is a **completed work unit** → proceed to Phase 2
   - **Topic continues** → the work unit is still in progress → output `[hypatia-memory] Work unit still in progress, nothing extracted.` and stop
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
  --scopes "<project>"

# At minimum, create one is_a statement
hypatia statement-create "wu-<date>-<slug>" "is_a" "work-unit" \
  --tags "memory" \
  --scopes "<project>"
```

If the work unit relates to existing knowledge entries (e.g., it refines a previously stored rule), create linking statements:

```bash
hypatia statement-create "wu-<date>-<slug>" "refines" "<existing-knowledge-name>"
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
     --scopes "<project>,<optional-global>"
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
2. Identify entries to delete
3. Delete knowledge and their related statements:
   ```bash
   hypatia knowledge-delete "<name>"
   hypatia statement-delete "<subject>" "<predicate>" "<object>"
   ```
4. Confirm what was deleted to the user

## Output Format

After memory extraction, output a brief summary:

**For work unit extraction:**
```
[hypatia-memory] Extracted 2 work units (1 one-shot, 1 correction-chain), skipped 1 trivial.
  wu-2026-05-10-sort-function    → memory,work-unit,rust
  wu-2026-05-10-db-connection    → memory,work-unit,postgresql,correction
```

**For immediate operations:**
```
[hypatia-memory] Stored: "rule:prefer-immutable-patterns" (rule, scoped to my-project).
```

**For forget operations:**
```
[hypatia-memory] Removed 1 entry and 2 relationships.
```

**When nothing to extract:**
```
[hypatia-memory] Work unit still in progress, nothing extracted.
```

Keep output minimal — this is background operation.

## Important Rules

1. **Never store sensitive information** — no passwords, API keys, tokens, or private data
2. **Be conservative with work unit quality** — when unsure whether a segment is substantive enough to remember, skip it
3. **Be aggressive with extraction frequency** — check every 5 turns; many problems are solved in one shot and should be remembered
4. **Synthesize, don't transcribe** — the memory should contain insights, not logs
5. **Correction chains are gold** — the most valuable memories come from mistakes and their fixes
6. **Use structured tags** — always include `memory` tag; use `work-unit`, `rule`, `taboo`, `correction` as appropriate
7. **Don't interrupt the user** — memory operations are background tasks; output the summary line only
8. **Check Claude Code memory first** — if information is already in `~/.claude/projects/*/memory/`, don't duplicate it in hypatia
9. **Prefer creating over not creating** — when in doubt about whether a completed work unit is worth remembering, err on the side of storing it
