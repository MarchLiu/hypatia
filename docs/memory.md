# Memory

Agent memory in Hypatia has two layers (see `skills/hypatia-memory/SKILL.md`):

## Layer 1: Conversation log (every message)

1. Every user and assistant message is stored as one `knowledge` entry, tag `message`.
2. When a session summary is available (compaction, end-of-session digest), store it as `knowledge` with tag `session`.
3. Messages in a session link to the session: `message belongTo session`.
4. After each new message, count messages without a level-1 summary link using `$not-summaried` (native JSE operator with LEFT JOIN). When count reaches **16**, synthesize one `summary-l1` and create `summarizes` statements to all 16 messages.
5. Repeat for level 2 (`summary-l2` over 16× `summary-l1`), level 3, … until a level has fewer than 16 unlinked items.
6. Batch order is **FIFO** (oldest unlinked items first).

## Layer 2: Semantic extraction (unchanged)

1. On explicit remember/forget mentions of hypatia → immediate extraction.
2. Otherwise every 5 turns, or session end / compact → work-unit extraction.
3. Extraction creates knowledge **and** relationship statements.
4. Forget also deletes related knowledge and statements.
5. Affirmative patterns → `rule`; negative patterns → `taboo`.
6. Every entry records `scopes` (project name; global rules use `""`).
7. New sessions load matching project + global rules and taboos.
8. Optional periodic review of default shelf for cross-session links.

## Predicates

| Predicate | Subject | Object |
|-----------|---------|--------|
| `belongTo` | message | session |
| `summarizes` | summary-lN | message or summary-l(N-1) |
| `is_a`, `refines`, `extends`, `supersedes`, `derivedFrom` | semantic layer | (as before) |
