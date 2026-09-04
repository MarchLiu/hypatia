---
name: hypatia-dream
description: "Run an incremental Hypatia graph-consolidation pass at a work-period boundary, like sleep-time knowledge organization. Trigger only for an explicit batch closeout or dream-time review, not ordinary knowledge CRUD, a one-off triple, or a lookup. Without an explicit --mode, these mean apply: '下班了', '收工了', '今天先到这', '本阶段结束', '阶段性收尾', '做个好梦', '睡前整理一下', 'done for the day', 'wrapping up', 'call it a night'. These mean report-only: '先整理看看', '做个梦看看', '睡前审阅一下', '看看知识之间的关系', '只出报告', '先别修改', 'just show me first', 'do not change anything yet'. Statement triples are the primary evidence; Knowledge bodies only disambiguate."
user-invocable: true
allowed-tools: Bash, Read, Grep, Glob
argument-hint: "[shelf] [--mode report|apply]"
---

# Hypatia Dream

Perform an incremental, relationship-focused consolidation pass over one Hypatia shelf. The purpose is to make the graph more coherent after new knowledge arrives, not to summarize, rewrite, delete, or otherwise curate the knowledge nodes themselves.

Treat the graph as a personal knowledge base: inspect its whole available relationship topology rather than imposing a small relationship window. Keep the analysis disciplined by treating Statement triples as the primary evidence. Knowledge bodies may clarify entity identity or resolve an apparent contradiction, but they are not a license to invent weak associations.

## Trigger And Mode Resolution

Trigger this skill only for a deliberate batch consolidation at a work-period boundary or an explicitly requested dream-time review. Do not trigger it for ordinary knowledge CRUD, a direct one-off triple request, an entity lookup, a relationship query, or automatic conversation-memory extraction; those belong to `hypatia` or `hypatia-memory`.

Interpret the optional shelf name from the request; use `default` when it is omitted.

Resolve the mode in this order:

1. An explicit `--mode report` or `--mode apply` always wins.
2. Without an explicit mode, treat these closeout signals and clear equivalents as `apply`: `下班了`, `收工了`, `今天先到这`, `本阶段结束`, `阶段性收尾`, `做个好梦`, `睡前整理一下`, `done for the day`, `wrapping up`, and `call it a night`.
3. Without an explicit mode, treat these preview signals and clear equivalents as `report`: `先整理看看`, `做个梦看看`, `睡前审阅一下`, `看看知识之间的关系`, `只出报告`, `先别修改`, `just show me first`, and `do not change anything yet`.
4. If both signal classes appear in one request, ask the user to choose a mode. Do not assume `apply`.
5. A direct but otherwise unspecified invocation defaults to `report`.

The modes have these effects:

- `report` reads the shelf and returns a structured report. It creates no state marker and changes no knowledge or statement.
- `apply` may add a missing statement or replace an explicitly contradicted statement. It never creates, edits, or deletes a regular Knowledge entry.

Before every Hypatia CLI command, briefly state the goal, the exact command, why that query or mutation is appropriate, and the expected result. Use one-shot commands only; do not use `hypatia repl`.

## Hypatia Conventions To Preserve

Use the actual Statement fields and CLI terminology: `head`, `relation`, and `tail`. Do not describe them as fields named `subject`, `predicate`, or `object` in commands or reports.

Prefer the relationship vocabulary already established by Hypatia's semantic-memory design:

- `is_a` for a clear category relationship.
- `refines` for a more precise form of an existing concept.
- `extends` for an addition that builds on an existing concept.
- `supersedes` for an explicit replacement of an older concept or decision.
- `derivedFrom` for a clear provenance relationship.

Preserve existing predicate spellings. `belongTo` and `summary` are operational relationships maintained by the memory system; do not reinterpret, delete, replace, or add semantic links to their message/session/summary endpoints. Likewise, do not reorganize archive metadata relationships such as `is_a archive`.

Do not use a vague catch-all relationship such as `related_to` merely because two entries share words or a topic. Omit uncertain candidates.

## Resolve The CLI And Snapshot Boundary

1. Resolve the binary in this order: `hypatia` on `PATH`, `./target/debug/hypatia`, then `./target/release/hypatia`. Stop and report a concrete blocker if none exists.
2. Set `SHELF` to the requested shelf or `default`, then confirm that shelf is connected before doing anything else:

```bash
hypatia list
```

Every Hypatia command fails when the target shelf is not connected, and the whole CLI fails at startup when any registered shelf uses the legacy DuckDB layout. Both are blockers to report, not conditions to work around.

3. Record a UTC run-start timestamp before reading graph data:

```bash
date -u '+%Y-%m-%d %H:%M:%S'
```

SQLite's `'now'` is UTC, so this is the same clock Hypatia stamps with. Hypatia writes `created_at` as `%Y-%m-%d %H:%M:%f`, so stored values carry milliseconds (`2026-09-02 08:50:08.790`) and compare as TEXT. A second-resolution boundary is therefore slightly conservative, deliberately so: `created_at <= "2026-09-02 08:50:08"` excludes `2026-09-02 08:50:08.790`, and because the watermark is written at the same resolution, the next run's `created_at > "2026-09-02 08:50:08"` picks that row back up. Do not pad the boundary with `.999` and do not truncate stored values to close the apparent gap — either one reintroduces a real gap.

4. Query the latest successful apply marker in the same shelf:

```bash
hypatia query '{"$knowledge":[["$contains","tags","hypatia-dream-run"]],"limit":1}' -s "<SHELF>"
```

Query results are ordered by `created_at DESC`, so `limit:1` returns the most recent marker. A marker is a system-tagged Knowledge entry created by this skill after a successful `apply`. Its content contains a `processed_through` timestamp. Read that value, not the marker entry's `created_at`: a run can create statements after its snapshot begins, and using the marker creation time could skip concurrent new information.

5. Define the collection interval as follows:

- If a valid marker exists, use `processed_through < created_at <= run_started_at`.
- If there is no valid marker, this is the first consolidation pass. Collect all pre-snapshot triples and knowledge entries.
- A malformed marker is not a reason to guess. Report it and treat the run as a first pass unless the user supplies an explicit baseline.

`report` mode must not advance this watermark. This lets a later `apply` reconsider the same proposed changes. Only a completed `apply` writes a marker.

## Gather The Working Set

Retrieve the complete pre-snapshot relationship graph. Use a large, explicit limit because the shelf is personal and the user requested no relationship-range cap:

```bash
hypatia query '{"$statement":[["$lte","created_at","<RUN_STARTED_AT>"]],"limit":10000}' -s "<SHELF>"
```

Retrieve the newly created triples with the collection interval. For a first pass, omit the lower-bound condition:

```bash
hypatia query '{"$statement":[["$and",["$gt","created_at","<BASELINE>"],["$lte","created_at","<RUN_STARTED_AT>"]]],"limit":10000}' -s "<SHELF>"
```

Also retrieve new Knowledge entries from the same interval. Exclude internal log, session, summary, and dream-marker entries by tag. For a first pass, omit the lower-bound condition:

```bash
hypatia query '{"$knowledge":[["$and",["$gt","created_at","<BASELINE>"],["$lte","created_at","<RUN_STARTED_AT>"],["$not",["$or",["$contains","tags","message"],["$contains","tags","session"],["$contains","tags","summary"],["$contains","tags","hypatia-dream-run"]]]]],"limit":10000}' -s "<SHELF>"
```

`limit` is a hard cap, not a promise. Compare the returned row count with the limit for each of the three queries above. If they are equal, the result was truncated by `ORDER BY created_at DESC` and the graph in hand is incomplete. Stop, report the truncation as a blocker, and apply nothing: a relation can only be judged contradicted, duplicate, or missing against the whole graph, and a truncated read would let this skill "correct" a relationship whose supporting evidence was simply cut off.

The new Knowledge set can include an unlinked node. It is eligible for one well-supported relationship to an older graph node, but only after its body and the prospective target's body have been read for disambiguation. Do not manufacture a relation solely to make an isolated node connected.

For each potentially affected node, fetch its Knowledge entry by exact name and inspect its one-hop incident triples from the full graph. Read only the bodies needed to explain an entity match, source lineage, category, refinement, extension, supersession, or direct contradiction. Never use conversation text, session summaries, or system markers as semantic evidence.

## Reason About Relationships

Work from the newly added triples and eligible new Knowledge nodes toward the full existing graph. For every candidate, identify:

- The exact new evidence: triple or Knowledge entry name and the relevant wording.
- The supporting old graph context: exact triples and, when needed, the old node body.
- The proposed triple in `(head, relation, tail)` form.
- A confidence level and a short explanation of why that relation is more precise than its alternatives.

Classify each outcome as exactly one of these:

1. **Add**: a missing, non-duplicate triple is directly supported by the new evidence and the existing graph vocabulary.
2. **Replace**: a specific old triple is directly contradicted by newer, unambiguous evidence. Name both triples. Prefer adding `supersedes` when the old idea remains historically useful; delete and replace only when the old relation itself is false.
3. **No change**: evidence is duplicate, weak, ambiguous, historical rather than contradictory, or depends on a protected/system relation.

A candidate must be high confidence before it may be applied. Similarity, shared tags, or lexical overlap alone are never high confidence.

Before a replacement, inspect the old statement's `content`, `tr_start`, and `tr_end`. A query result omits NULL columns entirely, so an absent `tr_start` or `tr_end` key means the statement has no temporal bound — not that the information is unavailable.

`statement-create` accepts only `--data`, `--synonyms`, and `--scopes`. It cannot set `tags`, cannot set `format`, cannot set `figures`, and cannot recreate a temporal bound. Report instead of replacing whenever the source statement has any of these:

- a populated `tr_start` or `tr_end`;
- a non-empty `content.tags`;
- a `content.format` other than `markdown`;
- a non-empty `content.figures`.

Otherwise the statement is eligible, and the replacement must carry over `content.data`, positional `content.synonyms`, and `content.scopes` from the source.

## Apply Changes

In `report` mode, stop after producing the report. Do not write a marker.

In `apply` mode, execute this sequence only after the analysis identifies high-confidence changes:

1. Re-query every proposed old and new triple immediately before mutating. Use the exact-key form, which matches on the `triple` primary key:

```bash
hypatia query '{"$statement":[["$triple","<HEAD>","<RELATION>","<TAIL>"]],"limit":1}' -s "<SHELF>"
```

Skip a stale plan, an addition that already exists, and a replacement whose source triple no longer matches the reviewed evidence.

2. If the run plans any replacement, export the shelf to a timestamped backup directory before the first replacement step and report the destination. `export` creates the destination directory itself, so do not pre-create it, and pass the shelf name as the first argument:

```bash
hypatia export "<SHELF>" "<BACKUP_DIR>"
```

3. For an addition, create only the exact reviewed triple. Do not create supporting nodes or generic fallback links.

```bash
hypatia statement-create "<HEAD>" "<RELATION>" "<TAIL>" -s "<SHELF>"
```

`statement.triple` is unique, so creating a triple that already exists fails with `UNIQUE constraint failed: statement.triple`. That means the relationship is already recorded: treat it as a no-op, count it as skipped, and continue. It is not a run failure.

4. For a replacement, **create the new triple first, verify it, and only then delete the source triple.** A replacement changes at least one of `head`, `relation`, and `tail`, so the two triples have different primary keys and can coexist for the moment in between. This ordering has no window in which the old relationship is already gone and the new one does not yet exist. Never delete first.

```bash
hypatia statement-create "<NEW_HEAD>" "<NEW_RELATION>" "<NEW_TAIL>" \
  -d "<SOURCE_DATA>" --synonyms '<SOURCE_SYNONYMS_JSON>' --scopes "<SOURCE_SCOPES>" -s "<SHELF>"
hypatia statement-delete "<OLD_HEAD>" "<OLD_RELATION>" "<OLD_TAIL>" -s "<SHELF>"
```

Carry the source metadata across exactly, and omit any flag whose source value is empty — `--synonyms ""` is a hard parse error, not an empty value. `--synonyms` takes the positional form `{"head":[...],"relation":[...],"tail":[...]}`. `--scopes` is comma-separated and a trailing comma means global scope, so source scopes `["project-a",""]` are reproduced as `--scopes "project-a,"` and `[""]` as `--scopes ","`.

If the create fails, nothing has been lost: report it and leave the source triple untouched. If the create succeeds but the delete fails, stop and report both triples as present — the graph is consistent but now carries a duplicate relationship, and a person must decide.

5. Re-query each created triple and every intended removal with the same `$triple` form. A created triple must come back; a removed triple must return `No results found.` Note that `statement-delete` exits non-zero with `not found` when the triple is already gone; that is an already-applied removal rather than a failure, but confirm it by query before continuing. If any verification fails, stop, report the partial outcome and the backup location, and do not create a watermark.

After all intended changes verify, write one run marker in the same shelf:

```bash
hypatia knowledge-create "hypatia-dream-run-<UTC_COMPACT_TIMESTAMP>" \
  -d "schema: 1
mode: apply
processed_through: <RUN_STARTED_AT>
added: <COUNT>
replaced: <COUNT>
report: Hypatia Dream completed successfully" \
  -t "system,hypatia-dream-run" \
  -s "<SHELF>"
```

The marker is operational metadata, not semantic knowledge. Never attach a Statement triple to it or use it as relationship evidence.

A marker is still an ordinary Knowledge entry: it is full-text indexed and vector-embedded like any other, so it will surface in the user's own `hypatia search` and `hypatia similar` results. Keep the body to the fields above for that reason — never store report prose, analysis, or candidate lists in it. This skill never deletes an old marker; only the most recent one is ever read.

If no high-confidence graph changes exist, `apply` may still write the marker after reporting a verified no-op. This records that the interval has been reviewed without falsely claiming a graph mutation.

## Required Report

Return the report in this structure, in the user's language:

```markdown
# Hypatia Dream Report

## Scope
- Shelf: `<name>`
- Mode: `report` or `apply`
- Baseline: first pass, or `<processed_through>`
- Snapshot boundary: `<run_started_at>`
- Reviewed: `<new knowledge count>` Knowledge entries and `<new statement count>` new triples against `<total statement count>` existing triples
- Graph read: `<returned row count>` of `<limit>` rows, complete or truncated

## Relationship Findings
### Add
- `(head, relation, tail)` [high confidence]
  Evidence: ...
  Existing context: ...

### Replace
- `(old head, old relation, old tail)` -> `(new head, new relation, new tail)` [high confidence]
  Evidence: ...
  Rationale: ...

### No Change
- Item: ...
  Reason: duplicate, insufficient evidence, protected system relation, or metadata/temporal safety restriction.

## Outcome
- Planned or applied additions: `<count>`
- Planned or applied replacements: `<count>`
- Skipped candidates: `<count>`
- Backup: `<path>` or `not needed`
- Watermark: `<processed_through>` written only after a verified `apply`, otherwise `unchanged`
```

Be concise but include every command that changed state and every exact triple it affected. Do not claim a relationship was corrected unless its post-mutation query verified the result.
