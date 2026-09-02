---
name: hypatia-dream
description: "Run a deliberate, incremental Hypatia graph-consolidation pass at the end of a work period, like sleep-time knowledge organization. Trigger only for an explicit batch closeout or sleep-time request, not ordinary knowledge CRUD or a one-off triple. With no explicit --mode, treat these Chinese closeout signals as apply: '下班了', '收工了', '今天先到这', '本阶段结束', '阶段性收尾', '做个好梦', and '睡前整理一下'. Treat these as report-only: '先整理看看', '做个梦看看', '睡前审阅一下', '看看知识之间的关系', '只出报告', and '先别修改'. Analyze Statement triples as primary evidence, read Knowledge bodies only for disambiguation, then report or safely apply high-confidence relationship additions and corrections."
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
2. Without an explicit mode, treat these closeout signals and clear equivalents as `apply`: `下班了`, `收工了`, `今天先到这`, `本阶段结束`, `阶段性收尾`, `做个好梦`, and `睡前整理一下`.
3. Without an explicit mode, treat these preview signals and clear equivalents as `report`: `先整理看看`, `做个梦看看`, `睡前审阅一下`, `看看知识之间的关系`, `只出报告`, and `先别修改`.
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
2. Set `SHELF` to the requested shelf or `default`.
3. Record a UTC run-start timestamp before reading graph data. Format it as `YYYY-MM-DD HH:MM:SS`, which compares correctly with Hypatia's `created_at` values.
4. Query the latest successful apply marker in the same shelf:

```bash
hypatia query '{"$knowledge":[["$contains","tags","hypatia-dream-run"]],"limit":1}' -s "<SHELF>"
```

A marker is a system-tagged Knowledge entry created by this skill after a successful `apply`. Its content contains a `processed_through` timestamp. Read that value, not the marker entry's `created_at`: a run can create statements after its snapshot begins, and using the marker creation time could skip concurrent new information.

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

Before a replacement, inspect the old statement's `content`, `tr_start`, and `tr_end`. Do not apply a replacement when `tr_start` or `tr_end` is populated: the CLI cannot recreate temporal bounds. Also do not apply a replacement if the original statement metadata cannot be faithfully preserved through `statement-create`; report it instead.

## Apply Changes

In `report` mode, stop after producing the report. Do not write a marker.

In `apply` mode, execute this sequence only after the analysis identifies high-confidence changes:

1. Re-query every proposed old and new triple immediately before mutating. Skip stale plans, existing additions, and replacements whose source triple no longer matches the reviewed evidence.
2. Before the first destructive deletion, export the shelf to a timestamped backup directory and report the destination. Use `hypatia export`; create the destination directory explicitly when needed.
3. For a replacement, delete only the exact reviewed source triple, then create the exact reviewed replacement. Preserve `data`, positional `synonyms`, and `scopes` from the source statement when they apply. Never delete first if that metadata or a temporal bound cannot be preserved.
4. For an addition, create only the exact reviewed triple. Do not create supporting nodes or generic fallback links.
5. Re-query each created triple and every intended removal. If verification fails, stop, report the partial outcome and backup location, and do not create a watermark.

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
