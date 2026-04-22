# /evaluate — Test claims against reality, resolve conflicts

You are running the evaluation phase of the current Wheat sprint. This is the honesty phase — where claims meet data.

Read CLAUDE.md for sprint context and claims.json for all existing claims.

## Process

1. **Run the compiler first**:

   ```bash
   npx @grainulation/wheat compile --summary
   ```

   Identify conflicts, weak evidence areas, and coverage gaps.

2. **Evaluate systematically**: For each topic with weak or conflicting evidence:

   - Run benchmarks, cost calculations, feature comparisons
   - Test prototypes against real conditions
   - Pull production metrics from connected tools if available
   - Cross-reference claims against each other

3. **Resolve conflicts**: When evaluation produces a clear answer:

   - Update the winning claim's evidence tier
   - Mark the losing claim as `superseded` with `resolved_by`
   - Add new evaluation claims (evidence: "tested") if needed

4. **Generate comparison dashboard**: Create `evidence/<topic-or-sprint-slug>.html` — a dashboard-style HTML page showing:
   - Side-by-side comparisons with real numbers
   - Conflict resolutions with evidence
   - Data tables, charts (CSS-only), metrics

## Claim updates

Evaluation claims use evidence tier `tested` or `production`:

```json
{
  "id": "e001",
  "type": "factual",
  "topic": "<what was evaluated>",
  "content": "<measured result — always include numbers>",
  "source": {
    "origin": "evaluation",
    "artifact": "evidence/<slug>.html",
    "connector": null
  },
  "evidence": "tested",
  "status": "active",
  "phase_added": "evaluate",
  "timestamp": "<ISO timestamp>",
  "conflicts_with": [],
  "resolved_by": null,
  "tags": []
}
```

Update `meta.phase` to "evaluate" in claims.json.

## Run the compiler again

```bash
npx @grainulation/wheat compile --summary
```

Verify that conflicts are resolved. If the compiler still shows blockers, tell the user what remains.

## Git commit

Stage claims.json and all new/changed files.

Commit: `wheat: /evaluate — resolved <conflict details>, added <claim IDs>`

## Tell the user

- Show the compilation status (ready or still blocked)
- Summarize what was evaluated and key findings
- Point them to the comparison dashboard HTML
- If ready: suggest `/brief` to compile the decision document
- If blocked: explain what conflicts remain and how to resolve them

$ARGUMENTS
