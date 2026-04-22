# /feedback — Incorporate stakeholder input

You are processing feedback from stakeholders for this Wheat sprint. Feedback introduces new claims (usually constraints or direction changes) that may trigger re-evaluation.

Read CLAUDE.md and claims.json for context.

## Process

1. **Parse the feedback**: The user's argument contains stakeholder input. This could be:

   - A new constraint ("CTO says prioritize speed over cost")
   - A correction ("the compliance team says we need SOC2 Type II, not Type I")
   - A direction change ("skip the custom build option, focus on Auth0 vs Clerk")
   - A question ("what about latency in EU regions?")

2. **Create claims**: Each piece of feedback becomes a claim:

```json
{
  "id": "f001",
  "type": "constraint|feedback",
  "topic": "<relevant topic>",
  "content": "<the feedback, attributed>",
  "source": {
    "origin": "stakeholder",
    "artifact": null,
    "connector": null
  },
  "evidence": "stated",
  "status": "active",
  "phase_added": "feedback",
  "timestamp": "<ISO timestamp>",
  "conflicts_with": [],
  "resolved_by": null,
  "tags": ["<stakeholder name or role>"]
}
```

3. **Check for conflicts**: Does this feedback contradict existing claims? If a stakeholder says "budget is $10K max" but existing research shows a solution costing $15K, that's a conflict. Mark it.

4. **Run the compiler**:

   ```bash
   npx @grainulation/wheat compile --summary
   ```

5. **Update CLAUDE.md** if the feedback changes constraints or audience.

## Git commit

Stage claims.json, CLAUDE.md, and any changed files.

Commit: `wheat: /feedback "<summary>" — added <claim IDs>`

## Tell the user

- Confirm what feedback was captured as claims
- Show any new conflicts introduced
- If compilation is now blocked, explain what needs re-evaluation
- Suggest next steps: `/research` for new questions, `/evaluate` to re-test, `/brief` to recompile

$ARGUMENTS
