# /init — Bootstrap a Wheat research sprint

You are initializing a new Wheat research sprint. Have a focused conversation with the user to establish:

1. **What are we figuring out?** Get a clear, specific question. Not "should we use X" but "should we use X given constraints Y and Z for audience A."
2. **Who is the audience?** Who needs to be convinced or informed? (engineering leads, CTO, product, finance, etc.)
3. **What constraints exist?** Budget, timeline, tech stack, compliance, team size, existing infrastructure.
4. **What does done look like?** What artifact ends this sprint? A recommendation? A prototype? A go/no-go decision?

Once you have answers:

## Step 1: Update CLAUDE.md

Update the Sprint section of CLAUDE.md with the question, audience, constraints, and success criteria.

## Step 2: Seed claims.json

Update claims.json with:

- `meta.question` — the sprint question
- `meta.initiated` — today's date (ISO format)
- `meta.audience` — array of audience labels
- `meta.phase` — set to "define"
- `meta.connectors` — empty array

Add constraint claims (type: "constraint") for each hard requirement identified. Use IDs starting with `d001`. Each claim needs:

```json
{
  "id": "d001",
  "type": "constraint",
  "topic": "<relevant topic>",
  "content": "<the constraint>",
  "source": { "origin": "stakeholder", "artifact": null, "connector": null },
  "evidence": "stated",
  "status": "active",
  "phase_added": "define",
  "timestamp": "<ISO timestamp>",
  "conflicts_with": [],
  "resolved_by": null,
  "tags": []
}
```

## Step 3: Run the compiler

```bash
npx @grainulation/wheat compile --summary
```

Verify compilation succeeds.

## Step 4: Generate problem statement

Generate `output/problem-statement.html` — a clean, self-contained HTML page summarizing the sprint question, audience, constraints, and success criteria. Use the dark theme from the explainer template in `templates/`. Keep it to a single page — this is the "here's what we're investigating" artifact that can be shared immediately.

## Step 5: Git commit

Stage claims.json, CLAUDE.md, output/problem-statement.html, and any other changed files.

Commit with message: `wheat: /init "<sprint question short>" — seeded <claim IDs>`

## Step 6: Tell the user what's next

Tell them:

- Their problem statement is in `output/problem-statement.html` — they can open it in a browser and share it
- Next step: `/research <topic>` to start exploring, or `/connect` to link org tools
- Remind them that `/status` shows the dashboard at any time

$ARGUMENTS
