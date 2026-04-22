# /prototype — Build something testable

You are building a proof-of-concept for the current Wheat sprint. Read CLAUDE.md for sprint context and claims.json for existing research claims.

## Process

1. **Determine what to prototype**: Based on the user's argument and existing research claims. If no argument given, look at the research and suggest the most promising option to test.

2. **Build it**: Create a working prototype in `prototypes/<name>/`. This should be:

   - Minimal — just enough to test the hypothesis
   - Runnable — include a README or run script
   - Measurable — produce output that can be evaluated

3. **Also generate a demo artifact**: Create `prototypes/<name>/demo.html` — a self-contained HTML page that shows what the prototype does, with screenshots, code snippets, or interactive elements. Non-technical stakeholders should be able to understand the prototype from this page alone.

## Claim updates

Every prototype finding becomes a claim with evidence tier `tested`:

```json
{
  "id": "p001",
  "type": "factual",
  "topic": "<what was tested>",
  "content": "<what we found — be specific with numbers>",
  "source": {
    "origin": "prototype",
    "artifact": "prototypes/<name>/",
    "connector": null
  },
  "evidence": "tested",
  "status": "active",
  "phase_added": "prototype",
  "timestamp": "<ISO timestamp>",
  "conflicts_with": [],
  "resolved_by": null,
  "tags": []
}
```

**Critical**: Check if any existing research claims (evidence: "web") are contradicted by prototype results. If so:

- Set `conflicts_with` on both claims
- The compiler will auto-resolve in favor of `tested` over `web`

Update `meta.phase` to "prototype" in claims.json if this is the first prototype.

## Run the compiler

```bash
npx @grainulation/wheat compile --summary
```

Report evidence upgrades (research claims superseded by prototype findings).

## Git commit

Stage claims.json and all new/changed files.

Commit: `wheat: /prototype "<name>" — added <claim IDs>`

## Tell the user

- Point them to the demo.html and the actual prototype code
- Summarize what was tested and what was found
- Highlight any research claims that were confirmed or contradicted
- Suggest: more `/prototype` for other options, `/evaluate` to compare, or `/status` to see progress

$ARGUMENTS
