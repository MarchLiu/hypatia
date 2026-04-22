# /research — Deep dive on a topic

You are researching a topic for the current Wheat sprint. Read CLAUDE.md for sprint context and claims.json for existing claims.

## Process

1. **Understand the request**: The user's argument tells you what to research. Could be a technology, a comparison, a question, a process.

2. **Research deeply**: Use web search, read documentation, check connected repos (see Connectors in CLAUDE.md). Be thorough — this is the foundation for later decisions.

3. **Extract claims**: Every finding becomes a typed claim. Be specific and verifiable. Bad: "Auth0 is popular." Good: "Auth0 serves 15,000+ customers as of 2025."

4. **Detect conflicts with existing claims**: Check claims.json. If your new findings contradict existing claims, set `conflicts_with` on both the new and existing claim.

## Adding claims

Append claims to claims.json with IDs continuing the `r###` sequence (check existing claims for the next number). Each claim:

```json
{
  "id": "r001",
  "type": "factual|estimate|risk|recommendation",
  "topic": "<topic category>",
  "content": "<specific, verifiable finding>",
  "source": {
    "origin": "research",
    "artifact": "research/<topic-slug>.md",
    "connector": null
  },
  "evidence": "web",
  "status": "active",
  "phase_added": "research",
  "timestamp": "<ISO timestamp>",
  "conflicts_with": [],
  "resolved_by": null,
  "tags": ["<relevant tags>"]
}
```

If the finding came from a connector (GitHub repo, Jira, etc.), set evidence to "documented" and fill in the connector field.

## Run the compiler

```bash
npx @grainulation/wheat compile --summary
```

Check for new conflicts introduced. Report them to the user.

## Generate HTML explainer

Create `research/<topic-slug>.html` — a self-contained HTML explainer using the dark scroll-snap template style. This should be:

- Beautiful and presentable (stakeholders will see this)
- Organized into logical sections (scroll-snap slides)
- Include key findings, comparisons, tradeoffs
- Reference claim IDs so findings are traceable

Also create `research/<topic-slug>.md` as the structured markdown source.

## Git commit

Stage claims.json and all new files.

Commit: `wheat: /research "<topic>" — added <claim IDs>`

## Tell the user

- Point them to the HTML file to open in browser
- Summarize key findings (3-5 bullets)
- Flag any conflicts with existing claims
- Suggest next steps: more `/research`, `/prototype`, or `/evaluate`

$ARGUMENTS
