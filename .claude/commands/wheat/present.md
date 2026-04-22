# /present — Generate a presentation from compiled claims

You are generating a presentation for this Wheat sprint. Same Bran compilation gate as /brief.

## Process

1. **Run the compiler with check**:

   ```bash
   npx @grainulation/wheat compile --check
   ```

   **If blocked, STOP.** Show errors, suggest fixes. Do not generate a presentation with unresolved conflicts.

2. **Read compilation.json** — use ONLY `resolved_claims`.

3. **Generate presentation HTML**: Create `output/presentation.html` — a self-contained, scroll-snap presentation using a dark theme.

   Structure the slides as:

   1. **Title slide**: Sprint question, date, audience
   2. **The Problem**: Why this research was needed (from constraint claims)
   3. **What We Found**: Key research findings (2-3 slides, from factual claims)
   4. **What We Tested**: Prototype and evaluation results (from tested claims)
   5. **Tradeoffs**: Risks and estimates
   6. **Recommendation**: The compiled recommendation with evidence
   7. **Next Steps**: Concrete actions
   8. **Appendix**: Compilation certificate, claim count, evidence summary

   Each slide should:

   - Be visually clean (use the dark theme, accent colors, cards)
   - Reference claim IDs subtly (small text at bottom of relevant sections)
   - Include data visualizations where relevant (CSS-only charts, comparison tables)
   - Work at any screen size (responsive)

## Key rules

- Same compilation gate as /brief — no presentation without passing compilation
- Cite claims, but more subtly than the brief (this is for presenting, not auditing)
- The presentation should tell a story: problem -> investigation -> evidence -> recommendation

## Git commit

Stage output/presentation.html.

Commit: `wheat: /present — generated presentation, certificate [hash]`

## Tell the user

- Presentation is at `output/presentation.html` — open in browser, works for screen share
- Mention the compilation certificate for reproducibility
- Suggest `/feedback` after they present to capture stakeholder responses

$ARGUMENTS
