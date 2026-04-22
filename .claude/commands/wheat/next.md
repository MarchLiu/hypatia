# /next -- Route next steps through Farmer

You just finished a slash command or a batch of work. The user monitors via Farmer on their phone and cannot see CLI text output.

Use AskUserQuestion to present what just happened and what comes next. This is the ONLY way the user sees results.

## Process

1. **Summarize** what was just produced in the question text (1-2 sentences -- claim counts, key findings, what changed)

2. **Pick 2-4 next steps** based on sprint state. Use this decision tree:

   - Unresolved conflicts exist -> suggest `/resolve`
   - Claim has no corroboration -> suggest `/witness <id> <url>`
   - Topic has weak evidence -> suggest `/research <topic>` or `/prototype`
   - Sprint is late-phase with gaps -> suggest `/blind-spot`
   - Sprint ready for output -> suggest `/brief`, `/present`, or `/handoff`
   - Multiple sprints exist -> suggest `/merge`

3. **Call AskUserQuestion** with the summary as the question and next steps as options. Keep option labels short (3-5 words), put detail in descriptions.

4. **Act on the user's choice** -- run whatever they picked.

## Rules

- Header should be contextual (e.g., "After /brief", "After research")
- Don't include a "do nothing" option unless the user might genuinely want to stop
- If agents just completed, mention how many and what they did
- multiSelect: false unless the options are independent parallel tasks
