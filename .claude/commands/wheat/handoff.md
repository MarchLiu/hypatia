# /handoff — Package Sprint for Transfer

You are generating a self-contained briefing optimized for a **successor** — someone who needs to continue this sprint, not a stakeholder making a decision. Read CLAUDE.md, claims.json, and compilation.json.

## Key distinction from other output commands

| Command    | Audience           | Optimized for                         |
| ---------- | ------------------ | ------------------------------------- |
| `/brief`   | Decision-makers    | "What should we do?"                  |
| `/present` | External audiences | Persuasion                            |
| `/status`  | Current researcher | Snapshot                              |
| `/handoff` | Successor          | "What do I need to know to continue?" |

## Process

1. **Run the compiler**:

   ```bash
   npx @grainulation/wheat compile --summary
   ```

2. **Read all data sources**:

   - `compilation.json` — current state
   - `claims.json` — all claims including superseded ones (the full history)
   - `git log --oneline claims.json` — the event log
   - `CLAUDE.md` — sprint context and conventions

3. **Build the reasoning chain**: For each topic, reconstruct the narrative:

   - What constraint or question initiated work on this topic?
   - What did research find?
   - Did prototyping confirm or contradict research?
   - Were there conflicts? How were they resolved?

4. **Identify open questions**: From compilation.json:

   - Unresolved conflicts
   - Coverage gaps
   - Unmitigated risks
   - Dismissed blind spots

5. **Generate the handoff document**: Create `output/handoff.md` and `output/handoff.html`.

6. **Print a summary** to the terminal.

## Git commit

Commit: `wheat: /handoff — generated sprint handoff document`

## Tell the user

- Point them to `output/handoff.md` and `output/handoff.html`
- Highlight the most important open questions
- Suggest: `/replay` for detailed timeline, `/blind-spot` for gap analysis

$ARGUMENTS
