# /status — Render the sprint dashboard

You are generating the current status dashboard for this Wheat sprint.

## Process

1. **Run the compiler**:

   ```bash
   npx @grainulation/wheat compile --summary
   ```

2. **Read compilation.json** for all dashboard data.

3. **Read git log** for recent activity:

   ```bash
   git log --oneline -20 claims.json
   ```

4. **Generate dashboard HTML**: Create/update `output/dashboard.html` — a self-contained dashboard page. The dashboard must show:

   **Header**: Sprint question, current phase, compilation status (ready/blocked), days since initiation

   **Phase Progress**: Visual progress through define -> research -> prototype -> evaluate -> compile. Show which phases have claims.

   **Evidence Strength by Topic**: For each topic in coverage, show:

   - Topic name
   - Number of claims
   - Highest evidence tier (with color coding: green=tested/production, amber=documented, red=web/stated)
   - Visual bar representing depth

   **Conflict Status**:

   - Resolved conflicts with winner/loser and reason
   - Unresolved conflicts highlighted as blockers
   - "Compilation readiness" indicator

   **Connected Sources**: List any connectors from meta

   **Recent Activity**: From git log — the sprint timeline

   **Claim Inventory**: Grouped by topic, showing type, evidence tier, and status for each claim

5. **Also print a text summary to the terminal** so the user gets immediate feedback without opening a file.

## Git commit

Only commit if the dashboard changed meaningfully. Don't commit for a status-only check.

Commit: `wheat: /status — updated dashboard`

## Tell the user

- Print the text summary (phase, claim counts, conflicts, readiness)
- Point them to `output/dashboard.html` for the full visual dashboard
- Suggest next actions based on what's needed (resolve conflicts, fill coverage gaps, etc.)

$ARGUMENTS
