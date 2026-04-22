# Wheat — Research Sprint

> This file configures Claude Code for Wheat research sprints. It is auto-maintained by `wheat init`. Edit with care.

## Sprint

**Question:** What is the architecture and design of Hypatia? Analyze its module structure, data flow, CLI design, storage patterns, and key design decisions.

**Audience:** engineers

**Constraints:**
- Rust codebase
- CLI tool for knowledge management

**Done looks like:** A comprehensive architecture overview with component descriptions, data flow diagrams, and design rationale

## Connectors

_No connectors configured. Use `/connect` to link org tools._

## Intent Router

When the user sends a plain message (no slash command), assess whether a Wheat command applies. If it does, announce which command you're running and execute it. If no command fits, respond normally.

**Route by intent:**

| User says something like...                                              | Route to                   | Why                         |
| ------------------------------------------------------------------------ | -------------------------- | --------------------------- |
| "look into X", "what about X", "explore X", "how does X work"            | `/research X`              | Information gathering       |
| "build X", "try X", "make a quick X", "test whether X"                   | `/prototype`               | Hands-on validation         |
| "is p001 really true?", "I doubt X", "what if X is wrong"                | `/challenge <id>`          | Adversarial testing         |
| "check this: <url>", "does <url> support X", "verify X against <url>"    | `/witness <id> <url>`      | External corroboration      |
| "what are we missing", "any gaps?", "what haven't we considered"         | `/blind-spot`              | Structural gap analysis     |
| "where are we", "what's the status", "show me the dashboard"             | `/status`                  | Sprint snapshot             |
| "write it up", "give me the recommendation", "summarize for the team"    | `/brief`                   | Decision document           |
| "make slides", "prepare for the meeting"                                 | `/present`                 | Stakeholder presentation    |
| "someone else is taking over", "hand this off", "document for successor" | `/handoff`                 | Knowledge transfer          |
| "combine with the other sprint", "merge these"                           | `/merge <path>`            | Cross-sprint merge          |
| "how did we get here", "show the history", "what changed over time"      | `/replay`                  | Sprint archaeology          |
| "we shipped, here's what happened", "actual results were X"              | `/calibrate --outcome "X"` | Prediction scoring          |
| "the stakeholder said X", "new constraint: X", "change of direction"     | `/feedback`                | Stakeholder input           |
| "resolve the conflict", "pick between X and Y"                           | `/resolve`                 | Conflict adjudication       |
| "connect to <repo/jira/docs>"                                            | `/connect <type> <target>` | External source linking     |
| "publish to confluence", "push to wiki", "sync to slack"                 | `/sync <target>`           | Artifact publishing         |
| "pull from deepwiki", "import from confluence", "backfill from repo"     | `/pull <source>`           | External knowledge backfill |

**When NOT to route:** Questions about the framework itself ("how does the compiler work?"), code edits to wheat files, general conversation, ambiguous intent. When in doubt, ask: "That sounds like it could be a `/research` -- want me to run it as a full research pass, or just answer the question?"

**Announce the routing:** Always tell the user what you're doing:

> Running as `/research "SSE scalability"` -- this will create claims and compile. Say "just answer" if you wanted a quick response instead.

This gives the user a chance to redirect before the full pipeline runs.

### Claims System (Bran IR)

- All findings are tracked as typed claims in `claims.json`
- Every slash command that produces findings MUST append claims
- Every slash command that produces output artifacts MUST run `wheat compile` first
- Output artifacts consume `compilation.json`, never `claims.json` directly
- The compiler is the enforcement layer -- if it says blocked, no artifact gets produced

### Claim Types

- `constraint` -- hard requirements, non-negotiable boundaries
- `factual` -- verifiable statements about the world
- `estimate` -- projections, approximations, ranges
- `risk` -- potential failure modes, concerns
- `recommendation` -- proposed courses of action
- `feedback` -- stakeholder input, opinions, direction changes

### Evidence Tiers (lowest to highest)

1. `stated` -- stakeholder said it, no verification
2. `web` -- found online, not independently verified
3. `documented` -- in source code, official docs, or ADRs
4. `tested` -- verified via prototype or benchmark
5. `production` -- measured from live production systems

### Claim ID Prefixes

- `d###` -- define phase (from /init)
- `r###` -- research phase (from /research)
- `p###` -- prototype phase (from /prototype)
- `e###` -- evaluate phase (from /evaluate)
- `f###` -- feedback phase (from /feedback)
- `x###` -- challenge claims (from /challenge)
- `w###` -- witness claims (from /witness)
- `burn-###` -- synthetic claims (from /control-burn, always reverted)
- `cal###` -- calibration claims (from /calibrate)
- `<sprint-slug>-<prefix>###` -- merged claims keep original prefix with sprint slug (from /merge)

### Next Command Hints

Every slash command MUST end its output with a "Next steps" section suggesting 2-4 concrete commands the user could run next, based on the current sprint state. Use this decision tree:

- Unresolved conflicts exist -> suggest `/resolve`
- Claim has no corroboration -> suggest `/witness <id> <relevant-url>`
- Topic has weak evidence -> suggest `/research <topic>` or `/prototype`
- Topic has type monoculture -> suggest `/challenge <id>` or `/research <topic>`
- Sprint is late-phase with gaps -> suggest `/blind-spot`
- Claims untested against reality -> suggest `/calibrate`
- Sprint ready for output -> suggest `/brief`, `/present`, or `/handoff`
- Sprint ready and external wiki configured -> suggest `/sync confluence`
- Need external context -> suggest `/pull deepwiki <repo>` or `/pull confluence`
- Multiple sprints exist -> suggest `/merge`
- Want to understand history -> suggest `/replay`

Format:

```
Next steps:
  /challenge p001     -- stress-test the zero-deps claim
  /witness r002 <url> -- corroborate fs.watch reliability
  /blind-spot         -- check for structural gaps
```

### Git Discipline

- Every slash command that modifies claims.json auto-commits
- Commit format: `wheat: /<command> <summary> -- added/updated <claim IDs>`
- `git log --oneline claims.json` = the sprint event log
- Compilation certificate references the claims hash for reproducibility

### Output Artifacts

- HTML files are self-contained (inline CSS/JS, no external deps)
- Use the dark scroll-snap template for explainers and presentations
- Use the dashboard template for status and comparisons
- PDFs generated via `node build-pdf.js <file.md>`

### Directory Structure

- `research/` -- topic explainers (HTML + MD)
- `prototypes/` -- working proof-of-concepts
- `evidence/` -- evaluation results and comparison dashboards
- `output/` -- compiled artifacts (briefs, presentations, dashboards)

### MCP Server Troubleshooting

- When the Wheat MCP server disconnects or fails to connect, do NOT retry more than twice
- Instead, immediately report the failure and suggest the user run: `claude mcp add wheat -- npx -y -p @grainulation/wheat wheat-mcp`
- If any MCP server is unresponsive after 2 attempts, fall back to direct file operations rather than looping on reconnection
- Run `/grainulator:healthcheck` to verify all servers before starting a session
