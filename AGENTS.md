# Wheat Research Sprint

**Question:** What is the architecture and design of Hypatia? Analyze its module structure, data flow, CLI design, storage patterns, and key design decisions.

**Audience:** engineers

**Constraints:**
- Rust codebase
- CLI tool for knowledge management

**Done looks like:** A comprehensive architecture overview with component descriptions, data flow diagrams, and design rationale

## Claims System

All findings are tracked as typed claims in `claims.json`. Claim types: constraint, factual, estimate, risk, recommendation, feedback. Evidence tiers (low to high): stated, web, documented, tested, production.

## Key Commands

- `wheat init` — bootstrap a research sprint
- `wheat compile` — validate and compile claims
- `wheat status` — sprint health dashboard
- `wheat search <query>` — search claims
- `wheat add-claim` — add a new claim
- `wheat resolve <id>` — resolve a conflicting claim
