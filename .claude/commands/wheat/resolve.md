# /resolve — Manually adjudicate a conflict

You are manually resolving a conflict between claims that the compiler couldn't auto-resolve (same evidence tier).

## Process

1. **Run the compiler** to see current conflicts:

   ```bash
   npx @grainulation/wheat compile --summary
   ```

2. **If the user specified claim IDs** (e.g., `/resolve r012 e003`), focus on that conflict. Otherwise, show all unresolved conflicts and ask which to resolve.

3. **Present both sides**: Show the conflicting claims with full context — content, evidence tier, source, when they were added.

4. **Ask the user to decide** (or decide based on additional research if the user asks you to investigate).

5. **Update claims.json**:

   - Set the winning claim's status to `active`
   - Set the losing claim's status to `superseded` and `resolved_by` to the winner's ID
   - Remove the conflict references from `conflicts_with`

6. **Run the compiler again**:

   ```bash
   npx @grainulation/wheat compile --summary
   ```

   Verify the conflict is resolved.

## Git commit

Stage claims.json.

Commit: `wheat: /resolve <winner> over <loser> — "<reason>"`

## Tell the user

- Confirm which claim won and why
- Show updated compilation status
- If all conflicts resolved, suggest `/brief` or `/present`

$ARGUMENTS
