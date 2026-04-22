# /merge — Combine Claim Sets Across Sprints

You are merging claims from another sprint into the current one. This is for when two teams researched the same problem independently and need to combine their findings.

## Process

1. **Parse the argument**: The user provides a path to another sprint's claims.json.

   - Example: `/merge ../auth-sprint/claims.json`
   - If no path given, ask for it.

2. **Validate both claim sets**:

   - Read the current `claims.json`
   - Read the incoming claims file
   - Validate both against the compiler schema:
     ```bash
     npx @grainulation/wheat compile --input <incoming-path> --output /tmp/wheat-merge-incoming.json
     ```

3. **Determine the sprint slug**: Derive from the incoming sprint's `meta.question`.

4. **Resolve ID collisions**: Prefix all incoming claim IDs with the sprint slug:

   - `r001` -> `auth-r001`
   - Also update all `conflicts_with` and `resolved_by` references.

5. **Align topics**: Present probable topic mappings for user confirmation.

6. **Detect cross-sprint conflicts** and **identify evidence upgrades**.

7. **Write merged claims.json** and compile:
   ```bash
   npx @grainulation/wheat compile --summary
   ```

## Git commit

Commit: `wheat: /merge <slug> — merged <N> claims from <source path>`

## Tell the user

- How many claims were merged
- Topic alignment results
- Cross-sprint conflicts detected
- Suggest: `/resolve` for conflicts, `/blind-spot` for cross-sprint gaps

## Cleanup

```bash
rm -f /tmp/wheat-merge-*.json
```

$ARGUMENTS
