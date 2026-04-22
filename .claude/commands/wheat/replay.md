# /replay — Time-Travel Through Sprint Evolution

You are reconstructing the historical evolution of this sprint by recompiling every version of claims.json from git history.

## Process

1. **Get the git history of claims.json**:

   ```bash
   git log --oneline claims.json
   ```

   This gives every commit that touched claims.json — the sprint event log.

2. **Extract each historical version**: For each commit hash:

   ```bash
   git show <hash>:claims.json > /tmp/wheat-replay-<N>.json
   ```

3. **Recompile each version** with the current compiler:

   ```bash
   npx @grainulation/wheat compile --input /tmp/wheat-replay-<N>.json --output /tmp/wheat-comp-<N>.json
   ```

4. **Compute deltas** between consecutive compilations:

   ```bash
   npx @grainulation/wheat compile --diff /tmp/wheat-comp-<N-1>.json /tmp/wheat-comp-<N>.json
   ```

5. **Identify interesting moments** in each delta:

   - Phase transitions (define -> research -> prototype -> evaluate)
   - First time compilation went "ready"
   - Peak conflict count
   - Evidence tier jumps (topic going web -> tested)
   - Claims added then superseded (the sprint changed its mind)

6. **Generate replay HTML**: Create `output/replay.html` — a self-contained timeline visualization using a dark scroll-snap template. Include:

   - Frame-by-frame scrubbing (each commit = one frame)
   - Highlighted pivotal moments
   - Coverage evolution chart
   - Summary statistics per frame

7. **Print a text summary** to the terminal with the key narrative moments.

## Git commit

Commit: `wheat: /replay — generated sprint timeline (N frames)`

## Tell the user

- How many frames (commits) were found
- The most interesting moments
- Point them to `output/replay.html` for the full interactive timeline
- Suggest: `/handoff` to package this narrative for a successor

## Cleanup

Remove temporary files from /tmp after generating the output:

```bash
rm -f /tmp/wheat-replay-*.json /tmp/wheat-comp-*.json
```

$ARGUMENTS
