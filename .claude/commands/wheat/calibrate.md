# /calibrate — Score Past Predictions Against Reality

You are checking what actually happened after a sprint's recommendations were implemented. This closes the feedback loop by comparing predictions to outcomes.

## Process

1. **Parse the outcome**: The user provides outcome data, either as:

   - Free text: `/calibrate --outcome "Shipped Auth0. Took 3 weeks not 2. Costs $18K/year not $15K."`
   - Claim-specific: `/calibrate e003 "actual: 3 weeks, $18K/year"`

2. **Read the sprint data** and match outcomes to original predictions.

3. **Create calibration claims** (`cal###` prefix, evidence: `production`).

4. **Compute accuracy scorecard** by evidence tier, source origin, and claim type.

5. **Write/update calibration.json** and add claims to claims.json:

   ```bash
   npx @grainulation/wheat compile --summary
   ```

6. **Print the scorecard** to the terminal.

## The meta-insight

This is the only command that validates the framework itself. If `tested` claims are right 95% of the time and `web` 65%, the tier system works.

## Git commit

Commit: `wheat: /calibrate — scored <N> predictions against outcomes`

## Tell the user

- The accuracy scorecard
- Which predictions were wrong and by how much
- Whether the evidence tier hierarchy is predictive
- Suggest: future sprints should weight evidence tiers based on this data

$ARGUMENTS
