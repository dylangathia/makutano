## Charges that stuck

1. **It explains a base that has never run.** No footage, no trained RL agent and no SUMO run exist, so today the layer would explain a stand-in pressure rule. All 12 jurors leaned on this.
2. **The pitch overclaims.** "Can prove why it made every decision" goes beyond what logs of inputs, outputs and shield firings can show, and the defense conceded this. 10 jurors cited it in their reasons.
3. **The LLM side has no plan.** The case names no model, cost, offline fallback or labelling effort. 5 jurors raised it.

## What the defense got right

1. The evidence fields (measured queues, forecast, choice, shield rule fired) already exist in the pipeline and console, so the layer reuses data instead of starting a new field.
2. The evaluation can be tested: ground truth comes from the logs, and the verifier's catch rate can be measured. The case is still silent on how non-numeric claims get checked.

## Conditions for acquittal

1. **Write the order of work into the proposal:** the explain layer starts only after the trained RL agent has produced results in calibrated SUMO (all 12 jurors).
2. **Reword the pitch** to something like "shows the logged evidence behind every decision" in a simulated junction, dropping "prove why" and "runs a Nairobi junction" (all 12).
3. **Name the LLM, its cost or offline fallback, and a verifier test set written by someone other than you** (6 jurors across these asks).

## The sentence

The court finds the idea sound but the case overstated and out of order. This week, install SUMO, Gymnasium and Stable-Baselines3 and get the core running. Revise the proposal so the explain layer comes last, under an honest pitch, with its model and test set named. Then bring it back.
