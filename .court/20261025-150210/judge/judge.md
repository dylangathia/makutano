## Charges that stuck

1. **There is no model to test yet.** The "agent" in the console is a hand-written stand-in rule, so a "model vs. real junction" difference would misstate what was tested (10 jurors).
2. **"Roughly" has no number.** The step-4 check that simulated "today's plan" matches the video has no tolerance, so no gap can be told apart from replay error (6 jurors in their reasons; 10 asked for a fixed tolerance).
3. **The detector has never run and is unchecked.** The same See output supplies both the demand and the "actual" performance, with no hand count to catch its errors (6 jurors).

## What the defense got right

1. It would be the pipeline's first real end-to-end run. Even the prosecution called this the defense's best point.
2. The record already treats the clip as a demonstration, not the main evaluation, so a failed check costs a demo and not the project.

## Conditions for acquittal

1. Rewrite the plan so the agent-versus-today comparison runs only with the trained agent in the calibrated SUMO model, and label any earlier run as "stand-in rule". This week, install SUMO, Gymnasium and Stable-Baselines3 so that path exists.
2. Write down a numeric match tolerance for "today's plan" now, before any footage is recorded, and report a gap only if it is larger than that tolerance.
3. Add a hand-count check of See against a few minutes of the clip, log arrivals upstream of the queue, and confirm filming permission and the junction's real control mode.

## The sentence

Record the footage and run it through See. That part is sound. Do not show a "difference" until the real agent exists and the yardstick is fixed in advance. Unblock the core experiments first.
