# Rebuttal: Defense

**1. No model yet: survives on timing only.** The user called it "an issue for another day," and the stand-ins "are meant to be replaced by the trained agent and forecaster later." Running the trained model once it exists, and labelling any earlier run honestly, answers this.

**2. Unvalidated detector: partly survives.** No hand-count check is in the record, a real gap. But step 4 already requires simulated "today's plan" to reproduce the video's delays, so corrupted See output would likely fail that test, not stay invisible. A short hand count closes the rest.

**3. Replay builds in today's signal: partly survives.** Signal-shaped arrivals are a fair concern, and the record has no fix for it. The toy-simulator point is weaker: the record names a calibrated, validated SUMO model as the control environment, so the replay need not stay in the console. The undefined "roughly" tolerance should be set in advance.

**4. Thin reference timings: fails as fatal.** The record already treats the clip as "a demonstration rather than the main evaluation," with report numbers from "the multi-seed experiments." A failed check costs a demo, not the project. Whether the junction runs a fixed plan, and whether filming is permitted, is not in the record and should be confirmed first.

**Deciding point:** Point 2 against the step-4 check. If a "today's plan" match with a tolerance fixed in advance guards the See output, the demo is sound as a bounded add-on.
