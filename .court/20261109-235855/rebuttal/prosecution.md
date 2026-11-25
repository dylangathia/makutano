# Rebuttal: Prosecution

1. **Fails.** Impressive depth comes from modules that actually run. Here three of them have not: no footage, no SUMO, no RL training. Adding a fourth spreads the work thinner. And checking an answer's numbers against the logs is consistency matching. The record gives no reason to call that a "formal verifier".

2. **Survives in part.** The fields do exist. But the console's "agent" is a stand-in pressure rule, so today the logs record a heuristic, not an RL policy. The records hold the inputs and the choice. They do not hold the policy's reasons.

3. **Fails.** A rule-based stand-in is easy to explain by design. If the RL agent never runs, Option A ends up explaining a heuristic. The thesis "an RL agent... can prove why" is then never tested. That does not hedge the risk. It changes the subject.

4. **Survives narrowly.** The metrics can be measured. But they score the LLM and the verifier, not the signal control. The ground truth is logs the system writes about itself, and the same author writes the injected errors. The record does not say how anyone outside would judge whether the explanations are good.

5. **Fails.** Hackathon judging is not the same as assessment of a final-year degree project. The record has nothing showing that the supervisor or the department values this pattern.

**Concessions:** The defense admits "prove" overstates what the verifier does. That agrees with the charge that the case fails *as stated*.

**The deciding point:** Can logs of inputs and outputs explain *why* an RL policy acted? If they cannot, the verifier only certifies that answers match the logs. It does not certify explanations, and the case's central claim fails as described.
