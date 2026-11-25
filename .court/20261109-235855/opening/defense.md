# Opening: Defense

The charge says the case should not go ahead as described. It should. The addition is small, sits on top of what already exists, and meets the user's own stated standard.

1. **It meets the stated bar, which is depth, not impact.** The user wrote that the project "has to show competency in Computing principles and Fundamentals ... it just needs to be impressive." Option A puts a fourth discipline on top of vision, forecasting and RL: grounded LLM question answering with a formal verifier. The obvious objection is "scope creep". But the bar is impressiveness, and verified explanation of an RL controller is the kind of systems integration that bar is asking for.

2. **It builds on existing infrastructure; it does not replace any.** The console already has "an event log of decisions", and the safety shield already reports "which safety-shield rule fired". The evidence records ("queues the camera measured, the forecast, the agent's choice") are fields the pipeline already computes. The verifier checks against data the system writes itself, so no new data collection is needed.

3. **It can be tested even if the hard parts slip.** The case admits that detection, RL training and SUMO "have not run yet" and that no footage exists. That is the strongest argument *for* Option A. The log-and-verify layer runs against the stand-in console today, and its evaluation ("answer accuracy against ground truth from the logs, citation precision, ... deliberately injected wrong answers") needs no footage. It hedges the project's biggest risk.

4. **The evaluation is concrete and falsifiable.** Ground truth comes straight from the logs, and injected fabrications measure the verifier's catch rate directly. This matches the existing project's discipline of baselines, ablations and stress tests. It is not a demo-only feature.

5. **The pattern has been shown to work elsewhere.** Hackathon winners shared "every AI claim checked against evidence" and "Every decision cites its evidence". The pitch ("can prove why it made every decision") gives a clear, defensible thesis.

**Concessions:** The record does not say how much time is left, so the time pressure from adding a fourth module cannot be ruled out. The "prove" in the pitch overstates what the verifier does: it checks consistency with the logs, not that the decisions were correct. It should be reworded.
