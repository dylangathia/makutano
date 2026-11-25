# Rebuttal: Defense

**1. Base never run: survives only as a sequencing point, not as a reason to reject.** The logging layer works on any controller: the console already produces "an event log of decisions" from stand-ins, so the layer can be built and tested now and pointed at the trained agent once it exists. The real risk is the order of work. Nothing in the record says the layer has to come before the core results.

**2. "Prove why": partly survives.** Shield firings are causal and can be logged. The learned policy's reasons are not. The fix is to reword the pitch ("can show the evidence behind every decision"). That is a wording change, not a flaw in the design.

**3. Verifier is circular: mostly fails.** Checking answers against the logs is exactly the stated goal: answers grounded in the records, not in the truth of the world. Camera accuracy is already checked separately "against manual counts of the same video." Two parts stand: the record doesn't say how non-numeric claims are verified, and self-injected errors are a weak test. Both can be fixed with a held-out or third-party error set.

**4. Spread too thin: survives in part.** The record has no LLM model, cost or offline plan, and that gap is real. But grounding, verification and measuring citation precision are computing fundamentals that fit the user's "impressive" bar. The layer also reuses data the pipeline already produces instead of starting a new field from scratch.

**The deciding point:** point 1, read as a question of order. If the jury accepts that the layer is built only after the core RL and SUMO results run, the remaining objections can be fixed by rewording and adding scope details.
