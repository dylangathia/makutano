Keep the Makutano final-year project (video → forecast → reinforcement learning signal control in a calibrated SUMO simulation) and add an evidence-grounded "explain why" layer: every signal decision is logged with its evidence, an LLM agent answers operator questions only from those records, and a verifier rejects any answer whose numbers or claims don't match the logs.

## Who and what this is for

- The user is a 4th-year BSc Computer Science student at the University of Nairobi (2026/27). This is their final-year project. Supervisor: Elisha O. Abade.
- The user's stated bar for the project: "a final year project has to show competency in Computing principles and Fundamentals, not necessarily helping people, it just needs to be impressive."
- Proposals have not yet been presented. The user asked the court to set aside how much time has passed when judging this.

## The existing project (Makutano)

Title: "MAKUTANO: A Video-Driven Machine Learning Pipeline for Predictive Adaptive Signal Control at a Nairobi Intersection". Three modules:
1. **See:** a YOLO detector fine-tuned on local footage plus ByteTrack tracking turns video into per-approach arrivals and queues for six road-user classes (matatu, boda boda, private car, truck/bus, handcart, pedestrian). Counts are checked against manual counts of the same video.
2. **Predict:** a GRU forecaster predicts arrivals 1, 5 and 15 minutes ahead plus a time-of-day profile, compared with persistence, historical-average and gradient-boosted-tree baselines on an unseen recording day.
3. **Control:** a deep RL agent (PPO/DQN, Stable-Baselines3) behind a formal safety shield (minimum and maximum green, amber and all-red, maximum waits) controls the signals in a SUMO model calibrated against the footage. It is compared with today's fixed plan, Webster re-timing, actuated and rule-based control over repeated simulation runs, with an ablation with and without forecasts, and stress tests (heavy demand, surge, blocked lane, pedestrian surge, camera failure).

Current state: the pipeline code is written with automated tests (37 passing). Detection, RL training and the SUMO experiments have not run yet because SUMO, Gymnasium and Stable-Baselines3 are not installed, and no footage has been recorded. A live web console exists (Python FastAPI + WebSocket backend, browser frontend) showing a simulated junction with vehicles moving, signals switching through the safety shield, queues, forecasts and an event log of decisions. In the console, the "RL agent" and forecaster are currently stand-ins (a forecast-aware pressure rule; a time-of-day profile scaled by recent arrivals) and its simulator is a small built-in microsimulation standing in for SUMO.

## Where the new idea came from

The user looked at winners of an online "Open Agent Hackathon 2026" (GenAI.works). Winning projects shared a pattern: a narrow user and decision, every AI claim checked against evidence ("No evidence, no finding"; "Every count must match the source's own count"; "Every decision cites its evidence"), and showing the reasoning trail. Examples: Proven (AI code reviewer that must show evidence for each finding), GreenTrace (checks cocoa suppliers' deforestation-free claims and shows why), Fieldwork (answers plain-English questions about a city from its public records, with counts checked against the source), FocoCerto (helps a fire coordinator decide which wildfires to answer first).

## The proposed addition (Option A)

- **Evidence records:** every signal decision is logged with its evidence: the queues the camera measured, the forecast, the agent's choice, and which safety-shield rule fired, if any.
- **Question answering:** an operator asks in plain English, e.g. "Why did east–west get 55 s of green at 07:42?" or "Which approach waited longest this morning, and why?" An LLM agent answers only from those records.
- **Verifier:** checks every answer; each number must match the logged data and each claim must cite a record, or the answer is rejected.
- **Evaluation:** answer accuracy against ground truth from the logs, citation precision, and how often the verifier catches fabricated claims (including deliberately injected wrong answers).
- **Proposed one-line pitch:** "Makutano runs a Nairobi junction's signals with a reinforcement learning agent, and can prove why it made every decision."

## Alternative considered

Switching instead to a Kenyan version of a winner, e.g. satellite-based checks of whether Kenyan coffee/tea farm plots meet the EU Deforestation Regulation, with an agent explaining each verdict. That alternative is not the case on trial.
