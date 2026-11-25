Record about 20 minutes of real video at a Nairobi junction, then have the project's model run the traffic and signal control on that same traffic, and see the difference against how the junction actually performed.

## Context the user has given

- This is the user's 4th-year BSc Computer Science final-year project at the University of Nairobi (2026/27). Final presentation is due 24 April 2027. Supervisor: Elisha O. Abade.
- Project title: "MAKUTANO: A Video-Driven Machine Learning Pipeline for Predictive Adaptive Signal Control at a Nairobi Intersection". It has three modules:
  1. **See:** a YOLO detector fine-tuned on local footage plus ByteTrack tracking turns video into per-approach arrivals and queues (vehicle classes include car, matatu, boda boda, bus, truck, tuk-tuk, pedestrian). A homography maps pixels to metres.
  2. **Predict:** an LSTM/GRU forecasts arrivals for the next 1, 5 and 15 minutes plus a time-of-day profile, compared against persistence, historical-average and gradient-boosted-tree baselines, split by recording day.
  3. **Control:** a deep RL agent (PPO/DQN with Stable-Baselines3) behind a safety shield (minimum/maximum green, amber and all-red, maximum waits), trained in a SUMO model of the junction calibrated on the morning peak and validated on the evening peak. It is compared with today's fixed plan, a Webster re-timed plan, actuated control and a rule-based controller, with an ablation with and without predictions, and stress tests (heavy demand, surge, blocked lane, pedestrian surge, camera failure). The planned evaluation runs every strategy over many random seeds with paired differences and 95% confidence intervals.
- Current state of the code: the pipeline code is written. The detection, RL training and experiments have not run yet, because SUMO, Gymnasium and Stable-Baselines3 are not installed. No footage has been recorded yet.
- A "live console" was just built: a Python (FastAPI + WebSocket) backend with a small built-in traffic microsimulation standing in for SUMO, and a browser frontend showing cars moving, signals switching automatically through the project's safety shield, queues, forecasts and run comparisons. In it, the "RL agent" is currently a stand-in rule (a forecast-aware pressure rule) and the forecaster is a stand-in (time-of-day profile scaled by recent arrivals). They are meant to be replaced by the trained agent and forecaster later.

## The idea

The user said: "what i'd like to do, is record, say for 20 minutes, and then have this model do the car and traffic control and see the difference, but that's an issue for another day."

The suggested way to do it, as discussed in the conversation:
1. Record about 20 minutes from a high, fixed camera position covering all approaches, and note the real signal timings (for example with a stopwatch).
2. Run the clip through the See module to get each vehicle's arrival time, approach and movement, plus measured queues and delays.
3. Add a "recorded" scenario to the live console that replays those exact arrivals instead of random demand.
4. Run "today's plan" with the measured timings on those arrivals and check it reproduces roughly the delays the video showed; then run the agent on the same arrivals and compare.

Points already raised in the conversation:
- One 20-minute clip is a single sample, so it would be a demonstration rather than the main evaluation; the report's numbers would still come from the multi-seed experiments.
- If the simulated "today's plan" does not roughly match the delays measured on video, any improvement shown would not be credible.
