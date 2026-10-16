# What each file does

The code follows the project's three modules: **see**, **predict**, **control**.
A fourth package, **sim**, holds the SUMO tools they share. Each stage reads the
previous stage's output.

```
see:      pick_points ─► config ; video ─► detect_track ─► tracks.csv ─► measure ─► series.csv + volumes
predict:  series.csv ─► evaluate ─► forecaster.pkl + profile.csv
sim:      volumes ─► sumo_demand ─► routes ─► run_sim ─► compare   (calibrate and validate the SUMO junction)
control:  webster ─► comparison plans ; scenarios ─► training and stress routes ;
          env + shield ─► train ─► agent.zip ─► experiments ─► results
```

## Module 1: See (`makutano/see/`)

| File | What it does | Reads | Writes |
|---|---|---|---|
| [pick_points.py](makutano/see/pick_points.py) | Opens a video frame; you click points and it prints their pixel coordinates, for zones, stop lines and ground control points in the config | a video | coordinates in the terminal |
| [homography.py](makutano/see/homography.py) | Fits the mapping from image pixels to metres on the road from ground control points, and reports its error, including the leave-one-out error (the figure to trust) | config | error report |
| [detect_track.py](makutano/see/detect_track.py) | YOLO detection and ByteTrack tracking; one row per box per frame. Maps standard (COCO) classes to project classes for pre-trained weights | video, YOLO weights | `tracks.csv` |
| [measure.py](makutano/see/measure.py) | Cleans tracks and converts them to metres, then computes each vehicle's movement, delay, speed and arrival time, the queue on each approach, and the 5-second series of arrivals and queues per approach | `tracks.csv`, config | `movement_volumes.csv`, `approach_volumes.csv`, `delay.csv`, `speeds.csv`, `queues.csv`, `passes.csv`, `series.csv` |

## Module 2: Predict (`makutano/predict/`)

| File | What it does | Reads | Writes |
|---|---|---|---|
| [features.py](makutano/predict/features.py) | Turns series into forecasting examples (recent arrivals and queue, time of day; targets are arrivals in the next 1, 5 and 15 minutes). Also builds the single live example the controller uses, with the same code, so training and use can't drift apart. Splits data by recording day | `series.csv` | — |
| [models.py](makutano/predict/models.py) | The forecasters: persistence and historical average (baselines), gradient-boosted trees (scikit-learn), and a GRU (PyTorch). All share `fit` / `predict` | — | — |
| [evaluate.py](makutano/predict/evaluate.py) | Trains every forecaster, chooses on the validation day, scores on the unseen test day, and builds the time-of-day profile | series files, config | `metrics.csv`, `metrics_by_approach.csv`, `profile.csv`, `profile_metrics.csv`, `forecaster.pkl` |

## Module 3: Control (`makutano/control/`)

| File | What it does | Reads | Writes |
|---|---|---|---|
| [shield.py](makutano/control/shield.py) | The safety shield every controller goes through: minimum and maximum green, amber and all-red, and maximum waits for every approach and for pedestrians | — | — |
| [rule_based.py](makutano/control/rule_based.py) | Hand-designed controller: serves the phase with the biggest queues and longest waits. A comparison strategy and the fallback if RL fails | — | — |
| [webster.py](makutano/control/webster.py) | Re-times the fixed plan from measured flows (Webster's method) and writes the "webster" and "actuated" signal programs | network, volumes, config | `programs.add.xml` |
| [scenarios.py](makutano/control/scenarios.py) | Samples varied training days from the time-of-day profile, and applies scenarios: demand scale, surge, pedestrian scale (route changes), blocked lane and camera outage (during the run) | profile, volumes, config | training route files |
| [env.py](makutano/control/env.py) | The calibrated junction as a Gymnasium environment: camera-like observations (limited range, realistic noise, predictions), actions through the shield, reward = reduced waiting time | network, routes, forecaster | — |
| [train.py](makutano/control/train.py) | Trains the PPO or DQN agent with Stable-Baselines3, with or without predictions | training routes, forecaster | `agent.zip`, `agent.json`, training log |
| [experiments.py](makutano/control/experiments.py) | Runs every strategy in every scenario on the same evaluation seeds; reports delay, throughput, queue, pedestrian waiting and time to clear queues, plus paired differences with 95% confidence intervals | routes, programs, forecaster, agents | `results.csv`, `paired_differences.csv` |

## Shared SUMO tools (`makutano/sim/`)

| File | What it does | Reads | Writes |
|---|---|---|---|
| [sumo_demand.py](makutano/sim/sumo_demand.py) | Turns movement counts into a SUMO route file: one flow per movement, class and interval | `movement_volumes.csv`, config | `*.rou.xml` |
| [run_sim.py](makutano/sim/run_sim.py) | Runs SUMO over several random seeds and tabulates simulated indicators shaped like the measured ones | network, routes, config | `sim_volumes.csv`, `sim_delay.csv`, `sim_queues.csv` |
| [compare.py](makutano/sim/compare.py) | Scores simulation against measurement: GEH per movement, % error for delay and queue length, confidence intervals over seeds. This is the calibration and validation check | measured and simulated tables | `compare_*.csv` |

## Other files

| File | What it does |
|---|---|
| [makutano/config.py](makutano/config.py) | Loads the YAML config and checks approach names (they are encoded into SUMO vehicle ids, so no `__` or `.`) |
| [config/site.example.yaml](config/site.example.yaml) | Template for a junction: ground control points, zones, SUMO edges and vehicle types, plus the `predict`, `control` and `scenarios` sections. Every value is a placeholder |
| [tests/](tests/) | Synthetic checks per module (`test_see_sim.py`, `test_predict.py`, `test_control.py`); run with `python -m pytest`. No video, GPU or SUMO needed |
| [requirements.txt](requirements.txt) | Core packages, including scikit-learn for the gradient-boosting forecaster |
| [requirements-detect.txt](requirements-detect.txt) | Adds Ultralytics (and PyTorch) for detection; install where there's a GPU |
| [requirements-ml.txt](requirements-ml.txt) | Adds PyTorch, Gymnasium and Stable-Baselines3 for the GRU and the RL agent |
| [docs/Makutano_Proposal.docx](docs/Makutano_Proposal.docx) | The project proposal |
| [README.md](README.md) | Setup and the full workflow, module by module |
| [.gitignore](.gitignore) | Keeps the virtual environment, outputs, model weights and raw footage out of git |

## Folders created as you work

| Folder | Contents |
|---|---|
| `data/raw/` | Original video. Never commit it |
| `outputs/` | Tracks, measurements, forecasts and experiment results (ignored by git) |
| `sumo/` | Network, route files and signal programs; worth committing |
| `models/` | Trained agents (ignored by git) |
