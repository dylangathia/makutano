# What each file does

The pipeline runs top to bottom: each stage reads the previous stage's output.

```
pick_points ─► config/site.yaml
                     │
video ─► detect_track ─► tracks.csv ─► indicators ─► measured tables ─► compare
                                                          │                ▲
                                                          ▼                │
                                                    sumo_demand ─► routes ─► run_sim
```

## Pipeline code (`makutano/`)

| File | What it does | Reads | Writes |
|---|---|---|---|
| [pick_points.py](makutano/pick_points.py) | Opens a video frame; you click points and it prints their pixel coordinates. Used to fill in zones, stop lines and ground control points in the config. | a video | coordinates printed to the terminal |
| [homography.py](makutano/homography.py) | Fits the mapping from image pixels to metres on the road, using ground control points (lat/lon from Google Earth, or metres). Run on its own, it reports the fit's error, including the leave-one-out error, which is the figure to trust. | `config/site.yaml` | error report in the terminal |
| [detect_track.py](makutano/detect_track.py) | Runs YOLO detection and ByteTrack tracking on a video. Writes one row per box per frame. With pre-trained weights it maps standard (COCO) classes onto project classes; with fine-tuned weights it uses their own class names. Needs a GPU for long videos. | video, YOLO weights | `tracks.csv` |
| [indicators.py](makutano/indicators.py) | Turns tracks into intersection indicators. Cleans tracks (fixes flickering classes, removes riders counted as pedestrians, drops very short tracks), converts positions to metres, computes speeds, then works out each vehicle's movement, delay and the queue on each approach. | `tracks.csv`, config | `movement_volumes.csv`, `approach_volumes.csv`, `delay.csv`, `speeds.csv`, `queues.csv`, `passes.csv` |
| [sumo_demand.py](makutano/sumo_demand.py) | Turns measured movement counts into a SUMO route file: one flow per movement, class and interval. Vehicle types come from the config. `--scale` changes demand for scenarios. | `movement_volumes.csv`, config | `*.rou.xml` |
| [run_sim.py](makutano/run_sim.py) | Runs SUMO over several random seeds and turns its outputs into tables shaped like the measured ones. `--additional` adds files such as an alternative signal plan for a scenario. | network, routes, config | `sim_volumes.csv`, `sim_delay.csv`, `sim_queues.csv`, raw SUMO XML |
| [compare.py](makutano/compare.py) | Scores the simulation against the measurements: GEH per movement for volumes, percentage error for delay and queue length, and 95% confidence intervals over seeds. This is the calibration and validation check. | measured and simulated tables | `compare_volumes.csv`, `compare_delay.csv`, `compare_queues.csv`, a summary in the terminal |
| [config.py](makutano/config.py) | Loads the YAML config and checks approach names (they are encoded into SUMO vehicle ids, so they must not contain `__` or `.`). | config | — |
| [\_\_init\_\_.py](makutano/__init__.py) | Marks `makutano/` as a Python package so modules run with `python -m makutano.<module>`. | — | — |

## Configuration

| File | What it does |
|---|---|
| [config/site.example.yaml](config/site.example.yaml) | Template for describing a junction: ground control points, study zone, each approach's entry/exit/queue zones and stop line, the SUMO edge ids, indicator settings (interval length, free-flow speeds, queue threshold) and SUMO vehicle types. Every value is a placeholder. Copy it to `config/site.yaml` and fill in your own. The `sumo.vtypes` section is what calibration adjusts. |

## Tests

| File | What it does |
|---|---|
| [tests/test_pipeline.py](tests/test_pipeline.py) | Checks on synthetic data that run without a GPU, video or SUMO. They cover homography recovery, point-in-polygon tests, rider removal, indicator values for a car at a known speed and a queued car at a known distance, route generation, parsing of SUMO outputs, and GEH. Run with `python -m pytest`. |

## Project files

| File | What it does |
|---|---|
| [README.md](README.md) | Setup, a step-by-step pilot run, what the pilot should tell you, and known simplifications to revisit. |
| [requirements.txt](requirements.txt) | Packages for everything except detection: NumPy, pandas, OpenCV, PyYAML, pytest. |
| [requirements-detect.txt](requirements-detect.txt) | Adds Ultralytics (which pulls in PyTorch) for detection and tracking. Install it where there is a GPU, for example Colab. |
| [pyproject.toml](pyproject.toml) | Project metadata, and pytest settings so tests can import `makutano`. |
| [.gitignore](.gitignore) | Keeps the virtual environment, outputs, model weights and raw footage (`data/raw/`) out of git. Raw footage shows identifiable people and number plates. |
| FILES.md | This file. |

## Folders created as you work

| Folder | Contents |
|---|---|
| `data/raw/` | Original video. Never commit it. |
| `outputs/` | Tracks, indicator tables and simulation results for each run. |
| `sumo/` | The junction network (`.net.xml`), route files and signal-plan files. These are worth committing once they exist; only `outputs/` and `data/raw/` are ignored. |
