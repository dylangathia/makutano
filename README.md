# Makutano

A machine learning pipeline for one Nairobi intersection, in three modules:

1. **See:** a YOLO detector fine-tuned on local footage, plus tracking, turns video
   into arrivals and queues on each approach.
2. **Predict:** a forecaster predicts each approach's arrivals for the next
   1, 5 and 15 minutes, plus a time-of-day profile of the peak.
3. **Control:** a reinforcement learning agent runs the signals inside a SUMO
   model of the junction calibrated against the video, behind a safety shield.
   It is compared with fixed, re-timed, actuated and rule-based control, then
   stress-tested under heavy demand, surges, a blocked lane, a pedestrian surge
   and a camera failure.

```
                ┌──────────────── see ────────────────┐   ┌──── predict ────┐   ┌────────── control ──────────┐
video ──► detect_track ──► tracks.csv ──► measure ──► series.csv ──► evaluate ──► forecaster.pkl ──► train (RL) ──► experiments
                                             │                         │                    ▲             ▲
                                             ▼                         ▼ profile.csv        │             │
                                    movement_volumes.csv ──► sim: sumo_demand / run_sim / compare  (calibrate the SUMO junction)
                                                                       └──► scenarios ──► training & stress-test routes
```

## Layout

| Package | Module | What it holds |
|---|---|---|
| `makutano/see/` | 1. See | point picker, homography, detection and tracking, measurement (indicators and the per-approach series) |
| `makutano/predict/` | 2. Predict | feature building, forecasters (persistence, historical average, gradient boosting, GRU), evaluation |
| `makutano/control/` | 3. Control | safety shield, rule-based controller, Webster timing, scenarios, the RL environment, training, experiments |
| `makutano/sim/` | Shared | SUMO demand generation, runs over many seeds, calibration checks (GEH and % error) |

[FILES.md](FILES.md) describes every file.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt          # measurement, prediction baselines, SUMO tools, tests
pip install -r requirements-detect.txt   # detection: pulls in PyTorch; better on Colab
pip install -r requirements-ml.txt       # GRU and reinforcement learning
python -m pytest                         # synthetic checks, no video, GPU or SUMO needed
```

Install [SUMO](https://sumo.dlr.de/docs/Downloads.php) and set `SUMO_HOME`.
Copy `config/site.example.yaml` to `config/site.yaml`; every value in the
example is a placeholder.

## 1. See

```
python -m makutano.see.pick_points --video data/raw/day1_am.mp4 --time 30
python -m makutano.see.homography --config config/site.yaml
python -m makutano.see.detect_track --video data/raw/day1_am.mp4 --out outputs/day1_am/tracks.csv
python -m makutano.see.measure --config config/site.yaml --tracks outputs/day1_am/tracks.csv `
    --out outputs/day1_am --start 2026-10-20T07:00:00
```

- **Homography:** check the leave-one-out error, not the in-sample one. If it's
  above about 1 m, add or spread out the control points.
- **`--start`:** give the wall-clock time of the first frame. The predictor needs
  the time of day and the date.
- **Checking the output:** spot-check `passes.csv` against the video, and compare
  the counts with a manual count. That comparison is the first validation result.

## 2. Predict

Record on at least three different days: one is held out for choosing a model,
and one for the final test.

```
python -m makutano.predict.evaluate --config config/site.yaml --series "outputs/*/series.csv" --out outputs/predict [--gru]
```

This prints the test error of every forecaster at 1, 5 and 15 minutes, and
saves the one with the lowest validation error as `forecaster.pkl`. If none
beats persistence or the historical average, that is a finding, and the best
baseline is the one the controller uses.

## 3. Control

1. **Build and calibrate the SUMO junction.** Build the network from
   OpenStreetMap with netconvert, correct it in netedit, and clip it to the
   camera's view. Set the edge ids in the config. Then:

   ```
   python -m makutano.sim.sumo_demand --config config/site.yaml --volumes outputs/day1_am/movement_volumes.csv --out sumo/eval.rou.xml
   python -m makutano.sim.run_sim --config config/site.yaml --routes sumo/eval.rou.xml --out outputs/sim --seeds 10
   python -m makutano.sim.compare --config config/site.yaml --measured outputs/day1_am --sim outputs/sim
   ```

   Calibrate on the morning peak, then rerun `compare` on the evening peak with
   no further changes. That second result is the validation.

2. **Fill in the `control` section** of the config:
   - the traffic light id and its green phases (each followed by its amber),
   - the approaches each phase serves,
   - the shield limits,
   - the camera range and observation noise, taken from the measurement validation.

3. **Make the comparison plans and the training days:**

   ```
   python -m makutano.control.webster --config config/site.yaml --volumes outputs/day1_am/movement_volumes.csv --out sumo/programs.add.xml
   python -m makutano.control.scenarios --config config/site.yaml --profile outputs/predict/profile.csv `
       --volumes outputs/day1_am/movement_volumes.csv --out sumo/train --n 30
   ```

4. **Train both agents**, with and without predictions:

   ```
   python -m makutano.control.train --config config/site.yaml --routes "sumo/train/*.rou.xml" --forecaster outputs/predict/forecaster.pkl --out models/rl_pred
   python -m makutano.control.train --config config/site.yaml --routes "sumo/train/*.rou.xml" --no-prediction --out models/rl_nopred
   ```

   Watch `monitor.monitor.csv`: episode reward should rise and level off. Open
   a few episodes in `sumo-gui` to check the policy isn't exploiting a
   simulator quirk.

5. **Run the comparison and the stress tests:**

   ```
   python -m makutano.control.experiments --config config/site.yaml --routes sumo/eval.rou.xml `
       --programs sumo/programs.add.xml --forecaster outputs/predict/forecaster.pkl `
       --rl models/rl_pred/agent.zip --rl-nopred models/rl_nopred/agent.zip --out outputs/experiments
   ```

   `paired_differences.csv` answers the research questions directly:
   - **does RL beat the alternatives?** `rl` minus each other strategy,
   - **does prediction help?** `rl` minus `rl_nopred`,
   - **how do controllers cope under pressure?** the stress scenarios, including `clear_time_s`.

## Status

| Part | State |
|---|---|
| Measurement, series, forecasting baselines, gradient boosting, shield, rule-based controller, Webster timing, scenarios, experiment statistics | Covered by the synthetic tests |
| GRU forecaster | Written; needs PyTorch to run |
| RL environment, training, experiment runs | Written and compile-checked; need SUMO, Gymnasium and Stable-Baselines3, so first run them on the pilot model and fix what breaks |

## Known simplifications, to revisit

| Area | Current behaviour | Later |
|---|---|---|
| Ground position | Bottom-centre of the box | Correct for vehicle height / box geometry |
| Riders | Person boxes mostly overlapping a two-wheeler are dropped | Check on annotated data |
| Arrivals | First appearance in the approach's queue zone | Check the zone reaches beyond the longest queue |
| Pedestrians | Not counted from video yet | Crossing counts per crosswalk, to set SUMO pedestrian demand |
| Time-of-day profile | Historical average across training days | A learned profile model, if the data supports it |
| Signal states in SUMO | Set directly as state strings; all-red shows every signal red | Check against the junction's real phase plan |
| Stress results | Outside the calibrated range | Report as behaviour under pressure, not as exact delays |
| ID switches | A switch splits one vehicle into two partial passes | Measure their rate on annotated clips |
| Matatu stops | Not modelled | `busStop` additional file with dwell times measured from video |

## Data protection

Raw footage stays out of version control (`data/raw/` is ignored). Blur faces
and number plates before keeping or sharing any footage. The pipeline keeps
only boxes and tracks, never image crops, and reports only aggregate figures.
