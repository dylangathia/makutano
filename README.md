# Makutano

A digital twin of a Nairobi traffic intersection, built from ordinary video:
detection and tracking, then intersection indicators, then a calibrated SUMO
model, then signal-plan scenarios.

This is the rough end-to-end pipeline. Every stage runs, using pre-trained
models and simple methods, so the full chain can be tested on a pilot clip
before any annotation or fine-tuning.

```
video ──► detect_track ──► tracks.csv ──► indicators ──► measured tables ──┬──► compare ◄── run_sim ◄── sumo_demand
          (YOLO+ByteTrack)               (homography,                       │       (GEH, %error)    (SUMO, N seeds)
                                          zones)                            └──► dashboard (later)
```

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt          # indicators, SUMO tools, tests
pip install -r requirements-detect.txt   # detection: pulls in PyTorch; better on Colab
python -m pytest                         # synthetic checks, no video or SUMO needed
```

Install [SUMO](https://sumo.dlr.de/docs/Downloads.php) and set `SUMO_HOME`.

## Pilot run (one 10–30 minute clip)

1. **Configure the site.** Copy `config/site.example.yaml` to `config/site.yaml`.
   Use the point picker to read pixel coordinates off a frame for ground control
   points, the study zone, each approach's zones, and stop lines:

   ```
   python -m makutano.pick_points --video data/raw/pilot.mp4 --time 30
   ```

   Find the same ground control points on Google Earth and enter them as
   `[lat, lon]`.

2. **Check the homography.** Look at the leave-one-out error, not the
   in-sample one. If it is more than about 1 m, add points or spread them
   out, especially toward the far edge of the area you measure.

   ```
   python -m makutano.homography --config config/site.yaml
   ```

3. **Detect and track** (on a GPU if you can):

   ```
   python -m makutano.detect_track --video data/raw/pilot.mp4 --out outputs/pilot/tracks.csv --max-seconds 600
   ```

4. **Compute indicators:**

   ```
   python -m makutano.indicators --config config/site.yaml --tracks outputs/pilot/tracks.csv --out outputs/pilot
   ```

   Open `passes.csv` next to the video and spot-check a few dozen vehicles. Also
   do a manual count of one approach for one interval. That comparison is the
   first validation result.

5. **Build the SUMO network.** Export the junction area from OpenStreetMap and
   convert it, then fix the geometry by hand in `netedit` against the video.
   Clip it to roughly the camera's study zone so simulated delay is measured
   over the same area as video delay.

   ```
   netconvert --osm-files sumo/junction.osm --output-file sumo/junction.net.xml --geometry.remove --junctions.join --tls.guess-signals
   ```

   Put the edge ids into each approach's `sumo_from_edge` / `sumo_to_edge`.
   Write the signal timings you **observed** on video (not the posted plan)
   into a `<tlLogic>` additional file, and list it under `sumo.additional_files`.

6. **Generate demand, simulate, compare:**

   ```
   python -m makutano.sumo_demand --config config/site.yaml --volumes outputs/pilot/movement_volumes.csv --out sumo/pilot.rou.xml
   python -m makutano.run_sim --config config/site.yaml --routes sumo/pilot.rou.xml --out outputs/pilot/sim --seeds 10
   python -m makutano.compare --config config/site.yaml --measured outputs/pilot --sim outputs/pilot/sim
   ```

## What the pilot should tell you

- Whether the camera position sees every approach and the back of its queues.
  If queues regularly run out of frame, queue length is censored. Fix that by
  changing the vantage point or adding a camera now, before full data collection.
- Which classes the pre-trained detector misses or confuses, so annotation
  effort goes where it matters (matatus come out as `bus_matatu` or
  `private_car` until fine-tuning).
- Whether the signals run a regular cycle or are being overridden by hand,
  which decides what the SUMO baseline has to represent.

## Known simplifications, to revisit

| Area | Current behaviour | Later |
|---|---|---|
| Ground position | Bottom-centre of the box | Correct for vehicle height / box geometry |
| Riders | Person boxes mostly overlapping a two-wheeler are dropped | Check on annotated data |
| Pedestrians | Excluded from vehicle indicators | Crossing counts and waiting time for the pedestrian-priority scenario |
| Queue length | Distance to the furthest slow vehicle in the queue zone | Add half a vehicle length; flag intervals where the queue reaches the zone edge |
| ID switches | A switch splits one vehicle into two partial passes | Measure their rate on annotated clips, and their effect on counts |
| Simulated volumes | Binned by SUMO departure time | Bin both sides by stop-line crossing |
| Matatu stops | Not modelled | `busStop` additional file with dwell times measured from video |

## Data protection

Raw footage stays out of version control (`data/raw/` is ignored). Blur faces
and number plates before keeping or sharing any footage. The pipeline keeps
only boxes and tracks, never image crops.
