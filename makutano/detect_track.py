"""Detect and track road users in a video, writing one row per box per frame.

    python -m makutano.detect_track --video data/raw/pilot.mp4 --out outputs/pilot/tracks.csv

With the default pre-trained COCO weights, COCO classes are mapped onto the
project's classes. COCO has no matatu class, so matatus come out as
`bus_matatu` (when detected as buses) or `private_car`; separating them is what
fine-tuning is for. Fine-tuned weights are used with their own class names.

Needs `pip install -r requirements-detect.txt`; run it on a GPU (Colab) for
anything longer than a short clip.
"""
import argparse
from pathlib import Path

import cv2
import pandas as pd

TRACK_COLUMNS = ["frame", "t_s", "track_id", "cls", "conf", "x1", "y1", "x2", "y2"]

COCO_TO_LOCAL = {
    0: "pedestrian",
    1: "bicycle",
    2: "private_car",
    3: "boda_boda",
    5: "bus_matatu",
    7: "truck",
}


def class_map_for(model):
    names = model.names
    if names.get(2) == "car" and names.get(3) == "motorcycle":
        return COCO_TO_LOCAL
    return dict(names)


def video_fps(video):
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if not fps or fps <= 0:
        raise ValueError(f"could not read frame rate from {video}")
    return fps


def track_video(video, weights, conf=0.25, imgsz=1280, vid_stride=1, tracker="bytetrack.yaml",
                device=None, max_seconds=None):
    from ultralytics import YOLO

    fps = video_fps(video)
    model = YOLO(weights)
    class_map = class_map_for(model)
    results = model.track(
        source=str(video), stream=True, persist=True, tracker=tracker, conf=conf,
        imgsz=imgsz, vid_stride=vid_stride, classes=sorted(class_map), device=device,
        verbose=False,
    )
    rows = []
    for i, r in enumerate(results):
        frame = i * vid_stride
        t = frame / fps
        if max_seconds is not None and t > max_seconds:
            break
        if i % 500 == 0:
            print(f"  frame {frame} ({t:.0f}s)", flush=True)
        boxes = r.boxes
        if boxes is None or boxes.id is None:
            continue
        xyxy = boxes.xyxy.cpu().numpy()
        ids = boxes.id.int().cpu().numpy()
        classes = boxes.cls.int().cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        for (x1, y1, x2, y2), tid, c, p in zip(xyxy, ids, classes, confs):
            rows.append((frame, t, int(tid), class_map[int(c)], float(p), x1, y1, x2, y2))
    return pd.DataFrame(rows, columns=TRACK_COLUMNS)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", required=True)
    p.add_argument("--out", required=True, help="tracks CSV to write")
    p.add_argument("--weights", default="yolo11s.pt", help="pre-trained or fine-tuned YOLO weights")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--imgsz", type=int, default=1280, help="large sizes help with small, distant objects")
    p.add_argument("--vid-stride", type=int, default=1, help="process every Nth frame (2 halves the work)")
    p.add_argument("--tracker", default="bytetrack.yaml")
    p.add_argument("--device", default=None, help="e.g. 0 for the first GPU, cpu")
    p.add_argument("--max-seconds", type=float, default=None, help="stop after this much video")
    args = p.parse_args()

    df = track_video(args.video, args.weights, args.conf, args.imgsz, args.vid_stride,
                     args.tracker, args.device, args.max_seconds)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"wrote {len(df)} boxes, {df['track_id'].nunique()} tracks to {out}")


if __name__ == "__main__":
    main()
