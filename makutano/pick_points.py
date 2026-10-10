"""Click points on a video frame and print them as YAML.

Use it to read off pixel coordinates for ground control points, zones and stop
lines in config/site.yaml.

    python -m makutano.pick_points --video data/raw/pilot.mp4 --time 30

Left click adds a point, `u` undoes, `c` prints and clears (start the next
polygon), `q` prints and quits.
"""
import argparse

import cv2


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", required=True)
    p.add_argument("--time", type=float, default=0.0, help="seconds into the video")
    p.add_argument("--max-width", type=int, default=1600, help="downscale the window to fit the screen")
    args = p.parse_args()

    cap = cv2.VideoCapture(args.video)
    cap.set(cv2.CAP_PROP_POS_MSEC, args.time * 1000)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"could not read a frame at {args.time}s from {args.video}")

    scale = min(1.0, args.max_width / frame.shape[1])
    points = []

    def redraw():
        view = frame.copy()
        for i, (x, y) in enumerate(points):
            cv2.circle(view, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(view, str(i), (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        if len(points) > 1:
            for a, b in zip(points, points[1:]):
                cv2.line(view, a, b, (0, 255, 255), 1)
        cv2.imshow("pick_points", cv2.resize(view, None, fx=scale, fy=scale))

    def on_mouse(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Store full-resolution coordinates regardless of window scaling.
            points.append((round(x / scale), round(y / scale)))
            redraw()

    def dump():
        print("[" + ", ".join(f"[{x}, {y}]" for x, y in points) + "]", flush=True)

    cv2.namedWindow("pick_points")
    cv2.setMouseCallback("pick_points", on_mouse)
    redraw()
    while True:
        key = cv2.waitKey(50) & 0xFF
        if key == ord("u") and points:
            points.pop()
            redraw()
        elif key == ord("c"):
            dump()
            points.clear()
            redraw()
        elif key == ord("q"):
            dump()
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
