"""Image-to-ground homography fitted from ground control points.

Run as a script to check how good the fit is before trusting any speed or
queue-length number:

    python -m makutano.homography --config config/site.yaml
"""
import argparse

import cv2
import numpy as np

from .config import load_config

EARTH_RADIUS_M = 6_371_000.0


def latlon_to_local(latlon, origin=None):
    """Project [lat, lon] pairs to local east/north metres.

    Equirectangular projection around `origin` (default: centroid). The error is
    negligible over the few hundred metres of a junction.
    """
    latlon = np.asarray(latlon, dtype=float)
    lat0, lon0 = latlon.mean(axis=0) if origin is None else origin
    x = np.radians(latlon[:, 1] - lon0) * EARTH_RADIUS_M * np.cos(np.radians(lat0))
    y = np.radians(latlon[:, 0] - lat0) * EARTH_RADIUS_M
    return np.column_stack([x, y])


def ground_points(h_cfg):
    pts = h_cfg["ground_points"]
    if h_cfg.get("ground_units", "metres") == "latlon":
        return latlon_to_local(pts, h_cfg.get("origin"))
    return np.asarray(pts, dtype=float)


def fit_homography(image_pts, ground_pts):
    src = np.asarray(image_pts, dtype=np.float64)
    dst = np.asarray(ground_pts, dtype=np.float64)
    if len(src) < 4 or len(src) != len(dst):
        raise ValueError("need at least 4 matching image/ground point pairs")
    H, _ = cv2.findHomography(src, dst, 0)
    if H is None:
        raise ValueError("homography fit failed: are three or more points collinear?")
    return H


def to_ground(H, pts):
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, H).reshape(-1, 2)


def homography_from_config(cfg):
    h = cfg["homography"]
    return fit_homography(h["image_points"], ground_points(h))


def reprojection_errors(H, image_pts, ground_pts):
    return np.linalg.norm(to_ground(H, image_pts) - np.asarray(ground_pts), axis=1)


def leave_one_out_errors(image_pts, ground_pts):
    """Error at each point when it is left out of the fit.

    This is the honest accuracy figure: in-sample error always looks better.
    """
    image_pts = np.asarray(image_pts, dtype=float)
    ground_pts = np.asarray(ground_pts, dtype=float)
    if len(image_pts) < 5:
        return None
    errors = []
    for i in range(len(image_pts)):
        keep = np.arange(len(image_pts)) != i
        H = fit_homography(image_pts[keep], ground_pts[keep])
        errors.append(np.linalg.norm(to_ground(H, image_pts[i:i + 1])[0] - ground_pts[i]))
    return np.array(errors)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    args = p.parse_args()

    h = load_config(args.config)["homography"]
    img, gnd = np.asarray(h["image_points"], float), ground_points(h)
    H = fit_homography(img, gnd)
    np.set_printoptions(precision=6, suppress=True)
    print("H =\n", H)
    errs = reprojection_errors(H, img, gnd)
    for i, e in enumerate(errs):
        print(f"  point {i}: in-sample error {e:.2f} m")
    print(f"in-sample RMSE: {np.sqrt(np.mean(errs ** 2)):.2f} m")
    loo = leave_one_out_errors(img, gnd)
    if loo is None:
        print("add a 5th point or more to get a leave-one-out error")
    else:
        print(f"leave-one-out RMSE: {np.sqrt(np.mean(loo ** 2)):.2f} m (worst point {loo.argmax()}: {loo.max():.2f} m)")


if __name__ == "__main__":
    main()
