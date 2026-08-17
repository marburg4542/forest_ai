"""Per-tree metrics: DBH, height, crown dimensions, and their quality flags."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull

from .fitting import ransac_circle, principal_axis


def _basis(axis):
    """Two unit vectors spanning the plane perpendicular to `axis`."""
    a = axis / np.linalg.norm(axis)
    tmp = np.array([0.0, 0.0, 1.0]) if abs(a[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = np.cross(a, tmp)
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(a, e1)


def _stem_axis(seed):
    """Unit vector along the stem, from the stacked circle centres."""
    chain = seed["chain"]
    if len(chain) < 2:
        return np.array([0.0, 0.0, 1.0]), np.array([seed["x"], seed["y"], chain[0]["h"]])
    ctr = np.array([[c["cx"], c["cy"], c["h"]] for c in chain])
    return principal_axis(ctr), ctr.mean(axis=0)


def measure_dbh(pts, h, seed, cfg, rng):
    """Fit a circle on a slice taken perpendicular to the stem axis.

    Selecting the slice by normalised height keeps breast height exact, while
    projecting onto the plane perpendicular to the axis removes the 1/cos(lean)
    inflation a leaning stem would otherwise get on a horizontal cut.
    """
    axis, origin = _stem_axis(seed)
    e1, e2 = _basis(axis)
    lean = float(np.degrees(np.arccos(np.clip(abs(axis[2]), 0, 1))))
    half = cfg.dbh_slice_thickness / 2

    for hz in cfg.dbh_fallback_heights:
        m = np.abs(h - hz) <= half
        if m.sum() < 12:
            continue
        q = pts[m] - origin
        uv = np.column_stack([q @ e1, q @ e2])
        # Keep only points close to the stem axis itself.  The label for a tree
        # legitimately contains understorey foliage and, where segmentation
        # leaked, bits of a neighbour; anchoring on the axis (which the circle
        # stack already located) rather than on the centroid of the label is
        # what stops the fit latching onto a foliage clump.
        lim = max(0.25, 2.5 * seed["stack_diameter"])
        uv = uv[np.hypot(uv[:, 0], uv[:, 1]) < lim]
        if len(uv) < 12:
            continue
        fit = ransac_circle(uv, tol=cfg.ransac_tol, iters=cfg.ransac_iters,
                            min_r=cfg.stem_min_diameter / 2,
                            max_r=min(lim, cfg.stem_max_diameter / 2), rng=rng)
        if fit is None or fit["rmse"] > cfg.stem_max_rmse * 1.5:
            continue
        return {
            "dbh_cm": 200 * fit["r"], "dbh_height_used": hz,
            "dbh_rmse_cm": 100 * fit["rmse"], "dbh_arc": fit["arc"],
            "dbh_n_pts": fit["n"], "stem_lean_deg": lean, "dbh_source": "axis_fit",
            "fit_u": fit["cx"], "fit_v": fit["cy"],
            "dbh_vs_stack": 2 * fit["r"] / max(seed["stack_diameter"], 1e-6),
        }

    return {
        "dbh_cm": 100 * seed["stack_diameter"], "dbh_height_used": np.nan,
        "dbh_rmse_cm": 100 * seed["stack_rmse"], "dbh_arc": seed["stack_arc"],
        "dbh_n_pts": 0, "stem_lean_deg": lean, "dbh_source": "stack_interp",
        "fit_u": np.nan, "fit_v": np.nan, "dbh_vs_stack": 1.0,
    }


def _crown(pts, h, seed, dbh_cm):
    """Crown base height, diameter, projected area and hull volume."""
    out = {"crown_base_m": np.nan, "crown_diameter_m": np.nan,
           "crown_area_m2": np.nan, "crown_volume_m3": np.nan}
    if len(pts) < 30:
        return out
    cx, cy = seed["x"], seed["y"]
    rad = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)

    # crown base: lowest 0.5 m layer whose 90th-percentile radius clearly
    # exceeds the stem itself
    thresh = max(0.75, 4 * dbh_cm / 100)
    edges = np.arange(1.0, np.nanmax(h), 0.5)
    base = np.nan
    for lo in edges:
        m = (h >= lo) & (h < lo + 0.5)
        if m.sum() < 20:
            continue
        if np.percentile(rad[m], 90) > thresh:
            base = float(lo)
            break
    out["crown_base_m"] = base

    top = pts[h >= base] if np.isfinite(base) else pts
    if len(top) >= 30:
        r = np.hypot(top[:, 0] - cx, top[:, 1] - cy)
        out["crown_diameter_m"] = float(2 * np.percentile(r, 95))
        try:
            out["crown_area_m2"] = float(ConvexHull(top[:, :2]).volume)
            out["crown_volume_m3"] = float(ConvexHull(top).volume)
        except Exception:
            pass
    return out


def measure_trees(xyz, h, labels, seeds, cfg, verbose=True):
    """Build the tree table.  One row per detected stem."""
    rng = np.random.default_rng(cfg.random_seed)
    n_trees = len(seeds)
    order = np.argsort(labels, kind="stable")
    ls = labels[order]
    lo = np.searchsorted(ls, np.arange(n_trees), "left")
    hi = np.searchsorted(ls, np.arange(n_trees), "right")

    rows = []
    for tid in range(n_trees):
        sel = order[lo[tid]:hi[tid]]
        seed = seeds[tid]
        pts, hh = xyz[sel].astype(np.float64), h[sel].astype(np.float64)
        row = {"tree_id": tid, "x": seed["x"], "y": seed["y"],
               "n_points": len(sel), "n_stem_layers": seed["n_layers"]}
        if len(sel) >= 20:
            row["height_m"] = float(np.nanpercentile(hh, cfg.height_percentile))
            row["height_max_m"] = float(np.nanmax(hh))
        else:
            row["height_m"] = row["height_max_m"] = np.nan
        row.update(measure_dbh(pts, hh, seed, cfg, rng))
        row.update(_crown(pts, hh, seed, row["dbh_cm"]))
        rows.append(row)

    df = pd.DataFrame(rows)
    df["basal_area_m2"] = np.pi * (df["dbh_cm"] / 200) ** 2
    df["h_d_ratio"] = df["height_m"] / (df["dbh_cm"] / 100)

    # a single confidence label makes the table usable without reading five
    # separate quality columns
    # Arc coverage is the decisive flag on a single-scan cloud.  Below roughly
    # a third of the circumference a circle fit is free to slide outwards along
    # the arc, which is exactly how a 14 cm stem ends up reported at 45 cm, and
    # the residual stays small the whole time - so residual alone cannot catch
    # it.  The independent circle-stack diameter is used as a cross-check.
    fitted = df["dbh_source"] == "axis_fit"
    consistent = df["dbh_vs_stack"].between(1 / 1.6, 1.6)
    tall_enough = df["height_m"] >= cfg.min_tree_height
    good = (fitted & (df["dbh_arc"] >= 0.60) & (df["dbh_rmse_cm"] <= 2.0)
            & (df["n_stem_layers"] >= 6) & consistent & tall_enough)
    fair = (fitted & (df["dbh_arc"] >= 0.35) & (df["dbh_rmse_cm"] <= 3.0)
            & df["dbh_vs_stack"].between(1 / 2.2, 2.2) & tall_enough)
    df["quality"] = np.where(good, "good", np.where(fair, "fair", "poor"))

    if verbose:
        print(f"    measured {len(df)} trees | quality "
              + " ".join(f"{k}={v}" for k, v in df["quality"].value_counts().items()))
    return df
