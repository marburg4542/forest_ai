"""The pipeline as one callable, shared by the CLI and the web app.

Keeping this in one place matters: if the web layer re-implemented the stage
order or the area calculation, the two would drift and the numbers on screen
would stop matching the numbers in the CSV.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from . import las_io, preprocess, ground, segment, measure, features, qc
from .config import Config

Progress = Callable[[float, str], None]


@dataclass
class Result:
    cfg: Config
    las_path: str
    header: dict
    xyz: np.ndarray
    h: np.ndarray
    ok: np.ndarray
    dtm: np.ndarray
    valid: np.ndarray
    org: np.ndarray
    res: float
    seeds: list
    labels: np.ndarray
    df: pd.DataFrame
    feat: pd.DataFrame
    dtm_area_ha: float
    stand_area_ha: float
    hull: np.ndarray | None
    scan_centre: np.ndarray
    log: list = field(default_factory=list)

    @property
    def usable(self) -> pd.DataFrame:
        return self.df[self.df["quality"].isin(["good", "fair"])]

    @property
    def good(self) -> pd.DataFrame:
        return self.df[self.df["quality"] == "good"]


def cache_key(las_path: str, cfg: Config) -> str:
    """Identity of the file plus every setting that changes the DTM."""
    st = os.stat(las_path)
    key = "|".join(str(v) for v in [
        os.path.abspath(las_path), st.st_size, int(st.st_mtime),
        cfg.noise_voxel, cfg.noise_min_pts, cfg.ground_voxel, cfg.csf_cloth_res,
        cfg.csf_rigidness, cfg.csf_time_step, cfg.csf_iterations,
        cfg.csf_threshold, cfg.dtm_res])
    return hashlib.sha1(key.encode()).hexdigest()[:16]


def stand_area(df: pd.DataFrame, quantile=0.5):
    """Stocked area: convex hull of the stems buffered by half a crown radius.

    Per-hectare figures must be divided by the area that actually carries
    trees.  On this dataset the valid DTM also covers open ground beside the
    stand, and using it would understate density and basal area by 3x.
    """
    from scipy.spatial import ConvexHull

    pts = df.loc[df["quality"] != "poor", ["x", "y"]].values
    if len(pts) < 3:
        pts = df[["x", "y"]].values
    buf = float(np.nanquantile(df["crown_diameter_m"], quantile)) / 4
    buf = buf if np.isfinite(buf) else 1.5
    try:
        hull = ConvexHull(pts)
        poly = pts[hull.vertices]
        per = np.linalg.norm(np.diff(np.vstack([poly, poly[:1]]), axis=0), axis=1).sum()
        return (hull.volume + per * buf + np.pi * buf ** 2) / 10000, poly
    except Exception:
        lo, hi = pts.min(0), pts.max(0)
        return float((hi[0] - lo[0]) * (hi[1] - lo[1])) / 10000, None


def scan_centre(xyz, res=2.0):
    """XY of the densest cell - on a single-setup TLS this is the scanner."""
    lo = xyz[:, :2].min(0)
    idx = ((xyz[:, :2] - lo) / res).astype(np.int64)
    ny = idx[:, 1].max() + 1
    c = np.bincount(idx[:, 0] * ny + idx[:, 1])
    k = int(c.argmax())
    return lo + (np.array([k // ny, k % ny]) + 0.5) * res


def run(las_path: str, cfg: Config | None = None, cache_dir=".cache",
        force=False, groups=4, progress: Progress | None = None,
        verbose=True) -> Result:
    """Run every stage.  `progress(fraction, message)` is called between stages."""
    cfg = cfg or Config()
    log: list[str] = []

    def step(frac, msg):
        log.append(msg)
        if verbose:
            print(f"[{frac*100:3.0f}%] {msg}")
        if progress:
            progress(frac, msg)

    os.makedirs(cache_dir, exist_ok=True)
    gcache = os.path.join(cache_dir, f"ground_{cache_key(las_path, cfg)}.npz")
    header = las_io.describe(las_path)

    step(0.02, "reading point cloud")
    X = las_io.read_xyz(las_path)
    step(0.10, f"removing isolated points from {len(X):,}")
    X, _ = preprocess.remove_isolated(X, cfg.noise_voxel, cfg.noise_min_pts)
    bounds = (X[:, :2].min(0), X[:, :2].max(0))

    if os.path.exists(gcache) and not force:
        step(0.18, "loading cached ground model")
        d = np.load(gcache)
        dtm, valid, org, res = d["dtm"], d["valid"], d["org"], float(d["res"])
    else:
        step(0.18, "classifying ground (cloth simulation)")
        D = preprocess.voxel_downsample(X, cfg.ground_voxel)
        gmask, _, _ = ground.csf_ground(
            D, cfg.csf_cloth_res, cfg.csf_rigidness, cfg.csf_time_step,
            cfg.csf_iterations, cfg.csf_threshold, verbose=verbose)
        step(0.28, f"building DTM from {gmask.sum():,} ground points")
        dtm, valid, org, res = ground.build_dtm(D[gmask], cfg.dtm_res, bounds=bounds)
        np.savez_compressed(gcache, dtm=dtm, valid=valid, org=org, res=res)
        del D, gmask

    step(0.33, "normalising height above ground")
    h, ok = ground.normalize_height(X, dtm, valid, org, res)
    # outside the trusted DTM the height is meaningless, so mark it NaN rather
    # than letting it quietly bias the inventory
    h = np.where(ok, h, np.nan).astype(np.float32)

    step(0.38, "detecting stems (circle stacking)")
    seeds = segment.detect_stems(X, h, cfg, verbose=verbose)
    seeds = segment.merge_close_seeds(seeds, verbose=verbose)
    if not seeds:
        raise RuntimeError("no stems detected - loosen the stem_* settings")

    step(0.62, f"growing {len(seeds)} trees (geodesic segmentation)")
    labels, _, _ = segment.segment_trees(X, h, ok, seeds, cfg, verbose=verbose)

    step(0.80, "measuring DBH, height and crowns")
    df = measure.measure_trees(X, h, labels, seeds, cfg, verbose=verbose)
    dtm_area_ha = float(valid.sum() * res * res / 10000)
    area_ha, hull = stand_area(df)
    centre = scan_centre(X)
    df["dist_from_scan_centre_m"] = np.hypot(df["x"] - centre[0], df["y"] - centre[1])

    step(0.90, "extracting species features")
    feat = features.build_feature_table(X, h, labels, df, cfg, verbose=verbose)
    usable = df.set_index("tree_id")["quality"].isin(["good", "fair"])
    try:
        grp, _, _ = features.cluster_trees(feat, n_clusters=groups,
                                           seed=cfg.random_seed,
                                           quality_mask=usable.values)
        df = df.merge(grp.rename("structural_group"), left_on="tree_id",
                      right_index=True, how="left")
    except Exception as e:
        log.append(f"structural clustering skipped: {e}")

    step(1.0, f"done - {len(df)} trees")
    return Result(cfg=cfg, las_path=las_path, header=header, xyz=X, h=h, ok=ok,
                  dtm=dtm, valid=valid, org=org, res=res, seeds=seeds,
                  labels=labels, df=df, feat=feat, dtm_area_ha=dtm_area_ha,
                  stand_area_ha=area_ha, hull=hull, scan_centre=centre, log=log)


def summary(r: Result) -> dict:
    """Stand-level numbers, as a dict so both the CLI and the app can format it."""
    g, gg = r.usable, r.good
    out = {
        "valid DTM area (ha)": r.dtm_area_ha,
        "stocked area (ha)": r.stand_area_ha,
        "trees detected": len(r.df),
        "good": len(gg), "fair": len(g) - len(gg), "poor": len(r.df) - len(g),
        "stem density all (/ha)": len(r.df) / r.stand_area_ha,
        "stem density good+fair (/ha)": len(g) / r.stand_area_ha,
        "DBH mean (cm)": g["dbh_cm"].mean(),
        "DBH median (cm)": g["dbh_cm"].median(),
        "DBH good-only mean (cm)": gg["dbh_cm"].mean(),
        "height mean (m)": g["height_m"].mean(),
        "height median (m)": g["height_m"].median(),
        "basal area (m2/ha)": g["basal_area_m2"].sum() / r.stand_area_ha,
    }
    for name, sub in (("all", r.df), ("good-only", gg)):
        v = sub[["dbh_cm", "height_m"]].dropna()
        out[f"height-DBH corr {name}"] = (float(np.corrcoef(v.T)[0, 1])
                                          if len(v) > 3 else np.nan)
    return out


def summary_text(r: Result) -> str:
    s = summary(r)
    med = r.df.groupby("quality")["dist_from_scan_centre_m"].median()
    return "\n".join([
        "=" * 72, "STAND SUMMARY", "=" * 72,
        f"  valid DTM area          {s['valid DTM area (ha)']:.3f} ha",
        f"  stocked area            {s['stocked area (ha)']:.3f} ha  "
        f"(per-ha figures use this)",
        f"  trees detected          {s['trees detected']}   good={s['good']} "
        f"fair={s['fair']} poor={s['poor']}",
        f"  stem density            {s['stem density all (/ha)']:.0f} stems/ha "
        f"(all detections)",
        f"                          {s['stem density good+fair (/ha)']:.0f} stems/ha "
        f"(good+fair only - an undercount, the poor stems are real trees)",
        f"  DBH        mean {s['DBH mean (cm)']:6.1f} cm   "
        f"median {s['DBH median (cm)']:5.1f}   "
        f"good-only mean {s['DBH good-only mean (cm)']:.1f}",
        f"  height     mean {s['height mean (m)']:6.1f} m    "
        f"median {s['height median (m)']:5.1f}",
        f"  basal area              {s['basal area (m2/ha)']:.2f} m2/ha "
        f"(good+fair stems only - a lower bound)",
        "  quality vs range from the scanner (m):  "
        + "  ".join(f"{k}={v:.0f}" for k, v in med.items()),
        f"  height-DBH correlation  all={s['height-DBH corr all']:+.2f}   "
        f"good-only={s['height-DBH corr good-only']:+.2f}"
        "   (should be clearly positive)",
        "=" * 72,
    ])


def write_outputs(r: Result, out_dir: str, figures=True, segmented_las=True):
    """CSVs, QC figures and the labelled point cloud."""
    os.makedirs(out_dir, exist_ok=True)
    r.df.to_csv(os.path.join(out_dir, "trees.csv"), index=False)
    r.feat.to_csv(os.path.join(out_dir, "tree_features.csv"))
    written = ["trees.csv", "tree_features.csv"]

    if figures:
        qc.plot_ground(os.path.join(out_dir, "qc_01_ground.png"),
                       r.xyz, r.h, r.dtm, r.valid, r.org, r.res)
        qc.plot_stem_map(os.path.join(out_dir, "qc_02_stem_map.png"),
                         r.df, r.dtm, r.valid, r.org, r.res)
        qc.plot_inventory(os.path.join(out_dir, "qc_03_inventory.png"), r.df)
        yc = float(r.df.loc[r.df["quality"] != "poor", "y"].median())
        qc.plot_segmentation(os.path.join(out_dir, "qc_04_segmentation.png"),
                             r.xyz, r.labels, yc=yc)
        qc.plot_circle_fits(os.path.join(out_dir, "qc_05_dbh_fits.png"),
                            r.xyz, r.h, r.labels, r.df, r.seeds, r.cfg)
        written += [f"qc_0{i}_*.png" for i in range(1, 6)]

    if segmented_las:
        keep = r.labels >= 0
        las_io.write_las(os.path.join(out_dir, "Forest_segmented.las"),
                         r.xyz[keep], src_path=r.las_path,
                         extra={"tree_id": r.labels[keep].astype(np.uint32),
                                "height_norm": r.h[keep]})
        written.append("Forest_segmented.las")
    return written
