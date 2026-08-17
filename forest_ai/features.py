"""Per-tree descriptors for species classification.

Deliberately handcrafted rather than learned.  With no field labels yet, and
no RGB or intensity in this file, a compact geometric descriptor plus a
gradient-boosted / random-forest classifier is the baseline that works from a
few dozen labelled trees per species - point-based networks (PointMLP,
PointNeXt, Point Transformer) need hundreds, or a pretrained backbone such as
FOR-species20K to fine-tune from.

`build_feature_table` produces the design matrix; `train_species_model` is the
supervised entry point once labels exist; `cluster_trees` is the label-free
stand-in that groups structurally similar trees so a field crew only has to
identify a few representatives per group.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

N_HEIGHT_BINS = 10


def _eig_shape(pts):
    """Linearity / planarity / sphericity from the PCA eigenvalues."""
    if len(pts) < 4:
        return 0.0, 0.0, 0.0
    c = pts - pts.mean(0)
    ev = np.linalg.eigvalsh(np.cov(c.T))[::-1]
    ev = np.clip(ev, 1e-12, None)
    s = ev.sum()
    l1, l2, l3 = ev / s
    return float((l1 - l2) / l1), float((l2 - l3) / l1), float(l3 / l1)


def tree_features(pts, h, row):
    """Feature dict for one tree.  `row` is its row from the measurement table."""
    f = {}
    H = row["height_m"]
    if not np.isfinite(H) or H <= 0 or len(pts) < 30:
        return None

    cx, cy = row["x"], row["y"]
    rad = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)
    rel = np.clip(h / H, 0, 1)

    # --- size and gross shape ------------------------------------------------
    f["height_m"] = H
    f["dbh_cm"] = row["dbh_cm"]
    f["h_d_ratio"] = row["h_d_ratio"]
    f["crown_base_rel"] = row["crown_base_m"] / H
    f["crown_length_rel"] = 1 - row["crown_base_m"] / H
    f["crown_diam_rel"] = row["crown_diameter_m"] / H
    f["crown_diam_dbh"] = row["crown_diameter_m"] / max(row["dbh_cm"] / 100, 1e-3)
    f["crown_area_m2"] = row["crown_area_m2"]
    f["crown_volume_m3"] = row["crown_volume_m3"]
    # how much of the enclosing cylinder the crown actually fills - open,
    # spreading crowns score low, dense conical crowns score high
    cyl = np.pi * (row["crown_diameter_m"] / 2) ** 2 * max(H - row["crown_base_m"], 1e-3)
    f["crown_fill"] = row["crown_volume_m3"] / cyl if np.isfinite(cyl) else np.nan

    # --- vertical distribution of returns ------------------------------------
    for p in (10, 25, 50, 75, 90, 95, 99):
        f[f"relh_p{p}"] = float(np.percentile(rel, p))
    f["relh_mean"] = float(rel.mean())
    f["relh_std"] = float(rel.std())
    f["relh_skew"] = float(((rel - rel.mean()) ** 3).mean() / (rel.std() ** 3 + 1e-9))
    f["relh_kurt"] = float(((rel - rel.mean()) ** 4).mean() / (rel.std() ** 4 + 1e-9))

    # --- density and width profile, sampled up the tree ----------------------
    bins = np.linspace(0, 1, N_HEIGHT_BINS + 1)
    which = np.clip(np.digitize(rel, bins) - 1, 0, N_HEIGHT_BINS - 1)
    counts = np.bincount(which, minlength=N_HEIGHT_BINS).astype(float)
    dens = counts / counts.sum()
    rmax = max(rad.max(), 1e-6)
    for i in range(N_HEIGHT_BINS):
        f[f"dens_{i}"] = float(dens[i])
        m = which == i
        f[f"radius_{i}"] = float(np.percentile(rad[m], 90) / rmax) if m.sum() >= 5 else 0.0
    # where the widest part of the crown sits: low for conical conifers,
    # high or mid for broadleaves
    f["widest_rel_height"] = float(np.argmax([f[f"radius_{i}"] for i in range(N_HEIGHT_BINS)])
                                   / (N_HEIGHT_BINS - 1))
    f["top_taper"] = f["radius_9"] / max(f["radius_5"], 1e-6)

    # --- PCA shape of whole tree and of crown only ---------------------------
    lin, pla, sph = _eig_shape(pts)
    f["pca_linearity"], f["pca_planarity"], f["pca_sphericity"] = lin, pla, sph
    crown = pts[h >= row["crown_base_m"]] if np.isfinite(row["crown_base_m"]) else pts
    lin, pla, sph = _eig_shape(crown)
    f["crown_linearity"], f["crown_planarity"], f["crown_sphericity"] = lin, pla, sph

    f["stem_lean_deg"] = row["stem_lean_deg"]
    f["n_points"] = row["n_points"]
    return f


def build_feature_table(xyz, h, labels, df, cfg, verbose=True):
    """Feature matrix for every measured tree, aligned to `df.tree_id`."""
    order = np.argsort(labels, kind="stable")
    ls = labels[order]
    n = len(df)
    lo = np.searchsorted(ls, np.arange(n), "left")
    hi = np.searchsorted(ls, np.arange(n), "right")

    rows = []
    for tid in range(n):
        sel = order[lo[tid]:hi[tid]]
        f = tree_features(xyz[sel].astype(np.float64),
                          h[sel].astype(np.float64), df.iloc[tid])
        if f is None:
            f = {}
        f["tree_id"] = tid
        rows.append(f)
    out = pd.DataFrame(rows).set_index("tree_id")
    if verbose:
        print(f"    {out.shape[1]} features for {out.shape[0]} trees")
    return out


def cluster_trees(feat: pd.DataFrame, n_clusters=4, seed=42, quality_mask=None):
    """Label-free structural grouping (the stand-in until field data exists).

    These are *structural* groups, not species: two species with the same
    architecture will land together, and one species will split across groups
    if some individuals are suppressed.  The point is to cut field work - a
    crew identifies a handful of trees per group instead of every tree.
    """
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X = feat.drop(columns=[c for c in ("n_points",) if c in feat], errors="ignore")
    if quality_mask is not None:
        X = X[quality_mask]
    pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         PCA(n_components=0.95, random_state=seed))
    Z = pipe.fit_transform(X.values)
    km = KMeans(n_clusters=n_clusters, n_init=20, random_state=seed).fit(Z)
    return pd.Series(km.labels_, index=X.index, name="structural_group"), Z, pipe


def train_species_model(feat: pd.DataFrame, labels: pd.Series, seed=42):
    """Supervised baseline, to be run once field species labels are available.

    Reports stratified 5-fold cross-validated accuracy so the number is honest
    on a small dataset; a single train/test split on 200 trees is mostly noise.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.pipeline import make_pipeline

    common = feat.index.intersection(labels.index)
    X, y = feat.loc[common], labels.loc[common]
    clf = make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestClassifier(n_estimators=500, min_samples_leaf=2,
                               class_weight="balanced", random_state=seed, n_jobs=-1),
    )
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    pred = cross_val_predict(clf, X.values, y.values, cv=cv, n_jobs=1)
    clf.fit(X.values, y.values)
    importance = pd.Series(clf[-1].feature_importances_, index=X.columns
                           ).sort_values(ascending=False)
    return {
        "model": clf,
        "cv_report": classification_report(y, pred, zero_division=0),
        "confusion": confusion_matrix(y, pred),
        "importance": importance,
    }
