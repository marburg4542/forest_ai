"""Compare a tree list against a reference.

The reference can be a field survey or the output of another tool (FSCT,
TreeLS, TreeLearn, ...).  Either way the comparison is the same problem:
decide which predicted stem corresponds to which reference stem, then report
detection rates and measurement error on the matched pairs.

Matching is done by optimal assignment rather than greedy nearest-neighbour.
Greedy matching is order-dependent and, in a dense stand where stems are ~2 m
apart, it will happily pair two predictions to the wrong pair of references
and inflate the error.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


def match_trees(pred: pd.DataFrame, ref: pd.DataFrame, max_dist=2.0,
                xy=("x", "y")):
    """One-to-one match under a distance cap.

    Returns (matched_df, stats).  `matched_df` has the paired rows suffixed
    _pred / _ref plus the separation distance.
    """
    px = pred[list(xy)].values
    rx = ref[list(xy)].values
    if len(px) == 0 or len(rx) == 0:
        raise ValueError("both tables must contain at least one tree")

    cost = np.hypot(px[:, None, 0] - rx[None, :, 0],
                    px[:, None, 1] - rx[None, :, 1])
    big = max_dist * 10
    solvable = np.where(cost <= max_dist, cost, big)
    pi, ri = linear_sum_assignment(solvable)
    keep = cost[pi, ri] <= max_dist
    pi, ri = pi[keep], ri[keep]

    matched = pd.concat(
        [pred.iloc[pi].add_suffix("_pred").reset_index(drop=True),
         ref.iloc[ri].add_suffix("_ref").reset_index(drop=True)], axis=1)
    matched["match_dist_m"] = cost[pi, ri]

    tp, fp, fn = len(pi), len(pred) - len(pi), len(ref) - len(ri)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    stats = {
        "n_pred": len(pred), "n_ref": len(ref),
        "TP": tp, "FP": fp, "FN": fn,
        "precision": prec, "recall": rec,
        "f1": 2 * prec * rec / max(prec + rec, 1e-9),
        "mean_match_dist_m": float(matched["match_dist_m"].mean()) if tp else np.nan,
    }
    return matched, stats


def error_stats(matched: pd.DataFrame, field: str) -> dict:
    """Bias / RMSE / MAE / R2 for one measured attribute."""
    a = matched.get(f"{field}_pred")
    b = matched.get(f"{field}_ref")
    if a is None or b is None:
        return {}
    m = np.isfinite(a) & np.isfinite(b)
    a, b = np.asarray(a)[m], np.asarray(b)[m]
    if len(a) < 2:
        return {"n": int(len(a))}
    d = a - b
    ss_res = float((d ** 2).sum())
    ss_tot = float(((b - b.mean()) ** 2).sum())
    return {
        "n": int(len(a)),
        "bias": float(d.mean()),
        "rmse": float(np.sqrt((d ** 2).mean())),
        "mae": float(np.abs(d).mean()),
        "rel_rmse_pct": float(100 * np.sqrt((d ** 2).mean()) / b.mean()),
        "r2": 1 - ss_res / ss_tot if ss_tot > 0 else np.nan,
    }


def report(pred: pd.DataFrame, ref: pd.DataFrame, max_dist=2.0,
           fields=("dbh_cm", "height_m"), quality=None) -> str:
    """Full text report.  `quality` restricts the prediction set, e.g. "good"."""
    if quality is not None and "quality" in pred:
        pred = pred[pred["quality"].isin(np.atleast_1d(quality))]
    matched, st = match_trees(pred, ref, max_dist)

    lines = [
        "detection",
        f"  predicted {st['n_pred']}   reference {st['n_ref']}   "
        f"matched within {max_dist} m: {st['TP']}",
        f"  precision {st['precision']:.3f}   recall {st['recall']:.3f}   "
        f"F1 {st['f1']:.3f}",
        f"  commission (FP) {st['FP']}   omission (FN) {st['FN']}   "
        f"mean offset {st['mean_match_dist_m']:.2f} m",
        "",
        "measurement (matched pairs only)",
    ]
    for f in fields:
        e = error_stats(matched, f)
        if not e or e.get("n", 0) < 2:
            lines.append(f"  {f:<12} not comparable (missing in one table)")
            continue
        lines.append(
            f"  {f:<12} n={e['n']:<4} bias {e['bias']:+7.2f}  rmse {e['rmse']:6.2f}"
            f"  mae {e['mae']:6.2f}  rel.rmse {e['rel_rmse_pct']:5.1f}%  R2 {e['r2']:.3f}")
    return "\n".join(lines), matched, st


def plot_agreement(path, matched, fields=("dbh_cm", "height_m")):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(fields), figsize=(6 * len(fields), 5.5),
                             squeeze=False)
    for a, f in zip(axes[0], fields):
        x, y = matched.get(f"{f}_ref"), matched.get(f"{f}_pred")
        if x is None or y is None:
            a.axis("off")
            continue
        a.scatter(x, y, s=18, alpha=0.75, c="#3d5a80")
        lim = [min(np.nanmin(x), np.nanmin(y)), max(np.nanmax(x), np.nanmax(y))]
        a.plot(lim, lim, "k--", lw=1, label="1:1")
        e = error_stats(matched, f)
        a.set_xlabel(f"reference {f}")
        a.set_ylabel(f"predicted {f}")
        a.set_title(f"{f}\nbias {e.get('bias', float('nan')):+.2f}  "
                    f"rmse {e.get('rmse', float('nan')):.2f}  "
                    f"R2 {e.get('r2', float('nan')):.3f}")
        a.legend()
        a.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
