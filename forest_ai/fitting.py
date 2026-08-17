"""Circle / cylinder fitting primitives used for stem detection and DBH."""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares


def kasa_circle(p: np.ndarray):
    """Algebraic (Kasa) circle fit.  Fast, used to seed the geometric fit."""
    x, y = p[:, 0], p[:, 1]
    A = np.column_stack([2 * x, 2 * y, np.ones(len(p))])
    b = x * x + y * y
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy, c = sol
    r2 = c + cx * cx + cy * cy
    return cx, cy, np.sqrt(max(r2, 1e-12))


def refine_circle(p: np.ndarray, cx: float, cy: float, r: float):
    """Geometric least squares with a soft-L1 loss.

    The algebraic fit is biased when a stem is only seen from one side - which
    is the normal case for a single-scan TLS - so the geometric residual
    (distance to the circle) is minimised explicitly.
    """
    def resid(par):
        return np.hypot(p[:, 0] - par[0], p[:, 1] - par[1]) - par[2]

    try:
        out = least_squares(resid, [cx, cy, r], loss="soft_l1", f_scale=0.02,
                            max_nfev=200)
        cx, cy, r = out.x
    except Exception:
        pass
    rmse = float(np.sqrt(np.mean(resid([cx, cy, r]) ** 2)))
    return float(cx), float(cy), float(abs(r)), rmse


def ransac_circle(p: np.ndarray, tol=0.02, iters=300, min_r=0.02, max_r=1.2,
                  rng=None):
    """RANSAC circle fit.  Returns dict or None.

    Vectorised: all `iters` candidate circles are built from random point
    triplets at once and scored against every point in one distance matrix.
    """
    n = len(p)
    if n < 8:
        return None
    rng = rng or np.random.default_rng(0)
    idx = rng.integers(0, n, size=(iters, 3))
    a, b, c = p[idx[:, 0]], p[idx[:, 1]], p[idx[:, 2]]

    d = 2 * (a[:, 0] * (b[:, 1] - c[:, 1]) + b[:, 0] * (c[:, 1] - a[:, 1])
             + c[:, 0] * (a[:, 1] - b[:, 1]))
    ok = np.abs(d) > 1e-9
    if not ok.any():
        return None
    a2 = (a ** 2).sum(1)
    b2 = (b ** 2).sum(1)
    c2 = (c ** 2).sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        ux = (a2 * (b[:, 1] - c[:, 1]) + b2 * (c[:, 1] - a[:, 1])
              + c2 * (a[:, 1] - b[:, 1])) / d
        uy = (a2 * (c[:, 0] - b[:, 0]) + b2 * (a[:, 0] - c[:, 0])
              + c2 * (b[:, 0] - a[:, 0])) / d
    r = np.hypot(a[:, 0] - ux, a[:, 1] - uy)
    ok &= np.isfinite(ux) & np.isfinite(uy) & (r > min_r) & (r < max_r)
    if not ok.any():
        return None
    ux, uy, r = ux[ok], uy[ok], r[ok]

    dist = np.abs(np.hypot(p[None, :, 0] - ux[:, None],
                           p[None, :, 1] - uy[:, None]) - r[:, None])
    counts = (dist < tol).sum(1)
    best = int(counts.argmax())
    inl = dist[best] < tol
    if inl.sum() < 8:
        return None

    cx, cy, rr, rmse = refine_circle(p[inl], ux[best], uy[best], r[best])
    if not (min_r < rr < max_r):
        return None
    final_inl = np.abs(np.hypot(p[:, 0] - cx, p[:, 1] - cy) - rr) < tol
    return {
        "cx": cx, "cy": cy, "r": rr, "rmse": rmse,
        "n": int(final_inl.sum()),
        "inlier_frac": float(final_inl.mean()),
        "inliers": final_inl,
        "arc": arc_coverage(p[final_inl], cx, cy),
    }


def arc_coverage(p: np.ndarray, cx: float, cy: float, bins: int = 36) -> float:
    """Fraction of the 360 deg around the stem centre that actually has points.

    A one-sided scan of a stem can still be fitted, but the diameter is much
    less reliable; this is the flag that says so.
    """
    if len(p) == 0:
        return 0.0
    ang = np.arctan2(p[:, 1] - cy, p[:, 0] - cx)
    hit = np.zeros(bins, dtype=bool)
    hit[((ang + np.pi) / (2 * np.pi) * bins).astype(int) % bins] = True
    return float(hit.mean())


def principal_axis(p: np.ndarray) -> np.ndarray:
    """Unit vector of the dominant direction (first PCA component)."""
    q = p - p.mean(0)
    _, _, vt = np.linalg.svd(q, full_matrices=False)
    v = vt[0]
    return v if v[2] >= 0 else -v
