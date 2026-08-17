"""Ground classification (Cloth Simulation Filter), DTM, height normalisation.

CSF (Zhang et al. 2016) inverts the point cloud so the terrain becomes the
uppermost surface, then drops a grid of mass points ("cloth") onto it under
gravity while internal springs keep the cloth from folding into small gaps.
Points that end up close to the settled cloth are ground.

This is a vectorised NumPy re-implementation: the cloth is an (nx, ny) array
updated with Verlet integration, and the spring pass is a Jacobi-style
averaged correction over the four grid neighbours.  On a 167 x 142 m plot at
0.5 m cloth resolution that is a 336 x 286 array, so 500 iterations take a
couple of seconds.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

GRAVITY = 9.8


def _cell_extreme(values, cells, n_cells, mode="max"):
    """Per-cell max / min / median of `values`, returned as a dense array."""
    order = np.lexsort((values, cells))
    cs, vs = cells[order], values[order]
    starts = np.flatnonzero(np.r_[True, cs[1:] != cs[:-1]])
    ends = np.r_[starts[1:], len(cs)]
    if mode == "max":
        pick = ends - 1
    elif mode == "min":
        pick = starts
    else:  # median
        pick = starts + (ends - starts) // 2
    out = np.full(n_cells, np.nan, dtype=np.float64)
    out[cs[starts]] = vs[pick]
    return out


def _fill_nearest(grid):
    """Replace NaNs with the value of the nearest valid cell."""
    nan = np.isnan(grid)
    if not nan.any():
        return grid, nan
    _, idx = ndimage.distance_transform_edt(nan, return_indices=True)
    return grid[tuple(idx)], nan


def csf_ground(xyz, cloth_res=0.5, rigidness=3, time_step=0.65,
               iterations=500, threshold=0.15, tol=0.005, verbose=True):
    """Classify ground points.  Returns (ground_mask, cloth_grid, grid_meta)."""
    zi = (-xyz[:, 2]).astype(np.float64)          # inverted: ground is now on top
    lo = xyz[:, :2].min(axis=0) - 2 * cloth_res
    hi = xyz[:, :2].max(axis=0) + 2 * cloth_res
    nx = int(np.ceil((hi[0] - lo[0]) / cloth_res)) + 1
    ny = int(np.ceil((hi[1] - lo[1]) / cloth_res)) + 1

    fx = (xyz[:, 0] - lo[0]) / cloth_res
    fy = (xyz[:, 1] - lo[1]) / cloth_res
    ix = np.clip(fx.astype(np.int64), 0, nx - 1)
    iy = np.clip(fy.astype(np.int64), 0, ny - 1)
    cells = ix * ny + iy

    # Intersection height: the highest inverted point per cell, i.e. the
    # lowest real point - the surface the cloth is allowed to rest on.
    ihv = _cell_extreme(zi, cells, nx * ny, "max").reshape(nx, ny)
    ihv, was_empty = _fill_nearest(ihv)

    cur = np.full((nx, ny), ihv.max() + 1.0)
    prev = cur.copy()
    movable = np.ones((nx, ny), dtype=bool)
    step = GRAVITY * time_step * time_step

    pairs = [((slice(0, -1), slice(None)), (slice(1, None), slice(None))),
             ((slice(None), slice(0, -1)), (slice(None), slice(1, None)))]

    for it in range(iterations):
        # --- gravity (Verlet) ---
        nxt = np.where(movable, 2 * cur - prev - step, cur)
        prev = cur
        cur = nxt

        # --- collision with the terrain ---
        hit = cur < ihv
        cur = np.where(hit, ihv, cur)
        movable &= ~hit

        # --- internal spring constraints ---
        for _ in range(rigidness):
            corr = np.zeros_like(cur)
            cnt = np.zeros_like(cur)
            for sa, sb in pairs:
                a, b = cur[sa], cur[sb]
                ma, mb = movable[sa], movable[sb]
                diff = b - a
                both = ma & mb
                corr[sa] += np.where(both, 0.5 * diff, np.where(ma & ~mb, diff, 0.0))
                corr[sb] += np.where(both, -0.5 * diff, np.where(mb & ~ma, -diff, 0.0))
                cnt[sa] += 1
                cnt[sb] += 1
            cur = cur + np.where(movable, corr / np.maximum(cnt, 1), 0.0)
            cur = np.maximum(cur, ihv)          # springs must not pull it underground

        delta = np.abs(cur - prev).max()
        if delta < tol and it > 20:
            break

    if verbose:
        print(f"    CSF: cloth {nx}x{ny} @ {cloth_res} m, "
              f"{it + 1} iterations, final delta {delta:.4f} m, "
              f"{was_empty.sum()} empty cells filled")

    cloth_at_pt = ndimage.map_coordinates(cur, [fx, fy], order=1, mode="nearest")
    ground = (cloth_at_pt - zi) < threshold
    return ground, cur, (lo, cloth_res, nx, ny)


def build_dtm(xyz_ground, res=0.25, smooth_sigma=1.0, bounds=None,
              max_fill_dist=3.0, despike=1.0, despike_window=5.0, passes=2,
              min_obs_density=0.06):
    """Rasterise ground points to a DTM plus a validity mask.

    Two things make a naive "lowest point per cell" DTM fail on a single-scan
    TLS cloud like this one:

    * far from the scanner the ground is occluded by the understorey, so the
      lowest point in a cell can be a branch several metres up;
    * beyond the scan's useful radius there are no ground returns at all, and
      nearest-neighbour infilling then smears a wrong height a long way.

    So ground cells that sit `despike` metres above their local neighbourhood
    are rejected and re-interpolated, and cells further than `max_fill_dist`
    from any real observation are flagged invalid instead of being invented.
    Returns (dtm, valid_mask, origin, res).
    """
    lo = np.array(bounds[0], dtype=float) if bounds is not None else xyz_ground[:, :2].min(axis=0)
    hi = np.array(bounds[1], dtype=float) if bounds is not None else xyz_ground[:, :2].max(axis=0)
    lo = lo - res
    hi = hi + res
    nx = int(np.ceil((hi[0] - lo[0]) / res)) + 1
    ny = int(np.ceil((hi[1] - lo[1]) / res)) + 1

    ix = np.clip(((xyz_ground[:, 0] - lo[0]) / res).astype(np.int64), 0, nx - 1)
    iy = np.clip(((xyz_ground[:, 1] - lo[1]) / res).astype(np.int64), 0, ny - 1)
    cells = ix * ny + iy

    # median per cell: robust to the odd low outlier that survived CSF
    dtm = _cell_extreme(xyz_ground[:, 2].astype(np.float64), cells,
                        nx * ny, "median").reshape(nx, ny)
    observed0 = ~np.isnan(dtm)

    win = max(3, int(round(despike_window / res)) | 1)
    n_spikes = 0
    for _ in range(passes):
        filled, _ = _fill_nearest(dtm)
        local = ndimage.median_filter(filled, size=win, mode="nearest")
        spike = (~np.isnan(dtm)) & (dtm > local + despike)
        n_spikes += int(spike.sum())
        if not spike.any():
            break
        dtm[spike] = np.nan

    observed = ~np.isnan(dtm)
    dist = ndimage.distance_transform_edt(~observed) * res
    valid = dist <= max_fill_dist

    # A handful of ground returns way out in the sparse fringe would otherwise
    # each seed their own little island of "valid" DTM at a bogus height.
    # Require a minimum local density of real observations, then keep only the
    # component connected to the main scan area.
    dens_win = max(3, int(round(2 * max_fill_dist / res)) | 1)
    dens = ndimage.uniform_filter(observed.astype(np.float32), dens_win, mode="constant")
    valid &= dens >= min_obs_density

    lbl, n = ndimage.label(valid)
    if n > 1:
        sizes = np.bincount(lbl.ravel())
        sizes[0] = 0
        valid = lbl == sizes.argmax()
        print(f"    DTM: dropped {n - 1} disconnected valid-area island(s)")

    dtm, _ = _fill_nearest(dtm)
    if smooth_sigma > 0:
        dtm = ndimage.gaussian_filter(dtm, smooth_sigma)

    print(f"    DTM: {nx}x{ny} @ {res} m | observed {observed0.mean()*100:.1f}% of cells, "
          f"{n_spikes} spike cells rejected | valid area "
          f"{valid.sum()*res*res/10000:.2f} ha | z {dtm[valid].min():.2f}..{dtm[valid].max():.2f} m")
    return dtm, valid, lo, res


def normalize_height(xyz, dtm, valid, origin, res, chunk=4_000_000):
    """Height above the DTM, plus a per-point flag for "DTM is trustworthy here"."""
    h = np.empty(len(xyz), dtype=np.float32)
    ok = np.empty(len(xyz), dtype=bool)
    validf = valid.astype(np.float32)
    for s in range(0, len(xyz), chunk):
        e = min(s + chunk, len(xyz))
        fx = (xyz[s:e, 0] - origin[0]) / res
        fy = (xyz[s:e, 1] - origin[1]) / res
        h[s:e] = xyz[s:e, 2] - ndimage.map_coordinates(dtm, [fx, fy], order=1, mode="nearest")
        ok[s:e] = ndimage.map_coordinates(validf, [fx, fy], order=1, mode="nearest") > 0.99
    return h, ok
