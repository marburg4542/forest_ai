"""Stem detection and individual-tree instance segmentation.

Two stages:

1. `detect_stems` clusters a horizontal slice of the cloud around breast
   height and keeps only clusters that actually look like a stem (vertical,
   circular, plausible diameter).  Understorey foliage, which also lives in
   that band, is rejected here.

2. `segment_trees` grows those seeds through the whole cloud by
   multi-source Dijkstra on a kNN graph.  Geodesic distance, rather than
   plain Euclidean nearest-seed, is what lets interlocking crowns be split
   correctly: a point on a branch is close *through the wood* to its own
   stem even when a neighbouring tree's crown is physically nearer.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN

from . import preprocess
from .fitting import ransac_circle, principal_axis


class _Union:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, i):
        while self.p[i] != i:
            self.p[i] = self.p[self.p[i]]
            i = self.p[i]
        return i

    def union(self, i, j):
        ri, rj = self.find(i), self.find(j)
        if ri != rj:
            self.p[max(ri, rj)] = min(ri, rj)

    def groups(self):
        out: dict[int, list[int]] = {}
        for i in range(len(self.p)):
            out.setdefault(self.find(i), []).append(i)
        return list(out.values())


def _layer_circles(xyz, h, cfg, rng, verbose):
    """RANSAC circle per DBSCAN cluster, in every thin horizontal layer."""
    heights = np.arange(cfg.stem_slice_lo,
                        cfg.stem_slice_hi - cfg.stem_layer_thick + 1e-6,
                        cfg.stem_layer_step)
    circles = []
    for li, h0 in enumerate(heights):
        m = (h >= h0) & (h < h0 + cfg.stem_layer_thick)
        gidx = np.flatnonzero(m)
        if len(gidx) < cfg.stem_dbscan_min_pts:
            continue
        P = xyz[gidx]
        keep = preprocess.voxel_downsample(P, cfg.stem_voxel * 2, return_index=True)[1]
        P, gidx = P[keep], gidx[keep]

        lab = DBSCAN(eps=cfg.stem_dbscan_eps, min_samples=cfg.stem_dbscan_min_pts,
                     n_jobs=cfg.n_jobs).fit(P[:, :2]).labels_
        nk = lab.max() + 1
        if nk <= 0:
            continue
        order = np.argsort(lab, kind="stable")
        ls = lab[order]
        lo_i = np.searchsorted(ls, np.arange(nk), "left")
        hi_i = np.searchsorted(ls, np.arange(nk), "right")

        n_ok = 0
        for k in range(nk):
            sel = order[lo_i[k]:hi_i[k]]
            if len(sel) < cfg.stem_dbscan_min_pts:
                continue
            fit = ransac_circle(P[sel][:, :2].astype(np.float64),
                                tol=cfg.ransac_tol, iters=cfg.ransac_iters,
                                min_r=cfg.stem_min_diameter / 2,
                                max_r=cfg.stem_max_diameter / 2, rng=rng)
            if (fit is None or fit["rmse"] > cfg.stem_max_rmse
                    or fit["inlier_frac"] < cfg.ransac_min_inlier_frac):
                continue
            circles.append({
                "layer": li, "h": h0 + cfg.stem_layer_thick / 2,
                "cx": fit["cx"], "cy": fit["cy"], "r": fit["r"],
                "rmse": fit["rmse"], "arc": fit["arc"],
                "idx": gidx[sel[fit["inliers"]]],
            })
            n_ok += 1
        if verbose:
            print(f"      layer {h0:.1f}-{h0+cfg.stem_layer_thick:.1f} m: "
                  f"{len(gidx):>7,} pts, {nk:>4} clusters -> {n_ok:>3} circles")
    return circles, len(heights)


def detect_stems(xyz, h, cfg, verbose=True):
    """Find stems by stacking per-layer circle fits.

    A single thick slice projected to 2D smears a tapering or leaning stem
    into a wide annulus, which both defeats the circle fit and lets two
    neighbouring stems merge.  Fitting each 20 cm layer separately and then
    requiring the circles to stack vertically is far more selective: random
    foliage never produces four consistent circles above one another.
    """
    rng = np.random.default_rng(cfg.random_seed)
    circles, n_layers = _layer_circles(xyz, h, cfg, rng, verbose)
    if verbose:
        print(f"    {len(circles)} candidate circles over {n_layers} layers")
    if not circles:
        return []

    # link circles between adjacent layers
    uf = _Union(len(circles))
    by_layer: dict[int, list[int]] = {}
    for i, c in enumerate(circles):
        by_layer.setdefault(c["layer"], []).append(i)

    for li in sorted(by_layer):
        upper = by_layer.get(li + 1)
        if not upper:
            continue
        low = by_layer[li]
        a = np.array([[circles[i]["cx"], circles[i]["cy"]] for i in low])
        b = np.array([[circles[j]["cx"], circles[j]["cy"]] for j in upper])
        tb = cKDTree(b)
        for n, i in enumerate(low):
            tol = cfg.stem_link_dist + 0.35 * circles[i]["r"]
            for m in tb.query_ball_point(a[n], tol):
                j = upper[m]
                ratio = circles[i]["r"] / circles[j]["r"]
                if 1 / cfg.stem_link_radius_ratio <= ratio <= cfg.stem_link_radius_ratio:
                    uf.union(i, j)

    seeds, rej = [], {"short": 0, "tilted": 0, "size": 0}
    for members in uf.groups():
        layers = {circles[i]["layer"] for i in members}
        if len(layers) < cfg.stem_min_layers:
            rej["short"] += 1
            continue
        # one circle per layer: keep the lowest-residual one
        best: dict[int, dict] = {}
        for i in members:
            c = circles[i]
            if c["layer"] not in best or c["rmse"] < best[c["layer"]]["rmse"]:
                best[c["layer"]] = c
        chain = [best[k] for k in sorted(best)]
        ctr = np.array([[c["cx"], c["cy"], c["h"]] for c in chain])
        if abs(principal_axis(ctr)[2]) < cfg.stem_min_verticality:
            rej["tilted"] += 1
            continue
        rr = np.array([c["r"] for c in chain])
        hh = np.array([c["h"] for c in chain])
        d13 = 2 * float(np.interp(cfg.dbh_height, hh, rr))
        if not (cfg.stem_min_diameter <= d13 <= cfg.stem_max_diameter):
            rej["size"] += 1
            continue
        seeds.append({
            "x": float(np.interp(cfg.dbh_height, hh, ctr[:, 0])),
            "y": float(np.interp(cfg.dbh_height, hh, ctr[:, 1])),
            "stack_diameter": d13,
            "n_layers": len(chain),
            "stack_rmse": float(np.mean([c["rmse"] for c in chain])),
            "stack_arc": float(np.mean([c["arc"] for c in chain])),
            "axis": principal_axis(ctr),
            "chain": chain,
            "point_idx": np.concatenate([c["idx"] for c in chain]),
        })

    if verbose:
        print(f"    stems accepted: {len(seeds)}  (rejected: too few layers="
              f"{rej['short']}, not vertical={rej['tilted']}, bad size={rej['size']})")
    return seeds


def merge_close_seeds(seeds, min_sep=0.30, verbose=True):
    """A stem cut in two by an occluding branch can yield two parallel stacks."""
    if not seeds:
        return seeds
    xy = np.array([[s["x"], s["y"]] for s in seeds])
    uf = _Union(len(seeds))
    for i, j in cKDTree(xy).query_pairs(min_sep, output_type="ndarray"):
        uf.union(int(i), int(j))
    merged = []
    for members in uf.groups():
        if len(members) == 1:
            merged.append(seeds[members[0]])
            continue
        best = max(members, key=lambda i: seeds[i]["n_layers"])
        s = dict(seeds[best])
        s["point_idx"] = np.concatenate([seeds[i]["point_idx"] for i in members])
        merged.append(s)
    if verbose and len(merged) != len(seeds):
        print(f"    merged {len(seeds) - len(merged)} duplicate stem stack(s) "
              f"-> {len(merged)} stems")
    return merged


def segment_trees(xyz, h, ok, seeds, cfg, verbose=True):
    """Assign every vegetation point to a stem seed.

    Returns (labels_full, node_xyz, node_labels) where labels_full is -1 for
    unassigned points and 0..n_trees-1 otherwise.
    """
    veg = ok & (h > cfg.veg_min_height)
    V = xyz[veg]
    if verbose:
        print(f"    vegetation points: {len(V):,}")

    nodes, node_src = preprocess.voxel_downsample(V, cfg.seg_voxel, return_index=True)
    nodes = nodes.astype(np.float32)
    n = len(nodes)
    if verbose:
        print(f"    graph nodes @ {cfg.seg_voxel} m: {n:,}")

    tree = cKDTree(nodes)
    dist, nbr = tree.query(nodes, k=cfg.graph_k + 1, workers=cfg.n_jobs)
    dist, nbr = dist[:, 1:], nbr[:, 1:]

    src = np.repeat(np.arange(n), cfg.graph_k)
    dst = nbr.ravel()
    dxyz = nodes[dst] - nodes[src]
    # vertical travel is cheap, so a path prefers to run up the stem before
    # spreading sideways into the crown
    w = np.sqrt(dxyz[:, 0] ** 2 + dxyz[:, 1] ** 2
                + (cfg.graph_z_weight * dxyz[:, 2]) ** 2)
    keep = dist.ravel() <= cfg.graph_max_edge
    src, dst, w = src[keep], dst[keep], w[keep]
    g = coo_matrix((w, (src, dst)), shape=(n, n)).tocsr()
    g = g.maximum(g.T)          # symmetrise
    if verbose:
        print(f"    graph edges: {g.nnz//2:,} (cut {(~keep).mean()*100:.1f}% over "
              f"{cfg.graph_max_edge} m)")

    # map each seed's slice points onto graph nodes
    seed_nodes, seed_tree_id = [], []
    for tid, s in enumerate(seeds):
        pts = xyz[s["point_idx"]]
        _, idx = tree.query(pts, k=1, workers=cfg.n_jobs)
        u = np.unique(idx)
        seed_nodes.append(u)
        seed_tree_id.append(np.full(len(u), tid))
    seed_nodes = np.concatenate(seed_nodes)
    seed_tree_id = np.concatenate(seed_tree_id)
    # a node claimed by two seeds goes to the first; harmless, they are adjacent
    seed_nodes, uniq_pos = np.unique(seed_nodes, return_index=True)
    seed_tree_id = seed_tree_id[uniq_pos]

    d, _, sources = dijkstra(g, directed=False, indices=seed_nodes,
                             min_only=True, return_predecessors=True)
    node_lab = np.full(n, -1, dtype=np.int32)
    reached = np.isfinite(d)
    lookup = np.full(n, -1, dtype=np.int32)
    lookup[seed_nodes] = seed_tree_id
    node_lab[reached] = lookup[sources[reached]]
    if verbose:
        print(f"    geodesic growing: {reached.mean()*100:.1f}% of nodes reached")

    # push labels back onto every original point
    labeled = node_lab >= 0
    ktree = cKDTree(nodes[labeled])
    lab_of_node = node_lab[labeled]
    labels_full = np.full(len(xyz), -1, dtype=np.int32)
    veg_idx = np.flatnonzero(veg)
    for s in range(0, len(veg_idx), 2_000_000):
        e = min(s + 2_000_000, len(veg_idx))
        dd, ii = ktree.query(V[s:e], k=1, workers=cfg.n_jobs)
        lab = lab_of_node[ii]
        lab[dd > cfg.seg_voxel * 3] = -1
        labels_full[veg_idx[s:e]] = lab
    return labels_full, nodes, node_lab
