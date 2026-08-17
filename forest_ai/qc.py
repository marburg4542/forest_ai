"""Quality-control figures.  Every stage should be looked at, not just trusted."""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage


def plot_ground(path, xyz, h, dtm, valid, org, res, sections=(-20.0, 0.0)):
    fig, ax = plt.subplots(2, 2, figsize=(15, 11))
    ext = [org[0], org[0] + dtm.shape[0] * res, org[1], org[1] + dtm.shape[1] * res]
    im = ax[0, 0].imshow(np.where(valid, dtm, np.nan).T, origin="lower",
                         extent=ext, cmap="terrain")
    plt.colorbar(im, ax=ax[0, 0], label="z (m)")
    ax[0, 0].set_title("DTM over the valid area")
    ax[0, 0].set_aspect("equal")

    for k, yc in enumerate(sections):
        a = ax[0, 1] if k == 0 else ax[1, 0]
        s = np.abs(xyz[:, 1] - yc) < 2.0
        a.scatter(xyz[s, 0], xyz[s, 2], s=0.05, c="0.35", lw=0)
        xs = np.linspace(xyz[s, 0].min(), xyz[s, 0].max(), 400)
        fx = (xs - org[0]) / res
        fy = np.full_like(xs, (yc - org[1]) / res)
        gz = ndimage.map_coordinates(dtm, [fx, fy], order=1, mode="nearest")
        gv = ndimage.map_coordinates(valid.astype(float), [fx, fy], order=1, mode="nearest")
        a.plot(xs, np.where(gv > 0.99, gz, np.nan), "r-", lw=2, label="DTM")
        a.plot(xs, np.where(gv > 0.99, gz + 1.3, np.nan), "b--", lw=1, label="1.3 m")
        a.set_title(f"cross-section y = {yc} m (+-2 m)")
        a.set_xlabel("x (m)"); a.set_ylabel("z (m)")
        a.legend(); a.set_aspect("equal")

    fin = np.isfinite(h)
    ax[1, 1].hist(h[fin], bins=200, color="seagreen")
    ax[1, 1].set_yscale("log")
    ax[1, 1].set_title("height above ground")
    ax[1, 1].set_xlabel("m")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def plot_stem_map(path, df, dtm, valid, org, res):
    fig, ax = plt.subplots(figsize=(11, 10))
    ext = [org[0], org[0] + dtm.shape[0] * res, org[1], org[1] + dtm.shape[1] * res]
    ax.imshow(np.where(valid, dtm, np.nan).T, origin="lower", extent=ext,
              cmap="Greys", alpha=0.45)
    colors = {"good": "#1a7f37", "fair": "#d29922", "poor": "#cf222e"}
    for q, c in colors.items():
        s = df[df["quality"] == q]
        if len(s):
            ax.scatter(s["x"], s["y"], s=np.clip(s["dbh_cm"], 4, 80) * 2.2,
                       facecolor="none", edgecolor=c, lw=1.4,
                       label=f"{q} (n={len(s)})")
    ax.set_aspect("equal")
    ax.set_title("stem map - circle size proportional to DBH")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.legend(loc="upper right")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def plot_inventory(path, df):
    g = df[df["quality"].isin(["good", "fair"])]
    fig, ax = plt.subplots(2, 3, figsize=(17, 9))
    ax[0, 0].hist(g["dbh_cm"], bins=30, color="#3d5a80")
    ax[0, 0].set_title("DBH (cm)")
    ax[0, 1].hist(g["height_m"], bins=30, color="#3d5a80")
    ax[0, 1].set_title("tree height (m)")
    ax[0, 2].scatter(g["dbh_cm"], g["height_m"], s=14, c="#3d5a80", alpha=0.7)
    ax[0, 2].set_xlabel("DBH (cm)"); ax[0, 2].set_ylabel("height (m)")
    ax[0, 2].set_title("height-diameter relation")
    colors = {"good": "#1a7f37", "fair": "#d29922", "poor": "#cf222e"}
    ax[1, 0].hist([df.loc[df["quality"] == q, "dbh_arc"] for q in colors],
                  bins=25, stacked=True, color=list(colors.values()),
                  label=list(colors))
    ax[1, 0].set_title("stem arc coverage (1.0 = seen all round)")
    ax[1, 0].legend()
    if "dist_from_scan_centre_m" in df:
        for q, c in colors.items():
            s = df[df["quality"] == q]
            ax[1, 1].scatter(s["dist_from_scan_centre_m"], s["dbh_arc"],
                             s=16, c=c, alpha=0.75, label=q)
        ax[1, 1].set_xlabel("distance from scan centre (m)")
        ax[1, 1].set_ylabel("arc coverage")
        ax[1, 1].set_title("occlusion grows with range from the scanner")
        ax[1, 1].legend()
    else:
        ax[1, 1].hist(df["dbh_rmse_cm"], bins=30, color="#8d6a9f")
        ax[1, 1].set_title("circle-fit residual (cm)")
    ax[1, 2].scatter(g["dbh_cm"], g["crown_diameter_m"], s=14, c="#3d5a80", alpha=0.7)
    ax[1, 2].set_xlabel("DBH (cm)"); ax[1, 2].set_ylabel("crown diameter (m)")
    ax[1, 2].set_title("crown-diameter relation")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def plot_segmentation(path, xyz, labels, yc=0.0, band=3.0):
    """Coloured cross-section: the honest way to see if crowns were split well."""
    s = (np.abs(xyz[:, 1] - yc) < band) & (labels >= 0)
    fig, ax = plt.subplots(figsize=(16, 6))
    rng = np.random.default_rng(0)
    palette = rng.permutation(plt.cm.tab20(np.linspace(0, 1, 20)))
    ax.scatter(xyz[s, 0], xyz[s, 2], s=0.6, c=palette[labels[s] % 20], lw=0)
    u = (~s) & (labels < 0) & (np.abs(xyz[:, 1] - yc) < band)
    ax.scatter(xyz[u, 0], xyz[u, 2], s=0.3, c="0.8", lw=0)
    ax.set_title(f"tree instances, cross-section y = {yc} m (+-{band} m); "
                 "grey = unassigned")
    ax.set_xlabel("x (m)"); ax.set_ylabel("z (m)"); ax.set_aspect("equal")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def plot_circle_fits(path, xyz, h, labels, df, seeds, cfg, n=12):
    """Show the actual DBH slices - the check that catches a silently bad fit."""
    from .measure import _basis, _stem_axis
    # a spread across the DBH range, not the n biggest - the biggest are the
    # ones most likely to have swallowed a neighbour and are not typical
    pool = df[df["quality"].isin(["good", "fair"])].sort_values("dbh_cm")
    if len(pool) == 0:
        pool = df
    take = np.unique(np.linspace(0, len(pool) - 1, min(n, len(pool))).astype(int))
    sel = pool.iloc[take]

    rows = int(np.ceil(len(sel) / 4))
    fig, axes = plt.subplots(rows, 4, figsize=(15, 3.7 * rows), squeeze=False)
    for a in axes.ravel():
        a.axis("off")
    for a, (_, r) in zip(axes.ravel(), sel.iterrows()):
        tid = int(r["tree_id"])
        seed = seeds[tid]
        axis, origin = _stem_axis(seed)
        e1, e2 = _basis(axis)
        hz = r["dbh_height_used"] if np.isfinite(r["dbh_height_used"]) else cfg.dbh_height
        m = (labels == tid) & (np.abs(h - hz) <= cfg.dbh_slice_thickness / 2)
        q = xyz[m].astype(np.float64) - origin
        uv = np.column_stack([q @ e1, q @ e2])
        lim = max(0.25, 2.5 * seed["stack_diameter"])
        near = np.hypot(uv[:, 0], uv[:, 1]) < lim
        a.axis("on")
        a.scatter(uv[~near, 0], uv[~near, 1], s=2, c="0.78", lw=0)
        a.scatter(uv[near, 0], uv[near, 1], s=4, c="0.25", lw=0)
        rr = r["dbh_cm"] / 200
        cu = r["fit_u"] if np.isfinite(r["fit_u"]) else 0.0
        cv = r["fit_v"] if np.isfinite(r["fit_v"]) else 0.0
        t = np.linspace(0, 2 * np.pi, 200)
        a.plot(cu + rr * np.cos(t), cv + rr * np.sin(t), "r-", lw=1.6)
        a.set_xlim(cu - lim, cu + lim)
        a.set_ylim(cv - lim, cv + lim)
        a.set_aspect("equal")
        a.set_title(f"#{tid}  DBH {r['dbh_cm']:.1f} cm  [{r['quality']}]\n"
                    f"arc {r['dbh_arc']:.2f}  rmse {r['dbh_rmse_cm']:.1f} cm", fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)
