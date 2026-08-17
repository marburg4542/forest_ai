"""Interactive Plotly figures for the web interface.

The 3-D view deliberately shows a random subset rather than all 6.5 M assigned
points.  A browser can be pushed to a couple of hundred thousand markers before
orbiting becomes unusable, and a uniform random sample preserves exactly the
thing the view is for - seeing whether stems and crowns were split sensibly.
Anyone who needs every point should open the exported .las in CloudCompare.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

QUALITY_COLORS = {"good": "#1a7f37", "fair": "#d29922", "poor": "#cf222e"}

# a categorical palette that stays distinguishable when cycled over many trees
_TREE_PALETTE = np.array([
    "#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f", "#edc948",
    "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac", "#86bcb6", "#d37295",
    "#a0cbe8", "#ffbe7d", "#8cd17d", "#b6992d", "#499894", "#f1ce63",
    "#d4a6c8", "#fabfd2",
])


def sample_points(result, max_points=150_000, include_ground=False, seed=0):
    """Random subset of the cloud plus its tree labels."""
    rng = np.random.default_rng(seed)
    mask = result.labels >= 0
    if include_ground:
        mask = np.ones(len(result.labels), dtype=bool)
    idx = np.flatnonzero(mask)
    if len(idx) > max_points:
        idx = rng.choice(idx, max_points, replace=False)
        idx.sort()
    return result.xyz[idx], result.labels[idx], result.h[idx]


def _discrete_scale(colors):
    """Plotly colorscale with hard bands, one per entry of `colors`."""
    n = len(colors)
    scale = []
    for i, c in enumerate(colors):
        scale += [[i / n, c], [(i + 1) / n, c]]
    return scale


def cloud_figure(result, max_points=150_000, color_by="tree", include_ground=False,
                 point_size=1.4):
    """3-D scatter of the segmented cloud.

    Colours are sent as a numeric array plus a colorscale, never as an array of
    hex strings: with 150k points the string form is a megabyte-plus of JSON and
    roughly doubles the time to first paint, while a float array goes over the
    wire as binary.
    """
    xyz, lab, h = sample_points(result, max_points, include_ground)
    centre = xyz.mean(axis=0)
    p = (xyz - centre).astype(np.float32)

    if color_by == "tree":
        palette = ["#d0d0d0"] + list(_TREE_PALETTE)      # index 0 = unassigned
        vals = np.where(lab >= 0, lab % len(_TREE_PALETTE) + 1, 0).astype(np.float32)
        marker = dict(size=point_size, color=vals,
                      colorscale=_discrete_scale(palette),
                      cmin=0, cmax=len(palette), showscale=False)
    elif color_by == "quality":
        order = ["poor", "fair", "good"]
        q = (result.df.set_index("tree_id")["quality"]
             .reindex(range(len(result.seeds))).fillna("poor"))
        code = q.map({k: i + 1 for i, k in enumerate(order)}).values
        vals = np.where(lab >= 0, code[np.clip(lab, 0, None)], 0).astype(np.float32)
        palette = ["#d0d0d0"] + [QUALITY_COLORS[k] for k in order]
        marker = dict(size=point_size, color=vals,
                      colorscale=_discrete_scale(palette),
                      cmin=0, cmax=len(palette), showscale=False)
    else:  # height above ground
        marker = dict(size=point_size, color=np.nan_to_num(h).astype(np.float32),
                      colorscale="Viridis",
                      colorbar=dict(title="m above<br>ground"), cmin=0,
                      cmax=float(np.nanpercentile(h, 99)))

    fig = go.Figure(go.Scatter3d(
        x=p[:, 0], y=p[:, 1], z=p[:, 2], mode="markers", marker=marker,
        hoverinfo="skip", showlegend=False))
    fig.update_layout(
        scene=dict(aspectmode="data",
                   xaxis_title="x (m)", yaxis_title="y (m)", zaxis_title="z (m)",
                   xaxis=dict(backgroundcolor="rgba(0,0,0,0)"),
                   yaxis=dict(backgroundcolor="rgba(0,0,0,0)"),
                   zaxis=dict(backgroundcolor="rgba(0,0,0,0)")),
        margin=dict(l=0, r=0, t=0, b=0), height=680,
        uirevision="cloud")   # keep the camera when a widget changes
    return fig, len(xyz)


def single_tree_figure(result, tree_id, max_points=40_000, seed=0):
    """One tree in 3-D, with the rest of the stand faded around it."""
    rng = np.random.default_rng(seed)
    sel = np.flatnonzero(result.labels == tree_id)
    if len(sel) > max_points:
        sel = rng.choice(sel, max_points, replace=False)
    tree = result.xyz[sel]
    centre = tree.mean(axis=0)

    near = np.flatnonzero(
        (result.labels >= 0) & (result.labels != tree_id)
        & (np.abs(result.xyz[:, 0] - centre[0]) < 8)
        & (np.abs(result.xyz[:, 1] - centre[1]) < 8))
    if len(near) > 25_000:
        near = rng.choice(near, 25_000, replace=False)

    fig = go.Figure()
    if len(near):
        q = result.xyz[near] - centre
        fig.add_trace(go.Scatter3d(x=q[:, 0], y=q[:, 1], z=q[:, 2], mode="markers",
                                   marker=dict(size=1, color="#dcdcdc"),
                                   name="neighbours", hoverinfo="skip"))
    p = tree - centre
    fig.add_trace(go.Scatter3d(x=p[:, 0], y=p[:, 1], z=p[:, 2], mode="markers",
                               marker=dict(size=1.8, color="#1a7f37"),
                               name=f"tree {tree_id}", hoverinfo="skip"))
    fig.update_layout(scene=dict(aspectmode="data"),
                      margin=dict(l=0, r=0, t=0, b=0), height=560,
                      uirevision=f"tree{tree_id}")
    return fig


def stem_map_figure(result, colour="quality"):
    """Plan view of the stand: one marker per stem, hoverable."""
    df = result.df
    fig = go.Figure()

    if result.hull is not None:
        poly = np.vstack([result.hull, result.hull[:1]])
        fig.add_trace(go.Scatter(x=poly[:, 0], y=poly[:, 1], mode="lines",
                                 line=dict(color="#999", dash="dot"),
                                 name="stocked area", hoverinfo="skip"))

    hover = ("tree %{customdata[0]}<br>DBH %{customdata[1]:.1f} cm"
             "<br>height %{customdata[2]:.1f} m<br>arc %{customdata[3]:.2f}"
             "<br>range %{customdata[4]:.0f} m<extra></extra>")
    cols = ["tree_id", "dbh_cm", "height_m", "dbh_arc", "dist_from_scan_centre_m"]

    if colour == "quality":
        for q, c in QUALITY_COLORS.items():
            s = df[df["quality"] == q]
            if not len(s):
                continue
            fig.add_trace(go.Scatter(
                x=s["x"], y=s["y"], mode="markers", name=f"{q} ({len(s)})",
                marker=dict(size=np.clip(s["dbh_cm"], 5, 45) * 0.55,
                            color=c, opacity=0.85,
                            line=dict(width=0.5, color="white")),
                customdata=s[cols].values, hovertemplate=hover))
    else:
        s = df.dropna(subset=[colour])
        fig.add_trace(go.Scatter(
            x=s["x"], y=s["y"], mode="markers", name=colour,
            marker=dict(size=np.clip(s["dbh_cm"], 5, 45) * 0.55,
                        color=s[colour], colorscale="Viridis",
                        colorbar=dict(title=colour),
                        line=dict(width=0.5, color="white")),
            customdata=s[cols].values, hovertemplate=hover))

    fig.add_trace(go.Scatter(
        x=[result.scan_centre[0]], y=[result.scan_centre[1]], mode="markers+text",
        marker=dict(size=15, color="black", symbol="x"),
        text=["scanner"], textposition="top center", name="scan centre",
        hoverinfo="skip"))

    fig.update_layout(height=680, xaxis_title="x (m)", yaxis_title="y (m)",
                      legend=dict(orientation="h", y=1.02, yanchor="bottom"),
                      margin=dict(l=0, r=0, t=30, b=0), uirevision="stemmap")
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


def dbh_slice_figure(result, tree_id):
    """The actual points the DBH was fitted to, with the fitted circle."""
    from .measure import _basis, _stem_axis

    r = result.df[result.df["tree_id"] == tree_id]
    if not len(r):
        return None
    r = r.iloc[0]
    seed = result.seeds[tree_id]
    axis, origin = _stem_axis(seed)
    e1, e2 = _basis(axis)
    hz = r["dbh_height_used"] if np.isfinite(r["dbh_height_used"]) else result.cfg.dbh_height
    m = ((result.labels == tree_id)
         & (np.abs(result.h - hz) <= result.cfg.dbh_slice_thickness / 2))
    q = result.xyz[m].astype(np.float64) - origin
    uv = np.column_stack([q @ e1, q @ e2])
    lim = max(0.25, 2.5 * seed["stack_diameter"])
    near = np.hypot(uv[:, 0], uv[:, 1]) < lim

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=uv[~near, 0], y=uv[~near, 1], mode="markers",
                             marker=dict(size=3, color="#d8d8d8"),
                             name="outside search radius", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=uv[near, 0], y=uv[near, 1], mode="markers",
                             marker=dict(size=4, color="#333"),
                             name="fitted points", hoverinfo="skip"))
    t = np.linspace(0, 2 * np.pi, 200)
    rr = r["dbh_cm"] / 200
    cu = r["fit_u"] if np.isfinite(r["fit_u"]) else 0.0
    cv = r["fit_v"] if np.isfinite(r["fit_v"]) else 0.0
    fig.add_trace(go.Scatter(x=cu + rr * np.cos(t), y=cv + rr * np.sin(t),
                             mode="lines", line=dict(color="#cf222e", width=2.5),
                             name=f"DBH {r['dbh_cm']:.1f} cm"))
    fig.update_layout(height=430, margin=dict(l=0, r=0, t=30, b=0),
                      xaxis_title="m", yaxis_title="m",
                      # plotly renders titles as its own mini-markup and does
                      # not decode HTML entities, so use the character itself
                      title=f"slice at {hz:.2f} m · arc {r['dbh_arc']:.2f} "
                            f"· residual {r['dbh_rmse_cm']:.1f} cm "
                            f"· {r['quality']}")
    fig.update_xaxes(range=[cu - lim, cu + lim])
    fig.update_yaxes(range=[cv - lim, cv + lim], scaleanchor="x", scaleratio=1)
    return fig


def inventory_figure(result):
    """DBH / height distributions and the occlusion-versus-range relationship."""
    from plotly.subplots import make_subplots

    df, g = result.df, result.usable
    fig = make_subplots(rows=2, cols=2, subplot_titles=(
        "DBH distribution (good + fair)", "height distribution (good + fair)",
        "height-diameter relation", "arc coverage vs range from the scanner"))

    fig.add_trace(go.Histogram(x=g["dbh_cm"], nbinsx=30, marker_color="#3d5a80",
                               showlegend=False), 1, 1)
    fig.add_trace(go.Histogram(x=g["height_m"], nbinsx=30, marker_color="#3d5a80",
                               showlegend=False), 1, 2)
    for q, c in QUALITY_COLORS.items():
        s = df[df["quality"] == q]
        if not len(s):
            continue
        fig.add_trace(go.Scatter(x=s["dbh_cm"], y=s["height_m"], mode="markers",
                                 marker=dict(size=6, color=c), name=q,
                                 legendgroup=q), 2, 1)
        fig.add_trace(go.Scatter(x=s["dist_from_scan_centre_m"], y=s["dbh_arc"],
                                 mode="markers", marker=dict(size=6, color=c),
                                 name=q, legendgroup=q, showlegend=False), 2, 2)
    fig.update_xaxes(title_text="DBH (cm)", row=1, col=1)
    fig.update_xaxes(title_text="height (m)", row=1, col=2)
    fig.update_xaxes(title_text="DBH (cm)", row=2, col=1)
    fig.update_yaxes(title_text="height (m)", row=2, col=1)
    fig.update_xaxes(title_text="distance from scanner (m)", row=2, col=2)
    fig.update_yaxes(title_text="arc coverage", row=2, col=2)
    fig.update_layout(height=760, margin=dict(l=0, r=0, t=40, b=0),
                      legend=dict(orientation="h", y=1.06, yanchor="bottom"))
    return fig
