"""Which pipeline settings the web form exposes, and their sane ranges.

`Config` has 41 fields; most are internals that should not be poked at from a
browser.  This is the curated subset, with the bounds and the explanation each
one needs.  The front-end builds the form from this, so adding a control here
is the only change needed to expose a new setting.
"""

from __future__ import annotations

from dataclasses import fields

from forest_ai.config import Config

# (name, label, min, max, step, help)
GROUPS: dict[str, list[tuple]] = {
    "stem detection": [
        ("stem_slice_lo", "search band bottom (m)", 0.3, 1.2, 0.1,
         "lowest height above ground searched for stem cross-sections"),
        ("stem_slice_hi", "search band top (m)", 1.5, 4.0, 0.1,
         "highest height searched"),
        ("stem_min_layers", "min stacked layers", 2, 8, 1,
         "how many 20 cm layers must line up vertically to accept a stem; "
         "lower finds more distant stems but adds false positives"),
        ("stem_dbscan_eps", "cluster radius (m)", 0.03, 0.20, 0.01,
         "DBSCAN neighbourhood within one layer; too large merges "
         "neighbouring stems"),
        ("stem_min_diameter", "min diameter (m)", 0.02, 0.20, 0.01,
         "reject anything thinner than this"),
        ("stem_max_diameter", "max diameter (m)", 0.3, 3.0, 0.1,
         "reject anything fatter than this"),
    ],
    "segmentation": [
        ("seg_voxel", "graph voxel (m)", 0.05, 0.40, 0.05,
         "resolution of the kNN graph used to grow trees; smaller is more "
         "accurate and much slower"),
        ("graph_max_edge", "max graph edge (m)", 0.2, 1.0, 0.05,
         "edges longer than this are cut, so growth cannot jump across gaps"),
        ("graph_z_weight", "vertical travel cost", 0.1, 1.0, 0.05,
         "below 1 makes paths prefer running up the stem before spreading "
         "into the crown"),
    ],
    "ground / DTM": [
        ("csf_cloth_res", "cloth resolution (m)", 0.2, 1.5, 0.1,
         "grid spacing of the simulated cloth"),
        ("csf_rigidness", "cloth rigidness", 1, 3, 1,
         "3 for flat terrain, 1 for steep"),
        ("csf_threshold", "ground threshold (m)", 0.05, 0.5, 0.05,
         "point-to-cloth distance that still counts as ground"),
        ("dtm_res", "DTM cell size (m)", 0.1, 1.0, 0.05,
         "terrain raster resolution"),
    ],
}

_TYPES = {f.name: f.type for f in fields(Config)}
EXPOSED = {name for group in GROUPS.values() for (name, *_) in group}


def spec() -> list[dict]:
    """Form description for the front-end, with current defaults filled in."""
    default = Config()
    out = []
    for group, items in GROUPS.items():
        controls = []
        for name, label, lo, hi, step, help_ in items:
            controls.append({
                "name": name, "label": label, "min": lo, "max": hi,
                "step": step, "help": help_,
                "value": getattr(default, name),
                "integer": _TYPES[name] is int,
            })
        out.append({"group": group, "controls": controls})
    return out


def clean(values: dict) -> dict:
    """Keep only exposed keys and coerce to the type the dataclass declares.

    A browser sends every number as a JSON float; feeding 3.0 into an int field
    would make the cache key differ from the CLI's for identical settings.
    """
    out = {}
    for k, v in (values or {}).items():
        if k not in EXPOSED or v is None:
            continue
        out[k] = int(round(float(v))) if _TYPES[k] is int else float(v)
    return out
