"""Low-memory .las / .laz reading and writing."""

from __future__ import annotations

import numpy as np
import laspy


def describe(path: str) -> dict:
    """Header summary without loading any points."""
    with laspy.open(path) as f:
        h = f.header
        return {
            "version": f"{h.version.major}.{h.version.minor}",
            "point_format": h.point_format.id,
            "point_count": h.point_count,
            "scales": tuple(h.scales),
            "offsets": tuple(h.offsets),
            "mins": tuple(h.mins),
            "maxs": tuple(h.maxs),
            "extra_dims": [d.name for d in h.point_format.extra_dimensions],
            "dimensions": [d.name for d in h.point_format.dimensions],
            "crs": str(h.parse_crs()) if h.vlrs else None,
        }


def read_xyz(path: str, chunk_size: int = 4_000_000, dtype=np.float32) -> np.ndarray:
    """Read only X/Y/Z, in chunks, as an (N, 3) array.

    float32 keeps the whole 15.8 M point cloud under 200 MB.  With coordinates
    within +-1000 m the resolution is ~1e-4 m, an order of magnitude finer than
    the 1 mm scale factor of the file, so nothing measurable is lost.
    """
    with laspy.open(path) as f:
        n = f.header.point_count
        out = np.empty((n, 3), dtype=dtype)
        i = 0
        for pts in f.chunk_iterator(chunk_size):
            m = len(pts)
            out[i:i + m, 0] = pts.x
            out[i:i + m, 1] = pts.y
            out[i:i + m, 2] = pts.z
            i += m
    return out[:i]


def write_las(path: str, xyz: np.ndarray, src_path: str | None = None,
              extra: dict[str, np.ndarray] | None = None) -> None:
    """Write points plus optional per-point attributes as ExtraBytes.

    `extra` maps dimension name -> array.  Integer arrays become uint32,
    floating arrays become float32, which is what CloudCompare expects for
    scalar fields.
    """
    if src_path is not None:
        with laspy.open(src_path) as f:
            src = f.header
        header = laspy.LasHeader(version="1.4", point_format=6)
        header.scales = src.scales
        header.offsets = src.offsets
    else:
        header = laspy.LasHeader(version="1.4", point_format=6)
        header.scales = np.array([0.001, 0.001, 0.001])
        header.offsets = np.floor(xyz.min(axis=0)).astype(np.float64)

    extra = extra or {}
    for name, arr in extra.items():
        kind = "uint32" if np.issubdtype(arr.dtype, np.integer) else "float32"
        header.add_extra_dim(laspy.ExtraBytesParams(name=name, type=kind))

    las = laspy.LasData(header)
    las.x = xyz[:, 0].astype(np.float64)
    las.y = xyz[:, 1].astype(np.float64)
    las.z = xyz[:, 2].astype(np.float64)
    for name, arr in extra.items():
        target = np.uint32 if np.issubdtype(arr.dtype, np.integer) else np.float32
        las[name] = arr.astype(target)
    las.write(path)
