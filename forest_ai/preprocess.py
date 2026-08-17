"""Noise removal and voxel downsampling.

Both operations are done with integer voxel keys and np.unique rather than a
KD-tree: on 15.8 M points a kNN-based statistical outlier filter takes minutes
and several GB, while the voxel approach is a few seconds and one extra int64
array.
"""

from __future__ import annotations

import numpy as np


def _voxel_keys(xyz: np.ndarray, voxel: float) -> tuple[np.ndarray, np.ndarray]:
    """Integer voxel index per point, packed into a single int64 key."""
    origin = xyz.min(axis=0)
    idx = np.floor((xyz - origin) / voxel).astype(np.int64)
    dims = idx.max(axis=0) + 1
    if dims[0] * dims[1] * dims[2] > 2**62:
        raise ValueError("voxel grid too large for int64 packing; increase voxel size")
    key = (idx[:, 0] * dims[1] + idx[:, 1]) * dims[2] + idx[:, 2]
    return key, dims


def remove_isolated(xyz: np.ndarray, voxel: float = 0.5, min_pts: int = 4
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Drop points sitting in sparsely populated voxels.

    Returns (filtered_xyz, keep_mask).  This is what removes the handful of
    stray points far above the canopy that would otherwise ruin the tree-height
    statistic.
    """
    key, _ = _voxel_keys(xyz, voxel)
    uniq, inv, counts = np.unique(key, return_inverse=True, return_counts=True)
    keep = counts[inv] >= min_pts
    return xyz[keep], keep


def voxel_downsample(xyz: np.ndarray, voxel: float, return_index: bool = False):
    """Keep one representative point per voxel (the first encountered).

    Keeping an original point rather than the voxel centroid matters for stem
    fitting: centroids of a partially scanned stem drift inwards and shrink the
    fitted diameter.
    """
    key, _ = _voxel_keys(xyz, voxel)
    _, first = np.unique(key, return_index=True)
    first.sort()
    if return_index:
        return xyz[first], first
    return xyz[first]


def summary(xyz: np.ndarray) -> str:
    lo, hi = xyz.min(axis=0), xyz.max(axis=0)
    area = (hi[0] - lo[0]) * (hi[1] - lo[1])
    return (f"{len(xyz):,} pts | "
            f"x {lo[0]:.1f}..{hi[0]:.1f}  y {lo[1]:.1f}..{hi[1]:.1f}  z {lo[2]:.1f}..{hi[2]:.1f} | "
            f"{area/10000:.2f} ha | {len(xyz)/max(area,1):.0f} pts/m2")
