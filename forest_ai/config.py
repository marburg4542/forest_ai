"""Tunable parameters for the whole pipeline.

Values are defaults for dense ground-based scans (TLS/MLS/photogrammetry,
several hundred points per square metre).  Every stage prints the values it
actually used so a run can be reproduced.
"""

from dataclasses import dataclass, asdict, field


@dataclass
class Config:
    # ---- input ------------------------------------------------------------
    las_path: str = "Forest.las"
    out_dir: str = "outputs"

    # ---- 2. preprocessing --------------------------------------------------
    # isolated-point removal: drop points whose 0.5 m voxel holds < min_pts
    noise_voxel: float = 0.5
    noise_min_pts: int = 4
    # working resolution for ground filtering / segmentation
    ground_voxel: float = 0.10
    seg_voxel: float = 0.15
    # resolution used when measuring stems (needs all the detail we can get)
    stem_voxel: float = 0.01

    # ---- 3. ground / DTM ---------------------------------------------------
    csf_cloth_res: float = 0.5      # cloth grid spacing (m)
    csf_rigidness: int = 3          # 1 steep terrain .. 3 flat terrain
    csf_time_step: float = 0.65
    csf_iterations: int = 500
    csf_threshold: float = 0.15     # point-to-cloth distance for "ground" (m)
    dtm_res: float = 0.25           # DTM raster cell size (m)

    # ---- 4. segmentation ---------------------------------------------------
    # stems are found by fitting a circle in each thin horizontal layer and
    # then linking circles that stack vertically
    stem_slice_lo: float = 0.60     # bottom of the search band (m above ground)
    stem_slice_hi: float = 2.60     # top of the search band
    stem_layer_thick: float = 0.20  # layer thickness
    stem_layer_step: float = 0.20   # spacing between layers
    stem_dbscan_eps: float = 0.07   # 2-D clustering radius within a layer (m)
    stem_dbscan_min_pts: int = 12
    stem_link_dist: float = 0.12    # + 0.35*r, max centre shift between layers
    stem_link_radius_ratio: float = 1.8     # max r ratio between linked layers
    stem_min_layers: int = 4        # a stem must appear in this many layers
    stem_min_diameter: float = 0.04         # reject anything thinner than 4 cm
    stem_max_diameter: float = 1.50         # ... or fatter than 1.5 m
    stem_max_rmse: float = 0.030            # circle fit residual (m)
    stem_min_verticality: float = 0.80      # |axis . z| of the stacked centres
    veg_min_height: float = 0.30    # points below this are ground/litter, ignored
    graph_k: int = 10               # kNN graph degree for geodesic growing
    graph_max_edge: float = 0.40    # cut edges longer than this (m)
    graph_z_weight: float = 0.35    # <1 makes vertical travel cheap -> follow stems

    # ---- 5. measurement ----------------------------------------------------
    dbh_height: float = 1.30        # breast height along the stem axis (m)
    dbh_slice_thickness: float = 0.12
    dbh_fallback_heights: tuple = (1.30, 1.50, 1.00, 1.70)
    ransac_iters: int = 300
    ransac_tol: float = 0.02        # inlier band around the fitted circle (m)
    ransac_min_inlier_frac: float = 0.35
    height_percentile: float = 99.5  # robust tree top
    min_tree_height: float = 2.0     # discard detections shorter than this

    # ---- misc --------------------------------------------------------------
    random_seed: int = 42
    n_jobs: int = -1

    def dump(self) -> dict:
        return asdict(self)
