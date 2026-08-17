"""forest_ai - individual tree extraction from forest point clouds (.las/.laz).

Pipeline stages (see run_pipeline.py):
    1. las_io      read/write .las with low memory footprint
    2. preprocess  noise removal, voxel downsampling
    3. ground      CSF ground filter, DTM, height normalisation
    4. segment     stem detection + geodesic tree instance segmentation
    5. measure     per-tree DBH, height, crown metrics
    6. features    per-tree descriptors for species classification
"""

__version__ = "0.1.0"
