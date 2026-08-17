---
title: forest_ai
emoji: 🌲
colorFrom: green
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
short_description: Individual tree extraction, DBH and height from .las clouds
---

# forest_ai

Individual tree extraction from forest point clouds. Give it a `.las`/`.laz` file
and it classifies ground, builds a terrain model, finds every stem, splits the
cloud into individual trees, and measures **DBH, height and crown** for each one
— with an explicit quality flag on every measurement.

Pure NumPy/SciPy/scikit-learn. No GPU, no PDAL, no Open3D, no build step.

*[อ่านเป็นภาษาไทย →](README.th.md)*

---

## Quick start

```bash
pip install -r requirements.txt

python serve.py           # web interface at http://localhost:8000
python run_pipeline.py    # or the command line, writing to outputs/
```

Both call the same `forest_ai/pipeline.py`, so the numbers are always identical.
A cold run on 15.8 M points takes about 90 s on 8 cores; re-runs reuse the
cached terrain model and take about 12 s.

## What it needs

- **A ground-based scan** — terrestrial (TLS) or mobile (MLS) laser scanning, or
  ground-level photogrammetry. Stems have to be visible at breast height.
- **Not airborne data.** A drone or aircraft cloud sees the canopy from above and
  barely any stem, so DBH cannot be measured from it at all.
- Dense enough to resolve a stem cross-section — roughly 100 points/m² or better.

RGB, intensity and return number are all optional; the pipeline works from
geometry alone.

---

## What it produces

| File | Contents |
|---|---|
| `trees.csv` | one row per tree: position, DBH, height, crown metrics, quality flags |
| `tree_features.csv` | 51 geometric descriptors per tree, ready for species classification |
| `Forest_segmented.las` | the cloud with a `tree_id` and `height_norm` scalar field |
| `qc_01…05.png` | ground model, stem map, distributions, instance cross-section, DBH fits |

Key columns in `trees.csv`:

```
tree_id  x  y                stem position, in the cloud's own coordinates
dbh_cm                       diameter at 1.3 m
height_m                     99.5th percentile height (robust to stray points)
crown_base_m  crown_diameter_m  crown_area_m2  crown_volume_m3
basal_area_m2  h_d_ratio  stem_lean_deg
quality                      good / fair / poor   <-- always filter on this
dbh_arc                      0-1, how much of the stem circumference was seen
dbh_rmse_cm                  circle-fit residual
dbh_vs_stack                 agreement between two independent diameter estimates
n_stem_layers                how many 20 cm layers stacked (4-10)
dist_from_scan_centre_m      the strongest predictor of measurement quality
structural_group             K-means cluster — NOT a species prediction
```

**Always filter on `quality == "good"` for numbers you intend to rely on.** On
the development dataset the height–diameter correlation is **-0.12 across all
trees** (meaningless) but **+0.45 among good trees** (biologically sensible) —
that difference is the quality flag doing its job.

---

## How it works

**1 · Read and denoise.** X/Y/Z only, in chunks, as float32 — 15.8 M points fit in
190 MB. Isolated points are dropped with a voxel-count filter rather than a
KD-tree statistical filter: same result on stray points, dozens of times faster.

**2 · Ground, DTM, height normalisation.** A vectorised NumPy re-implementation of
the **Cloth Simulation Filter** (Zhang et al. 2016) — the cloud is inverted so
terrain becomes the top surface, then a grid of mass points falls onto it under
Verlet integration with spring constraints. Two failure modes specific to
single-scan data are handled explicitly: ground occluded by understorey makes the
lowest point in a cell a branch several metres up (rejected by comparing against
a 5 m local median), and beyond the scan's useful radius there is no ground at
all, so cells too far from any real observation are flagged invalid rather than
interpolated from nothing.

**3 · Stem detection by circle stacking.** Each 20 cm layer between 0.6 and 2.6 m
is clustered in 2-D and RANSAC-fitted to a circle; circles that stack vertically
are linked with union–find, and a stem must appear in at least 4 layers.
A single thick slice projected to 2-D — the obvious approach — smears a tapering
or leaning stem into a wide annulus that defeats the fit and lets neighbouring
stems merge. Random foliage never produces four consistent circles above one
another, which is what makes this selective.

**4 · Instance segmentation by geodesic growing.** A kNN graph over 15 cm voxels,
then **multi-source Dijkstra** from every stem seed, with vertical travel made
cheaper than horizontal so paths run up the stem before spreading into the crown.
Geodesic rather than Euclidean distance is what splits interlocking crowns
correctly: a point on a branch is close *through the wood* to its own stem even
when a neighbour's crown is physically nearer.

**5 · Measurement.** Height is the 99.5th percentile, not the maximum, so one
stray point cannot inflate it. DBH is fitted on a slice taken **perpendicular to
the stem axis**, not to the vertical — a stem leaning 30° measured on a
horizontal cut reads 15% too fat. RANSAC then a geometric least-squares refine
with a soft-L1 loss; the algebraic fit is biased when a stem is seen from one
side, which is the normal case for a single scan.

**6 · Features and grouping.** 51 descriptors per tree (size, crown proportions,
relative-height percentiles, density and radius profiles, PCA shape of the whole
tree and of the crown), plus K-means grouping as a label-free stand-in until real
species labels exist.

**7 · QC figures.** Every stage produces one. **Look at them before trusting any
number** — every bug found while building this pipeline showed up in a figure and
none of them showed up in the statistics.

---

## Quality flags

| Level | Criteria |
|---|---|
| **good** | axis fit + `arc ≥ 0.60` + residual ≤ 2 cm + ≥ 6 layers + two estimates agree within 60% |
| **fair** | axis fit + `arc ≥ 0.35` + residual ≤ 3 cm + two estimates agree within 120% |
| **poor** | anything else |

`dbh_arc` is the decisive one. Below roughly a third of the circumference a
circle is free to slide outwards along the arc — that is how a 14 cm stem gets
reported at 45 cm — and **the residual stays small the whole time**, so residual
alone cannot catch it.

---

## Validating against field data

```bash
python evaluate_against_reference.py field_plot.csv --quality good --max-dist 1.5
```

Works against a field survey or another tool's output (FSCT, TreeLS, TreeLearn).
Matching uses optimal assignment (Hungarian), not greedy nearest-neighbour: in a
stand where stems are ~2 m apart, greedy matching pairs predictions to the wrong
reference tree and inflates the error. Reports precision / recall / F1 /
commission / omission, plus bias, RMSE, MAE and R² for DBH and height.

The reference CSV needs `x` and `y` in the cloud's coordinates; use
`--map DBH=dbh_cm --map X=x` if the column names differ, or map them
interactively in the web interface's Validation view.

---

## Web interface

```bash
python serve.py --port 8000 --host 127.0.0.1
```

FastAPI backend, plain HTML/CSS/JS front-end, no build step. `plotly.min.js` is
copied out of the installed plotly package on first start, so the page works
offline and never calls a CDN.

Seven views: **Overview** (stand statistics and downloads) · **Stem map**
(hoverable plan view) · **3D view** (orbitable cloud, coloured by tree, height or
quality) · **Trees** (sortable, filterable table; click a row to see the actual
DBH slice it was fitted to) · **Species** · **Validation** · **QC figures**.

The 3-D view shows a random sample — 150 k of 6.5 M points by default. That is
enough to judge whether segmentation is sensible; for full resolution open the
exported `.las` in CloudCompare and colour by `tree_id`.

### Installing it as a desktop app

The interface is a PWA, so the browser can install it: start the server, open
`http://localhost:8000`, and use **Install as an app** in the sidebar (or the
browser's own install control). It then opens in its own window with its own
icon and no address bar.

Two things worth knowing:

- **It still needs the server.** Every measurement happens in the Python
  process, so `python serve.py` has to be running. The worker caches the
  interface, not the computation — open the app without the server and you get
  a page telling you which command to run rather than a browser error.
- **Installing only works over `localhost`.** Browsers require a secure context
  for service workers, and plain `http://` to a LAN address is not one, so
  reaching the app from another machine works but cannot be installed.

The ~5 MB shell (mostly `plotly.min.js`) is precached, so start-up is instant.
Nothing under `/api/` is ever cached — those responses are per-session and
change with every run.

### REST API

The front-end talks to the backend over plain REST, so scripts and QGIS can use
the same endpoints. Interactive docs at `/api/docs`.

```
GET  /api/clouds  /api/params  /api/config  /api/header?las=
POST /api/upload                        streaming, size-capped
POST /api/run     GET /api/job          start a run, poll progress
GET  /api/summary /api/trees /api/species
GET  /api/figure/{cloud,stemmap,inventory,dbh_slice/{id},tree3d/{id}}
POST /api/evaluate  /api/write_outputs
GET  /api/download/{trees,features}.csv
```

Every request carries an `X-Session-Id` header; results, uploads and outputs are
scoped to it.

### Configuration

| Variable | Default | Purpose |
|---|---|---|
| `FAI_MAX_SESSIONS` | 2 | results kept in memory at once (~700 MB each) |
| `FAI_MAX_UPLOAD_MB` | 300 | largest accepted upload |
| `FAI_ALLOW_LOCAL_CLOUDS` | true | offer `.las` files sitting next to the server |
| `FAI_ALLOW_SEGMENTED_LAS` | true | allow writing the ~250 MB segmented cloud |

The Docker image sets the last two to `false`, so a shared deployment neither
exposes the host's files nor fills its disk.

**Single worker only.** A result is ~700 MB of numpy arrays in process memory and
cannot be shared between workers; `serve.py` and the Dockerfile both enforce it.
Concurrency between visitors comes from the session table, and the pipeline
itself runs on a one-slot thread pool so a second visitor queues rather than
competing for cores.

---

## Deploying

```bash
docker build -t forest_ai .
docker run -p 7860:7860 forest_ai
```

Any host that runs a container works. The `app_port: 7860` and the YAML block at
the top of this file are there for **Hugging Face Spaces**:

```bash
git remote add space https://huggingface.co/spaces/<user>/<space>
git push space main
```

> **Hugging Face is no longer free for this.** Since July 2026, hosting Gradio or
> Docker Spaces on free `cpu-basic` requires a PRO subscription ($9/month); only
> Static Spaces remain free, and those cannot run Python. Plan accordingly, or
> pick another host — the same `Dockerfile` runs anywhere.

Sizing for any host: the pipeline peaks around **700 MB of RSS** on a 15.8 M
point cloud, so a 512 MB instance is not enough. Two vCPUs turn the ~90 s run
(measured on 8 cores) into a few minutes.

---

## Limitations

1. **Occlusion sets the ceiling, not the algorithm.** On the single-scan
   development dataset, measurement quality tracks range from the scanner almost
   perfectly — median 17 m for *good* stems, 24 m for *fair*, 31 m for *poor* —
   and past ~28 m almost nothing reaches `arc > 0.5`. Trees are visibly present
   in the cloud beyond that and still go undetected. **The only fix is scanning
   from more positions and registering them.** No algorithm recovers points that
   were never returned.

2. **Per-hectare figures need a stocked area.** They are divided by the convex
   hull of the detected stems, not by the extent of the valid terrain model,
   which on the development data also covers open ground and would understate
   density and basal area threefold.

3. **Species classification is not implemented, only prepared.** The feature
   extractor and a cross-validated RandomForest scaffold are there, but there are
   no labels. `structural_group` is a K-means grouping of geometry: two species
   with the same architecture land together, and one species splits across groups
   when some individuals are suppressed. Its purpose is to cut field work —
   identify 10–15 trees per group instead of all of them.

4. **`dbh_rmse_cm` is not very discriminative.** It clusters at 0.9–1.2 cm for
   nearly every tree because the soft-L1 loss saturates at `f_scale=0.02`. Filter
   on `dbh_arc`.

5. **Accuracy against ground truth is unmeasured.** The quality thresholds
   (`arc ≥ 0.60 / 0.35`) come from first principles, not from calibration. Until
   30–50 field-measured trees have been compared, the real RMSE is unknown.

---

## Layout

```
forest_ai/          the pipeline — no web framework anywhere in here
  config.py         every tunable parameter, in one dataclass
  pipeline.py       all stages as one call, with a progress callback
  las_io.py         chunked, low-memory .las reading and writing
  preprocess.py     voxel-key denoising and downsampling
  ground.py         CSF, DTM with a validity mask, height normalisation
  fitting.py        RANSAC circle, geometric refine, arc coverage, PCA axis
  segment.py        circle stacking, union-find linking, geodesic segmentation
  measure.py        per-tree DBH / height / crown and the quality flag
  features.py       51 descriptors, K-means grouping, RandomForest scaffold
  evaluate.py       optimal-assignment matching and error statistics
  qc.py             the five matplotlib QC figures
  webviz.py         interactive plotly figures
web/                a thin HTTP layer over pipeline.py
  server.py  sessions.py  params.py  vendor.py
  static/           index.html, app.css, app.js, sw.js, manifest, icons
tools/make_icons.py generates the PWA icon set
serve.py  run_pipeline.py  evaluate_against_reference.py  Dockerfile
```

Tuning happens in `forest_ai/config.py` (or the web sliders). The terrain model
is cached in `.cache/`, keyed on the input file's path, size and mtime plus every
setting that affects it, so pointing at a different cloud can never reuse the
wrong terrain.

---

## Notes

Built against a 15.8 M point single-scan TLS plot of a plantation stand
(167 × 142 m extent, ~666 points/m², no RGB or intensity). Point clouds are
excluded from this repository — they are far larger than GitHub allows.
