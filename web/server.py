"""FastAPI layer over forest_ai.pipeline.

The pipeline produces a `Result` holding ~700 MB of numpy arrays that cannot be
serialised or shared between processes, so it lives in the server process and
the app MUST run single-worker (`serve.py` and the Dockerfile both enforce it).

Everything a user owns — the result, the job, uploaded files, written outputs —
is scoped to their session (see `sessions.py`); nothing is global.  The heavy
run itself goes to a one-slot thread pool so two users cannot saturate the box,
and the browser polls /api/job.  A thread rather than a process keeps the
Result in the same interpreter that serves the figures.
"""

from __future__ import annotations

import io
import json
import os
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import numpy as np
import pandas as pd
from fastapi import (Depends, FastAPI, File, Form, Header, HTTPException, Query,
                     Request, UploadFile)
from fastapi.responses import (FileResponse, HTMLResponse, JSONResponse,
                               PlainTextResponse, Response)
from fastapi.staticfiles import StaticFiles

from forest_ai import pipeline, webviz, evaluate, las_io
from forest_ai.config import Config
from . import params
from .sessions import (ALLOW_LOCAL_CLOUDS, ALLOW_SEGMENTED_LAS,
                       MAX_SESSIONS, MAX_UPLOAD_MB, STORE, Session)
from .vendor import ensure_plotly_js

HERE = os.path.dirname(os.path.abspath(__file__))
STATIC = os.path.join(HERE, "static")
CACHE_DIR = ".cache"

# safety net for hosts that start uvicorn directly instead of through serve.py
try:
    ensure_plotly_js(quiet=True)
except RuntimeError as e:
    print(f"WARNING: {e} - the page will not render any figures")

app = FastAPI(title="forest_ai", docs_url="/api/docs", openapi_url="/api/openapi.json")

_POOL = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pipeline")
_SUBMIT_LOCK = threading.Lock()


# ---------------------------------------------------------------- plumbing ---
def session(x_session_id: str | None = Header(default=None)) -> Session:
    return STORE.get(x_session_id)


def need_result(s: Session):
    if s.result is None:
        raise HTTPException(409, "no result in this session - run the pipeline first")
    return s.result


def fig_response(fig) -> Response:
    """Plotly figures go out as their own JSON.

    `to_json` base64-encodes numpy arrays, so a 150k-point cloud travels as
    ~4 MB instead of the ~8.7 MB it would take as plain JSON lists.  Building
    the Response by hand avoids a needless decode/re-encode round trip.
    """
    return Response(fig.to_json(), media_type="application/json")


def jsonable(obj):
    """numpy/pandas scalars -> plain Python, NaN -> None."""
    if isinstance(obj, dict):
        return {k: jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return None if not np.isfinite(f) else f
    if isinstance(obj, np.ndarray):
        return jsonable(obj.tolist())
    if obj is pd.NaT or (isinstance(obj, float) and pd.isna(obj)):
        return None
    return obj


def safe_cloud(s: Session, path: str) -> str:
    """Only clouds this session may open.

    Membership of the session's own list is the whole check: it rules out path
    traversal and, just as importantly, another session's uploads.
    """
    norm = os.path.normpath(path or "").replace("\\", "/")
    if norm not in s.clouds():
        raise HTTPException(400, f"unknown point cloud: {path}")
    return norm


# ------------------------------------------------------------------ pages ---
@app.get("/", response_class=HTMLResponse)
def index():
    with open(os.path.join(STATIC, "index.html"), encoding="utf-8") as f:
        return HTMLResponse(f.read())


app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/api/config")
def api_config():
    """What the deployment allows, so the UI can describe itself accurately."""
    return {
        "max_upload_mb": MAX_UPLOAD_MB,
        "max_sessions": MAX_SESSIONS,
        "local_clouds": ALLOW_LOCAL_CLOUDS,
        "segmented_las": ALLOW_SEGMENTED_LAS,
    }


# -------------------------------------------------------------- inventory ---
@app.get("/api/clouds")
def api_clouds(s: Session = Depends(session)):
    return {"clouds": s.clouds(), "current": getattr(s.result, "las_path", None)}


@app.get("/api/params")
def api_params():
    return {"groups": params.spec(), "groups_default": 4}


@app.get("/api/header")
def api_header(las: str = Query(...), s: Session = Depends(session)):
    try:
        return jsonable(las_io.describe(safe_cloud(s, las)))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, f"could not read header: {e}")


@app.post("/api/upload")
async def api_upload(file: UploadFile = File(...), s: Session = Depends(session)):
    """Stream the upload to the session's own directory.

    Written in chunks so a several-hundred-megabyte cloud is never held in
    memory, and aborted the moment it goes over the limit — otherwise a public
    deployment would happily fill its disk.
    """
    name = os.path.basename(file.filename or "upload.las")
    if not name.lower().endswith((".las", ".laz")):
        raise HTTPException(400, "expected a .las or .laz file")

    os.makedirs(s.upload_dir, exist_ok=True)
    dest = os.path.join(s.upload_dir, name)
    limit = MAX_UPLOAD_MB << 20
    size = 0
    try:
        with open(dest, "wb") as out:
            while chunk := await file.read(8 << 20):
                size += len(chunk)
                if size > limit:
                    raise HTTPException(
                        413, f"file is larger than the {MAX_UPLOAD_MB} MB limit "
                             f"on this deployment")
                out.write(chunk)
    except BaseException:
        # never leave a truncated cloud behind for the user to pick from
        try:
            os.remove(dest)
        except OSError:
            pass
        raise
    return {"path": dest.replace("\\", "/"), "bytes": size}


# -------------------------------------------------------------------- run ---
def _worker(s: Session, las: str, cfg: Config, groups: int, force: bool):
    def on_progress(frac, msg):
        s.job.update(progress=float(min(frac, 1.0)), message=msg)

    try:
        s.job.update(state="running", progress=0.0, message="starting", error=None)
        r = pipeline.run(las, cfg, cache_dir=CACHE_DIR, force=force,
                         groups=groups, progress=on_progress, verbose=True)
        s.result = r
        s.job.update(state="done", progress=1.0,
                     message=f"{len(r.df)} trees", error=None)
    except Exception as e:
        traceback.print_exc()
        s.job.update(state="error", message="failed",
                     error=f"{type(e).__name__}: {e}")


@app.post("/api/run")
def api_run(body: dict, s: Session = Depends(session)):
    if s.job["state"] in ("running", "queued"):
        raise HTTPException(409, "this session already has a run in progress")
    las = safe_cloud(s, body.get("las") or "")
    cfg = replace(Config(las_path=las), **params.clean(body.get("config")))
    groups = int(body.get("groups") or 4)
    force = bool(body.get("force"))

    with _SUBMIT_LOCK:
        # one worker for the whole server, so a second user waits rather than
        # both runs fighting over two cores
        busy = any(o.job["state"] == "running" for o in STORE.snapshot() if o is not s)
        s.job.update(state="queued", progress=0.0, error=None,
                     message="waiting for the worker" if busy else "queued")
        _POOL.submit(_worker, s, las, cfg, groups, force)
    return {"state": s.job["state"], "las": las}


@app.get("/api/job")
def api_job(s: Session = Depends(session)):
    return {**s.job, "has_result": s.result is not None,
            "las": getattr(s.result, "las_path", None), "session": s.sid}


# ----------------------------------------------------------------- result ---
@app.get("/api/summary")
def api_summary(s: Session = Depends(session)):
    r = need_result(s)
    med = r.df.groupby("quality")["dist_from_scan_centre_m"].median().to_dict()
    return jsonable({
        "las": r.las_path,
        "summary": pipeline.summary(r),
        "text": pipeline.summary_text(r),
        "median_range_by_quality": med,
        "header": r.header,
        "counts": r.df["quality"].value_counts().to_dict(),
    })


@app.get("/api/trees")
def api_trees(s: Session = Depends(session)):
    """All rows at once - a couple of hundred trees is a trivial payload, and
    filtering client-side keeps the table instant."""
    r = need_result(s)
    df = r.df.replace({np.nan: None})
    return {"columns": list(df.columns), "rows": jsonable(df.to_dict("records"))}


@app.get("/api/species")
def api_species(s: Session = Depends(session)):
    r = need_result(s)
    if "structural_group" not in r.df:
        return {"available": False, "profile": []}
    g = r.df.dropna(subset=["structural_group"])
    prof = (g.groupby("structural_group")
             .agg(n=("tree_id", "count"), dbh_cm=("dbh_cm", "median"),
                  height_m=("height_m", "median"),
                  crown_diameter_m=("crown_diameter_m", "median"),
                  crown_base_m=("crown_base_m", "median"),
                  h_d_ratio=("h_d_ratio", "median"))
             .round(2).reset_index())
    return {"available": True, "profile": jsonable(prof.to_dict("records"))}


@app.get("/api/figure/cloud")
def fig_cloud(n: int = 150_000, color_by: str = "tree", ground: bool = False,
              size: float = 1.4, s: Session = Depends(session)):
    r = need_result(s)
    n = int(np.clip(n, 5_000, 500_000))
    fig, shown = webviz.cloud_figure(r, n, color_by, ground, size)
    resp = fig_response(fig)
    resp.headers["X-Points-Shown"] = str(shown)
    resp.headers["X-Points-Total"] = str(int((r.labels >= 0).sum()))
    return resp


@app.get("/api/figure/stemmap")
def fig_stemmap(colour: str = "quality", s: Session = Depends(session)):
    return fig_response(webviz.stem_map_figure(need_result(s), colour))


@app.get("/api/figure/inventory")
def fig_inventory(s: Session = Depends(session)):
    return fig_response(webviz.inventory_figure(need_result(s)))


@app.get("/api/figure/dbh_slice/{tree_id}")
def fig_dbh_slice(tree_id: int, s: Session = Depends(session)):
    fig = webviz.dbh_slice_figure(need_result(s), tree_id)
    if fig is None:
        raise HTTPException(404, f"no tree {tree_id}")
    return fig_response(fig)


@app.get("/api/figure/tree3d/{tree_id}")
def fig_tree3d(tree_id: int, s: Session = Depends(session)):
    return fig_response(webviz.single_tree_figure(need_result(s), tree_id))


# ------------------------------------------------------------ qc / output ---
QC_FILES = ["qc_01_ground.png", "qc_02_stem_map.png", "qc_03_inventory.png",
            "qc_04_segmentation.png", "qc_05_dbh_fits.png"]


@app.get("/api/qc")
def api_qc(s: Session = Depends(session)):
    return {"files": [f for f in QC_FILES
                      if os.path.exists(os.path.join(s.out_dir, f))]}


@app.get("/api/qc/{name}")
def api_qc_image(name: str, s: Session = Depends(session)):
    if name not in QC_FILES:
        raise HTTPException(404, "unknown figure")
    path = os.path.join(s.out_dir, name)
    if not os.path.exists(path):
        raise HTTPException(404, "not generated yet")
    return FileResponse(path, media_type="image/png")


@app.post("/api/write_outputs")
def api_write_outputs(body: dict | None = None, s: Session = Depends(session)):
    r = need_result(s)
    body = body or {}
    want_las = bool(body.get("segmented_las", True)) and ALLOW_SEGMENTED_LAS
    written = pipeline.write_outputs(r, s.out_dir,
                                     figures=body.get("figures", True),
                                     segmented_las=want_las)
    return {"written": written, "dir": s.out_dir.replace("\\", "/"),
            "segmented_las_allowed": ALLOW_SEGMENTED_LAS}


@app.get("/api/download/{what}.csv")
def api_download(what: str, s: Session = Depends(session)):
    r = need_result(s)
    if what == "trees":
        body, name = r.df.to_csv(index=False), "trees.csv"
    elif what == "features":
        body, name = r.feat.to_csv(), "tree_features.csv"
    else:
        raise HTTPException(404, "expected trees or features")
    return PlainTextResponse(body, media_type="text/csv", headers={
        "Content-Disposition": f'attachment; filename="{name}"'})


# ------------------------------------------------------------- validation ---
@app.post("/api/evaluate")
async def api_evaluate(file: UploadFile = File(...), mapping: str = Form("{}"),
                       max_dist: float = Form(1.5), quality: str = Form("good"),
                       s: Session = Depends(session)):
    r = need_result(s)
    try:
        ref = pd.read_csv(io.BytesIO(await file.read()))
    except Exception as e:
        raise HTTPException(400, f"could not parse the CSV: {e}")

    ren = {k: v for k, v in json.loads(mapping).items() if k and v}
    ref = ref.rename(columns=ren)
    missing = {"x", "y"} - set(ref.columns)
    if missing:
        raise HTTPException(400, f"reference has no {sorted(missing)} column(s); "
                                 f"it has {list(ref.columns)}")

    quals = [q for q in quality.split(",") if q] or None
    try:
        text, matched, stats = evaluate.report(r.df, ref, max_dist=max_dist,
                                               quality=quals)
    except Exception as e:
        raise HTTPException(400, f"comparison failed: {e}")

    errors = {f: evaluate.error_stats(matched, f)
              for f in ("dbh_cm", "height_m")
              if f"{f}_ref" in matched and f"{f}_pred" in matched}
    pairs = {f: {"ref": matched[f"{f}_ref"].tolist(),
                 "pred": matched[f"{f}_pred"].tolist()} for f in errors}
    return jsonable({"text": text, "stats": stats, "errors": errors,
                     "pairs": pairs, "columns": list(ref.columns)})


@app.post("/api/evaluate/columns")
async def api_evaluate_columns(file: UploadFile = File(...)):
    """Column names only, so the browser can offer a mapping before comparing."""
    try:
        ref = pd.read_csv(io.BytesIO(await file.read()), nrows=5)
    except Exception as e:
        raise HTTPException(400, f"could not parse the CSV: {e}")
    return {"columns": list(ref.columns),
            "preview": jsonable(ref.replace({np.nan: None}).to_dict("records"))}


@app.exception_handler(HTTPException)
def http_error(request: Request, exc: HTTPException):
    return JSONResponse({"error": exc.detail}, status_code=exc.status_code)
