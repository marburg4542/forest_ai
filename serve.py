"""Start the forest_ai web interface.

    python serve.py [--port 8000] [--host 127.0.0.1]

Single worker on purpose: a pipeline result is ~700 MB of numpy arrays held in
module state, which cannot be shared between processes.  With more than one
worker, requests would land on workers that hold no result and the UI would
flicker between "ready" and "run the pipeline first".  Concurrency between
visitors comes from the per-session table in web/sessions.py instead.

Environment variables (all optional):
    FAI_MAX_SESSIONS        results kept in memory at once (default 2)
    FAI_MAX_UPLOAD_MB       largest accepted upload (default 300)
    FAI_ALLOW_LOCAL_CLOUDS  offer .las files sitting next to the server
                            (default true locally, false in the Docker image)
    FAI_ALLOW_SEGMENTED_LAS allow writing the ~250 MB segmented cloud
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8000)))
    ap.add_argument("--reload", action="store_true", help="auto-reload on edits")
    args = ap.parse_args()

    os.chdir(pathlib.Path(__file__).parent)
    from web.vendor import ensure_plotly_js
    try:
        ensure_plotly_js()
    except RuntimeError as e:
        sys.exit(str(e))

    import uvicorn
    print(f"\n  forest_ai  ->  http://{args.host}:{args.port}\n")
    uvicorn.run("web.server:app", host=args.host, port=args.port,
                reload=args.reload, workers=1, log_level="info")


if __name__ == "__main__":
    main()
