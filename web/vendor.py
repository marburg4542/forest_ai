"""Make plotly.js available locally.

Serving it out of the installed plotly package rather than a CDN keeps the app
working offline, adds no third-party request to every page load, and guarantees
the plotly.js version matches the plotly.py that produced the figure JSON —
which matters, because the figures rely on the base64 typed-array encoding.

Kept free of heavy imports so both `serve.py` and the Docker build can call it
before anything else is loaded.
"""

from __future__ import annotations

import pathlib
import shutil

VENDOR = pathlib.Path(__file__).parent / "static" / "vendor" / "plotly.min.js"
MIN_BYTES = 1_000_000


def ensure_plotly_js(quiet: bool = False) -> pathlib.Path:
    """Copy plotly.min.js next to the front-end if it is not already there."""
    if VENDOR.exists() and VENDOR.stat().st_size > MIN_BYTES:
        return VENDOR
    try:
        import plotly
    except ImportError as e:
        raise RuntimeError("plotly is not installed: pip install plotly") from e
    src = pathlib.Path(plotly.__file__).parent / "package_data" / "plotly.min.js"
    if not src.exists():
        raise RuntimeError(f"plotly.min.js not found in the plotly package at {src}")
    VENDOR.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, VENDOR)
    if not quiet:
        print(f"vendored plotly.js -> {VENDOR} ({VENDOR.stat().st_size/1e6:.1f} MB)")
    return VENDOR
