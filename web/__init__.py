"""HTTP interface for forest_ai.

`server.py` is a thin FastAPI layer over `forest_ai.pipeline`; all the science
lives in the `forest_ai` package and is untouched by this module.  The
front-end under `static/` is plain HTML/CSS/JS with no build step.
"""
