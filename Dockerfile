# Image for Hugging Face Spaces (sdk: docker) and any other container host.
#
# Single worker on purpose: a pipeline result is ~700 MB of numpy arrays held in
# process memory, so it cannot be shared between workers.  Concurrency comes
# from the per-session table in web/sessions.py, not from more processes.

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg

# Spaces runs the container as uid 1000; everything it writes must be writable
RUN useradd -m -u 1000 app
WORKDIR /home/app

COPY --chown=app:app requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=app:app forest_ai/ ./forest_ai/
COPY --chown=app:app web/ ./web/
COPY --chown=app:app serve.py run_pipeline.py evaluate_against_reference.py ./

USER app

# plotly.min.js is copied out of the installed plotly package at build time, so
# the page never reaches for a CDN and works with no outbound network
RUN python -c "from web.vendor import ensure_plotly_js; ensure_plotly_js()"

# scratch directories the app writes into
RUN mkdir -p data outputs .cache

# defaults suited to a small shared box; override per deployment
ENV FAI_MAX_SESSIONS=2 \
    FAI_MAX_UPLOAD_MB=300 \
    FAI_ALLOW_LOCAL_CLOUDS=false \
    FAI_ALLOW_SEGMENTED_LAS=false

EXPOSE 7860
CMD ["uvicorn", "web.server:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
