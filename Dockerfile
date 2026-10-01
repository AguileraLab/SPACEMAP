# SPACEMAP container image.
#
# The base image runs the full scientific pipeline and the GUI. The AI
# assistant is not installed by default and is not required; build with
# --build-arg SPACEMAP_EXTRAS="ai" to include it.
FROM python:3.10-slim-bookworm

# Which optional dependency groups to install. Empty means pipeline only.
ARG SPACEMAP_EXTRAS=""
# CPU wheels keep the default image small. Set empty or a CUDA index for GPU use.
ARG SPACEMAP_TORCH_INDEX_URL="https://download.pytorch.org/whl/cpu"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
    MPLBACKEND=Agg \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    TOKENIZERS_PARALLELISM=false \
    # Everything the application writes goes to one mounted volume.
    SPACEMAP_HOME=/home/spacemap/.spacemap
# SPACEMAP_ASSISTANT is deliberately not set here. Setting it would override
# the in-app setting and make the Tab 0 switch inert; leaving it unset lets the
# user turn the assistant on and off from the interface, and the choice
# persists in the mounted volume.

# Native libraries for image processing, the geospatial stack, OpenBLAS and
# PyTorch. Kept in one layer so the cleanup actually shrinks the image.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libffi-dev \
        libgeos-dev \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        libhdf5-dev \
        libopenblas-dev \
        libopenslide0 \
        libproj-dev \
        libxml2-dev \
        libxslt1-dev \
        proj-bin \
        proj-data \
        gdal-bin \
        libgdal-dev \
        zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/spacemap

# Copy metadata first so dependency installation is cached independently of
# source edits.
COPY pyproject.toml README.md LICENSE marker_configs.json ./
COPY src ./src

RUN python -m pip install --upgrade pip setuptools wheel \
    && if [ -n "$SPACEMAP_TORCH_INDEX_URL" ]; then \
           python -m pip install torch torchvision --index-url "$SPACEMAP_TORCH_INDEX_URL"; \
       fi \
    && if [ -n "$SPACEMAP_EXTRAS" ]; then \
           python -m pip install ".[${SPACEMAP_EXTRAS}]"; \
       else \
           python -m pip install .; \
       fi \
    && python -m pip check

COPY notebooks ./notebooks
COPY docs ./docs
COPY scripts/check_installed_app.py scripts/check_runtime.py ./scripts/

RUN useradd --create-home --shell /bin/bash spacemap \
    && mkdir -p /workspace/projects "$SPACEMAP_HOME" \
    && chown -R spacemap:spacemap /opt/spacemap /workspace "$SPACEMAP_HOME"

USER spacemap
WORKDIR /workspace

EXPOSE 8501

# Streamlit's own endpoint, so an orchestrator can tell a slow start from a
# failed one.
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.build_opener(urllib.request.ProxyHandler({})).open('http://127.0.0.1:8501/_stcore/health', timeout=4).status==200 else 1)"

CMD ["spacemap", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
