FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    STREAMLIT_SERVER_HEADLESS=true

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml app.py ./
COPY configs/ configs/
COPY src/ src/
COPY dashboard/ dashboard/
COPY scripts/ scripts/
COPY docs/ docs/
# Pull Git LFS model binaries before building (`git lfs pull`) or run `python scripts/download_models.py`
# to bundle trained weights; set --build-arg REQUIRE_MODELS=true to fail the build on LFS pointer stubs.
ARG REQUIRE_MODELS=false
COPY models/ models/
COPY reports/ reports/

RUN REQUIRE_MODELS="${REQUIRE_MODELS}" python -c "import os, pathlib, sys; \
models_dir = pathlib.Path('/app/models'); \
prefix = b'version https://git-lfs.github.com/spec/v1'; \
req = ['xgb_model.json', 'scaler.pkl', 'label_encoder.pkl', 'best_mlp.keras', 'best_cnn.keras', 'best_autoencoder.keras']; \
stubs = sorted([p.name for p in models_dir.glob('*') if p.is_file() and p.stat().st_size <= 1024 and p.read_bytes()[:len(prefix)] == prefix]); \
missing = sorted([f for f in req if not (models_dir / f).exists()]); \
require_models = os.environ.get('REQUIRE_MODELS', 'false').strip().lower() in {'1', 'true', 'yes'}; \
if stubs or missing: \
    print('======================================================================', file=sys.stderr); \
    print('WARNING: Docker build detected unfetched Git LFS stubs or missing models!', file=sys.stderr); \
    print(f'  LFS pointer stubs: {stubs}', file=sys.stderr); \
    print(f'  Missing core files: {missing}', file=sys.stderr); \
    print('  Run git lfs pull or python scripts/download_models.py before building.', file=sys.stderr); \
    print('  Container will start in degraded demo/report mode unless models exist.', file=sys.stderr); \
    print('======================================================================', file=sys.stderr); \
    sys.exit(1 if require_models else 0)"

RUN mkdir -p data/raw data/processed data/synthetic

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=5)" || exit 1

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
