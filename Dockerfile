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
# Pull Git LFS model binaries before building (`git lfs pull`) to bundle trained weights;
# if models are absent or LFS pointers, the dashboard starts cleanly in degraded demo/report mode.
COPY models/ models/
COPY reports/ reports/

RUN mkdir -p data/raw data/processed data/synthetic

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=5)" || exit 1

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
