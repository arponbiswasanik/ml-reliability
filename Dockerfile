FROM python:3.11-slim

WORKDIR /app

# Copy requirements first: cheap layer, everything below caches on it
COPY requirements.txt .

# CPU-only torch, pinned (PyPI default wheel drags in CUDA -> ~6GB image)
RUN TORCH_VERSION=$(grep -i '^torch==' requirements.txt | cut -d= -f3) && \
    if [ -z "$TORCH_VERSION" ]; then \
        echo "ERROR: no torch== pin found in requirements.txt"; exit 1; \
    fi && \
    pip install --no-cache-dir "torch==${TORCH_VERSION}" \
      --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir -r requirements.txt

# App code + artifacts (build context = local files, not git)
COPY src/ src/
COPY evaluation/ evaluation/
COPY models/champion.pt models/
COPY models/calibrator.joblib models/
COPY data/processed/reference_snapshot.parquet data/processed/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]