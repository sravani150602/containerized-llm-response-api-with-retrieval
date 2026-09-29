# ---------- builder: install Python deps into a virtualenv ----------
FROM python:3.11-slim AS builder
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY requirements.txt .
# CPU-only torch keeps the image ~4x smaller than the default CUDA build
RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.3" \
 && pip install -r requirements.txt

# ---------- runtime ----------
FROM python:3.11-slim
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/models/hf-cache \
    TOKENIZERS_PARALLELISM=false
RUN useradd --create-home --uid 1000 app && mkdir -p /models && chown app /models
COPY --from=builder /opt/venv /opt/venv
WORKDIR /srv
COPY --chown=app app ./app
COPY --chown=app knowledge_base ./knowledge_base
COPY --chown=app artifacts ./artifacts
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
