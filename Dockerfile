# HF Docker Space image.
#
# Deliberately CPU-only and torch-free: the ONNX artifact is built ahead of
# time and committed/pulled, so the runtime never needs torch or optimum. That
# keeps the image ~2GB smaller and the cold boot correspondingly shorter, which
# matters because a Space is stateless and pays boot cost on every restart.

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/home/user/.cache/huggingface \
    OMP_NUM_THREADS=4 \
    RAG_THREADS=4 \
    RAG_BUDGET_MS=200

# HF Spaces run as uid 1000; writing as root leaves unreadable cache dirs.
RUN useradd -m -u 1000 user
WORKDIR /home/user/app

COPY --chown=user pyproject.toml ./
RUN pip install --upgrade pip && pip install --no-cache-dir \
    "fastapi>=0.115" "uvicorn[standard]>=0.30" "websockets>=13.0" \
    "pydantic>=2.9" "httpx>=0.27" "orjson>=3.10" "pyyaml>=6.0" \
    "onnxruntime>=1.19" "transformers>=4.44" "tokenizers>=0.20" \
    "usearch>=2.12" "bm25s>=0.2" "numpy>=1.26" "scipy>=1.13" \
    "pyarrow>=17.0" "pandas>=2.2" "huggingface_hub>=0.25" "groq>=1.6"

COPY --chown=user src ./src
COPY --chown=user web ./web
COPY --chown=user bench ./bench
COPY --chown=user docs ./docs

USER user

# SARVAM_API_KEY and GROQ_API_KEY come from Space secrets, never the image.
# RAG_ARTIFACTS / RAG_SLICE point at the index pulled at boot.
ENV RAG_ARTIFACTS=/home/user/app/artifacts \
    RAG_SLICE=/home/user/app/data/slice \
    RAG_LANGS=eng_Latn,hin_Deva,tam_Taml

EXPOSE 7860

# The readiness probe must not pass until warm boot completes (model session,
# tokenizer, BM25, HNSW, dummy inference). A cold first request would otherwise
# land in the p100 bucket and misrepresent steady-state latency.
HEALTHCHECK --interval=30s --timeout=5s --start-period=600s --retries=3 \
    CMD python -c "import urllib.request,json,sys; \
        d=json.load(urllib.request.urlopen('http://localhost:7860/health')); \
        sys.exit(0 if d.get('ready') else 1)"

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "7860", \
     "--workers", "1", "--timeout-keep-alive", "75"]
