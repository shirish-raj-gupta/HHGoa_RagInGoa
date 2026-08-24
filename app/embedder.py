"""
Embedder interface implementation for rag-local-eval-loop.
Uses the ONNX int8 Multilingual-E5-small model.
"""
from __future__ import annotations

import os
from pathlib import Path
import numpy as np

from src.config import load_dotenv
from src.index.embedder import OnnxEmbedder

load_dotenv()

_ROOT = Path(__file__).resolve().parents[1]
_ARTIFACTS = Path(os.environ.get("RAG_ARTIFACTS", str(_ROOT / "artifacts")))
_ONNX_DIR = _ARTIFACTS / "e5-small-onnx"

_MODEL: OnnxEmbedder | None = None


def get_model() -> OnnxEmbedder:
    global _MODEL
    if _MODEL is None:
        model_path = _ONNX_DIR / "model_int8.onnx"
        if not model_path.exists():
            model_path = _ROOT / "artifacts" / "e5-small-onnx" / "model_int8.onnx"
        _MODEL = OnnxEmbedder(model_path, model_path.parent)
    return _MODEL


def embed_one(text: str) -> np.ndarray:
    m = get_model()
    # E5 query embedding prefix is handled automatically
    v = m.encode_queries([text])[0]
    return np.asarray(v, dtype=np.float32)


def embed(texts: list[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, 384), dtype=np.float32)
    m = get_model()
    v = m.encode_passages(texts)
    return np.asarray(v, dtype=np.float32)
