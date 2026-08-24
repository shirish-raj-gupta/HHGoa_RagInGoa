"""
Embedder interface implementation for rag-local-eval-loop.
Uses the ONNX int8 Multilingual-E5-small model.

Includes re-ranking to push the true positive passage to rank 1,
and query-passage relevance scoring for answerability detection.
"""
from __future__ import annotations

import os
import re
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


def _tokenize(text: str) -> set[str]:
    """Extract meaningful words (length > 2) for overlap scoring."""
    return {w.lower() for w in re.findall(r"\w+", text.lower()) if len(w) > 2}


def _word_overlap_score(query: str, passage: str) -> float:
    """Weighted word overlap between query and passage."""
    q_words = _tokenize(query)
    p_words = _tokenize(passage)
    if not q_words:
        return 0.0
    overlap = len(q_words & p_words)
    return overlap / len(q_words)


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    dot = float(np.dot(a, b))
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a < 1e-9 or norm_b < 1e-9:
        return 0.0
    return dot / (norm_a * norm_b)


def rerank_results(query: str, results: list, top_k: int = 5) -> list:
    """
    Re-rank search results using combined embedding similarity + word overlap.
    
    This pushes the true positive passage to rank 1 by combining:
    1. Original FAISS score (embedding similarity)
    2. Query-passage word overlap (captures lexical match)
    3. Fresh query-passage embedding cosine (captures semantic match with proper prefixes)
    
    Args:
        query: The search query
        results: List of result objects with .text and .score attributes
        top_k: Number of results to return
    
    Returns:
        Re-ranked list of results
    """
    if not results or len(results) <= 1:
        return results

    m = get_model()
    
    # Get query embedding with proper prefix
    q_vec = m.encode_queries([query])[0]
    
    scored = []
    for r in results:
        text = getattr(r, "text", "")
        if not text:
            scored.append((r, 0.0))
            continue
        
        # 1. Original FAISS score (already normalized)
        original_score = getattr(r, "score", 0.0)
        
        # 2. Word overlap score
        word_score = _word_overlap_score(query, text)
        
        # 3. Fresh cosine similarity with proper prefixes
        p_vec = m.encode_passages([text])[0]
        cosine_score = _cosine_sim(q_vec, p_vec)
        
        # Combined score: weighted average
        # Heavy weight on cosine (semantic) + moderate on word overlap + some on original
        combined = 0.50 * cosine_score + 0.30 * word_score + 0.20 * original_score
        scored.append((r, combined))
    
    # Sort by combined score descending
    scored.sort(key=lambda x: x[1], reverse=True)
    
    return [r for r, _ in scored[:top_k]]


def query_passage_relevance(query: str, passages: list[str]) -> float:
    """
    Compute how relevant the top passages are to the query.
    Returns a score 0-1. Low scores indicate the passages don't address the query.
    Used for answerability detection.
    """
    if not passages:
        return 0.0
    
    m = get_model()
    q_vec = m.encode_queries([query])[0]
    
    max_sim = 0.0
    for p in passages[:3]:  # Check top 3 passages
        if not p:
            continue
        p_vec = m.encode_passages([p])[0]
        sim = _cosine_sim(q_vec, p_vec)
        max_sim = max(max_sim, sim)
    
    return max_sim
