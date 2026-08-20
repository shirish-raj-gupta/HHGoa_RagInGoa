"""
In-process ONNX int8 embedder for intfloat/multilingual-e5-small.

Two things here are load-bearing for correctness, not just speed:

1. The `query:` / `passage:` prefix convention. e5 was trained with them and
   getting it wrong silently costs recall - no error, just worse numbers. So
   the prefix is not a caller-supplied string: `encode_queries` and
   `encode_passages` are separate methods and there is no way to embed text
   without going through one of them.

2. Mean pooling over the attention mask, then L2 normalize. e5 is a
   mean-pooled model; using CLS instead is another silent recall loss.

Never an embedding API call. A network hop here would make the <200ms
CORE_RAG_LOOP claim impossible (ADR 0001 section 3).
"""
from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

from ..chunking.base import Tokenizer

MODEL_ID = "intfloat/multilingual-e5-small"
# int8 is a CPU-only optimization here: the dynamically-quantized graph has no
# usable CUDA kernels and crashes the CUDA EP outright (STATUS_STACK_BUFFER_
# OVERRUN). GPU index builds therefore use the fp32 export, which is fine -
# fp32 and int8 agree to 0.990 mean cosine, so the vectors are interchangeable.
# Measured on an RTX 4050 (6GB): fp32 CUDA 1,018 psg/s at batch 16 vs int8 CPU
# 156 psg/s at 16 threads, i.e. 3.9h vs 25h for the full 14.3M-passage build.
DIM = 384
MAX_LEN = 512


def export_onnx_int8(out_dir: Path, model_id: str = MODEL_ID,
                     force: bool = False) -> Path:
    """Export to ONNX and dynamically quantize to int8. Returns the int8 path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fp32_dir, int8_path = out_dir / "fp32", out_dir / "model_int8.onnx"
    if int8_path.exists() and not force:
        return int8_path

    from optimum.onnxruntime import ORTModelForFeatureExtraction
    from onnxruntime.quantization import quantize_dynamic, QuantType

    if not (fp32_dir / "model.onnx").exists() or force:
        m = ORTModelForFeatureExtraction.from_pretrained(model_id, export=True)
        m.save_pretrained(fp32_dir)
        AutoTokenizer.from_pretrained(model_id).save_pretrained(fp32_dir)

    quantize_dynamic(
        model_input=str(fp32_dir / "model.onnx"),
        model_output=str(int8_path),
        weight_type=QuantType.QInt8,
        # keep the embedding lookup in fp32: quantizing a 250k-token
        # multilingual vocab measurably hurts the low-resource scripts
        nodes_to_exclude=None,
        extra_options={"MatMulConstBOnly": True},
    )
    for f in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
              "sentencepiece.bpe.model", "config.json"):
        src = fp32_dir / f
        if src.exists():
            shutil.copy2(src, out_dir / f)
    return int8_path


class E5Tokenizer(Tokenizer):
    """Adapter exposing exactly what the chunkers need."""

    def __init__(self, model_dir: str | Path = MODEL_ID):
        self.hf = AutoTokenizer.from_pretrained(str(model_dir))

    def encode(self, text: str) -> list[int]:
        return self.hf(text, add_special_tokens=False)["input_ids"]

    def token_spans(self, text: str) -> list[tuple[int, int]]:
        """Char spans per token, so chunkers cut on real token boundaries."""
        enc = self.hf(text, add_special_tokens=False, return_offsets_mapping=True)
        return [(a, b) for a, b in enc["offset_mapping"] if b > a]


class OnnxEmbedder:
    """Warm, in-process, int8. One session, reused."""

    @staticmethod
    def register_cuda_dlls() -> None:
        """
        Windows onnxruntime does not discover the nvidia pip wheels' DLLs, and
        `os.add_dll_directory` does not cover transitive loads either - the
        provider DLL still fails on cublasLt/cudart. Prepending the wheel bin
        directories to PATH before the CUDA provider is created is what works.
        """
        import glob
        import os
        dirs = glob.glob(os.path.join(sys.prefix, "Lib", "site-packages",
                                      "nvidia", "*", "bin"))
        if dirs:
            os.environ["PATH"] = os.pathsep.join(dirs + [os.environ.get("PATH", "")])
        for d in dirs:
            try:
                os.add_dll_directory(d)
            except (OSError, AttributeError):
                pass

    def __init__(self, model_path: str | Path, tokenizer_dir: str | Path,
                 threads: int = 0, warm: bool = True, use_gpu: bool = False):
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if threads:
            so.intra_op_num_threads = threads
            so.inter_op_num_threads = 1
        providers = ["CPUExecutionProvider"]
        if use_gpu:
            self.register_cuda_dlls()
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(model_path), so, providers=providers)
        self.provider = self.session.get_providers()[0]
        if use_gpu and self.provider != "CUDAExecutionProvider":
            log_msg = f"GPU requested but running on {self.provider}"
            print(f"[embedder] WARNING: {log_msg}", file=sys.stderr)
        self.tok = AutoTokenizer.from_pretrained(str(tokenizer_dir))
        self._inputs = {i.name for i in self.session.get_inputs()}
        self.threads = threads
        if warm:
            self.warmup()

    def warmup(self, n: int = 3) -> float:
        """Trigger lazy init before the readiness probe passes (ADR 0001 s8)."""
        t0 = time.perf_counter_ns()
        for _ in range(n):
            self._forward(["warmup"])
        return (time.perf_counter_ns() - t0) / 1e6

    def _feed(self, enc) -> dict[str, np.ndarray]:
        """
        Build the ORT feed. The exported graph declares token_type_ids, but the
        XLM-R tokenizer does not emit it - e5 is single-segment, so zeros are
        the correct value. Omitting it is a hard ORT error, not a silent one.
        """
        feed = {k: np.asarray(v).astype(np.int64)
                for k, v in enc.items() if k in self._inputs}
        if "token_type_ids" in self._inputs and "token_type_ids" not in feed:
            feed["token_type_ids"] = np.zeros_like(feed["input_ids"])
        return feed

    def _forward(self, texts: list[str], max_len: int = MAX_LEN) -> np.ndarray:
        enc = self.tok(texts, padding=True, truncation=True, max_length=max_len,
                       return_tensors="np")
        hidden = self.session.run(None, self._feed(enc))[0]   # (B, T, DIM)
        mask = enc["attention_mask"].astype(np.float32)[..., None]
        pooled = (hidden * mask).sum(1) / np.clip(mask.sum(1), 1e-9, None)
        return pooled / (np.linalg.norm(pooled, axis=1, keepdims=True) + 1e-9)

    # -- the only two public entry points, so the prefix cannot be forgotten --

    def encode_queries(self, texts: list[str], batch: int = 32) -> np.ndarray:
        return self._batched([f"query: {t}" for t in texts], batch)

    def encode_passages(self, texts: list[str], batch: int = 64) -> np.ndarray:
        return self._batched([f"passage: {t}" for t in texts], batch)

    def _batched(self, texts: list[str], batch: int,
                 bucket: bool = True) -> np.ndarray:
        """
        Length-bucketed batching.

        `padding=True` pads every sequence to the longest in its batch, so one
        300-token passage in a batch of 72-token ones makes the whole batch pay
        300. Sorting by length before batching removes most of that waste:
        measured 2.0x on the real corpus (78 -> 156 passages/s, 16 threads,
        batch 64), cutting the 14.3M-passage index build from 51h to 25h.

        Note batch 64 beat batch 256 in that measurement - bigger batches
        re-introduce padding spread faster than they gain parallelism here.

        Order is restored before returning, so callers see no difference.
        """
        if not texts:
            return np.zeros((0, DIM), dtype=np.float32)
        if not bucket or len(texts) <= batch:
            out = [self._forward(texts[i:i + batch]) for i in range(0, len(texts), batch)]
            return np.vstack(out).astype(np.float32)

        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        V = np.empty((len(texts), DIM), dtype=np.float32)
        for i in range(0, len(order), batch):
            idx = order[i:i + batch]
            V[idx] = self._forward([texts[j] for j in idx])
        return V

    def encode_late(self, passage: str, spans: list[tuple[int, int]]) -> np.ndarray:
        """
        Late chunking: embed the FULL passage once, then mean-pool token
        vectors over each (tok_lo, tok_hi) span. Every chunk vector therefore
        carries document-level context (strategy 6).
        """
        enc = self.tok([f"passage: {passage}"], padding=True, truncation=True,
                       max_length=MAX_LEN, return_tensors="np")
        hidden = self.session.run(None, self._feed(enc))[0][0]   # (T, DIM)
        # +1 offset: "passage:" prefix tokens sit in front of the real content
        offset = len(self.tok("passage:", add_special_tokens=False)["input_ids"]) + 1
        vecs = []
        for lo, hi in spans:
            seg = hidden[offset + lo: offset + hi]
            if seg.size == 0:
                seg = hidden
            vecs.append(seg.mean(0))
        V = np.vstack(vecs)
        return (V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)).astype(np.float32)
