# Deployment

## Why it is split

The frontend is a Hugging Face **Static** Space; the backend runs on a separate
machine behind a tunnel. That is not a preference — it is what the account can
host. Tested against the live API rather than assumed:

| Space type | Result |
|---|---|
| Docker | `402 Payment Required` — requires PRO |
| Gradio + ZeroGPU | `402` — *"wait 30 days or request a community grant"* |
| **Static** | created fine |

A Static Space serves files and cannot run Python. The index is also **~15 GB
across 15 language partitions**, which does not belong on a Space's ephemeral
storage even with PRO: it would re-download on every cold start.

```
   judge's browser
        │
        ▼
   HF Static Space          srg101/raginggoa
   index.html + config.js   (UI only, no compute)
        │  fetch / WebSocket, cross-origin
        ▼
   Cloudflare Tunnel        https://<name>.trycloudflare.com
        │
        ▼
   local backend            FastAPI + 15 mmap'd partitions + SQLite text
```

## Running the backend

```bash
docker compose up --build          # API on :7860
# or, without Docker:
RAG_LANGS=eng_Latn RAG_INDEX=artifacts/index \
  python -m uvicorn src.api.app:app --host 0.0.0.0 --port 7860
```

`RAG_LANGS` selects which partitions load. All 15 is ~15 GB on disk; vectors are
memory-mapped, so resident memory tracks the hot set rather than the total.

Boot pages the graph in **before** the readiness probe passes. This matters: a
cold mmap served the first user a 412 ms query with *both* retrieval arms
blowing their timeouts and nothing retrieved. Warmed, the first query is
10.1 ms.

## Exposing it

```bash
cloudflared tunnel --url http://localhost:7860
# -> https://<random-words>.trycloudflare.com

python -m scripts.deploy_static_space --space srg101/raginggoa \
    --api https://<that-url> --push
```

### The quick-tunnel caveat, stated plainly

`trycloudflare.com` quick tunnels are free and **ephemeral**:

- **The URL changes every restart.** The Space has to be redeployed (or opened
  with `?api=...`) each time. For a submission link that is fragile.
- **They are throttled.** Measured on a 183-byte `/health`: local p50 384 ms,
  tunnel p50 1089 ms with a 561–2481 ms spread on identical requests. A
  `/query` that takes ~2.0 s locally took 9.6–23.2 s through the tunnel while
  the core loop stayed at ~60 ms. The latency is the tunnel, not the pipeline.

For anything that needs to stay up, use a **named** Cloudflare tunnel (free,
needs a Cloudflare account) — it gives a stable hostname and better throughput:

```bash
cloudflared tunnel login
cloudflared tunnel create raginggoa
cloudflared tunnel route dns raginggoa rag.<your-domain>
cloudflared tunnel run --url http://localhost:7860 raginggoa
```

### CORS

The backend allowlists origins via `RAG_ALLOW_ORIGINS` (comma-separated),
defaulting to the Space plus localhost. It is **not** `*`, deliberately: the
Space calls the backend with the operator's own Sarvam and Groq keys, and a
wildcard would let any page spend them.

## If the backend is down

The page renders a designed "backend unreachable" state explaining that the
frontend is static, the index is 15 GB, and how to point it elsewhere
(`?api=...`) or run it locally. It does not silently fail.

## Cost exposure

The Space is public and voice is enabled, so **every visitor spends the
operator's Sarvam STT and Groq token quota**. That was a deliberate choice for
the judging window. To close it: stop the tunnel, or redeploy with `--api ""`,
and the page falls back to the offline state.
