# Knowledge Graph Extractor

Turn any document, URL, or a zip of files into an interactive knowledge graph,
using a self-hosted LLM (Qwen3.6-35B-A3B-MTP) on a single NVIDIA L4 — **or
natively on an Apple Silicon Mac, on Metal, with no Docker and no NVIDIA GPU.**

> **This is a fork** of [hanxiao/knowledge-graph-extractor](https://github.com/hanxiao/knowledge-graph-extractor)
> that adds the Apple Silicon path: `llama.cpp` on Metal plus an optional
> [MLX](https://github.com/ml-explore/mlx-lm) backend (~6x faster prefill).
> Start at [`docs/MAC.md`](docs/MAC.md). Everything else tracks upstream;
> the Mac support is offered upstream in
> [PR #15](https://github.com/hanxiao/knowledge-graph-extractor/pull/15) and
> unmerged so far, so use this fork if you're on a Mac.

Live demo (upstream, NVIDIA): https://hanxiao.io/knowledge-graph

[![Knowledge Graph Extractor](assets/hero.png)](https://hanxiao.io/knowledge-graph)

Each extracted fact is one graph edge: a `(subject) --[predicate]--> (object)`
triple plus a title, description, evidence span, confidence, tags, and source
file. Facts stream into a force-directed graph; hover an edge for the full card.

## How it works

1. **Input** — paste text, a URL (fetched to markdown via Jina Reader), or a
   `.zip` (txt, md, html, pdf, docx, json, csv, code...). Oversized docs are
   chunked (not truncated) so the full text is processed.
2. **Extract** — the LLM emits atomic `(subject, predicate, object)` triples.
   The prompt forces canonical entity/value subjects and objects so nodes
   connect instead of becoming prose dead-ends.
3. **Dedup** (on by default) — semantic dedup via
   [jina-embeddings-v5-text-nano](https://huggingface.co/jinaai/jina-embeddings-v5-text-nano)
   on CPU, across rounds and across files.
4. **Visualize** — every unique fact is one edge; node names are normalized so
   variants merge. Download the result as JSONL.

## Job queue

The L4 has one llama slot, so jobs run one at a time via a single-slot scheduler:
a new submission preempts the running job, which is persisted and auto-resumes
from where it left off when the slot frees. Jobs (meta + facts.jsonl + input)
persist under `data/jobs/` so the list, JSONL reload, and resume survive
restarts.

## Stack

- **llama-server** — [llama.cpp](https://github.com/ggml-org/llama.cpp) with
  CUDA, serves the model over an OpenAI-compatible API (port 8080). On a Mac the
  same port is served by llama.cpp on Metal or by `mlx_lm.server`; the app only
  ever talks to `LLAMA_URL`, so no application logic differs between backends.
- **app** — FastAPI: extraction + scheduler + CPU dedup + UI (port 3000).

## Setup

Two supported paths: NVIDIA + Docker (upstream), or Apple Silicon natively.

### NVIDIA / Docker

Single NVIDIA L4 24GB GPU (e.g. GCP `g2-standard-8`). Needs Docker + the NVIDIA
Container Toolkit.

```bash
git clone https://github.com/soobrosa/knowledge-graph-extractor.git
cd knowledge-graph-extractor

cp .env.example .env          # add your JINA_API_KEY (https://jina.ai/api-key)
bash scripts/setup.sh         # downloads the model (~17GB) and starts both services
```

Then open `http://<your-ip>:3000`.

Manual model download + run:

```bash
mkdir -p models
pip install -q huggingface-hub
python3 -c "from huggingface_hub import hf_hub_download; \
hf_hub_download('unsloth/Qwen3.6-35B-A3B-MTP-GGUF', \
'Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf', local_dir='models')"
docker compose up -d --build
```

### Apple Silicon (Mac, Metal)

No NVIDIA GPU, no Docker. Homebrew's `llama-server` serves the model on Metal, the
FastAPI app and the CPU dedup embedder run in a local `uv` venv. Tested on an
M3 Pro / 36GB; **32GB+ unified memory recommended** (the Q3_K_XL model wires ~17GB).

```bash
brew install llama.cpp
curl -LsSf https://astral.sh/uv/install.sh | sh

git clone https://github.com/soobrosa/knowledge-graph-extractor.git
cd knowledge-graph-extractor
cp .env.example .env          # add your JINA_API_KEY (https://jina.ai/api-key)

bash scripts/mac-run.sh       # starts llama-server on :8080, then the app on :3000
```

Full walkthrough — Python deps, the ~17GB model download, memory guidance per
machine tier, and env knobs — is in [`docs/MAC.md`](docs/MAC.md); the launcher is
[`scripts/mac-run.sh`](scripts/mac-run.sh).

For large docs and zips (extraction re-reads the whole document each round, so it's
prefill-bound) the optional MLX backend is ~6x faster at prefill:

```bash
BACKEND=mlx bash scripts/mac-run.sh
```

It needs a one-time separate venv and a 4-bit MLX model; see
[MLX setup](docs/MAC.md#mlx-setup-one-time). Context is auto-capped (~75K) because
released `mlx_lm.server` has no KV-quantization flag.

## Configuration

llama-server flags live in `docker-compose.yml`. Key ones:

| Flag | Value | Why |
|------|-------|-----|
| `--ctx-size` | 16384 | Input capacity vs VRAM |
| `--spec-type draft-mtp` | — | MTP speculative decoding (large speedup on L4) |
| `--cache-reuse` | 256 | KV cache reuse across rounds on the same doc |
| `--flash-attn` | 1 | Flash attention |
| `--n-predict` | 8192 | Max generation length |

UI parameters: rounds per doc, dedup model (on/off), dedup field, dedup
threshold. Benchmark notes on quantization and decoding live in
[`autoresearch/`](autoresearch/REPORT.md).

## Layout

```
app.py             FastAPI app: extraction + UI + API
jobs.py            single-slot job scheduler (queue/preempt/backfill/persist)
Dockerfile         app container
docker-compose.yml both services + data volume
scripts/setup.sh   one-shot GCP L4 setup
scripts/mac-run.sh native Apple Silicon launcher (llama.cpp Metal or MLX)
docs/MAC.md        Apple Silicon setup + memory/backend notes
autoresearch/      throughput benchmark notes
data/              persisted jobs (gitignored)
models/            model files (gitignored)
```

## License

MIT
