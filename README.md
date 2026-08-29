# Knowledge Graph Extractor — Apple Silicon

Turn any document, URL, or a zip of files into an interactive knowledge graph,
using a self-hosted LLM (Qwen3.6-35B-A3B-MTP) running **natively on an Apple
Silicon Mac over Metal — no Docker, no NVIDIA GPU, nothing leaving the machine.**

> **Mac-only fork** of [hanxiao/knowledge-graph-extractor](https://github.com/hanxiao/knowledge-graph-extractor),
> which targets a single NVIDIA L4 via Docker. This fork drops the CUDA/Docker
> path entirely and replaces it with `llama.cpp` on Metal plus an optional
> [MLX](https://github.com/ml-explore/mlx-lm) backend (~6x faster prefill).
> Extraction, dedup, scheduler and UI are upstream's and still track it; the Mac
> support is offered back in
> [PR #15](https://github.com/hanxiao/knowledge-graph-extractor/pull/15).
> Need the NVIDIA path? Use upstream.

Each extracted fact is one graph edge: a `(subject) --[predicate]--> (object)`
triple plus a title, description, evidence span, confidence, tags, and source
file. Facts stream into a force-directed graph; hover an edge for the full card.

[![Knowledge Graph Extractor](assets/hero.png)](https://hanxiao.io/knowledge-graph)

*(Screenshot and [live demo](https://hanxiao.io/knowledge-graph) are upstream's, running on an L4. The UI is identical.)*

## Requirements

| | |
|---|---|
| Machine | Apple Silicon (M-series). **32GB+ unified memory recommended**; tested on M3 Pro / 36GB |
| Tools | `brew install llama.cpp` (build ≥ 9430 for MTP spec decoding), [`uv`](https://astral.sh/uv) |
| Disk | ~17GB for the Q3_K_XL GGUF |
| Keys | A free [Jina API key](https://jina.ai/api-key), only needed for URL inputs |

A 16GB machine won't fit this model; see the [memory table](docs/MAC.md#memory-notes).

## Quickstart

```bash
brew install llama.cpp
curl -LsSf https://astral.sh/uv/install.sh | sh

git clone https://github.com/soobrosa/knowledge-graph-extractor.git
cd knowledge-graph-extractor

bash scripts/mac-setup.sh     # machine checks, .venv + deps, .env, model download (~17GB)
bash scripts/mac-run.sh       # llama-server on :8080 (Metal), then the app on :3000
```

Then open **http://localhost:3000/**.

`mac-setup.sh` is idempotent: it verifies the machine and host tools, warns about
insufficient unified memory or a llama.cpp build too old for `draft-mtp`, creates
`.venv` from [`requirements.txt`](requirements.txt), seeds `.env`, and resumes the
model download if interrupted. Re-run it freely. Useful flags:

```bash
bash scripts/mac-setup.sh --skip-model   # set up everything but the 17GB download
bash scripts/mac-setup.sh --mlx          # also provision the MLX backend
```

Add your Jina key to `.env` (`JINA_API_KEY=`) before running; `mac-run.sh` refuses
to start without one. Stop the app with `Ctrl+C`, the model with `pkill -f llama-server`.

## How it works

1. **Input** — paste text, a URL (fetched to markdown via Jina Reader), or a
   `.zip` (txt, md, html, pdf, docx, json, csv, code...). Oversized docs are
   chunked (not truncated) so the full text is processed.
2. **Extract** — the LLM emits atomic `(subject, predicate, object)` triples.
   The prompt forces canonical entity/value subjects and objects so nodes
   connect instead of becoming prose dead-ends.
3. **Dedup** (on by default) — semantic dedup via
   [jina-embeddings-v5-text-nano](https://huggingface.co/jinaai/jina-embeddings-v5-text-nano),
   pinned to **CPU** so Metal's unified memory stays reserved for the LLM.
   Applies across rounds and across files.
4. **Visualize** — every unique fact is one edge; node names are normalized so
   variants merge. Download the result as JSONL.

## Job queue

There's one inference slot, so jobs run one at a time via a single-slot scheduler:
a new submission preempts the running job, which is persisted and auto-resumes
from where it left off when the slot frees. Jobs (meta + facts.jsonl + input)
persist under `data/jobs/` so the list, JSONL reload, and resume survive
restarts.

## Backends

Both serve an OpenAI-compatible API on `:8080`; the app only ever talks to
`LLAMA_URL`, so no application logic differs between them.

| `BACKEND` | Engine | Model | Prefill | Context ceiling |
|---|---|---|---|---|
| `llamacpp` *(default)* | llama.cpp (Metal, GGUF) | `models/Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf` | baseline | high (stable) |
| `mlx` | mlx-lm (`mlx_lm.server`) | `models/mlx/Qwen3.6-35B-A3B-UD-MLX-4bit` | ~6x | ~75-85K (auto-capped) |

Extraction is prefill-heavy (each round re-reads the whole document), so MLX is
the win that matters on large docs and zips:

```bash
BACKEND=mlx bash scripts/mac-run.sh
```

MLX needs its own venv (`--mlx` above) because mlx-lm bumps `transformers` and
breaks the dedup embedder if installed alongside it. Context is auto-capped
because released `mlx_lm.server` has no KV-quantization flag; the launcher
detects `--kv-bits` and raises the cap when a build finally ships it. Details and
caveats: [docs/MAC.md](docs/MAC.md#alternative-backend-mlx-faster-prefill).

## Configuration

Everything is env, read from `.env` by both scripts. Defaults live in
[`scripts/mac-run.sh`](scripts/mac-run.sh).

| Var | Default | Why |
|---|---|---|
| `BACKEND` | `llamacpp` | `llamacpp` or `mlx` |
| `MODEL_FILE` | `Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf` | GGUF under `models/` (a symlink to one elsewhere works) |
| `CTX_SIZE` | `16384` | Input capacity vs unified memory; auto-capped under `mlx` |
| `SPEC_ARGS` | `--spec-type draft-mtp ...` | MTP speculative decoding; set empty to disable on older llama.cpp |
| `NGL` | `999` | All layers on Metal (unified memory, no spill tradeoff) |
| `CACHE_REUSE` | `256` | KV cache reuse across rounds on the same doc |
| `LLAMA_URL` | `http://127.0.0.1:8080` | Where the app looks for the model server |

UI parameters: rounds per doc, dedup model (on/off), dedup field, dedup threshold.

## Layout

```
app.py               FastAPI app: extraction + UI + API
jobs.py              single-slot job scheduler (queue/preempt/backfill/persist)
requirements.txt     app dependencies (installed into .venv)
scripts/mac-setup.sh one-shot setup: checks, venv + deps, .env, model download
scripts/mac-run.sh   launcher: llama.cpp (Metal) or MLX on :8080, then the app
docs/MAC.md          full Mac reference: memory tiers, backends, env knobs, caveats
autoresearch/        upstream's quantization/decoding benchmarks (measured on an L4)
data/                persisted jobs (gitignored)
models/              model files (gitignored)
```

Note that [`autoresearch/`](autoresearch/REPORT.md) records upstream's benchmark
runs on NVIDIA L4 hardware. The quantization conclusions carry over; the
throughput numbers do not.

## License

MIT
