# Knowledge Graph Extractor — Apple Silicon

Turn any document, URL, or a zip of files into an interactive knowledge graph,
using a self-hosted LLM (Qwen3.8-27B, DSpark-accelerated) running **natively on
an Apple Silicon Mac over Metal — no Docker, no NVIDIA GPU, nothing leaving the
machine.**

> **Mac-only fork** of [hanxiao/knowledge-graph-extractor](https://github.com/hanxiao/knowledge-graph-extractor),
> which targets a single NVIDIA L4 via Docker. This fork drops the CUDA/Docker
> path entirely and serves Qwen3.8-27B through
> [mlx-dspark](https://github.com/ARahim3/mlx-dspark) — DeepSeek's DSpark and
> z-lab's DFlash speculative decoding, native on MLX, **lossless and up to
> ~2.6-4x faster decode** than the same model alone. `llama.cpp` (Metal) and
> `mlx-lm` remain as alternative backends for the previous Qwen3.6-35B-A3B
> model. Extraction, dedup, scheduler and UI are upstream's and still track it;
> the Mac support is offered back in
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
| Tools | [`uv`](https://astral.sh/uv); `mlx-dspark` (installed by setup, or `uv tool install mlx-dspark`) |
| Disk | ~16GB for Qwen3.8-27B-4bit, plus a few GB for the drafter |
| Keys | A free [Jina API key](https://jina.ai/api-key), only needed for URL inputs |

A 16GB machine won't fit this model; see the [memory table](docs/MAC.md#memory-notes).
The 8-bit Qwen3.8 build (higher spec-decode ratios, ~29GB peak) needs 48GB+.

## Quickstart

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh

git clone https://github.com/soobrosa/knowledge-graph-extractor.git
cd knowledge-graph-extractor

bash scripts/mac-setup.sh     # machine checks, .venv + deps, .env, mlx-dspark, model download
bash scripts/mac-run.sh       # mlx-dspark on :8080 (Metal), then the app on :3000
```

Then open **http://localhost:3000/**.

`mac-setup.sh` is idempotent: it verifies the machine and host tools, warns about
insufficient unified memory, creates `.venv` from [`requirements.txt`](requirements.txt),
seeds `.env`, installs mlx-dspark if it's missing, and pulls Qwen3.8-27B-4bit into the
Hugging Face cache (resumable). Re-run it freely. Useful flags:

```bash
bash scripts/mac-setup.sh --skip-model   # set up everything but the model download
bash scripts/mac-setup.sh --dspark       # force mlx-dspark into a project .venv-dspark
bash scripts/mac-setup.sh --mlx          # also provision the legacy mlx-lm backend
```

Add your Jina key to `.env` (`JINA_API_KEY=`) before running; `mac-run.sh` refuses
to start without one. Stop the app with `Ctrl+C`, the model with
`pkill -f mlx-dspark`.

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

All three serve an OpenAI-compatible API on `:8080`; the app only ever talks to
`LLAMA_URL`, so no application logic differs between them.

| `BACKEND` | Engine | Model | Decode | Notes |
|---|---|---|---|---|
| `dspark` *(default)* | [mlx-dspark](https://github.com/ARahim3/mlx-dspark) | `mlx-community/Qwen3.8-27B-4bit` (~18GB) | ~25-38 tok/s, **2.6-4x vs the same model alone** (DFlash2 drafter, `--mode auto`) | lossless spec decode; draft cap auto-calibrated per Mac; CPU co-prefill + prefix caching |
| `llamacpp` | llama.cpp (Metal, GGUF) | `models/Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf` | draft-MTP spec decode | the original Mac path |
| `mlx` | mlx-lm (`mlx_lm.server`) | `models/mlx/Qwen3.6-35B-A3B-UD-MLX-4bit` | ~6x llama.cpp prefill | legacy alternative |

The default is one command with no drafter flags: `--mode auto` resolves the
measured-best drafter for the target (for Qwen3.8-27B that's the
[DFlash2](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2) head), and the
draft cap is calibrated against *your* machine on first run. Thinking mode is
off by default — extraction wants facts, not reasoning; `ENABLE_THINKING=1`
turns it back on.

```bash
BACKEND=llamacpp bash scripts/mac-run.sh   # or BACKEND=mlx
```

## Configuration

Everything is env, read from `.env` by both scripts. Defaults live in
[`scripts/mac-run.sh`](scripts/mac-run.sh).

| Var | Default | Why |
|---|---|---|
| `BACKEND` | `dspark` | `dspark`, `llamacpp`, or `mlx` |
| `DSPARK_MODEL` | `mlx-community/Qwen3.8-27B-4bit` | HF repo or local path; drafter auto-resolves |
| `DSPARK_MODE` | `auto` | `auto` / `dspark` / `dflash` / `lookup` / `baseline` |
| `DSPARK_KV_BITS` | `0` | Target KV quantization; **must stay `0` for Qwen3.8-27B** (hybrid linear-attention targets reject `--kv-bits`) |
| `DSPARK_CTX` | `32768` | Server-side context window |
| `ENABLE_THINKING` | off | `1` lets Qwen3.8 reason before answering (slower) |
| `MODEL_FILE` | `Qwen3.6-...gguf` | GGUF under `models/` (llamacpp backend only; symlinks work) |
| `CTX_SIZE` | `16384` | App-side input chunk budget (llamacpp/`mlx` backends) |
| `SPEC_ARGS` | `--spec-type draft-mtp ...` | llama.cpp MTP spec decode; set empty to disable |
| `LLAMA_URL` | `http://127.0.0.1:8080` | Where the app looks for the model server |

UI parameters: rounds per doc, dedup model (on/off), dedup field, dedup threshold.

## Layout

```
app.py               FastAPI app: extraction + UI + API
jobs.py              single-slot job scheduler (queue/preempt/backfill/persist)
requirements.txt     app dependencies (installed into .venv)
scripts/mac-setup.sh one-shot setup: checks, venv + deps, .env, model download
scripts/mac-run.sh   launcher: mlx-dspark (default), llama.cpp, or mlx-lm on :8080
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
