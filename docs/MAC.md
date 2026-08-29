# Run on Apple Silicon (Mac, Metal)

This is the Mac-only fork's full reference. The extractor runs natively on an Apple Silicon Mac with
**no Docker and no NVIDIA GPU**. The default stack serves **Qwen3.8-27B** with
[mlx-dspark](https://github.com/ARahim3/mlx-dspark) - DeepSeek's DSpark and z-lab's DFlash
speculative decoding, native on MLX - and the FastAPI app + the `jina-embeddings-v5-text-nano`
dedup embedder run in a local `uv` virtualenv. No application logic changes are needed: the model
is decoupled behind the OpenAI-compatible `LLAMA_URL`, so backends are swappable without touching
the app.

Tested on an **M3 Pro / 36 GB**, macOS. At least **32 GB** of unified memory is recommended.

## What's different from upstream (NVIDIA / Docker)

Upstream [hanxiao/knowledge-graph-extractor](https://github.com/hanxiao/knowledge-graph-extractor)
targets a single NVIDIA L4 through `docker compose`. This fork removes that path entirely; the table
records what the Mac path does instead, and why.

| Area | Upstream (NVIDIA / Docker) | This fork (Apple Silicon) | Why |
| --- | --- | --- | --- |
| Serving | `llama.cpp:server-cuda` container | `mlx-dspark serve` (Metal), `llama-server` or `mlx_lm.server` as alternatives | No CUDA / `nvidia-container-toolkit` on macOS. |
| Model | Qwen3.6-35B-A3B (MoE, GGUF) | Qwen3.8-27B (dense, MLX 4-bit) | Dense model that mlx-dspark's drafters accelerate hard. |
| Speculative decode | draft-MTP (llama.cpp) / none (mlx-lm) | DSpark + DFlash2 drafters (`--mode auto`) | Lossless: the target verifies every drafted token. |
| GPU offload | implicit (CUDA) | all layers on Metal | Unified memory; `-ngl 999` on the llama.cpp path. |
| App + embedder | Docker container | `python app.py` in a `uv` venv; embedder on CPU | No GPU passthrough into Docker on macOS; CPU keeps Metal free for the LLM. |
| torch | CPU wheel baked into the image | `requirements.txt` into `.venv` (MPS/CPU build) | Installed locally; no image to build. |
| Setup | `scripts/setup.sh` (GCP L4) | `scripts/mac-setup.sh` | Machine checks, venv, deps, model download. |

## Prerequisites

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # uv (Python env)
```

You also need a free **Jina API key** (https://jina.ai/api-key, only used for URL inputs) and,
recommended, a free **Hugging Face read token** (https://huggingface.co/settings/tokens) for a
fast, stable model download. `mlx-dspark` is installed by setup if missing; install it yourself
with `uv tool install mlx-dspark`.

## Setup (scripted)

```bash
bash scripts/mac-setup.sh
```

One idempotent pass: verifies arm64 and unified memory, checks the tools the backend selected in
`.env` needs, creates `.venv` from `requirements.txt`, seeds `.env` from `.env.example`, installs
`mlx-dspark` if it's missing, and pulls the model into the HF cache (resumable). Re-run it after a
failure and it resumes rather than redoing.

```bash
HF_TOKEN=hf_your_token bash scripts/mac-setup.sh   # avoids the anonymous rate limit
bash scripts/mac-setup.sh --skip-model             # everything except the model download
bash scripts/mac-setup.sh --dspark                 # force mlx-dspark into project .venv-dspark
bash scripts/mac-setup.sh --mlx                    # also provision the legacy mlx-lm backend
```

Then put a real key in `.env` (`JINA_API_KEY=...`) - `mac-run.sh` refuses to start without one -
and skip to [Run](#run).

## Setup (manual)

Equivalent to the script, if you'd rather do it by hand.

### 1. Install the Python deps

`torch` pulls the Apple-Silicon (MPS/CPU) build.

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.txt
uv tool install mlx-dspark        # the default inference server (installs mlx >= 0.32)
```

### 2. Download the model (Qwen3.8-27B-4bit, ~16 GB)

mlx-dspark takes an HF repo id or a local path. Pre-pull the target so the first serve isn't a
download; the matched drafter (for Qwen3.8-27B: `incoai/Qwen3.8-27B-DFlash2`, a few GB) still
fetches on first serve:

```bash
HF_TOKEN=hf_your_token \
.venv/bin/python -c "from huggingface_hub import snapshot_download; \
snapshot_download('mlx-community/Qwen3.8-27B-4bit')"
```

Peak RAM: **~18 GB** (4-bit) / ~29 GB (8-bit - needs 48 GB+, but buys the project's best measured
spec-decode ratios: 3.63x mean with the same DFlash2 drafter). KV cache costs ~0.086 GB per 1k
tokens fp16. Note Qwen3.8-27B is a **hybrid linear-attention** target (only 16 of 64 layers hold a
KV cache), so `DSPARK_KV_BITS` must stay `0` for it - mlx-dspark rejects `--kv-bits` for hybrids.

### 3. Set your key

```bash
cp .env.example .env
sed -i '' 's/^JINA_API_KEY=.*/JINA_API_KEY=jina_your_real_key/' .env
```

Defaults are baked into `scripts/mac-run.sh`; override any in `.env`:

```bash
BACKEND=dspark
DSPARK_MODEL=mlx-community/Qwen3.8-27B-4bit
DSPARK_MODE=auto                  # auto|dspark|dflash|lookup|baseline
DSPARK_KV_BITS=0                  # 4/8 only for pure-attention targets; hybrids reject it
DSPARK_CTX=32768
ENABLE_THINKING=1                 # default off
LLAMA_URL=http://127.0.0.1:8080
JOBS_DIR=./data/jobs
```

Both scripts source `.env` before applying their defaults, so anything pinned there wins.

## Run

```bash
bash scripts/mac-run.sh
```

Starts `mlx-dspark serve` on `:8080`, waits for it to load (first run downloads the drafter and
calibrates this Mac's draft cap, ~5 s, cached), then starts the app on `:3000`.
Open **http://localhost:3000/**.

Stop the app with `Ctrl+C`; stop the model with `pkill -f mlx-dspark`.

## Smoke test

`scripts/smoke.sh` verifies the live stack end to end in ~1 minute: `/health`,
a real completion, an attached drafter (a `mode=lookup` result is reported as a
failure - that's the silent degradation an old mlx-dspark causes), the UI, and
a real extraction job whose facts must parse with subject/predicate/object set.
It exits 0/1 and leaves its test job in the job list (no delete API yet). Run
it standalone, or let `mac-run.sh --smoke` check the server before starting the
app.

### What mlx-dspark buys

- **Lossless speculation.** The target verifies every drafted token, so output is identical to
  plain decoding - every mode, every cap. Only the speed changes.
- **No tuning needed.** `--mode auto` resolves the measured-best drafter for the target (for
  Qwen3.8-27B: the [DFlash2](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2) head; the DSpark
  alternative is `DimInfer/Qwen3.8-27B-Dspark-v1` at 4-bit, `RadixArk/Qwen3.8-27B-DSpark` at
  8-bit). The draft cap is calibrated against *your* machine, not the M4 Pro the project's tables
  were measured on.
- **Prefill help, too.** CPU co-prefill (~1.3-1.4x on wide prefills) is on by default; prefix
  caching reuses document chunks across extraction rounds. `--cpu-split 0` disables the former if
  it misbehaves on your Mac.
- **Thinking is off** (`--no-thinking`) by default: extraction wants facts, not reasoning. Turn
  `ENABLE_THINKING=1` on in `.env` if you want the model to reason before answering.

Measured decode on Qwen3.8-27B (mlx-dspark's tables, M4 Pro): 4-bit **~25-38 tok/s** at 2.3-2.6x,
8-bit ~24-34 tok/s at 2.8-4.1x (chat is the low end, math/code the high). Extraction output is
structure-heavy and tends toward the high end.

## Memory notes

Qwen3.8-27B-4bit peaks at ~18 GB plus KV cache (~0.086 GB per 1k tokens fp16; only 16 of its 64
layers hold KV, and `--kv-bits` is rejected for this hybrid target). On a 36 GB machine it runs
comfortably at the default 32K context. If a long job pages heavily, lower `DSPARK_CTX`.

| Unified memory | Qwen3.8-27B-4bit (~18 GB + KV) | Guidance |
| --- | --- | --- |
| 16 GB | ✗ won't fit | Not supported; a 7-14B model is the ceiling on this tier. |
| 24 GB | ⚠ tight | Fits but keep `DSPARK_CTX` modest; close Chrome/IDEs. |
| 32-48 GB | ✓ comfortable (**tested on 36 GB**) | Recommended. |
| 48 GB+ | ✓ | Room for the 8-bit build (best ratios, ~29 GB peak) or much longer context. |

The embedder stays on CPU precisely so Metal's memory is reserved for the LLM.

## Alternative backend: llama.cpp (the original Mac path)

The fork's first backend, kept for parity with upstream's GGUF-based flow and its MTP
speculative decoding:

```bash
BACKEND=llamacpp bash scripts/mac-run.sh      # or set BACKEND=llamacpp in .env
```

Needs Homebrew's `llama-server` (`brew install llama.cpp`, build >= 9430 for draft-MTP) and the
Q3_K_XL GGUF under `models/` - `mac-setup.sh` downloads it when `BACKEND=llamacpp`. Key knobs
(`llama-server` flags mirror upstream's tuned CUDA config with Mac-appropriate values):
`--flash-attn on` (this build's flag takes `on|off|auto`, not `1`), `-ngl 999` (all layers on
Metal), `SPEC_ARGS` for draft-MTP, `CACHE_REUSE` for KV reuse across rounds. If your llama.cpp
build predates 9430, use the non-MTP GGUF (`unsloth/Qwen3.6-35B-A3B-GGUF`) and leave `SPEC_ARGS`
empty. A GGUF symlinked in from elsewhere works - see `MODEL_FILE` in `.env`.

## Alternative backend: MLX (legacy)

The intermediate backend before mlx-dspark landed. Serves the old Qwen3.6-35B-A3B model with
plain mlx-lm - prefill ~6x faster than llama.cpp, but no drafter:

```bash
BACKEND=mlx bash scripts/mac-run.sh
```

### MLX setup (one-time)

mlx-lm must live in its **own** venv - installing it into the app `.venv` bumps `transformers` and
can break the embedder. `mac-setup.sh --mlx` does this for you:

```bash
bash scripts/mac-setup.sh --mlx                    # creates .venv-mlx, installs mlx-lm
MLX_HF_REPO=<org/repo> bash scripts/mac-setup.sh --mlx --skip-model   # + fetch a 4-bit MLX build
```

Without `MLX_HF_REPO` it sets up the venv and then tells you how to supply the model, since there's
no single canonical 4-bit MLX repo for this one. By hand:

```bash
uv venv .venv-mlx
VIRTUAL_ENV=$PWD/.venv-mlx uv pip install mlx-lm
# Download or convert a 4-bit MLX build into models/mlx/Qwen3.6-35B-A3B-UD-MLX-4bit:
.venv-mlx/bin/mlx_lm.convert --hf-path <org/repo> -q --q-bits 4 \
  --mlx-path models/mlx/Qwen3.6-35B-A3B-UD-MLX-4bit
```

`mac-run.sh` checks both the venv and the model exist before starting and errors with the fix if not.

### MLX env knobs

| Var | Default | Meaning |
| --- | --- | --- |
| `MLX_MODEL` | `models/mlx/Qwen3.6-35B-A3B-UD-MLX-4bit` | Path to the MLX model dir. |
| `MLX_CTX_CAP` | `75000` (fp16 KV) / `85000` (4-bit KV) | Max context for MLX; `CTX_SIZE` is auto-capped to this. |
| `MODEL_NAME` | (auto = the backend's model) | Pinned so the app's request `model` field matches the loaded model. |

### MLX caveats

- **Context auto-cap.** Stock `mlx_lm.server` runs an **fp16 KV cache** with no KV-quantization
  flag (see upstream [ml-explore/mlx-lm#1043](https://github.com/ml-explore/mlx-lm/issues/1043)).
  As of mlx-lm 0.31.3 (latest release) there is still **no `--kv-bits` on `mlx_lm.server`**, so
  this is the path everyone gets today: fp16 KV OOMs at ~78-92K actual tokens on 36 GB, and the
  script caps context to `MLX_CTX_CAP` (75K) accordingly. The script is forward-looking - it
  auto-detects a `--kv-bits` flag and, when present, switches to 4-bit KV and raises the cap
  toward ~85K. That flag isn't in any released mlx-lm yet; I'm working on landing it upstream, and
  until it ships the fp16/75K path is the one in use.
- **`MODEL_NAME` is pinned to the model path.** `mlx_lm.server` resolves the request's `model`
  field against the loaded model and otherwise tries to fetch it from HuggingFace (a request for
  the friendly label `qwen3.6` -> 404). llama.cpp ignores the label; MLX needs the match, so the
  script exports `MODEL_NAME=$MLX_MODEL` for this backend. The dspark backend pins it to the repo
  id for the same reason.

Stop the MLX model with `pkill -f mlx_lm.server`.
