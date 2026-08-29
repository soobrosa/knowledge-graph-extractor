#!/bin/bash
# One-shot setup for the Knowledge Graph Extractor on an Apple Silicon Mac.
#
#   1. checks the machine (arm64, unified memory) and the two host tools (llama.cpp, uv)
#   2. seeds .env from .env.example
#   3. creates .venv and installs requirements.txt
#   4. downloads the GGUF model (~17GB) into models/
#
# Idempotent: every step is skipped if already satisfied, so re-running after a
# failure resumes rather than redoing. Launch with scripts/mac-run.sh afterwards.
#
# Flags:
#   --skip-model   set everything up but don't download the ~17GB GGUF
#   --mlx          also create .venv-mlx with mlx-lm (optional faster-prefill backend)
#   --help
#
# Env overrides: MODEL_REPO, MODEL_FILE, MLX_HF_REPO, MLX_MODEL, HF_TOKEN, PYTHON_VERSION.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"

SKIP_MODEL=0
WITH_MLX=0
for arg in "$@"; do
  case "$arg" in
    --skip-model) SKIP_MODEL=1 ;;
    --mlx) WITH_MLX=1 ;;
    -h|--help) sed -n '2,17p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "ERROR: unknown flag '$arg' (try --help)" >&2; exit 1 ;;
  esac
done

step() { printf '\n=== %s ===\n' "$1"; }
ok()   { printf '  ok: %s\n' "$1"; }

# --- 1. machine + host tools --------------------------------------------------
step "checking machine"
[ "$(uname -s)" = "Darwin" ] || { echo "ERROR: this script is macOS-only (got $(uname -s))." >&2; exit 1; }
if [ "$(uname -m)" != "arm64" ]; then
  echo "ERROR: Apple Silicon (arm64) required; this is $(uname -m). Metal offload needs an M-series Mac." >&2
  exit 1
fi
ok "$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo 'Apple Silicon'), macOS $(sw_vers -productVersion)"

MEM_GB=$(( $(sysctl -n hw.memsize) / 1024 / 1024 / 1024 ))
if [ "$MEM_GB" -lt 24 ]; then
  echo "  WARNING: ${MEM_GB}GB unified memory. A Q3 quant of this model wires ~17GB and won't fit comfortably." >&2
  echo "           See the memory table in docs/MAC.md; a 7-14B model is the realistic ceiling here." >&2
elif [ "$MEM_GB" -lt 32 ]; then
  echo "  WARNING: ${MEM_GB}GB unified memory - tight. Keep CTX_SIZE modest and close Chrome/IDEs." >&2
else
  ok "${MEM_GB}GB unified memory"
fi

step "checking host tools"
if ! command -v llama-server >/dev/null; then
  echo "ERROR: llama-server not found. Install it with:  brew install llama.cpp" >&2
  exit 1
fi
ok "llama-server: $(command -v llama-server)"
# Capture the help text instead of piping into `grep -q`: grep exits on first match,
# llama-server then dies of SIGPIPE, and `set -o pipefail` turns that into a failed
# pipeline - i.e. the feature reads as absent on builds that do support it.
LLAMA_HELP="$(llama-server --help 2>/dev/null || true)"
if [ "${LLAMA_HELP#*draft-mtp}" = "$LLAMA_HELP" ]; then
  echo "  WARNING: this llama.cpp build lacks --spec-type draft-mtp (needs build >= 9430)." >&2
  echo "           Either 'brew upgrade llama.cpp', or set SPEC_ARGS= in .env to disable spec decoding" >&2
  echo "           and use a non-MTP GGUF (MODEL_REPO=unsloth/Qwen3.6-35B-A3B-GGUF)." >&2
else
  ok "llama.cpp supports draft-mtp speculative decoding"
fi
if ! command -v uv >/dev/null; then
  echo "ERROR: uv not found. Install it with:  curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
fi
ok "uv: $(uv --version)"

# --- 2. configuration ---------------------------------------------------------
# Sourced before the defaults below, so .env wins over them (same precedence as
# mac-run.sh: a MODEL_FILE pinned in .env is the one this script looks for).
step "configuration (.env)"
if [ -f .env ]; then
  ok ".env exists (left untouched)"
else
  cp .env.example .env
  ok "created .env from .env.example"
fi
set -a
# shellcheck disable=SC1091  # .env is user-created, not in the repo
. ./.env
set +a
if [ -z "${JINA_API_KEY:-}" ] || [ "${JINA_API_KEY:-}" = "jina_xxxx" ]; then
  echo "  TODO: put a real JINA_API_KEY in .env (free: https://jina.ai/api-key)." >&2
  echo "        Only URL inputs need it - pasted text and zips don't - but mac-run.sh" >&2
  echo "        refuses to start until it's set." >&2
else
  ok "JINA_API_KEY is set"
fi

MODEL_REPO="${MODEL_REPO:-unsloth/Qwen3.6-35B-A3B-MTP-GGUF}"
MODEL_FILE="${MODEL_FILE:-Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf}"
MLX_MODEL="${MLX_MODEL:-$ROOT/models/mlx/Qwen3.6-35B-A3B-UD-MLX-4bit}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"

# --- 3. app venv --------------------------------------------------------------
step "python env (.venv)"
if [ -x "$ROOT/.venv/bin/python" ]; then
  ok ".venv exists ($("$ROOT/.venv/bin/python" --version))"
else
  uv venv --python "$PYTHON_VERSION" .venv
  ok "created .venv (python $PYTHON_VERSION)"
fi
uv pip install --quiet --python "$ROOT/.venv/bin/python" -r requirements.txt
ok "installed requirements.txt"

# --- 4. model -----------------------------------------------------------------
step "model (models/$MODEL_FILE)"
mkdir -p models
if [ -f "$ROOT/models/$MODEL_FILE" ]; then
  # -f and du -L follow symlinks, so a model linked in from elsewhere counts as present.
  ok "already present ($(du -hL "$ROOT/models/$MODEL_FILE" | cut -f1))"
elif [ "$SKIP_MODEL" = "1" ]; then
  echo "  skipped (--skip-model). Fetch it later by re-running this script."
else
  echo "  downloading $MODEL_REPO / $MODEL_FILE (~17GB, resumable)..."
  [ -n "${HF_TOKEN:-}" ] || echo "  note: no HF_TOKEN set; anonymous downloads are rate-limited (https://huggingface.co/settings/tokens)"
  "$ROOT/.venv/bin/python" - "$MODEL_REPO" "$MODEL_FILE" <<'PY'
import sys
from huggingface_hub import hf_hub_download
print("  saved:", hf_hub_download(sys.argv[1], sys.argv[2], local_dir="models"))
PY
  ok "model ready"
fi

# --- 5. optional MLX backend --------------------------------------------------
if [ "$WITH_MLX" = "1" ]; then
  step "MLX backend (.venv-mlx)"
  # mlx-lm must stay out of .venv: it bumps transformers and breaks the dedup embedder.
  if [ -x "$ROOT/.venv-mlx/bin/mlx_lm.server" ]; then
    ok ".venv-mlx already has mlx-lm"
  else
    [ -d "$ROOT/.venv-mlx" ] || uv venv .venv-mlx
    VIRTUAL_ENV="$ROOT/.venv-mlx" uv pip install --quiet mlx-lm
    ok "installed mlx-lm into .venv-mlx"
  fi
  if [ -d "$MLX_MODEL" ]; then
    ok "MLX model present: $MLX_MODEL"
  elif [ -n "${MLX_HF_REPO:-}" ]; then
    echo "  downloading MLX model from $MLX_HF_REPO ..."
    mkdir -p "$(dirname "$MLX_MODEL")"
    "$ROOT/.venv/bin/python" - "$MLX_HF_REPO" "$MLX_MODEL" <<'PY'
import sys
from huggingface_hub import snapshot_download
print("  saved:", snapshot_download(sys.argv[1], local_dir=sys.argv[2]))
PY
    ok "MLX model ready"
  else
    echo "  TODO: no MLX model at $MLX_MODEL and no MLX_HF_REPO given, so nothing was fetched."
    echo "        Either point at a pre-quantized 4-bit MLX repo:"
    echo "            MLX_HF_REPO=<org/repo> bash scripts/mac-setup.sh --mlx --skip-model"
    echo "        or convert one yourself:"
    echo "            .venv-mlx/bin/mlx_lm.convert --hf-path <org/repo> -q --q-bits 4 --mlx-path $MLX_MODEL"
    echo "        The llamacpp backend works without this; MLX is opt-in (see docs/MAC.md)."
  fi
fi

# --- done ---------------------------------------------------------------------
step "setup complete"
echo "  start it:   bash scripts/mac-run.sh"
[ "$WITH_MLX" = "1" ] && echo "  or on MLX:  BACKEND=mlx bash scripts/mac-run.sh"
echo "  then open:  http://localhost:3000/"
