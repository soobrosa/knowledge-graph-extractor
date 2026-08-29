#!/bin/bash
# One-shot setup for the Knowledge Graph Extractor on an Apple Silicon Mac.
#
#   1. checks the machine (arm64, unified memory) and the host tools the chosen backend needs
#   2. seeds .env from .env.example
#   3. creates .venv and installs requirements.txt
#   4. downloads the model for the backend selected in .env (BACKEND, default dspark)
#
# Idempotent: every step is skipped if already satisfied, so re-running after a
# failure resumes rather than redoing. Launch with scripts/mac-run.sh afterwards.
#
# Flags:
#   --skip-model   set everything up but don't download the model (~16-18GB)
#   --dspark       also provision mlx-dspark into .venv-dspark (the default backend)
#   --mlx          also create .venv-mlx with mlx-lm (alternative MLX backend)
#   --help
#
# Env overrides: BACKEND, DSPARK_MODEL, MODEL_REPO, MODEL_FILE, MLX_HF_REPO, MLX_MODEL,
# HF_TOKEN, PYTHON_VERSION.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"

SKIP_MODEL=0
WITH_MLX=0
WITH_DSPARK=0
for arg in "$@"; do
  case "$arg" in
    --skip-model) SKIP_MODEL=1 ;;
    --mlx) WITH_MLX=1 ;;
    --dspark) WITH_DSPARK=1 ;;
    -h|--help) sed -n '2,19p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "ERROR: unknown flag '$arg' (try --help)" >&2; exit 1 ;;
  esac
done

step() { printf '\n=== %s ===\n' "$1"; }
ok()   { printf '  ok: %s\n' "$1"; }

# --- 1. machine ---------------------------------------------------------------
step "checking machine"
[ "$(uname -s)" = "Darwin" ] || { echo "ERROR: this script is macOS-only (got $(uname -s))." >&2; exit 1; }
if [ "$(uname -m)" != "arm64" ]; then
  echo "ERROR: Apple Silicon (arm64) required; this is $(uname -m). Metal offload needs an M-series Mac." >&2
  exit 1
fi
ok "$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo 'Apple Silicon'), macOS $(sw_vers -productVersion)"

MEM_GB=$(( $(sysctl -n hw.memsize) / 1024 / 1024 / 1024 ))
if [ "$MEM_GB" -lt 24 ]; then
  echo "  WARNING: ${MEM_GB}GB unified memory. Neither the Qwen3.8-27B 4-bit build (~18GB peak)" >&2
  echo "           nor a Q3 GGUF of the 35B fits comfortably. See docs/MAC.md's memory notes." >&2
elif [ "$MEM_GB" -lt 32 ]; then
  echo "  WARNING: ${MEM_GB}GB unified memory - tight. Keep CTX_SIZE modest and close Chrome/IDEs." >&2
else
  ok "${MEM_GB}GB unified memory"
fi

# --- 2. configuration ---------------------------------------------------------
# Sourced before the defaults below, so .env wins over them (same precedence as
# mac-run.sh: a BACKEND or DSPARK_MODEL pinned in .env is the one this script provisions).
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

BACKEND="${BACKEND:-dspark}"
DSPARK_MODEL="${DSPARK_MODEL:-mlx-community/Qwen3.8-27B-4bit}"
MODEL_REPO="${MODEL_REPO:-unsloth/Qwen3.6-35B-A3B-MTP-GGUF}"
MODEL_FILE="${MODEL_FILE:-Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf}"
MLX_MODEL="${MLX_MODEL:-$ROOT/models/mlx/Qwen3.6-35B-A3B-UD-MLX-4bit}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"

# --- 3. host tools (only what the chosen backend needs) -----------------------
step "checking host tools ($BACKEND backend)"
if ! command -v uv >/dev/null; then
  echo "ERROR: uv not found. Install it with:  curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
fi
ok "uv: $(uv --version)"

if [ "$BACKEND" = "dspark" ] || [ "$WITH_DSPARK" = "1" ]; then
  if command -v mlx-dspark >/dev/null; then
    ok "mlx-dspark: $(command -v mlx-dspark)"
    # The Qwen3.8-27B registry rows (drafter auto-resolve) landed in mlx-dspark 0.13+;
    # older builds serve the target drafter-free in 'lookup' mode - no error, just slow.
    if ! mlx-dspark models 2>/dev/null | grep -q "Qwen3.8-27B"; then
      echo "  WARNING: this mlx-dspark build has no Qwen3.8-27B registry row (needs >= 0.13)." >&2
      echo "           'auto' mode will serve WITHOUT a drafter (lookup-only). Upgrade:" >&2
      echo "               uv tool upgrade mlx-dspark   # or: pip install -U mlx-dspark" >&2
    else
      ok "mlx-dspark knows Qwen3.8-27B (drafter auto-resolve)"
    fi
  else
    echo "  NOTE: mlx-dspark not on PATH; it will be provisioned into .venv-dspark below." >&2
  fi
fi
if [ "$BACKEND" = "llamacpp" ]; then
  if ! command -v llama-server >/dev/null; then
    echo "ERROR: llama-server not found (BACKEND=llamacpp needs it). Install:  brew install llama.cpp" >&2
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
fi

# --- 4. app venv --------------------------------------------------------------
step "python env (.venv)"
if [ -x "$ROOT/.venv/bin/python" ]; then
  ok ".venv exists ($("$ROOT/.venv/bin/python" --version))"
else
  uv venv --python "$PYTHON_VERSION" .venv
  ok "created .venv (python $PYTHON_VERSION)"
fi
uv pip install --quiet --python "$ROOT/.venv/bin/python" -r requirements.txt
ok "installed requirements.txt"

# --- 5. model -----------------------------------------------------------------
if [ "$BACKEND" = "dspark" ]; then
  step "model ($DSPARK_MODEL -> HF cache)"
  if [ -n "${HF_TOKEN:-}" ]; then ok "HF_TOKEN set"; else echo "  note: no HF_TOKEN set; anonymous downloads are rate-limited (https://huggingface.co/settings/tokens)" >&2; fi
  # snapshot_download is a cheap metadata check when the repo is already cached.
  "$ROOT/.venv/bin/python" - "$DSPARK_MODEL" <<'PY'
import sys
from huggingface_hub import snapshot_download
print("  saved:", snapshot_download(sys.argv[1]))
PY
  ok "target model in HF cache"
  echo "  note: the matched drafter (for Qwen3.8-27B: the DFlash2 head) downloads on first"
  echo "        serve - a few GB, once; mlx-dspark then calibrates this Mac's draft cap (~5s)."
elif [ "$BACKEND" = "llamacpp" ]; then
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
else
  step "model (BACKEND=$BACKEND)"
  echo "  handled by the $BACKEND backend section below."
fi

# --- 6. optional backends -----------------------------------------------------
if [ "$WITH_DSPARK" = "1" ]; then
  step "dspark backend (.venv-dspark)"
  if command -v mlx-dspark >/dev/null && [ -z "${FORCE_VENV_DSPARK:-}" ]; then
    ok "mlx-dspark already available on PATH (skipping .venv-dspark; set FORCE_VENV_DSPARK=1 to force)"
  elif [ -x "$ROOT/.venv-dspark/bin/mlx-dspark" ]; then
    ok ".venv-dspark already has mlx-dspark"
  else
    [ -d "$ROOT/.venv-dspark" ] || uv venv --python "$PYTHON_VERSION" .venv-dspark
    VIRTUAL_ENV="$ROOT/.venv-dspark" uv pip install --quiet mlx-dspark
    ok "installed mlx-dspark into .venv-dspark"
  fi
fi

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
    echo "        The default dspark backend works without this; MLX is opt-in (see docs/MAC.md)."
  fi
fi

# --- done ---------------------------------------------------------------------
step "setup complete"
echo "  start it:   bash scripts/mac-run.sh"
[ "$WITH_MLX" = "1" ] && echo "  or on MLX:  BACKEND=mlx bash scripts/mac-run.sh"
[ "$WITH_DSPARK" = "1" ] || [ "$BACKEND" = "dspark" ] || echo "  or on dspark: BACKEND=dspark bash scripts/mac-run.sh"
echo "  then open:  http://localhost:3000/"
