#!/bin/bash
# Smoke test for the Knowledge Graph Extractor stack. Run against a LIVE stack
# (scripts/mac-run.sh), from anywhere:
#
#   bash scripts/smoke.sh              # check server :8080 AND app :3000 end-to-end
#   bash scripts/smoke.sh --skip-app   # inference server only
#   bash scripts/smoke.sh --skip-server # app only (still needs the server, for extraction)
#
# What it proves, in order:
#   1. the inference server answers /health and actually generates tokens
#   2. (dspark) a real drafter is attached - NOT silent lookup-mode degradation
#   3. the app serves its UI
#   4. a real extraction job through the app returns parseable, well-formed facts
#
# Exit 0 only if every requested stage passes. The extraction job it creates
# stays in the job list ("SMOKE TEST - safe to delete"); there is no delete API.
#
# Knobs: SERVER_URL, APP_URL, SMOKE_TIMEOUT (s, default 180).
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
ROOT="$(pwd)"

SERVER_URL="${SERVER_URL:-http://127.0.0.1:8080}"
APP_URL="${APP_URL:-http://127.0.0.1:3000}"
SMOKE_TIMEOUT="${SMOKE_TIMEOUT:-180}"
SKIP_APP=0
SKIP_SERVER=0
for arg in "$@"; do
  case "$arg" in
    --skip-app) SKIP_APP=1 ;;
    --skip-server) SKIP_SERVER=1 ;;
    -h|--help) sed -n '2,17p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown flag '$arg' (try --help)" >&2; exit 1 ;;
  esac
done

PASS=0; FAIL=0
ok()   { printf '  \033[32mPASS\033[0m %s\n' "$1"; PASS=$((PASS+1)); }
bad()  { printf '  \033[31mFAIL\033[0m %s\n' "$1"; FAIL=$((FAIL+1)); }
info() { printf '  --  %s\n' "$1"; }
stage(){ printf '\n== %s ==\n' "$1"; }

PY=python3
command -v "$PY" >/dev/null 2>&1 || PY="$ROOT/.venv/bin/python"

# --- 1. inference server ------------------------------------------------------
if [ "$SKIP_SERVER" = "0" ]; then
  stage "inference server ($SERVER_URL)"

  H=$(curl -fsS -m 5 "$SERVER_URL/health" 2>/dev/null)
  if [ -n "$H" ]; then ok "/health responds"; else bad "/health unreachable - is the server up?"; fi

  # dspark-family servers expose drafter/mode on /health; llama.cpp doesn't.
  if printf '%s' "$H" | "$PY" -c "import json,sys; d=json.load(sys.stdin); sys.exit(0 if 'drafter' in d else 1)" 2>/dev/null; then
    MODE=$(printf '%s' "$H" | "$PY" -c "import json,sys; print(json.load(sys.stdin).get('mode'))")
    DRAFTER=$(printf '%s' "$H" | "$PY" -c "import json,sys; print(json.load(sys.stdin).get('drafter'))")
    if [ "$MODE" = "lookup" ] && [ "$DRAFTER" = "None" ]; then
      bad "drafter NOT attached: serving drafter-free 'lookup' mode (this was the silent mlx-dspark 0.7.0 degradation)"
      info "fix: uv tool upgrade mlx-dspark  (0.13+ knows Qwen3.8-27B and auto-resolves the DFlash2 drafter)"
    elif [ "$MODE" = "baseline" ]; then
      info "mode=baseline (spec decode deliberately off; if that's not intended, check DSPARK_MODE)"
    else
      ok "speculative decoding active: mode=$MODE drafter=$DRAFTER"
    fi
  else
    info "no drafter/mode on /health (llama.cpp-style server) - skipping spec-decode check"
  fi

  # A real completion: proves the server generates, not just that it's listening.
  MODEL=$(curl -fsS -m 5 "$SERVER_URL/v1/models" 2>/dev/null | "$PY" -c "import json,sys; print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null)
  MODEL="${MODEL:-${MODEL_NAME:-qwen3.8}}"
  R=$(curl -fsS -m 120 "$SERVER_URL/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with the single word OK.\"}],\"max_tokens\":16,\"temperature\":0}" 2>/dev/null)
  if printf '%s' "$R" | "$PY" -c "import json,sys; sys.exit(0 if json.load(sys.stdin)['choices'][0]['message']['content'].strip() else 1)" 2>/dev/null; then
    ok "completion round-trip works (model=$MODEL)"
  else
    bad "completion failed or returned empty content"
    info "last response: $(printf '%s' "$R" | head -c 300)"
  fi
fi

# --- 2. app + end-to-end extraction -------------------------------------------
if [ "$SKIP_APP" = "0" ]; then
  stage "app ($APP_URL)"

  if curl -fsS -m 10 "$APP_URL/" -o /dev/null 2>/dev/null; then
    ok "UI serves"
  else
    bad "app not reachable on $APP_URL"
  fi

  stage "end-to-end extraction"
  DOC='Daniel Molnar founded EagleEye, a VC intelligence startup based in Berlin, in 2023. EagleEye builds AI tooling for venture capital diligence.'
  TMPDOC=$(mktemp /tmp/kge-smoke.XXXXXX.txt)
  printf '%s' "$DOC" > "$TMPDOC"
  JOB=$(curl -fsS -m 15 -X POST "$APP_URL/api/jobs-text" \
    -F "file=@$TMPDOC;type=text/plain" -F "k=1" 2>/dev/null | "$PY" -c "import json,sys; print(json.load(sys.stdin).get('job_id',''))" 2>/dev/null)
  rm -f "$TMPDOC"
  if [ -n "$JOB" ]; then
    ok "job accepted ($JOB)"
  else
    bad "POST /api/jobs-text failed"
    JOB=""
  fi

  if [ -n "$JOB" ]; then
    # Poll the JSONL until at least one parseable fact lands, the job errors, or we time out.
    # (files_done can lag the actual facts, so facts - not progress - are the completion signal.)
    RESULT=""
    ELAPSED=0
    while [ "$ELAPSED" -lt "$SMOKE_TIMEOUT" ]; do
      sleep 4; ELAPSED=$((ELAPSED+4))
      RESP=$(curl -fsS -m 10 "$APP_URL/api/jobs/$JOB/jsonl" 2>/dev/null)
      [ -n "$RESP" ] || continue
      RESULT=$(printf '%s' "$RESP" | "$PY" -c '
import json, sys
try:
    d = json.load(sys.stdin)
except Exception:
    sys.exit(0)
meta = d.get("meta", {})
if meta.get("error"):
    print("ERROR:" + str(meta["error"])); sys.exit(0)
lines = [l for l in d.get("jsonl", "").splitlines() if l.strip()]
facts = []
for l in lines:
    try: facts.append(json.loads(l))
    except Exception: pass
if facts:
    bad_fields = [f for f in facts
                  if not (f.get("subject") and f.get("predicate") and f.get("object"))]
    print("OK:%d:%d" % (len(facts), len(bad_fields)))
' 2>/dev/null)
      case "$RESULT" in
        ERROR:*) bad "job errored: ${RESULT#ERROR:}"; break ;;
        OK:*) break ;;
      esac
    done

    case "$RESULT" in
      OK:*)
        NFACTS=${RESULT#OK:}; NFACTS=${NFACTS%%:*}; NBAD=${RESULT##*:}
        if [ "$NBAD" = "0" ]; then
          ok "extraction returned $NFACTS well-formed fact(s) in ~${ELAPSED}s"
        else
          bad "$NBAD of $NFACTS facts missing subject/predicate/object"
        fi
        ;;
      "")
        bad "no facts after ${SMOKE_TIMEOUT}s (job $JOB - check the UI/logs; model may be slow or wedged)"
        ;;
    esac
    info "cleanup: the job stays in the list as 'SMOKE TEST' output - no delete API exists yet"
  fi
fi

# --- verdict ------------------------------------------------------------------
printf '\n'
if [ "$FAIL" -eq 0 ]; then
  printf '\033[32mSMOKE PASS\033[0m (%s check(s))\n' "$PASS"
  exit 0
else
  printf '\033[31mSMOKE FAIL\033[0m - %s passed, %s failed\n' "$PASS" "$FAIL"
  exit 1
fi
