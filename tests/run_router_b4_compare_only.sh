#!/bin/bash
# Coordinator-only comparison retry. Reuses the synchronized activation from
# an earlier ordinary control and never launches ds4 or repeats prefill.
set -o pipefail

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
W=${W:-$ROOT}
M=${M:-}
G=${G:-}
CAP=${CAP:-}
O=${O:-${TMPDIR:-/tmp}/ds4-router-b4-compare-$(date -u +%Y%m%dT%H%M%SZ)}
LAYER=${LAYER:-24}
POS=${POS:-4096}
ROWS=${ROWS:-8192}

[ -n "$M" ] || { echo "M must name the target GGUF" >&2; exit 2; }
[ -n "$G" ] || { echo "G must name the GPU-lock helper" >&2; exit 2; }
[ -n "$CAP" ] || { echo "CAP must name the existing capture prefix" >&2; exit 2; }

if [ "${RUN_ROUTER_B4_GPU:-}" != 1 ]; then
    echo "RUN_ROUTER_B4_GPU=1 is required; this wrapper runs Metal work" >&2
    exit 2
fi
if [ -z "${GPU_LOCK_TOKEN:-}" ]; then
    echo "GPU_LOCK_TOKEN is required; the coordinator must acquire the GPU lock" >&2
    exit 2
fi
"$G" preflight "$GPU_LOCK_TOKEN" || exit 2
if [ -n "${DS4_GLM_EXACT:-}" ] && [ "${DS4_GLM_EXACT:-0}" != 0 ]; then
    echo "DS4_GLM_EXACT must be unset for the B4 arm" >&2
    exit 2
fi
for f in "$W/speed-bench/metal_router_b4_fixture" "$M"; do
    [ -s "$f" ] || { echo "missing required file: $f" >&2; exit 2; }
done

ACT="${CAP}_glm_ffn_norm-${LAYER}_pos${POS}.bin"
[ -s "$ACT" ] || { echo "missing synchronized activation: $ACT" >&2; exit 2; }
expected_bytes=$((ROWS * 4096 * 4))
actual_bytes=$(stat -f %z "$ACT") || exit 2
[ "$actual_bytes" = "$expected_bytes" ] || {
    echo "activation has $actual_bytes bytes, expected $expected_bytes ($ROWS rows)" >&2
    exit 2
}

mkdir -p "$O"
failures=0
fail() {
    echo "FAIL: $*" >&2
    failures=$((failures + 1))
}

DS4_GLM_ENABLE_ROUTER_SPLITK_B4=1 \
DS4_GLM_DISABLE_ROUTER_SPLITK_B4=0 \
"$W/speed-bench/metal_router_b4_fixture" \
    --gguf "$M" --capture-prefix "$CAP" --layer "$LAYER" --pos "$POS" \
    --expect-rows "$ROWS" --ref-rows 64 --out "$O/results" \
    >"$O/compare.out" 2>"$O/compare.err"
rc=$?
echo "rc=$rc" >"$O/compare.rc"
[ "$rc" = 0 ] || fail "A/B4 fixture comparison exited $rc"
[ -s "$O/results/summary.txt" ] || fail "comparison summary is missing"
grep -q "GLM router split-K B4 ENGAGED" "$O/compare.err" ||
    fail "comparison log lacks the B4 engagement receipt"
for f in "$O/results/selection-A.bin" "$O/results/selection-B4.bin"; do
    [ -s "$f" ] || fail "comparison selection artifact is missing: $f"
done

head_id=$(git -C "$W" rev-parse HEAD) || fail "could not resolve source HEAD"
target_bytes=$(stat -f %z "$M") || fail "could not stat target bytes"
target_mtime=$(stat -f %m "$M") || fail "could not stat target mtime"
{
    echo "tree=$W"
    echo "head=$head_id"
    echo "target=$M"
    echo "target_sha256_bound_prior=828f413cae8ceee74796606814295e00e04c1c7b151aca9a3c5c75db32cb73e0"
    echo "target_bytes=$target_bytes"
    echo "target_mtime=$target_mtime"
    echo "capture=$ACT"
    echo "capture_layer=$LAYER"
    echo "capture_pos=$POS"
    echo "capture_rows=$ROWS"
    echo "control_prefill_reused=1"
} >"$O/IDENTITY" || fail "could not write identity receipt"
shasum -a 256 "$ACT" "$W/speed-bench/metal_router_b4_fixture" >"$O/SHA256SUMS" ||
    fail "could not hash the activation or fixture executable"
[ -s "$O/results/summary.txt" ] && cat "$O/results/summary.txt"
echo "failures=$failures" >"$O/RESULT"
echo "artifacts: $O"
[ "$failures" = 0 ]
