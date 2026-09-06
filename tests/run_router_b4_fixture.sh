#!/bin/bash
# Coordinator-only GPU-window wrapper. It captures one synchronized real
# router activation from an ordinary 16k control, then runs the standalone
# A/B4 comparison. It is deliberately not a Makefile test and must never be
# folded into a timed expert-bank run: graph dumps synchronize, and the bank
# super-chunk refuses debug-dump mode.
set -o pipefail

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
W=${W:-$ROOT}
M=${M:-}
P62=${P62:-}
G=${G:-}
O=${O:-${TMPDIR:-/tmp}/ds4-router-b4-$(date -u +%Y%m%dT%H%M%SZ)}
LAYER=${LAYER:-24}
POS=${POS:-4096}
ROWS=${ROWS:-8192}

[ -n "$M" ] || { echo "M must name the target GGUF" >&2; exit 2; }
[ -n "$P62" ] || { echo "P62 must name the long prompt" >&2; exit 2; }
[ -n "$G" ] || { echo "G must name the GPU-lock helper" >&2; exit 2; }

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
for f in "$W/ds4" "$W/speed-bench/metal_router_b4_fixture" "$M" "$P62"; do
    [ -s "$f" ] || { echo "missing required file: $f" >&2; exit 2; }
done

mkdir -p "$O"
# This 69,254-byte prefix is the established 16,224-token control.  Its
# ordinary chunk plan is 4,096 + 8,192 + 3,936 tokens.  The exact dump name
# below fixes the compared middle chunk at pos=4,096, while its byte count
# proves that the synchronized activation contains all 8,192 rows.
head -c 69254 "$P62" >"$O/p16k.txt"
[ "$(stat -f %z "$O/p16k.txt")" = 69254 ] || {
    echo "failed to create the exact 69,254-byte control slice" >&2
    exit 2
}
CAP="$O/capture"
failures=0
fail() {
    echo "FAIL: $*" >&2
    failures=$((failures + 1))
}

# Correctness fixture only. This forced ordinary control has B4 and the bank
# schedule off, and the existing dump hook synchronizes before reading the
# GPU-produced ffn_norm rows.
env DS4_METAL_MODEL_UNTRACKED=1 \
    DS4_GLM53_MEMORY_CEILING_GB=280 \
    DS4_GLM53_PREFILL_CHUNK=8192 \
    DS4_GLM_DISABLE_SUPERCHUNK=1 \
    DS4_GLM_DISABLE_EXPERT_BANK=1 \
    DS4_GLM_DISABLE_ROUTER_SPLITK_B4=1 \
    DS4_METAL_GRAPH_DUMP_PREFIX="$CAP" \
    DS4_METAL_GRAPH_DUMP_NAME=glm_ffn_norm \
    DS4_METAL_GRAPH_DUMP_LAYER="$LAYER" \
    DS4_METAL_GRAPH_DUMP_POS="$POS" \
    "$W/ds4" -m "$M" --metal --nothink --temp 0 -c 70000 -n 1 \
    --prompt-file "$O/p16k.txt" >"$O/control.out" 2>"$O/control.err"
rc=$?
echo "rc=$rc" >"$O/control.rc"
[ "$rc" = 0 ] || fail "ordinary 16k control exited $rc"

ACT="${CAP}_glm_ffn_norm-${LAYER}_pos${POS}.bin"
[ -s "$ACT" ] || fail "expected synchronized activation dump missing: $ACT"
expected_bytes=$((ROWS * 4096 * 4))
actual_bytes=missing
if [ -s "$ACT" ]; then
    actual_bytes=$(stat -f %z "$ACT") || fail "could not stat activation dump"
    [ "$actual_bytes" = "$expected_bytes" ] ||
        fail "activation dump has $actual_bytes bytes, expected $expected_bytes ($ROWS rows)"
fi
grep -q "dumped glm_ffn_norm layer $LAYER pos $POS" "$O/control.err" ||
    fail "control log lacks the synchronized layer=$LAYER pos=$POS dump receipt"
grep -q "GLM prefill:" "$O/control.err" || fail "control log lacks the completed prefill receipt"
grep -q "GLM router split-K B4 ENGAGED" "$O/control.err" &&
    fail "B4 unexpectedly engaged in the control capture"
grep -q "GLM SUPER-CHUNK prefill ENGAGED" "$O/control.err" &&
    fail "expert-bank super-chunk unexpectedly engaged in the control capture"

# Do not start the comparison with a failed or wrong-shape fixture.
if [ "$failures" != 0 ]; then
    echo "failures=$failures" >"$O/RESULT"
    echo "artifacts: $O"
    exit 1
fi

DS4_GLM_ENABLE_ROUTER_SPLITK_B4=1 \
DS4_GLM_DISABLE_ROUTER_SPLITK_B4=0 \
"$W/speed-bench/metal_router_b4_fixture" \
    --gguf "$M" --capture-prefix "$CAP" --layer "$LAYER" --pos "$POS" \
    --expect-rows "$ROWS" --ref-rows 64 --out "$O/compare" \
    >"$O/compare.out" 2>"$O/compare.err"
rc=$?
echo "rc=$rc" >"$O/compare.rc"
[ "$rc" = 0 ] || fail "A/B4 fixture comparison exited $rc"
[ -s "$O/compare/summary.txt" ] || fail "comparison summary is missing"
grep -q "GLM router split-K B4 ENGAGED" "$O/compare.err" ||
    fail "comparison log lacks the B4 engagement receipt"
for f in "$O/compare/selection-A.bin" "$O/compare/selection-B4.bin"; do
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
    echo "capture_layer=$LAYER"
    echo "capture_pos=$POS"
    echo "capture_rows=$ROWS"
    echo "prompt_slice_bytes=$(stat -f %z "$O/p16k.txt")"
    echo "prompt_slice_tokens_bound_prior=16224"
    echo "ordinary_chunk_plan_bound_prior=4096,8192,3936"
} >"$O/IDENTITY" || fail "could not write identity receipt"
shasum -a 256 "$P62" "$O/p16k.txt" "$ACT" "$W/ds4" \
    "$W/speed-bench/metal_router_b4_fixture" >"$O/SHA256SUMS" ||
    fail "could not hash the prompt, activation, or executables"
grep -E "dumped glm_ffn_norm|GLM prefill:" "$O/control.err" || true
[ -s "$O/compare/summary.txt" ] && cat "$O/compare/summary.txt"
echo "failures=$failures" >"$O/RESULT"
echo "artifacts: $O"
[ "$failures" = 0 ]
