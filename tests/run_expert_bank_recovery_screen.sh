#!/bin/bash
# One-window expert-bank discriminator.  The caller owns the GPU lock; this
# script only verifies that token and runs either the cheap expansion pair or
# one instrumented 16k native prefill.  MODE=packed16 keeps the super-chunk
# schedule but deliberately skips bank expansion and uses packed operands.
# MODE=async16 enables deferred completion for resident-HC pass-1 slices.
# MODE=fused16 also holds the final slice and appends expansion + pass23.
# MODE=stage0-16 runs only the retained-routing L24 routed-segment cross screen.
# MODE=ledgeroff16/ledgerbank16/ledgerfused16 collect mode-1 GPU spans.
set -o pipefail

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
W=${W:-$ROOT}
M=${M:-}
P62=${P62:-}
IDS=${IDS:-}
G=${G:-}
MODE=${MODE:-native16}
O=${O:-${TMPDIR:-/tmp}/ds4-expert-bank-$(date -u +%Y%m%dT%H%M%SZ)}

[ -n "$M" ] || { echo "M must name the target GGUF" >&2; exit 2; }
[ -n "$G" ] || { echo "G must name the GPU-lock helper" >&2; exit 2; }

if [ -z "${GPU_LOCK_TOKEN:-}" ]; then
    echo "GPU_LOCK_TOKEN is required; the coordinator must acquire the GPU lock" >&2
    exit 2
fi
"$G" preflight "$GPU_LOCK_TOKEN" || exit 2
mkdir -p "$O"

case "$MODE" in
expansion)
    [ -n "$IDS" ] || { echo "IDS must name the routed-id fixture" >&2; exit 2; }
    "$W/tests/test_glm_expert_bank" \
        --gguf "$M" --ids "$IDS" --layer 24 \
        --off-gate 91109299648 --off-up 92468254144 --off-down 93827208640 \
        --expansion-only >"$O/expansion.out" 2>"$O/expansion.err"
    rc=$?
    echo "rc=$rc" >"$O/expansion.rc"
    grep -E "model pages|expansion |bank allocation reuse|expansion-only" \
        "$O/expansion.out"
    ;;
stage0-16)
    [ -n "$IDS" ] || { echo "IDS must name the routed-id fixture" >&2; exit 2; }
    "$W/tests/test_glm_expert_bank" \
        --gguf "$M" --ids "$IDS" --layer 24 --tokens 16224 \
        --off-gate 91109299648 --off-up 92468254144 --off-down 93827208640 \
        --passes 5 --cross-screen --no-compare >"$O/stage0-16.out" \
        2>"$O/stage0-16.err"
    rc=$?
    echo "rc=$rc" >"$O/stage0-16.rc"
    grep -E "routing:|expansion |allocation reuse|cross-schedule|packed production|bank whole|routed saving|net," \
        "$O/stage0-16.out"
    ;;
native16|packed16|async16|fused16)
    [ -n "$P62" ] || { echo "P62 must name the long prompt" >&2; exit 2; }
    head -c 69254 "$P62" >"$O/p16k.txt"
    packed_env=
    if [ "$MODE" = packed16 ]; then
        packed_env=DS4_GLM_EXPERT_BANK_DIAGNOSTIC_PACKED=1
    fi
    async_env=
    if [ "$MODE" = async16 ]; then
        async_env=DS4_GLM_EXPERT_BANK_ASYNC_SLICES=1
    fi
    fused_env=
    if [ "$MODE" = fused16 ]; then
        fused_env=DS4_GLM_EXPERT_BANK_FUSED_LAYER_CB=1
    fi
    env DS4_METAL_MODEL_UNTRACKED=1 \
        DS4_GLM53_MEMORY_CEILING_GB=280 \
        DS4_GLM_ENABLE_EXPERT_BANK=1 \
        DS4_GLM_EXPERT_BANK_TIMING=1 \
        DS4_GLM_GEN_COUNTERS=1 \
        DS4_GLM_IGNORE_EOS=1 \
        $packed_env \
        $async_env \
        $fused_env \
        "$W/ds4" -m "$M" --metal --nothink --temp 0 -c 70000 -n 16 \
        --prompt-file "$O/p16k.txt" >"$O/native16.out" 2>"$O/native16.err"
    rc=$?
    echo "rc=$rc" >"$O/native16.rc"
    grep -E "GLM SUPER-CHUNK (prefill ENGAGED|expert-bank expansion timing|wall timing|layer completion timing|pass23 timing|pass1 sliced timing|done)|GLM prefill:" \
        "$O/native16.err"
    ;;
ledgeroff16|ledgerbank16|ledgerfused16)
    [ -n "$P62" ] || { echo "P62 must name the long prompt" >&2; exit 2; }
    head -c 69254 "$P62" >"$O/p16k.txt"
    bank_env=DS4_GLM_ENABLE_EXPERT_BANK=1
    if [ "$MODE" = ledgeroff16 ]; then
        bank_env=DS4_GLM_DISABLE_EXPERT_BANK=1
    fi
    fused_env=
    if [ "$MODE" = ledgerfused16 ]; then
        fused_env=DS4_GLM_EXPERT_BANK_FUSED_LAYER_CB=1
    fi
    env DS4_METAL_MODEL_UNTRACKED=1 \
        DS4_GLM53_MEMORY_CEILING_GB=280 \
        DS4_GLM_GEN_COUNTERS=1 \
        DS4_GLM_IGNORE_EOS=1 \
        DS4_KERNEL_LEDGER=1 \
        DS4_KERNEL_LEDGER_DUMP="$O/ledger.tsv" \
        $bank_env \
        $fused_env \
        "$W/ds4" -m "$M" --metal --nothink --temp 0 -c 70000 -n 16 \
        --prompt-file "$O/p16k.txt" >"$O/native16.out" 2>"$O/native16.err"
    rc=$?
    echo "rc=$rc" >"$O/native16.rc"
    grep -E "GLM SUPER-CHUNK (prefill ENGAGED|done)|GLM prefill:" \
        "$O/native16.err"
    grep -E '^#LEDGER|expert_bank|mul_mm_id' "$O/ledger.tsv"
    ;;
*)
    echo "MODE must be expansion, stage0-16, native16, packed16, async16, fused16, ledgeroff16, ledgerbank16, or ledgerfused16" >&2
    exit 2
    ;;
esac

echo "artifacts: $O"
exit "$rc"
