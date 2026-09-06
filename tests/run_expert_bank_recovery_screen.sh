#!/bin/bash
# One-window expert-bank discriminator.  The caller owns the GPU lock; this
# script only verifies that token and runs either the cheap expansion pair or
# one instrumented 16k native prefill.  MODE=packed16 keeps the super-chunk
# schedule but deliberately skips bank expansion and uses packed operands.
# MODE=async16 enables deferred completion for resident-HC pass-1 slices.
set -o pipefail

W=${W:-/Users/mark/src/ds4-wt-astral-expertbank}
M=${M:-/Users/mark/src/ds4-glm/gguf/GLM-5.3-Flash-Q4_K-9ab7053.gguf}
P62=${P62:-/Users/mark/megakernel-refs/prompt-backup/needle-64k.txt}
IDS=${IDS:-/Users/mark/megakernel-refs/public-artifact/PREFILL-Q8-20260906T152843Z/expertbank/moe-ids-62k-public.bin}
G=${G:-/Users/mark/megakernel-refs/gpulock.sh}
MODE=${MODE:-native16}
O=${O:-/Users/mark/megakernel-refs/public-artifact/EXPERTBANK-RECOVERY-$(date -u +%Y%m%dT%H%M%SZ)}

if [ -z "${GPU_LOCK_TOKEN:-}" ]; then
    echo "GPU_LOCK_TOKEN is required; the coordinator must acquire the GPU lock" >&2
    exit 2
fi
"$G" preflight "$GPU_LOCK_TOKEN" || exit 2
mkdir -p "$O"

case "$MODE" in
expansion)
    "$W/tests/test_glm_expert_bank" \
        --gguf "$M" --ids "$IDS" --layer 24 \
        --off-gate 91109299648 --off-up 92468254144 --off-down 93827208640 \
        --expansion-only >"$O/expansion.out" 2>"$O/expansion.err"
    rc=$?
    echo "rc=$rc" >"$O/expansion.rc"
    grep -E "model pages|expansion |bank allocation reuse|expansion-only" \
        "$O/expansion.out"
    ;;
native16|packed16|async16)
    head -c 69254 "$P62" >"$O/p16k.txt"
    packed_env=
    if [ "$MODE" = packed16 ]; then
        packed_env=DS4_GLM_EXPERT_BANK_DIAGNOSTIC_PACKED=1
    fi
    async_env=
    if [ "$MODE" = async16 ]; then
        async_env=DS4_GLM_EXPERT_BANK_ASYNC_SLICES=1
    fi
    env DS4_METAL_MODEL_UNTRACKED=1 \
        DS4_GLM53_MEMORY_CEILING_GB=280 \
        DS4_GLM_ENABLE_EXPERT_BANK=1 \
        DS4_GLM_EXPERT_BANK_TIMING=1 \
        DS4_GLM_GEN_COUNTERS=1 \
        DS4_GLM_IGNORE_EOS=1 \
        $packed_env \
        $async_env \
        "$W/ds4" -m "$M" --metal --nothink --temp 0 -c 70000 -n 16 \
        --prompt-file "$O/p16k.txt" >"$O/native16.out" 2>"$O/native16.err"
    rc=$?
    echo "rc=$rc" >"$O/native16.rc"
    grep -E "GLM SUPER-CHUNK (prefill ENGAGED|expert-bank expansion timing|wall timing|layer completion timing|pass23 timing|pass1 sliced timing|done)|GLM prefill:" \
        "$O/native16.err"
    ;;
*)
    echo "MODE must be expansion, native16, packed16, or async16" >&2
    exit 2
    ;;
esac

echo "artifacts: $O"
exit "$rc"
