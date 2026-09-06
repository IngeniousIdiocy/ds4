#!/bin/sh
# Root/GPU-owner only. Four short runs: serial IDs, suffix A, repeat A,
# suffix B. The accepted prefix stays fixed; the entire rejected suffix
# changes, including its first row. The block straddles two four-token pools.
# Post-restore serial steps cross the next pool boundary, exposing a bad tail.
# MODEL, DFLASH required; DS4, OUT, PROMPT, NGEN, CTX optional.
# This is causal-integrity instrumentation, never a throughput benchmark.
set -eu
unset DS4_DFLASH_SCRIPT_SERIAL DS4_DFLASH_FORCE_REPLAY DS4_DFLASH_NO_WIDE_ROLLBACK
DS4="${DS4:-./ds4}"
MODEL="${MODEL:?set MODEL}"
DFLASH="${DFLASH:?set DFLASH}"
OUT="${OUT:-dflash-prefix-$(date -u +%Y%m%dT%H%M%SZ)}"
PROMPT="${PROMPT:-Explain how a bicycle stays upright, in three sentences.}"
NGEN="${NGEN:-40}"
CTX="${CTX:-8192}"
HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
mkdir -p "$OUT"
OUT=$(CDPATH= cd -- "$OUT" && pwd)
# The shader loader reads metal/ relative to the working tree; run this
# script from the same worktree as the supplied binary.
set -- --model "$MODEL" --dflash "$DFLASH" --ctx "$CTX" -n "$NGEN" --temp 0 -p "$PROMPT"
{
    git rev-parse HEAD
    git diff --stat
    shasum -a 256 "$DS4" ds4.c ds4_metal.m ds4_dflash_glm.inc ds4_glm53_prefix.h ds4_dflash_rollback.h metal/dsv4_misc.metal
    printf 'model=%s\ndflash=%s\nctx=%s\nngen=%s\n' "$MODEL" "$DFLASH" "$CTX" "$NGEN"
} >"$OUT/identity.txt"
echo 'serial ID reference'
env DS4_DFLASH_SCRIPT=0:0:1 DS4_DFLASH_SCRIPT_SERIAL=1 \
    DS4_DFLASH_SCRIPT_IDS_OUT="$OUT/serial.ids" \
    "$DS4" "$@" >"$OUT/serial.txt" 2>"$OUT/serial.err"
test -s "$OUT/serial.ids"
BASE=$(head -1 "$OUT/serial.ids" | cut -d' ' -f1)
LAST=$(tail -1 "$OUT/serial.ids" | cut -d' ' -f1)
ALIGN=$((BASE + 4))
ALIGN=$((ALIGN + (4 - ALIGN % 4) % 4))
POS=$((ALIGN + 1))
test "$LAST" -ge "$((POS + 11))" || { echo 'continuation too short' >&2; exit 2; }
{ echo "$BASE"; cut -d' ' -f2 "$OUT/serial.ids"; } >"$OUT/ids.txt"
printf '%s:4:7\n' "$POS" >"$OUT/blocks.txt"
for ARM in a repeat b; do
    mkdir "$OUT/$ARM"
    SALT=0
    if [ "$ARM" = b ]; then SALT=1729; fi
    echo "prefix arm $ARM, suffix salt $SALT"
    env DS4_DFLASH_STATS=1 DS4_DFLASH_SCRIPT="$POS:4:7" \
        DS4_DFLASH_SCRIPT_IDS="$OUT/ids.txt" DS4_DFLASH_SCRIPT_SUFFIX="$SALT" \
        DS4_DFLASH_SCRIPT_IDS_OUT="$OUT/$ARM.ids" \
        DS4_DFLASH_SCRIPT_DUMP="$OUT/$ARM" DS4_DFLASH_SCRIPT_DUMP_CACHE=1 \
        "$DS4" "$@" >"$OUT/$ARM.txt" 2>"$OUT/$ARM.err"
done
python3 "$HERE/dflash_prefix_compare.py" "$OUT" "$POS"
echo "artifacts: $OUT"
