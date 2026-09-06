#!/bin/sh
# Deterministic DFlash restore+replay oracle coverage. For direct prefix
# snapshots and suffix contamination, use dflash_prefix_harness.sh.
#
# Three CLI passes over the same short prompt. The prefix is re-prefilled by
# each pass -- this is not payload reuse, it is just cheap at this size -- and
# what IS reused is the recorded continuation: pass 1 writes down the token ids
# ordinary decoding produces, and passes 2 and 3 propose from them.
#
#   1. control    every position decodes serially, through the same binary and
#                 the same code path the speculative arm falls back to. Records
#                 the continuation's token ids.
#   2. control+   the same, with the real block window, so its dumps of the
#                 frontier logits and the complete speculative state cover the
#                 frontiers the scripted arm will reach. Must reproduce pass 1.
#   3. scripted   speculative blocks at chosen positions, drafting the recorded
#                 continuation for K rows and a known-wrong token at row K. The
#                 verifier decides; the script only proposes.
#
# Coverage: rejection after 0, 1, 2 and 3 accepted drafts, at start positions
# covering all four residues of the 4-token pool boundary, the shortest block
# that can reject, a block whose rejection lands on the boundary row, and --
# LAST, so no partial comparison happens downstream of a batched-verify state
# -- a full-accept block.
#
# The verdict is tests/dflash_rejection_compare.py, which has its own planted
# CPU fixtures (make dflash-compare-fixtures). It refuses by default: missing
# files, incomplete tensor coverage, read failures, malformed or duplicated
# records, non-finite values, differing token ids, and any partial-frontier
# state or logits that are not exactly equal.
#
# Usage: DS4=./ds4 MODEL=target.gguf DFLASH=draft.gguf \
#        [PROMPT="..."] [OUT=dir] [NGEN=80] sh tests/dflash_rejection_harness.sh
#
# Runs the model: hold the GPU lock.

set -eu

DS4="${DS4:-./ds4}"
MODEL="${MODEL:?set MODEL to the target gguf}"
DFLASH="${DFLASH:?set DFLASH to the drafter gguf}"
PROMPT="${PROMPT:-Explain how a bicycle stays upright, in three sentences.}"
NGEN="${NGEN:-80}"
CTX="${CTX:-8192}"
OUT="${OUT:-dflash-reject-$(date -u +%Y%m%dT%H%M%SZ)}"
HERE=$(dirname "$0")

mkdir -p "$OUT/control" "$OUT/scripted"
common="--model $MODEL --dflash $DFLASH --ctx $CTX -n $NGEN --temp 0"

# The plan, in offsets from the first block. Fixed here so the required
# continuation length is known before the model is loaded.
#   offset  K  N     what it covers
PLAN='0:0:7 9:1:7 18:2:7 27:3:7 36:0:1 41:3:4 50:7:7'
SPAN=58           # offsets used, plus room for the last block's drafts

REQUIRED=$((SPAN + 8))
if [ "$NGEN" -lt "$REQUIRED" ]; then
    echo "NGEN=$NGEN is shorter than the $REQUIRED tokens this plan needs" >&2
    echo "raise NGEN (80 is the default and is sufficient)" >&2
    exit 2
fi

# ---------------------------------------------------------------- control
echo "== pass 1: control (serial), recording the continuation"
DS4_DFLASH_STATS=1 \
DS4_DFLASH_SCRIPT="0:0:1" \
DS4_DFLASH_SCRIPT_SERIAL=1 \
DS4_DFLASH_SCRIPT_IDS_OUT="$OUT/control.ids" \
  $DS4 $common -p "$PROMPT" >"$OUT/control.txt" 2>"$OUT/control.err" || {
    echo "control arm failed; see $OUT/control.err" >&2; exit 1; }

if [ ! -s "$OUT/control.ids" ]; then
    echo "no continuation recorded: is this build the dflash branch?" >&2
    exit 1
fi

BASE=$(head -1 "$OUT/control.ids" | cut -d' ' -f1)
LAST=$(tail -1 "$OUT/control.ids" | cut -d' ' -f1)
NREC=$(wc -l <"$OUT/control.ids" | tr -d ' ')
{ echo "$BASE"; cut -d' ' -f2 "$OUT/control.ids"; } >"$OUT/ids.txt"
echo "   continuation: $NREC tokens, positions $BASE..$LAST"

# ------------------------------------------------------------ block list
FIRST=$((BASE + 4))
ALIGN=$((FIRST + (4 - FIRST % 4) % 4))     # first position with pos mod 4 == 0
BLOCKS=""
for e in $PLAN; do
    off=${e%%:*}; rest=${e#*:}; k=${rest%%:*}; n=${rest##*:}
    BLOCKS="${BLOCKS:+$BLOCKS,}$((ALIGN + off)):$k:$n"
done
NEEDED=$((ALIGN + SPAN))
echo "   blocks: $BLOCKS"
echo "   plan needs positions up to $NEEDED; continuation reaches $LAST"
if [ "$NEEDED" -gt "$LAST" ]; then
    echo "continuation too short for the plan: raise NGEN" >&2
    exit 2
fi

# --------------------------------------------------------- control dumps
echo "== pass 2: control dumps over the block window"
DS4_DFLASH_SCRIPT="$BLOCKS" \
DS4_DFLASH_SCRIPT_SERIAL=1 \
DS4_DFLASH_SCRIPT_IDS="$OUT/ids.txt" \
DS4_DFLASH_SCRIPT_IDS_OUT="$OUT/control2.ids" \
DS4_DFLASH_SCRIPT_DUMP="$OUT/control" \
  $DS4 $common -p "$PROMPT" >"$OUT/control2.txt" 2>"$OUT/control2.err" || {
    echo "control dump arm failed; see $OUT/control2.err" >&2; exit 1; }
cmp -s "$OUT/control.ids" "$OUT/control2.ids" ||
    { echo "the control arm is not reproducible between passes" >&2; exit 1; }

# --------------------------------------------------------------- scripted
echo "== pass 3: scripted (deterministic drafts)"
DS4_DFLASH_STATS=1 \
DS4_DFLASH_FORCE_REPLAY=1 \
DS4_DFLASH_SCRIPT="$BLOCKS" \
DS4_DFLASH_SCRIPT_IDS="$OUT/ids.txt" \
DS4_DFLASH_SCRIPT_IDS_OUT="$OUT/scripted.ids" \
DS4_DFLASH_SCRIPT_DUMP="$OUT/scripted" \
  $DS4 $common -p "$PROMPT" >"$OUT/scripted.txt" 2>"$OUT/scripted.err" || {
    echo "scripted arm failed; see $OUT/scripted.err" >&2; exit 1; }

# ---------------------------------------------------------------- verdict
echo "== compare"
echo "$BLOCKS" >"$OUT/blocks.txt"
python3 "$HERE/dflash_rejection_compare.py" "$OUT" "$BLOCKS" --all-replay
rc=$?
echo "artifacts: $OUT"
exit "$rc"
