#!/bin/zsh
# Prefill Tier 2 gate, criterion 2: score control and candidate on the 1k manifest (and the 100-prompt
# pin set), then compare. Run from the repo/worktree root with exclusive GPU use (no ds4-server).
#   CAND_ENV="VAR=1 ..."   env that turns the candidate feature on (control = same binary, env unset)
#   M                      model GGUF (default gguf/GLM-5.3-Flash-Q4_K.gguf)
#   GATE_LOCK_DIR          optional GPU-lock directory; when set, refuse unless it exists
#   OUTDIR                 results dir (default bench/fidelity/gate-<ts>)
#   SKIP_100=1             skip the 100-prompt runs (control TSV for the pin check then comes from CTRL_100)
set -u
[[ -z "${GATE_LOCK_DIR:-}" || -d $GATE_LOCK_DIR ]] || { echo "refusing: lock $GATE_LOCK_DIR not held"; exit 2; }
pgrep -x ds4-server >/dev/null && { echo "refusing: ds4-server running"; exit 2; }
export DS4_METAL_MODEL_UNTRACKED=1 DS4_GLM53_MEMORY_CEILING_GB=400
M=${M:-gguf/GLM-5.3-Flash-Q4_K.gguf}
[[ -f $M ]] || { echo "refusing: model $M not found (set M=<path to GGUF>)"; exit 2; }
S=./gguf-tools/quality-testing/score_official
[[ -x $S ]] || { echo "build the scorer first: make -C gguf-tools quality-score"; exit 2; }
OUTDIR=${OUTDIR:-bench/fidelity/gate-$(date +%Y%m%d-%H%M%S)}; mkdir -p $OUTDIR
M1K=bench/fidelity/manifest-1k/manifest.tsv
M100=gguf-tools/quality-testing/data/glm53-flash-openrouter-zai-fp8-100/manifest.tsv
run() { local tag=$1 envs=$2 man=$3; echo "== $tag $(date +%T)"; env ${=envs} $S $M $man $OUTDIR/$tag.tsv > $OUTDIR/$tag.log 2>&1; echo "   exit=$? $(tail -1 $OUTDIR/$tag.log | cut -c1-120)"; grep -qE "function not found|pipeline failed" $OUTDIR/$tag.log && echo "   WARNING: missing kernel in $tag"; }
run control-1k "" $M1K
run cand-1k "${CAND_ENV:?set CAND_ENV}" $M1K
if [[ "${SKIP_100:-0}" != 1 ]]; then run control-100 "" $M100; run cand-100 "$CAND_ENV" $M100; fi
if [[ "${SKIP_100:-0}" != 1 ]]; then python3 bench/compare_1k.py --manifest $M1K $OUTDIR/control-1k.tsv $OUTDIR/cand-1k.tsv $OUTDIR/control-100.tsv $OUTDIR/cand-100.tsv; else python3 bench/compare_1k.py --manifest $M1K $OUTDIR/control-1k.tsv $OUTDIR/cand-1k.tsv; fi | tee $OUTDIR/verdict.txt
