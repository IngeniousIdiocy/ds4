#!/bin/zsh
# Interleaved A/B prefill speed gate (the prefill campaign's standard gate).
#   A_BIN / B_BIN   binaries (default: both ./ds4, i.e. the tree this script is run from)
#   M               model GGUF (default gguf/GLM-5.3-Flash-Q4_K.gguf, upstream's download_model.sh glm53-q4 name)
#   A_ENV / B_ENV   extra "VAR=val VAR2=val" applied to that arm (kill switches)
#   REPS            interleaved repetitions (default 4)
#   BUCKETS         comma list of prompts: 62k,30k,512,128,111 (default 62k,30k,111)
#   OUTDIR          results dir (default bench/results/gate-<ts>)
#   PROMPT_DIR      directory holding the bucket prompts needle-64k.txt, prefill-30k.txt,
#                   prefill-512.txt, prefill-128.txt, prefill-111.txt (default bench/prompts; none are tracked)
#   GATE_LOCK_DIR   optional GPU-lock directory; when set the script refuses to run unless it exists
#   GATE_LOCK_OWNER substring that must appear in $GATE_LOCK_DIR/owner (refuse otherwise)
# Runs base, variant, base, variant … per bucket; reports prefill t/s, seconds to
# first token (tokens ÷ prefill t/s), and byte identity of the 16-token greedy
# output between arms. The caller is responsible for exclusive GPU use (no ds4-server
# or other model process during the run).
set -u
# ds4 compiles its Metal library from metal/*.metal RELATIVE TO THE CWD, so a
# worktree binary run from another tree silently compiles that tree's kernels
# (lever 1 found its arm B falling back to stock pipelines this way).  Each arm
# therefore runs from the directory that contains its own binary; relative
# A_BIN/B_BIN resolve against the caller's CWD.
A_BIN=${A_BIN:-./ds4}; B_BIN=${B_BIN:-./ds4}
A_BIN=${A_BIN:A}; B_BIN=${B_BIN:A}
export DS4_METAL_MODEL_UNTRACKED=1 DS4_GLM53_MEMORY_CEILING_GB=280
M=${M:-gguf/GLM-5.3-Flash-Q4_K.gguf}; M=${M:A}
[[ -f $M ]] || { echo "refusing: model $M not found (set M=<path to GGUF>)"; exit 2; }
A_ENV=${A_ENV:-}; B_ENV=${B_ENV:-}
REPS=${REPS:-4}; BUCKETS=${BUCKETS:-62k,30k,111}
OUTDIR=${OUTDIR:-bench/results/gate-$(date +%Y%m%d-%H%M%S)}; mkdir -p $OUTDIR
[[ -e $OUTDIR/raw.tsv ]] && { echo "refusing: $OUTDIR/raw.tsv exists (rows would append to an old run)"; exit 2; }
GATE_LOCK_DIR=${GATE_LOCK_DIR:-}
if [[ -n $GATE_LOCK_DIR ]]; then
  [[ -d $GATE_LOCK_DIR ]] || { echo "refusing: $GATE_LOCK_DIR not held"; exit 2; }
  [[ -n "${GATE_LOCK_OWNER:-}" ]] && ! grep -q "$GATE_LOCK_OWNER" $GATE_LOCK_DIR/owner 2>/dev/null && { echo "refusing: lock held by $(cat $GATE_LOCK_DIR/owner), not $GATE_LOCK_OWNER"; exit 2; }
fi
pgrep -x ds4-server >/dev/null && { echo "refusing: ds4-server is running"; exit 2; }
typeset -A PFILE PCTX PTOK
PROMPT_DIR=${PROMPT_DIR:-bench/prompts}
PFILE[62k]=$PROMPT_DIR/needle-64k.txt;  PCTX[62k]=70000
PFILE[30k]=$PROMPT_DIR/prefill-30k.txt; PCTX[30k]=40000
PFILE[512]=$PROMPT_DIR/prefill-512.txt; PCTX[512]=8192
PFILE[128]=$PROMPT_DIR/prefill-128.txt; PCTX[128]=8192
PFILE[111]=$PROMPT_DIR/prefill-111.txt; PCTX[111]=8192
ntok() { (cd ${A_BIN:h} && ./${A_BIN:t} -m $M --dump-tokens --nothink --prompt-file $1 2>/dev/null | head -1 | python3 -c "import sys,json;print(len(json.loads(sys.stdin.readline())))"); }
run() { # arm rep bucket
  local arm=$1 rep=$2 b=$3 bin envs
  if [[ $arm == A ]]; then bin=$A_BIN; envs=$A_ENV; else bin=$B_BIN; envs=$B_ENV; fi
  local tag=$b-$arm-$rep
  (cd ${bin:h} && env ${=envs} ./${bin:t} -m $M --metal --nothink -n 16 --temp 0 -c ${PCTX[$b]} --prompt-file ${PFILE[$b]} \
      > $OUTDIR/$tag.out 2> $OUTDIR/$tag.err)
  grep -qE "function not found|pipeline failed|was not found in the library" $OUTDIR/$tag.err && echo "  WARNING $tag: a Metal function or pipeline was not found (wrong tree?)"
  local pf=$(grep -oE "prefill: [0-9.]+" $OUTDIR/$tag.err | awk '{print $2}')
  grep -q "generation:" $OUTDIR/$tag.err || pf=FAIL
  echo "$b $arm $rep $pf" >> $OUTDIR/raw.tsv
  echo "  $tag prefill=$pf t/s"
}
echo "gate: A=[$A_BIN $A_ENV] B=[$B_BIN $B_ENV] reps=$REPS buckets=$BUCKETS -> $OUTDIR"
for b in ${(s:,:)BUCKETS}; do
  [[ -f ${PFILE[$b]} ]] || { echo "missing prompt ${PFILE[$b]} (a plain-text prompt of that nominal token count; see bench/README.md)"; continue; }
  PTOK[$b]=$(ntok ${PFILE[$b]}); echo "bucket $b: ${PTOK[$b]} tokens"
  for r in $(seq 1 $REPS); do run A $r $b; run B $r $b; done
  cmp -s $OUTDIR/$b-A-1.out $OUTDIR/$b-B-1.out && bytes=identical || bytes=DIFFERENT
  for r in $(seq 2 $REPS); do (( r >= 2 && r <= REPS )) || continue; cmp -s $OUTDIR/$b-A-1.out $OUTDIR/$b-A-$r.out || bytes="$bytes(A-nondeterministic)"; cmp -s $OUTDIR/$b-B-1.out $OUTDIR/$b-B-$r.out || bytes="$bytes(B-nondeterministic)"; done
  python3 - $OUTDIR/raw.tsv $b ${PTOK[$b]} "$bytes" <<'PY'
import sys,statistics as st
raw,b,tok,bytes_=sys.argv[1],sys.argv[2],int(sys.argv[3]),sys.argv[4]
A=[];B=[]
for l in open(raw):
    bb,arm,rep,pf=l.split()
    if bb!=b or pf=='FAIL': continue
    (A if arm=='A' else B).append(float(pf))
def s(x): return (st.mean(x), (max(x)-min(x)) if len(x)>1 else 0.0)
ma,sa=s(A); mb,sb=s(B)
pairs=[y-x for x,y in zip(A,B)]
print(f"RESULT {b}: A {ma:.2f} t/s (spread {sa:.2f}) -> B {mb:.2f} (spread {sb:.2f}); delta {mb-ma:+.2f} t/s ({100*(mb-ma)/ma:+.1f}%), pairs positive {sum(p>0 for p in pairs)}/{len(pairs)}; TTFT A {tok/ma:.1f} s -> B {tok/mb:.1f} s; bytes {bytes_}")
PY
done
echo "raw: $OUTDIR/raw.tsv"
