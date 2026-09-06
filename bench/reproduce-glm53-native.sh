#!/usr/bin/env bash
# Reproduce one fixed-horizon native GLM-5.3 CLI gate on the validated M3 Ultra profile.
# Run only with exclusive GPU access. This script deliberately does not hash the 185 GB model.
set -euo pipefail

: "${MODEL:?set MODEL to the validated 185299232064-byte target GGUF}"

BIN=${BIN:-./ds4}
CASE=${CASE:-62k}
OUT=${OUT:-native-${CASE}-$(date -u +%Y%m%dT%H%M%SZ)}

case "$CASE" in
  62k)  CTX=70000;  PROMPT=${PROMPT:-bench/prompts/needle-64k.txt}; EXPECTED_PROMPT_SHA=e15ce96e009e4684d7006be75e6c24d2f8bfabf0a1ebacf9bc06a850034a6b9b ;;
  300k) CTX=320000; PROMPT=${PROMPT:-bench/prompts/prompt-300k.txt}; EXPECTED_PROMPT_SHA=7a66f69f497d09e7c5c8dbe4295b8956b46e7874d03b7209c48b555bfc7efc9c ;;
  *) echo "CASE must be 62k or 300k" >&2; exit 2 ;;
esac

[[ -x "$BIN" ]] || { echo "missing executable: $BIN" >&2; exit 2; }
[[ -f "$MODEL" && -f "$PROMPT" ]] || { echo "missing MODEL or PROMPT" >&2; exit 2; }
[[ $(stat -f %z "$MODEL") == 185299232064 ]] || { echo "unexpected model size" >&2; exit 2; }
PROMPT_SHA=$(shasum -a 256 "$PROMPT" | awk '{print $1}')
[[ "$PROMPT_SHA" == "$EXPECTED_PROMPT_SHA" ]] || { echo "unexpected prompt digest" >&2; exit 2; }

mkdir -p "$OUT"

# Remove inherited tuning and diagnostic variables. The two variables added below are
# fixed-horizon instrumentation; the guarded M3 Ultra profile resolves by itself.
while IFS='=' read -r name _; do
  case "$name" in DS4_*|MTL_*) unset "$name" ;; esac
done < <(env)

{
  printf 'case=%s\n' "$CASE"
  printf 'model_bytes=%s\n' "$(stat -f %z "$MODEL")"
  printf 'model_known_sha256=%s\n' 828f413cae8ceee74796606814295e00e04c1c7b151aca9a3c5c75db32cb73e0
  printf 'prompt_sha256=%s\n' "$PROMPT_SHA"
  shasum -a 256 "$BIN"
} >"$OUT/identity.txt"

set +e
DS4_GLM_GEN_COUNTERS=1 DS4_GLM_IGNORE_EOS=1 \
  "$BIN" -m "$MODEL" --metal --nothink --temp 0 \
  -c "$CTX" -n 2048 --prompt-file "$PROMPT" >"$OUT/output.txt" 2>"$OUT/run.err"
RC=$?
set -e
printf '%s\n' "$RC" >"$OUT/exit-code.txt"
exit "$RC"
