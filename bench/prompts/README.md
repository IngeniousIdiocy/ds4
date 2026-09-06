# Benchmark prompts

The two prompt files in this directory are the exact bytes used for every admitted
native throughput record of the GLM-5.3 M3 Ultra branch (`docs/GLM53_M3ULTRA.md`,
"Results"; the receipt directories are named there). They are stored verbatim and must
never be reformatted: a different byte is a different workload, and no result measured
on one file may be attached to another. `bench/prefill-gate.sh` reads `needle-64k.txt`
from here as its 62k bucket; the other buckets it names are not tracked.

## Identity

| file | sha256 | bytes | input tokens (measured) | native recipe |
|---|---|---|---|---|
| `needle-64k.txt` | `e15ce96e009e4684d7006be75e6c24d2f8bfabf0a1ebacf9bc06a850034a6b9b` | 262,815 | **62,174** | `-c 70000 -n 2048` |
| `prompt-300k.txt` | `7a66f69f497d09e7c5c8dbe4295b8956b46e7874d03b7209c48b555bfc7efc9c` | 1,268,635 | **300,000** | `-c 320000 -n 2048` |

**Tokenizer and template.** The token counts are what `ds4` itself reports for the
file passed as `--prompt-file` with `--nothink`. Neither file begins with a rendered
chat marker, so `ds4_cli.c` (`build_prompt`) treats the whole file as the user message
and encodes it with `ds4_encode_chat_prompt` — the GLM-5.3 chat template from the
model file's tokenizer metadata, thinking disabled — and the count includes the
template's own tokens. The tokenizer is the one carried by the public artifact
(`gguf/GLM-5.3-Flash-Q4_K.gguf`, sha256 `828f413c…`, produced as described in
`docs/GLM53_M3ULTRA.md`). The receipts bind the count as `prompt_len` on the
`GLM gen counters` line of each run's `*.err`:

- 62k: `BASE62-v3-20260906T133801Z` — identity line `prompt=…/needle-64k.txt
  sha256=e15ce96e… ctx=70000 npred=2048 depth=62k`; counters
  `n_generated=2048 n_decode_eval=2047 prompt_len=62174 final_pos=64222 ctx=70000
  n_predict=2048 … stop=predict_limit`.
- 300k: `BASE300-v3-20260906T140109Z` (and `BASE300-v3xr8-20260906T151438Z`, same
  prompt) — identity line `prompt=…/prompt-300k.txt sha256=7a66f69f… ctx=320000
  npred=2048 depth=300k`; counters `n_generated=2048 n_decode_eval=2047
  prompt_len=300000 final_pos=302048 ctx=320000 n_predict=2048 … stop=predict_limit`.

**Final native reference recipe** (the configuration used by the final receipts):

```sh
export DS4_GLM_GEN_COUNTERS=1 DS4_GLM_IGNORE_EOS=1
./ds4 -m gguf/GLM-5.3-Flash-Q4_K.gguf --metal --nothink --temp 0 \
      -c 70000  -n 2048 --prompt-file bench/prompts/needle-64k.txt
./ds4 -m gguf/GLM-5.3-Flash-Q4_K.gguf --metal --nothink --temp 0 \
      -c 320000 -n 2048 --prompt-file bench/prompts/prompt-300k.txt
```

**Timer convention.** A block is valid only when `n_generated=2048` and
`stop=predict_limit`. The loop evaluates 2,047 tokens after the first sampled token
(`n_decode_eval=2047`); the generated rate quoted for a block is `2048 / decode_s`
(the `generation:` figure on the `GLM prefill: … generation: …` line), `ms_per_eval`
is `1000 * decode_s / 2047`, and prefill is timed and reported separately and never included
in the decode figure. With no `--dflash` weights, bare startup resolves serial mode and allocates no drafter.
`DS4_GLM_IGNORE_EOS=1` keeps decoding to `-n` past EOS so both
arms of a comparison get the same window; it changes the generated text, which is why
the output bytes of every block are kept with the receipt.

## Provenance

Both files are synthetic and carry nothing that cannot be published. Their upstream
fixture and deterministic generator are covered by this repository's MIT license; the
reflow, planted fictional code, repetition and byte cut add no external source text.

`needle-64k.txt` is derived from upstream ds4's own long-context regression fixture
`tests/long_context_story_prompt.txt`, which `tests/generate_long_context_story_prompt.py`
writes deterministically (seed 20260513) from five scene templates about the fictional
harbor town of Bellwether; that fixture is in the `9ab7053` tree this branch is based
on. The derivation: the fixture's text (including its DeepSeek-style marker strings,
which here are plain text because the file does not start with one) was reflowed to
one sentence per line, each line tagged `(note NNNNN)`, the whole fixture repeated
twice (the second copy truncated at note 02249), preceded by the line "You will answer
a question at the end about this archive." and followed by a blank line and one
question. One sentence was planted at line 1129 as the needle — a fictional
maintenance code, `VELVET-OSPREY-6401`, for an "orbital greenhouse" — and the closing
question asks for it. The prompt is used for throughput; nothing about the branch's
claims depends on the answer. The reflow script itself was not retained; the file is
therefore shipped as the bytes that were measured, not regenerated.

`prompt-300k.txt` is a byte cut of `needle-64k.txt`: six copies joined by
`\n\n[section N]\n\n` (N = 1…6), truncated by a token-count bisection to exactly
300,000 prompt tokens, which fell at byte 1,268,635. It is reproduced exactly by

```sh
for i in 1 2 3 4 5 6; do cat bench/prompts/needle-64k.txt; printf '\n\n[section %d]\n\n' $i; done \
  | head -c 1268635 | shasum -a 256      # 7a66f69f…
```

Because it is a cut of repeated copies, it ends mid-story, while the needle and question recur earlier inside it; that makes it unfit as a task or recall
item, and it is used only as a 300,000-token throughput workload.
