# Third-party code, weights and dependencies on the GLM-5.3 M3 Ultra branch

The repository licence (`LICENSE`, MIT) is unchanged. `third_party/iris/LICENSE`,
`cuda/mmq/VENDOR.md` and `licenses/` are upstream's and unchanged. This file records
what the branch adds on top of them.

## DFlash / DFlash2 draft engine (`ds4_dflash2.inc`, `ds4_dflash_glm.inc`, `ds4_dflash_seed.inc`, `ds4_dflash_selector.inc`, `ds4_dflash_golden.inc`, `metal/dflash2.metal`, `gguf-tools/dflash2_to_gguf.py`)

A GLM-5.3 port of audreyt's DFlash/DFlash2 draft engine from the `ornith15` branch
submitted to upstream as **antirez/ds4 pull request #844** ("Ornith 1.5 + Qwen 3.8 27B
hybrid runtime + DFlash speculative decoding on Metal", author audreyt, open at the time
of writing). The port credits are in the header comment of `ds4_dflash2.inc`. Changes
made here: a self-contained scratch pool, BF16 drafter weights, GGUF-parameterized rope
base and RMS epsilon, and a persistent sliding-window context-KV cache for DFlash2. The
candidate selector (`ds4_dflash_selector.inc`) implements the z-lab `CandidateSelector`
contract; the draft-model contracts follow z-lab's `DFlashDraftModel` /
`DFlash2DraftModel`.

The pull request carries no licence statement of its own (checked 2026-09-06: PR
still open, author @audreyt, no licence or origin terms in the description); it was
submitted for inclusion in antirez/ds4, which is MIT-licensed, and this branch treats the
ported code under that licence on the inbound=outbound basis of a contribution to an
MIT repository. If the contribution's licence status is settled differently upstream,
this section is to be updated before publication.

**Drafter weights are not redistributed.** The DFlash2 drafter for GLM-5.3-Flash
(`incoai/GLM-5.3-Flash-DFlash2` on Hugging Face) is published under **CC BY-NC-ND 4.0**
(`license: cc-by-nc-nd-4.0` in its model card, checked via the Hub API 2026-09-06). `gguf-tools/dflash2_to_gguf.py` converts it
for local research use only; the converted GGUF must not be redistributed, and nothing
of it is tracked in this repository. Historical DFlash2 results in `bench/README.md`
depended on that file. On this branch DFlash2 is loaded but refused at run time (see
`docs/GLM53_M3ULTRA.md`, "Feature status").

## KDA kernels (`metal/glm53_kda.metal`)

Kimi Delta Attention kernels, adapted from upstream's `kimi-k3` branch (same repository,
same licence).

## Model weights

The validated weights are a conversion of the official zai-org/GLM-5.3-Flash checkpoint
with upstream's own `gguf-tools/glm53_quantize.py`; the model's licence is the model
card's, and the weights are not part of this repository.

## Test fixtures

- `bench/fidelity/manifest-1k/` contains, and `bench/fidelity/manifest-long/generate.py`
  regenerates, passages from Project Gutenberg editions of *The Complete Works of William
  Shakespeare* and *I promessi sposi* (public domain; the Project Gutenberg header, which
  carries its own redistribution terms, is retained where a case begins at the top of the
  file), from upstream's long-context test prompts and, for the long manifest only, from
  this repository's own documentation.
- `gguf-tools/quality-testing/data/*` are upstream's reference outputs (see upstream's
  `gguf-tools/quality-testing/README.md` for their provenance).

## Tooling dependencies added by the branch

- **jinja2** (with `jinja2.ext.loopcontrols`), used only by
  `tests/glm_tool_result_reorder_ref.py` to render the GLM chat template as the reference
  for the server's tool-result reorder test. Not needed to build or run ds4; no install
  is attempted.
- **numpy**, used by `gguf-tools/dflash2_to_gguf.py` and `gguf-tools/glm53_requant_kda.py`
  (already a dependency of upstream's gguf tooling).
