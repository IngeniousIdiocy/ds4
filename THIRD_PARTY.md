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

The imported source is distributed under MIT. Audrey Tang (`@audreyt`) submitted the
implementation from `audreyt/ds4`'s `ornith15` branch in
[antirez/ds4 PR #844](https://github.com/antirez/ds4/pull/844); the DFlash2
implementation begins at commit
[`b02d5ecb243fff092a00ed8981b1ad16d525183d`](https://github.com/audreyt/ds4/commit/b02d5ecb243fff092a00ed8981b1ad16d525183d).
The upstream base named by the PR (`84cc882`) and the inspected contributor head
(`bde007a5757e2701761ce30c869fa7f82771857e`) contain the same MIT `LICENSE` blob
(`5973a4c99a8b7c0f2e0a58fbcd7f93b230da6b78`). GitHub's
[contribution terms](https://docs.github.com/en/site-policy/github-terms/github-terms-of-service#6-contributions-under-repository-license)
also apply the repository license to contributions unless a separate agreement says
otherwise. No contrary license or separate agreement was found in the inspected
sources.

Source and method credit are distinct: the C/Metal implementation ported here comes
from Audrey Tang's PR; the DFlash algorithm and the `DFlashDraftModel`,
`DFlash2DraftModel` and `CandidateSelector` reference contracts come from
[Z Lab's MIT-licensed DFlash project](https://github.com/z-lab/dflash/tree/07ebd93db9f472af339b644bb70221ad8428328a).
Source distributions retain this repository's MIT `LICENSE` and the
[Z Lab MIT notice](licenses/Z-LAB-DFLASH-MIT.txt).

**Drafter weights are not redistributed.** The pinned
[`incoai/GLM-5.3-Flash-DFlash2@dc77ff1c99eeb2df044ee3d4f0094eb033fee410`](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2/blob/dc77ff1c99eeb2df044ee3d4f0094eb033fee410/README.md)
model card declares **CC BY-NC-ND 4.0** and describes the checkpoint as released for
research and evaluation, with separate contact information for commercial licensing.
`gguf-tools/dflash2_to_gguf.py` supports a local, non-commercial copy. This repository
distributes neither the original checkpoint nor the converted output; keep the converted
GGUF local and do not share it. Nothing from the checkpoint is tracked here: no weights,
derivative tensors or golden weight data. Historical DFlash2 results in
`bench/README.md` depended on a local copy.
`docs/DFLASH_GLM53.md` documents how to obtain and convert the drafter yourself, pinned
to revision `dc77ff1c99eeb2df044ee3d4f0094eb033fee410` (`model.safetensors` sha256
`b33c03475ba7322cf398828f2d8d1be376df30dc05c6b40c28c8ea8da23e410b`), with the model-card
and licence links; on this branch `--dflash` is an opt-in mode whose status is recorded
in `docs/GLM53_M3ULTRA.md`, "Feature status".

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
