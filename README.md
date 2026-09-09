<p align="center">
  <img src="logo.svg" alt="DwarfStar logo" width="220">
</p>

**DwarfStar** is a small native inference engine optimized first for
**DeepSeek V4 Flash** (including the experimental vision model).
It also supports **GLM 5.2 and 5.3**, **GLM 5.3 Flash**, and
**DeepSeek V4 PRO**. It is self-contained and
deliberately narrow, not a general GGUF runner. Model loading, prompt rendering,
tool calls, KV state, the HTTP server, and the coding agent are built and tested together.
The repository also includes tools and data for GGUF, imatrix, quality, and speed.

Supported backends:

* **Metal**, the primary target, on Macs with 96 GB or more. Smaller machines
  can use SSD streaming.
* **NVIDIA CUDA**, including multi-GPU systems and DGX Spark.
* **ROCm** on Strix Halo systems such as the Framework Desktop.

This project would not exist without **llama.cpp and GGML**, make sure to read
the acknowledgements section, a big thank you to Georgi Gerganov and all the
other contributors.

Model support is intentionally opportunistic. The project follows the best open
weights for useful local machine sizes, especially 128 GB laptops and 512 GB
workstations. A model may be removed when a better replacement arrives.

The project has first class support for SSD streaming of weights, so it is
possible to run models bigger than RAM while often still getting decent
performances, and even running very large models (like the full GLM 5.3 or DeepSeek v4 PRO)
on systems with just 128GB of RAM at a slower speed, but fast enough for
QA-style chats.

# So, what can I do with this software?

* You can run a very capable models in your consumer hardware, a MacBook, a DGX Spark, or a Strix Halo for example. Even if you have not enough RAM, with SSD streaming, you can run it at a decent speed.
* You can use multiple CUDA cards as a multi-user LLM server. Ada Lovelace, including L40S, is supported: newer models can run here even when their other inference implementations require newer GPUs. Our eight-L40S Flash setup has reached about 126 t/s aggregate generation with 16 sessions.
* Using two 128 GB Macs connected with RDMA, you can run 4-bit DeepSeek Flash or GLM 5.3 Flash with tensor parallelism. Larger GLM 5.2 quants need larger machines, such as Mac Studios.
* You can also use pipeline paralellism to glue together multiple systems to sum their RAM and run larger models.

## Motivations

* Capable open-weight models now fit on high-end personal machines.
* DeepSeek V4 Flash and PRO, GLM 5.2, tolerate aggressive routed-expert quantization.
* Compressed KV caches and fast local SSDs make long contexts practical.
* The idea of an inference system specialized for a few models.

# AI full disclosure

* This software is developed with **strong assistance from GPT 5.5, 5.6, Claude Fable** and with humans leading the ideas, testing, and debugging. We say this openly because it shaped how the project was built. If you are not happy with AI-developed code, this software is not for you. The acknowledgement below is equally important: this would not exist without `llama.cpp` and GGML, largely written by hand.

## Acknowledgements to llama.cpp and GGML

`ds4.c` does not link against GGML, but it **exists thanks to the path opened by the
llama.cpp project and the kernels, quantization formats, GGUF ecosystem, and hard-won
engineering knowledge developed there**.
We are thankful and indebted to [`llama.cpp`](https://github.com/ggml-org/llama.cpp)
and its contributors. Their implementation, kernels, tests, and design choices were
an essential reference while building this DeepSeek V4 specific inference path.
Some source-level pieces are retained or adapted here under the MIT license: GGUF
quant layouts and tables, CPU quant/dot logic, and certain kernels. For this
reason, and because we are genuinely grateful, we keep the GGML authors copyright
notice in our `LICENSE` file.

## Status

The software is currently very fast changing. Consider it beta quality.
Before each release, a big QA run is executed, however instabilities
are definitely possible.

# How to use this project?

I (Salvatore) believe that the way projects should be shipped and used changed because of AI. The main differences today are:

1. With AI, users can modify the software in significant ways with low efforts, costs, and even lacking deep domain knowledge about the task they want to accomplish. For instance, a DwarfStar user with a specific hardware setup can ask a coding agent to improve the inference speed of this software for the specific hardware setup, asking the model to reach the maximum prefill and generation speed without impacting correctness, and also asking to do a deep QA pass.
2. Similiarly, because of "1", software may be shipped in a different way than before. It must be more a working template for the biggest use cases, without trying to cover every possible setup. If DwarfStar showcases a few good implementations of tensor parallel execution, the code will work as a rail for implementing the same feature in specific conditions, for a new model, and so forth.

So, while this project attempts to be usable for the featured models and the most common hardware setups, I ask you, if you have access to coding agents, to consider using coding agents as an interface to discover the project, make modifications, create personalized setups. This way you can likely do more than what we ship, and certain things that are not documented or implemented, and that you require, are potentially very easy to achieve.

## Start Here

```sh
git clone https://github.com/antirez/ds4.git
cd ds4
```

Choose your build. The platform guides cover prerequisites, memory sizing,
and hardware-specific setups:

| Platform guide | Build |
| --- | --- |
| [Metal on Apple Silicon](docs/METAL.md) | `make` |
| [DGX Spark](docs/DGX_SPARK.md) | `make cuda-spark` |
| [Strix Halo / Framework Desktop](docs/STRIX_HALO.md) | `make strix-halo` |
| [One or more CUDA cards, including Ada/L40S](docs/CUDA_MULTI_GPU.md) | `make cuda-generic` |

For a first run on a 96 or 128 GB machine, download DeepSeek V4 Flash Q2:

```sh
./download_model.sh ds4f-q2
```

Downloads go in `gguf/`. Repeat the command to resume an interrupted download.
Leave memory for the context and runtime buffers as well as the model.
See [other models](docs/MODELS.md) or use [SSD streaming](docs/SSD_STREAMING.md)
on a smaller Mac.

## Everyday Use

Once built and with a model downloaded:

```sh
./ds4
./ds4 -p "Explain Redis streams in one paragraph."
./ds4-agent
./ds4-server --ctx 32768
```

The default model is `ds4flash.gguf`, a link updated by main-model downloads.
Pass `-m FILE` to choose explicitly. Commands normally run from the repository
root; use `--chdir /path/to/ds4` when launching elsewhere.

The server listens at `http://127.0.0.1:8000` by default; see [serving](docs/SERVER.md)
for API access and multiple sessions.

The interactive CLI keeps a multi-turn conversation. Use `/help`, `/read FILE`,
`/ctx N`, and `/quit`. Ctrl+C interrupts generation and returns to the prompt.
Run each binary with `--help` for its full options.

### Native coding agent

`ds4-agent` runs inference directly, without a separate HTTP server. It keeps
the token history and live model state together, shows prefill progress, and
uses the model's native tool format. DeepSeek and GLM have their own templates.

Sessions are stored in `~/.ds4/kvcache`:

| Command | Action |
| --- | --- |
| `/save` | Save the current session |
| `/list` | List saved sessions |
| `/switch <sha>` | Resume a session |
| `/del <sha>` | Delete a saved session |
| `/strip <sha>` | Keep text and title, removing the large KV payload |

Compatible local KV snapshots avoid rebuilding the prompt. Stripped sessions
and network TP restores require prefill. Sessions containing images cannot yet
be saved. Saved conversations and traces may contain private information.

For Pi, OpenCode, Codex CLI, or Claude Code, use `ds4-server` instead and follow
the [client setup guide](docs/CLIENTS.md).

### Models, images, and speculation

[Models and vision](docs/MODELS.md) lists the supported downloads and memory
requirements. DeepSeek Vision Experimental uses a different checkpoint from
Flash 0731; GLM 5.3 Flash adds vision to the same text model.

With the matching encoder passed as `--vision FILE`, use `/read image.png`
in the CLI or `view_image` in the native agent.

Speculative decoding is opt-in. GLM uses `--mtp`; Flash DSpark needs a matching
support GGUF. GLM-5.3 also supports an optional, separately obtained DFlash2
drafter. A bare startup is serial; supplying `--dflash FILE` selects the
conservative windowed-confidence scheduler unless `--dflash-mode` says otherwise.
Both DFlash public modes run the trained anchor-plus-seven proposal and require
target verification. Conservative admits a confident prefix of at least four
positions, verifies the full block, judges attempts in windows of three against
measured serial cost and backs off when a window loses; reasoning decodes serially.
Speculative continuously offers all seven draft positions whenever the causal
context and response room permit; it has no confidence, economics, retry,
backoff, or meter policy. DFlash is greedy-only. Speculative can be much slower
on an unfavorable request, and either verified mode can take a numerically
different valid continuation from the ordinary serial path. Read
[speculative decoding](docs/SPECULATIVE_DECODING.md) for MTP/DSpark and the
[GLM-5.3 DFlash2 guide](docs/DFLASH_GLM53.md) for the drafter recipe and startup
policies.

### Output and power

Thinking is enabled by default. Use `--nothink` or `/nothink` for direct
answers, and `--think` or `/think` to enable it again.
The normal sampling defaults are temperature 1, top-p 1, and min-p 0.05;
`--temp 0` selects greedy output.

For DeepSeek, `--power N` trades throughput for lower sustained GPU load.
The default is 100. GLM currently requires `--power 100`.

DeepSeek Flash and GLM 5.3 Flash also support directional steering. Load a
vector with `--dir-steering-file FILE`; `/steer F` adjusts its scale for
subsequent tokens in a local CLI or agent session, without rebuilding the
existing KV cache. See [steering documentation](dir-steering/README.md).

`--prefix-file FILE` preloads complete `USER:` / `ASSISTANT:` pairs before
the live conversation. A turn marker must start a line, roles must alternate,
and the last turn must be `ASSISTANT:`.

## Capability Evaluation

`ds4-eval` runs embedded capability regression tests against a real GGUF.
These are DwarfStar integration checks, not official leaderboard scores.

```sh
./ds4-eval -m ds4flash.gguf --trace /tmp/ds4-eval.txt
./ds4-eval -m ds4flash.gguf --suite hard-smoke
./ds4-eval -m ds4flash.gguf --suite hard --retry-incomplete
```

The default suite is `core`; `--suite all` runs core and hard cases.
`--list-cases` lists tests without loading a model. `--plain` selects
non-interactive output, and `--regrade-trace FILE` scores an existing trace
without generating again. Sources and licenses are in [EVAL_DATA.md](EVAL_DATA.md).
For inference correctness and release checks, read [testing](docs/TESTING.md).

## Speed

This recorded DeepSeek V4 Flash Q2 sweep uses an M5 Max with 128 GB RAM,
2048-token continued-prefill intervals, and 128 greedy generation tokens per
frontier. It is a baseline, not a fresh benchmark of every commit.

![M5 Max Flash Q2 throughput](speed-bench/m5_max_ts.svg)

See [performance and benchmarking](docs/PERFORMANCE.md) for the full numbers,
DGX Spark results, comparison conditions, and benchmark commands.

## Detailed Guides

- [Models and vision](docs/MODELS.md): Flash, PRO, GLM, and matching encoders.
- [SSD streaming](docs/SSD_STREAMING.md): run larger than RAM and size the cache.
- [Inference across machines](docs/DISTRIBUTED.md): two-Mac TP/RDMA and layer pipelines.
- [Speculative decoding](docs/SPECULATIVE_DECODING.md): DSpark, GLM MTP, and sampling.
- [Serving](docs/SERVER.md): APIs, images, batching, and disk KV caches.
- [Coding agent clients](docs/CLIENTS.md): Pi, OpenCode, Codex CLI, and Claude Code.
- [Performance](docs/PERFORMANCE.md): reproducible measurements and recorded baselines.
- [Testing and development](docs/TESTING.md): regression tests, debugging, and model-building tools.

Read [CONTRIBUTING.md](CONTRIBUTING.md) before sending a pull request.

<a id="glm53_m3ultra"></a>
## GLM-5.3-Flash on the M3 Ultra (this branch)

This branch carries a Metal-side decode/prefill kernel set, server changes, an
optional DFlash2 speculative decoder and a fidelity harness for GLM-5.3-Flash
(Q4_K, 185 GB) on a 512 GB M3 Ultra Mac Studio. It is based on upstream `9ab7053`
and has not been rebased onto later upstream commits. Start with
[the GLM-5.3 M3 Ultra guide](docs/GLM53_M3ULTRA.md) — tested configuration, the
measured scope of `DS4_GLM_EXACT=1`, feature status, switch reference and known
issues — and read [CHANGES-GLM53.md](CHANGES-GLM53.md) for what changed relative
to upstream and why.

Measured on the final build (`524c8a1`), bare defaults, one cold run each, the GPU
sampled idle before launch. The upstream column is unmodified `9ab7053` (plus a
disclosed 48-line counters patch) on the same machine, file, prompts and 2,048-token
horizon; see
[upstream-baseline.json](bench/receipts/glm53-m3ultra/upstream-baseline.json):

| measurement | upstream `9ab7053` | this branch |
|---|---:|---:|
| prefill, 62,174-token prompt | 365.7 t/s | **550.4 t/s** |
| serial decode after that prompt, 2,048 tokens | 23.75 t/s | **38.07 t/s** |
| prefill, 300,000-token prompt | 316.8 t/s | **473.9 t/s** |
| serial decode after that prompt, 2,048 tokens | 21.64 t/s | **37.39 t/s** |
| serial decode, short prompt, 512 tokens | 29.2 t/s | **40.9 t/s** |
| DFlash2 conservative, SQL / JSON 512-token fixtures | — | **62.2 / 49.4 t/s**, output byte-identical to serial |

The 62k and 300k outputs are byte-identical to the retained reference blocks and to
the previous public candidate. DFlash2 is optional and needs a locally converted
drafter; no drafter weights are redistributed. The
[DFlash2 guide](docs/DFLASH_GLM53.md) is the complete recipe: the pinned
`incoai/GLM-5.3-Flash-DFlash2` revision and checksums (section 2), the
`gguf-tools/dflash2_to_gguf.py` conversion to the BF16 GGUF (section 3), and the
`llama-quantize` steps for the Q8_0 drafter the numbers above were measured with
(section 9). On a real coding-agent workload the default conservative profile
measured +4.3% output throughput over serial, while unfavorable prose fixtures
lose 2-3%. These are single cold runs on one machine, not averages
over workloads or run-to-run variance. See the
[release evidence index](bench/RELEASE-EVIDENCE.md) for receipts, scope and the
remaining open items. `./ds4 --help glm53` lists the supported controls and kill
switches.

**Accuracy of the changes.** Every kernel change is classified in
[bench/FIDELITY.md](bench/FIDELITY.md): Tier 1 changes keep the same arithmetic in
the same order and are proven bit-identical in a randomized harness (at least 50,000
poisoned draws, every output word compared, 1,000-run determinism); Tier 2 changes
alter a summation order, ship behind their own kill switch, and are registered so
`DS4_GLM_EXACT=1` turns all of them off at once. The
[exact-mode diagnostic](docs/GLM53_M3ULTRA.md#exact-mode-diagnostic) and
[bench/EXACT-MODE-PLAN.md](bench/EXACT-MODE-PLAN.md) record the measured
differences against upstream rather than assuming identity: on three long-context
cases exact-mode total NLL differed from upstream by 0.000238, 0.004171 and 0.000048.
The quality receipts are in the evidence index: the focused task screen scored
[77/77 in both arms](bench/receipts/glm53-m3ultra/taskcheck-quality.json) with 22 of
23 outputs byte-identical to upstream, while the
[token-weighted NLL screen](bench/receipts/glm53-m3ultra/e6-quality.json) landed
0.0019 above upstream against a provisional 0.0005 criterion, which is disclosed as
unmet, not waived. DFlash2's causal rollback is covered by
[tests/DFLASH-PREFIX.md](tests/DFLASH-PREFIX.md), and on every fixture in this
release the DFlash outputs were byte-identical to serial.

## Logo

The DwarfStar logo was designed by hand by Salvatore Sanfilippo, made more
graphical with AI, and manually reworked by Ben Gnomino, whose human touch made
it rock.
