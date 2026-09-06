#include "ds4_help.h"

#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

typedef struct {
    const char *off;
    const char *cyan;
    const char *title;
    const char *yellow;
    const char *grey;
    const char *red;
    const char *white;
    const char *bright;
} help_colors;

static help_colors help_make_colors(FILE *fp) {
    bool color = isatty(fileno(fp));
    help_colors c = {0};
    if (!color) return c;
    c.off = "\x1b[0m";
    c.cyan = "\x1b[38;5;81m";
    c.title = "\x1b[1;38;5;250m";
    c.yellow = "\x1b[38;5;179m";
    c.grey = "\x1b[38;5;240m";
    c.red = "\x1b[38;5;203m";
    c.white = "\x1b[38;5;252m";
    c.bright = "\x1b[1;38;5;231m";
    return c;
}

static void title(FILE *fp, const help_colors *c, const char *s) {
    fprintf(fp, "%s%s%s\n", c->title ? c->title : "", s, c->off ? c->off : "");
}

static void title_red(FILE *fp, const help_colors *c, const char *s) {
    fprintf(fp, "%s%s%s\n", c->red ? c->red : "", s, c->off ? c->off : "");
}

static bool option_name_has_switch(const char *name) {
    bool word_start = true;
    while (*name) {
        if (word_start && (*name == '-' || *name == '/')) return true;
        word_start = (*name == ' ');
        name++;
    }
    return false;
}

static void print_colored_option_name(FILE *fp, const help_colors *c, const char *name) {
    bool has_switch = option_name_has_switch(name);
    bool word_start = true;
    while (*name) {
        const char *start = name;
        while (*name && *name != ' ') name++;
        bool is_option = !has_switch || *start == '-' || *start == '/' ||
                         (word_start && has_switch && *start != '[');
        const char *color = is_option ? c->cyan : c->bright;
        if (color) fputs(color, fp);
        fwrite(start, 1, (size_t)(name - start), fp);
        if (color && c->off) fputs(c->off, fp);
        if (*name == ' ') {
            fputc(*name++, fp);
            word_start = false;
        }
    }
}

static void opt(FILE *fp, const help_colors *c, const char *name, const char *desc) {
    if (c->cyan) {
        fputs("  ", fp);
        print_colored_option_name(fp, c, name);
        fprintf(fp, " %s|%s ", c->grey ? c->grey : "", c->grey ? c->off : "");
        fprintf(fp, "%s%s%s\n", c->white ? c->white : "", desc,
                c->white ? c->off : "");
        return;
    }

    const int col = 30;
    int n = (int)strlen(name);
    if (n > col) {
        fprintf(fp, "  %s\n      %s\n", name, desc);
    } else {
        fprintf(fp, "  %-30s %s\n", name, desc);
    }
}

static void para(FILE *fp, const help_colors *c, const char *s) {
    fprintf(fp, "%s%s%s\n",
            c->yellow ? c->yellow : "", s, c->yellow ? c->off : "");
}

static bool streq(const char *a, const char *b) {
    return a && b && strcmp(a, b) == 0;
}

static bool topic_is(const char *topic, const char *name) {
    return topic && strcmp(topic, name) == 0;
}

static const char *tool_name(ds4_help_tool tool) {
    switch (tool) {
    case DS4_HELP_DS4: return "ds4";
    case DS4_HELP_SERVER: return "ds4-server";
    case DS4_HELP_AGENT: return "ds4-agent";
    case DS4_HELP_BENCH: return "ds4-bench";
    case DS4_HELP_EVAL: return "ds4-eval";
    }
    return "ds4";
}

static const char *tool_usage(ds4_help_tool tool) {
    switch (tool) {
    case DS4_HELP_DS4:
        return "Usage: ds4 [(-p PROMPT | --prompt-file FILE)] [options]";
    case DS4_HELP_SERVER:
        return "Usage: ds4-server [options]";
    case DS4_HELP_AGENT:
        return "Usage: ds4-agent [options]";
    case DS4_HELP_BENCH:
        return "Usage: ds4-bench (--prompt-file FILE | --chat-prompt-file FILE) [options]";
    case DS4_HELP_EVAL:
        return "Usage: ds4-eval [options]";
    }
    return "Usage: ds4 [options]";
}

static const char *tool_summary(ds4_help_tool tool) {
    switch (tool) {
    case DS4_HELP_DS4:
        return "Chat with a local DwarfStar model, run one-shot prompts, inspect models, or coordinate distributed inference.";
    case DS4_HELP_SERVER:
        return "Serve one loaded DwarfStar model through OpenAI, Responses, Anthropic, and completion-compatible HTTP APIs.";
    case DS4_HELP_AGENT:
        return "Run the native terminal coding agent with live tools, session save/restore, and a responsive prompt while the model works.";
    case DS4_HELP_BENCH:
        return "Measure prefill, decode, context growth, and KV-cache size across repeatable context frontiers.";
    case DS4_HELP_EVAL:
        return "Run the built-in reasoning, math, science, and security evaluation harness with a live terminal UI.";
    }
    return "";
}

static void print_model_runtime(FILE *fp, const help_colors *c,
                                ds4_help_tool tool, bool full) {
    title(fp, c, "Model And Runtime");
    opt(fp, c, "-m, --model FILE", "GGUF model path. Default: ds4flash.gguf");
    if (tool == DS4_HELP_DS4 || tool == DS4_HELP_AGENT || tool == DS4_HELP_SERVER) {
        opt(fp, c, "--vision FILE", "Vision encoder GGUF for the selected model.");
    }
#ifdef DS4_ROCM_BUILD
    opt(fp, c, "--metal | --rocm | --cpu", "Select the backend explicitly.");
    opt(fp, c, "--backend NAME", "Backend name: metal, rocm, or cpu.");
#else
    opt(fp, c, "--metal | --cuda | --cpu", "Select the backend explicitly.");
    opt(fp, c, "--backend NAME", "Backend name: metal, cuda, or cpu.");
    opt(fp, c, "--gpu-vram N[,N,...]|auto", "CUDA VRAM budgets per device, in GiB, or auto-detect free VRAM.");
    opt(fp, c, "--gpu-devices N[,N,...]", "CUDA device indices used by multi-GPU placement.");
    if (tool != DS4_HELP_EVAL) {
        opt(fp, c, "--cuda-tensor-parallel", "Enable the paired DeepSeek tensor/expert path on an even multi-GPU CUDA placement.");
    }
#endif
    if (tool != DS4_HELP_BENCH) {
        opt(fp, c, "-c, --ctx N", "Allocated context tokens.");
    }
    if (tool == DS4_HELP_SERVER) {
        opt(fp, c, "-n, --tokens N", "Default max output tokens when clients omit a limit.");
    }
    opt(fp, c, "-t, --threads N", "CPU helper threads for host-side/reference work.");
    opt(fp, c, "--power N", "GPU duty-cycle target, 1..100. Default: 100");
    opt(fp, c, "--ssd-streaming", "Metal/CUDA/ROCm: opt in to SSD-backed model streaming instead of full residency.");
    opt(fp, c, "--ssd-streaming-cold", "SSD streaming: skip default popularity-based expert-cache preload.");
    opt(fp, c, "--ssd-streaming-cache-experts N|NGB", "SSD streaming cache target. N requests dynamic expert slots; NGB also reserves two full prefill layers. Either may be reduced to fit the model, graph, context, and backend working set.");
    opt(fp, c, "--ssd-streaming-full-layers N", "GLM Metal streaming: keep the first N routed layers fully resident. Default: auto from NGB expert budget; use 0 to disable.");
    opt(fp, c, "--ssd-streaming-preload-experts N", "SSD streaming: upfront popularity preload count. DeepSeek auto-seeds by default; GLM demand-fills unless N is explicit.");
    opt(fp, c, "--simulate-used-memory NGB", "Diagnostic: lock N GiB before model load to simulate a smaller-memory machine.");
    opt(fp, c, "--prefill-chunk N", "Graph prefill chunk size. Default: CUDA TP 2048; PRO long prompts 8192; others 4096.");
    if (full) {
        if (tool == DS4_HELP_EVAL || tool == DS4_HELP_BENCH) {
            opt(fp, c, "--mtp-model FILE", "External MTP or DSpark support GGUF.");
        }
        if (tool == DS4_HELP_DS4 || tool == DS4_HELP_AGENT || tool == DS4_HELP_SERVER) {
            opt(fp, c, "--mtp", "Enable model-embedded MTP speculation.");
            opt(fp, c, "--mtp-model FILE", "External MTP or DSpark support GGUF.");
            opt(fp, c, "--mtp-draft N", "Maximum autoregressive MTP draft tokens. Default: 1");
            opt(fp, c, "--mtp-margin F", "Verifier confidence margin for fast MTP acceptance. Default: 3");
            opt(fp, c, "--mtp-timing", "Enable embedded MTP and print acceptance/timing counters.");
            opt(fp, c, "--dspark", "Enable DSpark using the support GGUF passed with --mtp-model.");
            opt(fp, c, "--dspark-confidence F", "Enable DSpark with confidence pruning threshold 0..1. Greedy/opportunistic default: Metal 0.6, CUDA/ROCm 0.7; exact sampling: 0.8");
            opt(fp, c, "--mtp-exact-sampling", "Preserve the ordinary temperature distribution instead of accepting target-matching greedy drafts directly.");
            opt(fp, c, "--dspark-strict", "Load DSpark support but keep target-only decode.");
            if (tool == DS4_HELP_DS4 || tool == DS4_HELP_SERVER) {
                opt(fp, c, "--dflash FILE", "GLM-5.3: load a DFlash2 draft GGUF and speculate (optional; drafter weights are not bundled). See docs/DFLASH_GLM53.md.");
                opt(fp, c, "--dflash-mode MODE", "DFlash scheduling: conservative (default with --dflash), speculative (uncapped), or serial (do not load the drafter).");
            }
        } else if (tool == DS4_HELP_BENCH) {
            opt(fp, c, "--dspark", "Benchmark greedy DSpark using the support GGUF passed with --mtp-model.");
            opt(fp, c, "--dspark-confidence F", "DSpark confidence pruning threshold 0..1.");
        }
        opt(fp, c, "--quality", "Prefer exact kernels where faster approximate paths exist.");
        opt(fp, c, "--warm-weights", "Touch mapped tensor pages at startup to reduce first-use stalls.");
        if (tool == DS4_HELP_DS4 || tool == DS4_HELP_BENCH) {
            opt(fp, c, "--expert-profile FILE", "Metal-only: write routed expert locality/cache simulation JSON.");
        }
    }
    fputc('\n', fp);
}

static void print_sampling(FILE *fp, const help_colors *c, bool full) {
    title(fp, c, "Prompt And Sampling");
    opt(fp, c, "-n, --tokens N", "Maximum generated tokens.");
    opt(fp, c, "--temp F", "Sampling temperature. 0 is greedy/deterministic.");
    opt(fp, c, "--top-p F", "Nucleus sampling probability.");
    opt(fp, c, "--min-p F", "Keep tokens scoring at least F times the top token.");
    opt(fp, c, "--seed N", "Sampling seed for reproducible non-greedy runs.");
    para(fp, c, "GLM CLI and agent runs default to temperature 1.0, top-p 0.95, and min-p 0 unless those options are set explicitly.");
    opt(fp, c, "--think", "Use normal thinking mode.");
    opt(fp, c, "--think-max", "Use Think Max when context is large enough.");
    opt(fp, c, "--nothink", "Disable thinking and ask for direct replies.");
    if (full) {
        opt(fp, c, "-sys, --system TEXT", "System prompt. Empty string disables the default where supported.");
        opt(fp, c, "-p, --prompt TEXT", "One-shot prompt text.");
        opt(fp, c, "--prompt-file FILE", "Read one-shot prompt text from FILE.");
        opt(fp, c, "--raw-prompt", "Tokenize the one-shot prompt without chat markers.");
    }
    fputc('\n', fp);
}

static void print_steering(FILE *fp, const help_colors *c) {
    title(fp, c, "Directional Steering");
    opt(fp, c, "--dir-steering-file FILE", "Load one f32 direction vector per layer.");
    opt(fp, c, "--dir-steering-ffn F", "Apply steering after FFN outputs. Default with file: 1");
    opt(fp, c, "--dir-steering-attn F", "Apply steering after attention outputs. Default: 0");
    fputc('\n', fp);
}

static void print_distributed(FILE *fp, const help_colors *c) {
    title(fp, c, "Distributed Inference");
    fputc('\n', fp);
    para(fp, c, "Distributed mode runs one logical session across several machines by assigning contiguous model layer ranges to workers. Workers own their layer slice and KV-cache shard; the coordinator owns the prompt, sampling loop, and client/API flow. Start workers first, then start the coordinator. The coordinator waits for a complete route and streams hidden states through the workers.");
    fputc('\n', fp);
    opt(fp, c, "--role ROLE", "Distributed role: coordinator or worker.");
    opt(fp, c, "--layers A:B", "Inclusive layer slice, e.g. 0:20 or 21:output.");
    opt(fp, c, "--listen HOST PORT", "Coordinator listen address; workers may use it for their data listener.");
    opt(fp, c, "--coordinator HOST PORT", "Coordinator address for --role worker.");
    opt(fp, c, "--dist-prefill-chunk N", "Coordinator prefill pipeline chunk size. Default: session cap.");
    opt(fp, c, "--dist-prefill-window N", "Max prefill chunks in flight. Default: workers+2, capped at 8.");
    opt(fp, c, "--dist-activation-bits N", "Hidden-state transport width: 32, 16, or 8. Default: 32");
    opt(fp, c, "--dist-replay-check", "Diagnostic: reset and replay prompt, then compare logits.");
    opt(fp, c, "--debug", "Print coordinator route/debug logs.");
    fputc('\n', fp);
    title(fp, c, "Tensor Parallelism");
    fputc('\n', fp);
    para(fp, c, "Tensor parallelism uses the same coordinator/worker addresses as distributed mode, but always runs one 50/50 worker. Add --tensor-parallel, omit --layers, start the worker, then start the coordinator.");
    fputc('\n', fp);
    opt(fp, c, "--tensor-parallel", "Switch --role/--listen/--coordinator to two-machine tensor parallelism.");
    opt(fp, c, "--transport auto|rdma|tcp", "Tensor gate transport. Default: auto");
    opt(fp, c, "--rdma-device NAME", "Select a verbs device when auto-detection is ambiguous.");
    opt(fp, c, "--rdma-gid-index N", "Select the local verbs GID index.");
    opt(fp, c, "--tensor-parallel-token-prefill", "GLM diagnostic: prefill one token at a time for exact arithmetic.");
    opt(fp, c, "--debug-hash N", "Cross-check hidden state every N tokens.");
    fputc('\n', fp);
}

static void print_cli_diagnostics(FILE *fp, const help_colors *c);

static void print_cli_specific(FILE *fp, const help_colors *c, bool full) {
    title(fp, c, "CLI Modes");
    opt(fp, c, "ds4", "Start the interactive prompt.");
    opt(fp, c, "ds4 -p TEXT", "Run one prompt and exit.");
    opt(fp, c, "ds4 --prompt-file FILE", "Run a long prompt from a file and exit.");
    opt(fp, c, "--prefix-file FILE", "Preload complete alternating USER:/ASSISTANT: turns before the live conversation.");
    fputc('\n', fp);
    if (full) {
        print_cli_diagnostics(fp, c);
    }
}

static void print_cli_diagnostics(FILE *fp, const help_colors *c) {
    title(fp, c, "Diagnostics And Data Collection");
    opt(fp, c, "--inspect", "Load the model and print a summary only.");
    opt(fp, c, "--dump-tokens", "Print the exact CLI prompt token stream, then exit. Use --raw for literal text.");
    opt(fp, c, "--dump-logits FILE", "Write full next-token logits as JSON.");
    opt(fp, c, "--dump-logprobs FILE", "Write greedy continuation top-logprobs as JSON.");
    opt(fp, c, "--logprobs-top-k N", "Alternatives stored by --dump-logprobs. Default: 20");
    opt(fp, c, "--decode-consistency N", "Compare N-token decode logits with a fresh full prefill.");
    opt(fp, c, "--expert-profile FILE", "Metal-only: write routed expert locality/cache simulation JSON.");
    opt(fp, c, "--perplexity-file FILE", "Score raw text with teacher-forced NLL.");
    opt(fp, c, "--imatrix-dataset FILE", "Rendered prompt dataset for imatrix collection.");
    opt(fp, c, "--imatrix-out FILE", "Write llama-compatible routed-MoE imatrix .dat.");
    opt(fp, c, "--imatrix-max-prompts N", "Stop imatrix collection after N prompts.");
    opt(fp, c, "--imatrix-max-tokens N", "Stop imatrix collection after N prompt tokens.");
    opt(fp, c, "--imatrix-min-expert-samples N", "Continue until every routed expert has N samples.");
    opt(fp, c, "--head-test", "Run the output HC/logits head after the native slice.");
    opt(fp, c, "--first-token-test", "Run exact CPU whole-model pass for the first prompt token.");
    opt(fp, c, "--metal-graph-test", "Compare first GPU-resident graph stages with CPU.");
    opt(fp, c, "--metal-graph-full-test", "Run the GPU-resident self-token graph across all layers.");
    opt(fp, c, "--metal-graph-prompt-test", "Compare CPU and GPU graph logits for the full prompt.");
    fputc('\n', fp);
}

static void print_cli_commands(FILE *fp, const help_colors *c) {
    title_red(fp, c, "Interactive Commands");
    opt(fp, c, "/help", "Show interactive commands.");
    opt(fp, c, "/think, /think-max, /nothink", "Switch thinking mode.");
    opt(fp, c, "/ctx N", "Restart the interactive session with a new context size.");
    opt(fp, c, "/power N", "Set GPU duty cycle percentage, 1..100.");
    opt(fp, c, "/read FILE", "Submit a text file, PNG, or JPEG as the next user message.");
    opt(fp, c, "/quit, /exit", "Leave the prompt.");
    opt(fp, c, "Ctrl+C", "Stop current generation and return to ds4>.");
    fputc('\n', fp);
}

static void print_agent_specific(FILE *fp, const help_colors *c) {
    title(fp, c, "Agent Options");
    opt(fp, c, "-p, --prompt TEXT", "Submit an initial prompt after startup.");
    opt(fp, c, "--prompt-file FILE", "Read the initial prompt from FILE.");
    opt(fp, c, "--prefix-file FILE", "Preload complete alternating USER:/ASSISTANT: turns before the live task.");
    opt(fp, c, "--non-interactive", "Run without TUI. With an initial prompt: one turn; otherwise: repeated stdin prompts.");
    opt(fp, c, "--raw-prompt", "Non-interactive initial prompt only: omit agent chat/tool text.");
    opt(fp, c, "--edit-upto", "Enable anchored [upto] edits and automatic marker insertion.");
    opt(fp, c, "-sys, --system TEXT", "Extra system prompt. Empty disables extra text.");
    opt(fp, c, "--trace FILE", "Write prompt, token, and DSML debug trace.");
    opt(fp, c, "--chdir DIR", "Change working directory before loading runtime assets.");
    fputc('\n', fp);
}

static void print_agent_sessions(FILE *fp, const help_colors *c) {
    title(fp, c, "Agent Runtime Commands");
    opt(fp, c, "/save", "Save the current session in ~/.ds4/kvcache.");
    opt(fp, c, "/compact", "Compact the current session context now.");
    opt(fp, c, "/list", "List saved sessions, sorted by recent update time.");
    opt(fp, c, "/switch ID", "Load a saved session and show recent history.");
    opt(fp, c, "/del ID", "Delete a saved session.");
    opt(fp, c, "/strip ID", "Remove KV payload; the text history can be rebuilt later.");
    opt(fp, c, "/history [N]", "Show N recent user turns from the current session.");
    opt(fp, c, "/power N", "Set GPU duty cycle percentage, 1..100.");
    opt(fp, c, "/new", "Start a fresh session from the system prompt.");
    opt(fp, c, "/quit, /exit", "Exit.");
    fputc('\n', fp);
}

static void print_server_api(FILE *fp, const help_colors *c) {
    title(fp, c, "HTTP API");
    opt(fp, c, "--host HOST", "Bind address. Default: 127.0.0.1");
    opt(fp, c, "--port N", "Bind port. Default: 8000");
    opt(fp, c, "--cors", "Add Access-Control-Allow-* headers for browser JS clients.");
    opt(fp, c, "--trace FILE", "Write prompts, cache decisions, output, and tool calls.");
    opt(fp, c, "--batched-session N", "Keep N resident sessions and batch decode-ready requests.");
    opt(fp, c, "--mixed-prefill-quantum N", "Prefill chunk while generations are active. Default: 128; GLM-5.3 minimum: 1024");
    para(fp, c, "Endpoints: /v1/chat/completions, /v1/responses, /v1/completions, and /v1/messages.");
    para(fp, c, "Model endpoint aliases include deepseek-v4-flash and deepseek-v4-pro; both serve the loaded GGUF.");
    fputc('\n', fp);
}

static void print_server_thinking(FILE *fp, const help_colors *c) {
    title(fp, c, "Server Thinking Defaults");
    para(fp, c, "DeepSeek-compatible chat requests default to high-effort thinking.");
    para(fp, c, "reasoning_effort=max or output_config.effort=max requests Think Max.");
    para(fp, c, "Think Max requires --ctx >= 393216; smaller contexts use high.");
    para(fp, c, "thinking={type:disabled}, think=false, or model=deepseek-chat selects non-thinking mode.");
    para(fp, c, "In thinking mode, client sampling knobs are ignored like the official API.");
    fputc('\n', fp);
}

static void print_kv_cache(FILE *fp, const help_colors *c) {
    title(fp, c, "Disk KV Cache");
    opt(fp, c, "--kv-disk-dir DIR", "Enable disk KV checkpoints in DIR.");
    opt(fp, c, "--kv-disk-space-mb N", "Disk budget. Default when enabled: 4096");
    opt(fp, c, "--kv-cache-min-tokens N", "Do not save/load checkpoints shorter than N. Default: 512");
    opt(fp, c, "--kv-cache-cold-max-tokens N", "Save cold first prompts up to N tokens. 0 disables. Default: 30000");
    opt(fp, c, "--kv-cache-continued-interval-tokens N", "Save aligned continued frontiers. 0 disables. Default: 10000");
    opt(fp, c, "--kv-cache-boundary-trim-tokens N", "Trim tail tokens for cold boundary saves. Default: 32");
    opt(fp, c, "--kv-cache-boundary-align-tokens N", "Align cold boundary saves to this multiple. Default: 2048");
    opt(fp, c, "--kv-cache-reject-different-quant", "Reject checkpoints written with different routed-expert quantization.");
    opt(fp, c, "--disable-exact-dsml-tool-replay", "Disable exact sampled DSML tool replay map.");
    opt(fp, c, "--tool-memory-max-ids N", "Exact tool-call IDs kept in RAM. Default: 100000");
    fputc('\n', fp);
}

static void print_bench_specific(FILE *fp, const help_colors *c) {
    title(fp, c, "Benchmark Input");
    opt(fp, c, "--prompt-file FILE", "Raw benchmark text; token sequence is sliced at each frontier.");
    opt(fp, c, "--chat-prompt-file FILE", "Render FILE as one no-thinking chat user message.");
    opt(fp, c, "-sys, --system TEXT", "System prompt used only with --chat-prompt-file.");
    fputc('\n', fp);
    title(fp, c, "Benchmark Sweep");
    opt(fp, c, "--ctx-start N", "First measured frontier. Default: 2048");
    opt(fp, c, "--ctx-max N", "Last measured frontier. Default: 32768");
    opt(fp, c, "--ctx-alloc N", "Allocated context. Default: ctx-max + gen-tokens + 1");
    opt(fp, c, "--step-mul F", "Multiplicative step. Default: 1");
    opt(fp, c, "--step-incr N", "Linear step when --step-mul is 1. Default: 2048");
    opt(fp, c, "--gen-tokens N", "Greedy decode tokens per frontier. 0 for pure prefill. Default: 128");
    opt(fp, c, "--teacher-forced-decode", "Decode the following prompt tokens instead of each predicted argmax.");
    opt(fp, c, "--csv FILE", "Write CSV there instead of stdout.");
    opt(fp, c, "--dump-frontier-logits-dir DIR", "Write one full-logit JSON file per frontier.");
    fputc('\n', fp);
}

static void print_eval_specific(FILE *fp, const help_colors *c) {
    title(fp, c, "Evaluation");
    opt(fp, c, "--suite NAME", "core, hard, all, or hard-smoke. Default: core");
    opt(fp, c, "--source NAME", "Run only cases from this source.");
    opt(fp, c, "--domain NAME", "Run only cases in this domain.");
    opt(fp, c, "--case-id ID", "Run the case with this source ID.");
    opt(fp, c, "--list-cases", "List selected cases without loading a model.");
    opt(fp, c, "--validate-cases", "Validate all embedded cases and exit.");
    opt(fp, c, "-n, --tokens N", "Override the generation budget for every question.");
    opt(fp, c, "--questions N", "Run only the first N selected questions.");
    opt(fp, c, "--case-sequence LIST", "Run 1-based case numbers in this comma-separated order.");
    opt(fp, c, "--retry-incomplete", "Retry a missing final answer once with twice the budget.");
    opt(fp, c, "--trace FILE", "Write questions, outputs, and grading decisions.");
    opt(fp, c, "--regrade-trace FILE", "Regrade a prior trace without loading the model.");
    opt(fp, c, "--soft-limit-reply-budget N", "Soft close thinking near the end of reply budget. Default: 1024");
    opt(fp, c, "--hard-limit-reply-budget N", "Force </think> with N tokens left. Default: 512");
    opt(fp, c, "--soft-limit-think-close-rank N", "Soft-close when </think> is in top N tokens. Default: 3");
    opt(fp, c, "--pause-ms N", "Pause after each result in the TTY UI. Default: 350");
    opt(fp, c, "--plain", "Disable split-screen ANSI UI.");
    opt(fp, c, "--self-test-extractors", "Run answer-extractor self-tests and exit.");
    fputc('\n', fp);
}

static void print_glm53(FILE *fp, const help_colors *c) {
    title(fp, c, "GLM-5.3 Metal Switches");
    para(fp, c, "Environment variables added by the GLM-5.3 M3 Ultra work. Defaults are the shipped fast mode; nothing needs to be set. Registry entries are the DS4_GLM_EXACT list in bench/EXACT-MODE-PLAN.md; the full reference, including opt-in experiments, is docs/GLM53_M3ULTRA.md.");
    fputc('\n', fp);
    title(fp, c, "Supported Controls");
    opt(fp, c, "DS4_ANTHROPIC_DEFAULT_EFFORT", "Default reasoning effort for Anthropic-protocol requests that carry none (e.g. Claude Code); explicit request fields still win.");
    opt(fp, c, "DS4_DFLASH_CTX_CAP", "Caps the drafter's context rows (default about 256; more rows cost draft latency).");
    opt(fp, c, "DS4_DFLASH_DISABLE", "Ignores a loaded DFlash2 drafter and decodes serially.");
    opt(fp, c, "DS4_GLM53_MEMORY_CEILING_GB", "Clamps the GLM-5.3 memory-guard budget to N GB (used to keep a 512 GB machine's other workloads safe).");
    opt(fp, c, "DS4_GLM53_PREFILL_CHUNK", "Upper bound on prefill chunk tokens (default 8192; 4096 and 2048 restore earlier shipped chunks).");
    opt(fp, c, "DS4_GLM_DSA_TAIL_CHECKED", "=0 restores the legacy unchecked ragged tail in DSA attention (default 1: bounds-checked; registry entry 4).");
    opt(fp, c, "DS4_GLM_ENABLE_BF16_LOWRANK_SPLITK", "=0 turns the BF16 low-rank split-K off (default on).");
    opt(fp, c, "DS4_GLM_ENABLE_DSA_BLOCKED_SOFTMAX", "=1 forces the blocked softmax on (default on).");
    opt(fp, c, "DS4_GLM_ENABLE_HCX_PTAIL", "=1 explicitly selects the ptail HC-expand epilogue (already the default).");
    opt(fp, c, "DS4_GLM_ENABLE_ROUTED_DOWN_SPLIT", "=1 forces the expert-parallel routed down split on (default on).");
    opt(fp, c, "DS4_GLM_ENABLE_ROUTER_SHARED_FOLD", "=1 forces the router/shared-expert fold on (it is the default; the kill switch wins).");
    opt(fp, c, "DS4_GLM_ENABLE_TOPK_FAST", "No-op kept for scripts that set it (the fast path is on by default).");
    opt(fp, c, "DS4_GLM_EXACT", "=1 exact mode: every registered floating-point-order change off, numerics equal to upstream at the pin (see EXACT-MODE-PLAN.md).");
    opt(fp, c, "DS4_GLM_HC_PRE_ALGEBRA_A", "=0 turns off hc_pre algebra half A (unscaled split-K dot, scale in the tail; default on, registry entry 6).");
    opt(fp, c, "DS4_GLM_HC_PRE_SLICES", "=8|16|32 hc_pre split-K slice count (default 16; 32 is faster but out of the fidelity budget).");
    opt(fp, c, "DS4_GLM_SCORER_XREDUCE_MIN_ROWS", "Candidate-row threshold below which the production scorer is kept even when xr8 is enabled (default 37500; 0 disables the gate).");
    opt(fp, c, "DS4_GLM_SPLIT8_BLOCK_ROWS_DEEP", "Block rows for split8 DSA attention at depth (default 128; registry entry 7).");
    opt(fp, c, "DS4_GLM_SPLIT8_BLOCK_ROWS_SHALLOW", "Block rows for split8 DSA attention at shallow depth (default 32; registry entry 7).");
    opt(fp, c, "DS4_GLM_T2S_BF16_NSG", "Simdgroups per threadgroup of the BF16 matvec (registry entry 9; pinned under DS4_GLM_EXACT).");
    opt(fp, c, "DS4_GLM_T2S_Q8NSG_<FAM>", "Per-family override of the Q8_0 matvec simdgroup count (families KDA, HCX, SDN, SHG, DSAF; registry entry 8; pinned under DS4_GLM_EXACT).");
    opt(fp, c, "DS4_GLM_TOPK_FAST_MIN_COMP", "Minimum candidate width for the top-k fast path (default 12288; 0 removes the gate).");
    opt(fp, c, "DS4_SERVER_CHECKPOINT_ON_LENGTH", "=0 stops recording the thinking checkpoint for turns truncated by max_tokens (default: recorded, so the next turn continues from live KV).");
    opt(fp, c, "DS4_SERVER_CHECKPOINT_WITH_TOOLS", "=0 stops recording the thinking checkpoint for tool-context turns (default: recorded).");
    opt(fp, c, "DS4_TRACE_MAX_MB", "Caps the live --trace segment at N MB; the previous segment is kept at <path>.1.");
    fputc('\n', fp);
    title(fp, c, "Kill Switches");
    para(fp, c, "Each turns one default-on change off, for A/B measurement and bisection. Unless the meaning says otherwise a switch is read as set to any non-empty value.");
    opt(fp, c, "DS4_DFLASH_NO_ADAPTIVE", "Legacy fallback when --dflash-mode is omitted: uncapped DFlash experiment with prefill seeding; no 2% slowdown claim.");
    opt(fp, c, "DS4_DFLASH_BUDGET_MS", "Diagnostic positive full refresh/proposal budget estimate; default 500ms on the calibrated M3 Ultra public-model profile. Invalid values stay serial.");
    opt(fp, c, "DS4_DFLASH_NO_SELECTOR", "Disables the DFlash2 candidate selector (coherent-chain tracing).");
    opt(fp, c, "DS4_DFLASH_SDPA_SCALAR", "Forces the scalar SDPA drafter kernel instead of the simdgroup one.");
    opt(fp, c, "DS4_GLM_DISABLE_BF16_LOWRANK_SPLITK", "Ordinary mm kernel for the BF16 low-rank prefill matmuls instead of split-K (registry entry 2).");
    opt(fp, c, "DS4_GLM_DISABLE_DENSE_HALF_COPY", "Disables the dense-layer half-copy path and its ring.");
    opt(fp, c, "DS4_GLM_DISABLE_DENSE_HALF_RING", "Disables the half-copy ring alone.");
    opt(fp, c, "DS4_GLM_DISABLE_DSA_BATCH_HOIST", "Batched DSA prefill attention: no hoisting of the shared loads.");
    opt(fp, c, "DS4_GLM_DISABLE_DSA_BATCH_NOROPE", "Batched DSA prefill attention: keep the (dead at this shape) rope path.");
    opt(fp, c, "DS4_GLM_DISABLE_DSA_BATCH_SKIP_RESCALE", "Batched DSA prefill attention: no rescale skipping.");
    opt(fp, c, "DS4_GLM_DISABLE_DSA_BLOCKED_SOFTMAX", "Row-at-a-time softmax in batched DSA attention instead of the blocked online softmax (registry entry 3).");
    opt(fp, c, "DS4_GLM_DISABLE_DSA_GATHER_WIDE", "8-byte gather copies in the blocked DSA kernel instead of 16-byte (bit-identical).");
    opt(fp, c, "DS4_GLM_DISABLE_DSA_INDEXER_FOLD", "Separate dispatches for q_a, kv_a and the three DSA indexer projections instead of one (bit-identical fold).");
    opt(fp, c, "DS4_GLM_DISABLE_DSA_TAIL_CHECKED", "Kill switch for the checked ragged tail.");
    opt(fp, c, "DS4_GLM_DISABLE_HCX_PTAIL", "Production HC-expand epilogue instead of the eight-lane ptail epilogue (Tier 1 default; DS4_GLM_EXACT also selects production).");
    opt(fp, c, "DS4_GLM_DISABLE_HC_ATTN_TAILFUSE", "Residual add + HC expand as separate dispatches after the attention-output matvec.");
    opt(fp, c, "DS4_GLM_DISABLE_HC_CHAIN_COLLAPSE", "Prefill HC chain: separate collapse and weighted RMSNorm passes.");
    opt(fp, c, "DS4_GLM_DISABLE_HC_CHAIN_SCALE", "Prefill HC chain: the RMSNorm materializes its own scaled copy instead of the expand publishing the per-row scale.");
    opt(fp, c, "DS4_GLM_DISABLE_HC_MIX_SPLITK", "Row-per-simdgroup HC mixer instead of the split-K mixer (an FP-order change).");
    opt(fp, c, "DS4_GLM_DISABLE_HC_NORM_FUSE", "Unfused HC norm dispatches.");
    opt(fp, c, "DS4_GLM_DISABLE_HC_NORM_MIX_FUSE", "Unfused HC norm+mix dispatches.");
    opt(fp, c, "DS4_GLM_DISABLE_HC_PHASEC_HOIST", "Kill switch for the phase-C hoist experiment.");
    opt(fp, c, "DS4_GLM_DISABLE_HC_PRE_ALGEBRA_A", "Kill switch for algebra half A.");
    opt(fp, c, "DS4_GLM_DISABLE_HC_PRE_ALGEBRA_B", "Kill switch for algebra half B.");
    opt(fp, c, "DS4_GLM_DISABLE_HC_PRE_ONE_DISPATCH", "Kill switch for the one-dispatch hc_pre experiment.");
    opt(fp, c, "DS4_GLM_DISABLE_HC_PRE_SINGLE", "Unfused hc_pre ladder instead of the single-dispatch fused kernel.");
    opt(fp, c, "DS4_GLM_DISABLE_HC_PRE_WIDE", "hc_pre split-K at the decode default slice count instead of 16 (exact-mode registry entry 1).");
    opt(fp, c, "DS4_GLM_DISABLE_HC_REFUSE", "Four-dispatch hc_pre instead of the fused rms+split-K / reduce+wsum pair.");
    opt(fp, c, "DS4_GLM_DISABLE_HC_TAILFUSE", "Residual add + HC expand as separate dispatches after the MoE shared-down / dense-down matvec.");
    opt(fp, c, "DS4_GLM_DISABLE_HC_TAIL_W4", "Scalar split-K reduce in the hc_pre tail instead of float4.");
    opt(fp, c, "DS4_GLM_DISABLE_HC_TAIL_WIDE", "Single 1024-thread hc_pre tail instead of the replicated wide tail.");
    opt(fp, c, "DS4_GLM_DISABLE_INDEXED_SPLIT8", "Mono DSA decode attention kernel for every n_selected instead of the split8 kernel above 64 rows.");
    opt(fp, c, "DS4_GLM_DISABLE_INDEXER_CAUSAL_GRID", "Full rectangular indexer score grid in prefill instead of the causal staircase plus fill kernel.");
    opt(fp, c, "DS4_GLM_DISABLE_INDEXER_HEADFOLD", "Disables the indexer head-fold variant selection (prefill indexer projections).");
    opt(fp, c, "DS4_GLM_DISABLE_INDEXER_PAIR", "Separate dispatches for the indexer k projection and compressor gate.");
    opt(fp, c, "DS4_GLM_DISABLE_INDEXER_SCORE_STREAM", "Non-streamed DSA indexer scorer (query re-read per threadgroup).");
    opt(fp, c, "DS4_GLM_DISABLE_KDA_LOWRANK_PACK", "Undoes the KDA low-rank pack (f_a+g_a+beta flat3, f_b+g_b pair2in; de-aliased kda_lowrank2).");
    opt(fp, c, "DS4_GLM_DISABLE_KDA_LOWRANK_PROLOGUE", "f_b/g_b as separate dispatches instead of in prep_state's per-head prologue.");
    opt(fp, c, "DS4_GLM_DISABLE_KDA_OUT_FOLD", "KDA decode_out as its own dispatch instead of folded into the state kernel's tail.");
    opt(fp, c, "DS4_GLM_DISABLE_KDA_PREFILL_C16", "KDA prefill at the C=8 configuration instead of the 16-column recurrence.");
    opt(fp, c, "DS4_GLM_DISABLE_KDA_PREFILL_FAST", "Production three-kernel KDA prefill instead of the fast path.");
    opt(fp, c, "DS4_GLM_DISABLE_KDA_QKV_LOWRANK_FOLD", "Separate dispatch for the three low-rank rows instead of riding in the q/k/v grid's tail.");
    opt(fp, c, "DS4_GLM_DISABLE_KDA_REFUSE", "Three-dispatch KDA decode (prep / state / out) instead of the two-dispatch fused prep+state.");
    opt(fp, c, "DS4_GLM_DISABLE_KDA_SPLIT", "Single fused KDA decode kernel instead of the split form (falls back one level further).");
    opt(fp, c, "DS4_GLM_DISABLE_LAZY_BATCH_WS", "=1 allocates every batch workspace tensor eagerly instead of on first use.");
    opt(fp, c, "DS4_GLM_DISABLE_MOE_BLOCK_DATAFLOW", "Kill switch for the MoE block dataflow kernel.");
    opt(fp, c, "DS4_GLM_DISABLE_PREFILL_FOLD_FFNADD", "Prefill: FFN residual add as its own pass instead of folded (bit-exact).");
    opt(fp, c, "DS4_GLM_DISABLE_PREFILL_FOLD_HCEXPAND", "Prefill: width-1 HC expand instead of width-4.");
    opt(fp, c, "DS4_GLM_DISABLE_PREFILL_FOLD_MOEMAP", "Prefill: single-threadgroup routed work map instead of per-expert threadgroups.");
    opt(fp, c, "DS4_GLM_DISABLE_PREFILL_FOLD_MOESWIGLU", "Prefill: width-1 SwiGLU passes instead of width-4.");
    opt(fp, c, "DS4_GLM_DISABLE_QAKV_FUSE", "Separate dispatches for attn q_a and kv_a.");
    opt(fp, c, "DS4_GLM_DISABLE_QKLOW_BATCH_TILE", "Per-(head, token) qk low-rank prefill kernel instead of the token-tiled one.");
    opt(fp, c, "DS4_GLM_DISABLE_QKLOW_SG", "Thread-per-row qk low-rank decode kernel instead of the coalesced simdgroup kernel at qk_nope=256.");
    opt(fp, c, "DS4_GLM_DISABLE_REDUCE_Q8_U16", "Byte-wise Q8_0 value-row loads in the split8 reduce instead of ushort pairs.");
    opt(fp, c, "DS4_GLM_DISABLE_REDUCE_WIDE_TG", "256-thread split8 reduce dispatch instead of 512.");
    opt(fp, c, "DS4_GLM_DISABLE_ROUTED_ASTAGE_DB", "Disables A-stage double buffering in the routed-expert prefill GEMM (default off at compile time; see DS4_GLM_ROUTED_ASTAGE_DB).");
    opt(fp, c, "DS4_GLM_DISABLE_ROUTED_DEQ_WIDE", "Scalar quant-byte loads in the routed prefill GEMM dequant instead of 16-byte loads.");
    opt(fp, c, "DS4_GLM_DISABLE_ROUTED_DOWN_SPLIT", "Sequential routed-expert down matvec instead of the expert-parallel split (Tier 2; also off under DS4_GLM_EXACT).");
    opt(fp, c, "DS4_GLM_DISABLE_ROUTED_GATEUP_WIDE", "Production routed gate+up kernel instead of the ushort4-load variant.");
    opt(fp, c, "DS4_GLM_DISABLE_ROUTED_TILE_NARROW", "Plain 32-row routed prefill tile instead of the narrow tile selection.");
    opt(fp, c, "DS4_GLM_DISABLE_ROUTER_BATCH_TILE", "Per-token router matvec in prefill instead of the token-tiled kernel.");
    opt(fp, c, "DS4_GLM_DISABLE_ROUTER_SELECT_FAST", "Full 512-wide bitonic sort for router top-k instead of the iterative selection (bit-identical).");
    opt(fp, c, "DS4_GLM_DISABLE_ROUTER_SHARED_FOLD", "Router not folded into the head of the shared-expert gate+up grid (two dispatches instead of one).");
    opt(fp, c, "DS4_GLM_DISABLE_ROUTER_TAIL_FOLD", "Top-8 router selection as its own dispatch instead of folded onto the logits matvec tail.");
    opt(fp, c, "DS4_GLM_DISABLE_SCORER_HALF", "Forces the production DSA scorer kernel regardless of any variant request.");
    opt(fp, c, "DS4_GLM_DISABLE_SCORER_XREDUCE", "Kill switch for the xr* scorers.");
    opt(fp, c, "DS4_GLM_DISABLE_SINKHORN_PAR", "Sinkhorn comb back behind the barrier it used to wait on (no overlap with the collapse).");
    opt(fp, c, "DS4_GLM_DISABLE_TOOL_RESULT_REORDER", "Disables the GLM tool-result reorder that renders parallel tool results in the order the chat template expects.");
    opt(fp, c, "DS4_GLM_DISABLE_TOPK_FAST", "Disables the bounded-radix DSA top-k fast path (Tier 1; its only switch; not clamped by DS4_GLM_EXACT).");
    opt(fp, c, "DS4_GLM_DISABLE_TOPK_ONESHOT", "Disables the one-shot DSA top-k selection path.");
    opt(fp, c, "DS4_GLM_MTP_NO_ROWSNAP", "Disables the MTP row-boundary KDA snapshot fast path (inert on real GLM-5.3 graphs on this branch).");
    opt(fp, c, "DS4_KDA_CONCURRENT_DISABLE", "Runs the six KDA first-level prefill projections serially instead of in one concurrent dispatch level.");
    opt(fp, c, "DS4_KV_EVICT_RAW", "=1 restores the raw full-length KV store on eviction instead of trimming it to the last client-transcript position (which keeps disk keys reproducible).");
    opt(fp, c, "DS4_METAL_DISABLE_GLM53_Q8_QKV", "Disables the fused Q8_0 KDA q/k/v decode matvec (separate projections instead).");
    opt(fp, c, "DS4_SLOT_SCORE_LEGACY", "Restores the legacy slot placement instead of eviction-cost scoring (only matters with --batched-session >= 2).");
    opt(fp, c, "DS4_STREAM_GUARD_LEGACY", "Restores the old streaming guard that held all answer text until a second </think> when thinking and tools were both enabled.");
    fputc('\n', fp);
}
static bool tool_has_topic(ds4_help_tool tool, const char *topic) {
    if (!topic) return true;
    if (streq(topic, "all")) return true;
    if (streq(topic, "runtime") || streq(topic, "distributed") || streq(topic, "glm53")) return true;
    if (streq(topic, "sampling"))
        return tool == DS4_HELP_DS4 || tool == DS4_HELP_AGENT || tool == DS4_HELP_EVAL;
    if (streq(topic, "steering"))
        return tool == DS4_HELP_DS4 || tool == DS4_HELP_SERVER || tool == DS4_HELP_AGENT;
    switch (tool) {
    case DS4_HELP_DS4:
        return streq(topic, "diagnostics") || streq(topic, "commands");
    case DS4_HELP_SERVER:
        return streq(topic, "api") || streq(topic, "kv-cache") || streq(topic, "thinking");
    case DS4_HELP_AGENT:
        return streq(topic, "sessions") || streq(topic, "commands") || streq(topic, "tools");
    case DS4_HELP_BENCH:
        return streq(topic, "benchmark");
    case DS4_HELP_EVAL:
        return streq(topic, "evaluation");
    }
    return false;
}

static void more_line(FILE *fp, const help_colors *c, const char *label, const char *topic) {
    static const char *colors[] = {
        "\x1b[38;5;81m", "\x1b[38;5;114m", "\x1b[38;5;179m",
        "\x1b[38;5;141m", "\x1b[38;5;147m"
    };
    static size_t idx;
    const char *on = c->cyan ? colors[idx++ % (sizeof(colors) / sizeof(colors[0]))] : "";
    if (streq(label, "Interactive commands:") && c->red) on = c->red;
    const char *off = c->off ? c->off : "";
    fprintf(fp, "    %s%-26s%s --help %s\n", on, label, off, topic);
}

static void print_more_info(FILE *fp, const help_colors *c, ds4_help_tool tool) {
    title(fp, c, "More Info");
    more_line(fp, c, "Runtime full info:", "runtime");
    if (tool_has_topic(tool, "sampling"))
        more_line(fp, c, "Sampling full info:", "sampling");
    more_line(fp, c, "Distributed inference:", "distributed");
    more_line(fp, c, "GLM-5.3 switches:", "glm53");
    if (tool_has_topic(tool, "steering"))
        more_line(fp, c, "Steering full info:", "steering");
    if (tool == DS4_HELP_DS4) {
        more_line(fp, c, "Interactive commands:", "commands");
        more_line(fp, c, "Diagnostics:", "diagnostics");
    } else if (tool == DS4_HELP_SERVER) {
        more_line(fp, c, "HTTP API:", "api");
        more_line(fp, c, "Disk KV cache:", "kv-cache");
        more_line(fp, c, "Thinking behavior:", "thinking");
    } else if (tool == DS4_HELP_AGENT) {
        more_line(fp, c, "Agent sessions:", "sessions");
        more_line(fp, c, "Agent commands:", "commands");
        more_line(fp, c, "Agent tool system:", "tools");
    } else if (tool == DS4_HELP_BENCH) {
        more_line(fp, c, "Benchmark sweep:", "benchmark");
    } else if (tool == DS4_HELP_EVAL) {
        more_line(fp, c, "Evaluation options:", "evaluation");
    }
    fputc('\n', fp);
}

static void print_examples(FILE *fp, const help_colors *c, ds4_help_tool tool, const char *topic) {
    title(fp, c, "Examples");
    if (topic_is(topic, "distributed")) {
        opt(fp, c, "worker", "./ds4 --role worker --layers 21:output --coordinator 192.168.0.181 9000 -m ds4flash.gguf");
        opt(fp, c, "coordinator", "./ds4 --role coordinator --layers 0:20 --listen 0.0.0.0 9000 -p \"Hello\" -m ds4flash.gguf");
    } else if (topic_is(topic, "runtime")) {
        if (tool == DS4_HELP_SERVER) {
            opt(fp, c, "Metal API", "./ds4-server -m ds4flash.gguf --metal --ctx 100000");
            opt(fp, c, "quiet API", "./ds4-server --power 60 --host 127.0.0.1 --port 8000");
        } else if (tool == DS4_HELP_AGENT) {
            opt(fp, c, "agent", "./ds4-agent -m ds4flash.gguf --ctx 100000");
            opt(fp, c, "quiet agent", "./ds4-agent --power 50");
        } else if (tool == DS4_HELP_BENCH) {
            opt(fp, c, "bench", "./ds4-bench --prompt-file long.txt --ctx-max 32768");
            opt(fp, c, "quiet bench", "./ds4-bench --prompt-file long.txt --power 70");
        } else if (tool == DS4_HELP_EVAL) {
            opt(fp, c, "eval", "./ds4-eval --questions 10 --ctx 100000");
            opt(fp, c, "CPU debug", "./ds4-eval --cpu --questions 1 --tokens 32");
        } else {
            opt(fp, c, "Metal", "./ds4 -m ds4flash.gguf --metal -c 100000");
            opt(fp, c, "quiet thermals", "./ds4 -p \"Summarize README\" --power 50");
        }
    } else if (topic_is(topic, "steering")) {
        opt(fp, c, "steer FFN", "./ds4 -p \"Write tersely\" --dir-steering-file dir.bin --dir-steering-ffn 0.8");
    } else if (topic_is(topic, "glm53")) {
        opt(fp, c, "exact mode", "DS4_GLM_EXACT=1 ./ds4 -m gguf/GLM-5.3-Flash-Q4_K.gguf --metal -p \"Hello\"");
        opt(fp, c, "memory ceiling", "DS4_GLM53_MEMORY_CEILING_GB=280 ./ds4-server -m gguf/GLM-5.3-Flash-Q4_K.gguf --metal --ctx 409600");
    } else if (tool == DS4_HELP_SERVER || topic_is(topic, "api") || topic_is(topic, "kv-cache")) {
        opt(fp, c, "local API", "./ds4-server --ctx 100000 --kv-disk-dir ~/.ds4/server-kv --kv-disk-space-mb 8192");
        opt(fp, c, "curl", "curl http://127.0.0.1:8000/v1/models");
    } else if (tool == DS4_HELP_AGENT || topic_is(topic, "sessions") || topic_is(topic, "tools")) {
        opt(fp, c, "interactive", "./ds4-agent");
        opt(fp, c, "one shot", "./ds4-agent --non-interactive -p \"Create /tmp/hello.c\"");
    } else if (tool == DS4_HELP_BENCH || topic_is(topic, "benchmark")) {
        opt(fp, c, "csv", "./ds4-bench --prompt-file long.txt --ctx-max 32768 --csv speed.csv");
        opt(fp, c, "prefill only", "./ds4-bench --prompt-file long.txt --gen-tokens 0");
    } else if (tool == DS4_HELP_EVAL || topic_is(topic, "evaluation")) {
        opt(fp, c, "first 10", "./ds4-eval --questions 10 --trace eval.trace");
        opt(fp, c, "plain", "./ds4-eval --plain --nothink --tokens 512");
    } else {
        opt(fp, c, "chat", "./ds4");
        opt(fp, c, "one shot", "./ds4 -p \"Explain mmap in C\"");
        opt(fp, c, "long prompt", "./ds4 --think-max --prompt-file prompt.txt --ctx 393216");
    }
    fputc('\n', fp);
}

static void print_topic(FILE *fp, const help_colors *c, ds4_help_tool tool, const char *topic) {
    if (streq(topic, "all")) {
        print_model_runtime(fp, c, tool, true);
        if (tool_has_topic(tool, "sampling")) print_sampling(fp, c, true);
        if (tool_has_topic(tool, "steering")) print_steering(fp, c);
        print_distributed(fp, c);
        print_glm53(fp, c);
        if (tool == DS4_HELP_DS4) {
            print_cli_specific(fp, c, true);
            print_cli_commands(fp, c);
        } else if (tool == DS4_HELP_SERVER) {
            print_server_api(fp, c);
            print_server_thinking(fp, c);
            print_kv_cache(fp, c);
        } else if (tool == DS4_HELP_AGENT) {
            print_agent_specific(fp, c);
            print_agent_sessions(fp, c);
        } else if (tool == DS4_HELP_BENCH) {
            print_bench_specific(fp, c);
        } else if (tool == DS4_HELP_EVAL) {
            print_eval_specific(fp, c);
        }
        return;
    }

    if (streq(topic, "runtime")) print_model_runtime(fp, c, tool, true);
    else if (streq(topic, "sampling")) print_sampling(fp, c, true);
    else if (streq(topic, "steering")) print_steering(fp, c);
    else if (streq(topic, "distributed")) print_distributed(fp, c);
    else if (streq(topic, "glm53")) print_glm53(fp, c);
    else if (tool == DS4_HELP_DS4 && streq(topic, "diagnostics")) print_cli_diagnostics(fp, c);
    else if (tool == DS4_HELP_DS4 && streq(topic, "commands")) print_cli_commands(fp, c);
    else if (tool == DS4_HELP_SERVER && streq(topic, "api")) print_server_api(fp, c);
    else if (tool == DS4_HELP_SERVER && streq(topic, "kv-cache")) print_kv_cache(fp, c);
    else if (tool == DS4_HELP_SERVER && streq(topic, "thinking")) print_server_thinking(fp, c);
    else if (tool == DS4_HELP_AGENT && streq(topic, "sessions")) print_agent_sessions(fp, c);
    else if (tool == DS4_HELP_AGENT && streq(topic, "commands")) print_agent_sessions(fp, c);
    else if (tool == DS4_HELP_AGENT && streq(topic, "tools")) {
        title(fp, c, "Agent Tool System");
        para(fp, c, "The agent can read, search, write, edit, run bash, and browse through Chrome-backed web tools.");
        para(fp, c, "DeepSeek-family models emit DSML tool calls; GLM models use native <tool_call> syntax. Both are rendered live in the terminal.");
        para(fp, c, "Edit uses exact old/new replacement. --edit-upto enables anchored replacements between a unique head and tail.");
        fputc('\n', fp);
    } else if (tool == DS4_HELP_BENCH && streq(topic, "benchmark")) print_bench_specific(fp, c);
    else if (tool == DS4_HELP_EVAL && streq(topic, "evaluation")) print_eval_specific(fp, c);
}

static void print_default(FILE *fp, const help_colors *c, ds4_help_tool tool) {
    print_model_runtime(fp, c, tool, false);

    if (tool == DS4_HELP_DS4) {
        print_cli_specific(fp, c, true);
        print_sampling(fp, c, false);
    } else if (tool == DS4_HELP_SERVER) {
        print_server_api(fp, c);
        print_kv_cache(fp, c);
    } else if (tool == DS4_HELP_AGENT) {
        print_agent_specific(fp, c);
        print_agent_sessions(fp, c);
    } else if (tool == DS4_HELP_BENCH) {
        print_bench_specific(fp, c);
    } else if (tool == DS4_HELP_EVAL) {
        print_eval_specific(fp, c);
    }
}

void ds4_help_print(FILE *fp, ds4_help_tool tool, const char *topic) {
    help_colors c = help_make_colors(fp);
    if (topic && !tool_has_topic(tool, topic)) {
        fprintf(fp, "%s: unknown help topic '%s'\n\n", tool_name(tool), topic);
        topic = NULL;
    }

    fprintf(fp, "%s%s%s\n", c.bright ? c.bright : "", tool_name(tool), c.off ? c.off : "");
    fprintf(fp, "%s\n\n", tool_summary(tool));
    fprintf(fp, "%s\n\n", tool_usage(tool));

    if (topic) print_topic(fp, &c, tool, topic);
    else {
        print_default(fp, &c, tool);
        print_more_info(fp, &c, tool);
    }
    print_examples(fp, &c, tool, topic);
}
