#!/usr/bin/env python3
"""Build the LONG-context manifest for the DSA checked-tail quality assessment (see
bench/FIDELITY.md, "Known upstream defect").
The 1k manifest never leaves the model's 4,096-token dense-attention window, so it never executes the
indexed-attention kernel the correction changes.  This manifest cuts prompts of ~6k / 12k / 30k / 62k
tokens from the same corpora (Shakespeare, I Promessi Sposi, the repo's markdown docs), each followed
by a fixed continuation of ~230 tokens (the 1k manifest's convention: 'continue this passage', the
continuation is the text that actually follows, cut at word boundaries).  Windows never overlap, within
or across depths, and no two prompts share a 1,024-character prefix.  Deterministic (seeded).
Token counts are exact: each corpus is tokenized once with the model's own tokenizer on the CPU
(`ds4 --dump-tokens --raw`), windows are cut at token boundaries, and every finished prompt is
re-counted through the scorer's chat renderer (`--dump-tokens --nothink --system ''`, which matches
score_official's ds4_encode_chat_prompt(NULL system) - verified on case_0056: 691 both ways).
No GPU: --dump-tokens opens the GGUF for its vocabulary only (~60 MB RSS).
Environment: DS4_ROOT (repository root; default: derived from this file), DS4_MODEL (GGUF whose
tokenizer is used; default gguf/GLM-5.3-Flash-Q4_K.gguf), MANIFEST_CORPUS_DIR (untracked corpora,
default bench/fidelity/corpus; needs shakespeare.txt), MANIFEST_LONG_WORK (scratch dir for the
tokenized corpora, default <manifest dir>/tok).
The generated prompts, continuations and manifest TSVs are not tracked (see bench/README.md,
"Fixtures"): run this script to rebuild them.  provenance.json is retained from the original
generation; the Shakespeare and Promessi Sposi cases regenerate identically from the same corpora
(same seed, same order), the three repo_docs cases are cut from whatever the public docs contain."""
import json, os, random, re, subprocess, sys
random.seed(20260905)
ROOT = os.environ.get("DS4_ROOT") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
OUT = os.path.join(ROOT, "bench/fidelity/manifest-long")
REL = "bench/fidelity/manifest-long"
WORK = os.environ.get("MANIFEST_LONG_WORK", os.path.join(OUT, "tok"))
DS4 = os.path.join(ROOT, "ds4")
MODEL = os.environ.get("DS4_MODEL", os.path.join(ROOT, "gguf/GLM-5.3-Flash-Q4_K.gguf"))
CORPUS = os.environ.get("MANIFEST_CORPUS_DIR", os.path.join(ROOT, "bench/fidelity/corpus"))
INSTR = "Continue the following text exactly in the same style. Output only the continuation.\n\n"
CONT_TOKENS = 230          # scored continuation, ~200-260 after the word-boundary extension
TOL = 0.005                # prompt token count within +-0.5% of nominal
DEPTHS = {"62k": 62000, "30k": 30720, "12k": 12288, "6k": 6144}   # nominal PROMPT tokens (chat-rendered)
# composition: (source, depth) -> cases.  Sizes in tokens: shakespeare 1,457k, promessi 413k, repo_docs 81k.
PLAN = {
    "shakespeare":       {"62k": 11, "30k": 12, "12k": 16, "6k": 16},
    "promessi_sposi_it": {"62k": 2,  "30k": 4,  "12k": 8,  "6k": 6},
    "repo_docs":         {"62k": 1,  "30k": 0,  "12k": 0,  "6k": 2},
}
SOURCES = {  # same corpora as manifest-1k/generate.py (deep-math / story / security are not long enough or gone)
    "shakespeare": os.path.join(CORPUS, "shakespeare.txt"),
    "promessi_sposi_it": os.path.join(ROOT, "speed-bench/promessi_sposi.txt"),
    "repo_docs": None,
}
def clean(t):
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    return re.sub(r"[ \t]+\n", "\n", t)
def load(label):
    if label == "repo_docs":   # the repository's top-level Markdown plus the bench and gguf-tools READMEs
        parts = []
        for f in sorted(os.listdir(ROOT)):
            if f.endswith(".md"):
                parts.append(open(os.path.join(ROOT, f), encoding="utf-8", errors="replace").read())
        for f in ["bench/README.md", "bench/FIDELITY.md",
                  "gguf-tools/README.md", "gguf-tools/quality-testing/README.md"]:
            p = os.path.join(ROOT, f)
            if os.path.exists(p): parts.append(open(p, encoding="utf-8", errors="replace").read())
        return "\n\n".join(parts)
    return open(SOURCES[label], encoding="utf-8", errors="replace").read()
def bytes_to_unicode():   # GPT-2 byte-level BPE alphabet; the vocab strings use it
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]; n = 0
    for b in range(256):
        if b not in bs: bs.append(b); cs.append(256 + n); n += 1
    return dict(zip(bs, [chr(c) for c in cs]))
U2B = {v: k for k, v in bytes_to_unicode().items()}
def dump_tokens(path, chat):
    env = dict(os.environ, DS4_METAL_MODEL_UNTRACKED="1")
    cmd = [DS4, "-m", MODEL, "--dump-tokens", "--prompt-file", path] + (["--nothink", "--system", ""] if chat else ["--raw"])
    r = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True)
    if r.returncode != 0: sys.exit(f"tokenizer failed on {path}: {r.stderr.decode(errors='replace')[-400:]}")
    return r.stdout
def tokenize_corpus(label, text):
    """token boundaries as byte offsets into the cleaned utf-8 corpus: offs[j] = start of token j."""
    src = text.encode("utf-8")
    txt = os.path.join(WORK, label + ".txt"); tok = os.path.join(WORK, label + ".tok")
    if not (os.path.exists(txt) and open(txt, "rb").read() == src): open(txt, "wb").write(src)
    if not os.path.exists(tok) or os.path.getmtime(tok) < os.path.getmtime(txt):
        open(tok, "wb").write(dump_tokens(txt, chat=False))
    raw = open(tok, "rb").read()
    nl = raw.index(b"\n"); ids = json.loads(raw[:nl]); lines = raw[nl + 1:].split(b"\n")
    offs, p = [0], 0
    for j in range(len(ids)):
        st = lines[j][8:].decode("utf-8")
        b = bytes(U2B[c] for c in st)
        if src[p:p + len(b)] != b:
            # a special-token literal in the text (repo docs quote e.g. <|begin_of_sentence|>): resync on the next token
            nxt = bytes(U2B[c] for c in lines[j + 1][8:].decode("utf-8")) if j + 1 < len(ids) else b""
            q = src.find(nxt, p, p + 256) if nxt else -1
            if q < 0: sys.exit(f"{label}: cannot align token {j} ({st!r}) at byte {p}")
            p = q; offs.append(p); continue
        p += len(b); offs.append(p)
    if p != len(src): sys.exit(f"{label}: reconstructed {p} of {len(src)} bytes")
    return src, offs, ids, lines
def word_start(lines, j):   # token j begins a word (space) or a line (newline)
    st = lines[j][8:9]
    return st in (b"\xc4",) and lines[j][8:10].decode("utf-8")[0] in ("Ġ", "Ċ")
def next_word_boundary(lines, j, n):
    while j < n and not word_start(lines, j): j += 1
    return j
def count_chat(prompt_text):
    tmp = os.path.join(WORK, "probe.txt"); open(tmp, "w", encoding="utf-8").write(prompt_text)
    return len(json.loads(dump_tokens(tmp, chat=True).split(b"\n", 1)[0]))
def main():
    os.makedirs(WORK, exist_ok=True)
    os.makedirs(os.path.join(OUT, "prompts"), exist_ok=True); os.makedirs(os.path.join(OUT, "continuations"), exist_ok=True)
    overhead = count_chat(INSTR + "x") - 1      # chat template + instruction tokens (one probe)
    print(f"template+instruction overhead: {overhead} tokens", file=sys.stderr)
    cases, seen_prefix, counters = [], set(), {d: 0 for d in DEPTHS}
    for label in PLAN:
        src, offs, ids, lines = tokenize_corpus(label, clean(load(label)))
        n = len(ids)
        print(f"{label}: {n} tokens, {len(src)} bytes", file=sys.stderr)
        order = [d for d in DEPTHS for _ in range(PLAN[label][d])]
        random.shuffle(order)
        j = 0     # next free token index
        for depth in order:
            N = DEPTHS[depth]
            # start at a line start that is a token boundary
            while j < n and not (offs[j] > 0 and src[offs[j] - 1:offs[j]] == b"\n" or offs[j] == 0): j += 1
            if j >= n: sys.exit(f"{label}: corpus exhausted before {depth} case {counters[depth]}")
            want = N - overhead
            for attempt in range(4):
                e = next_word_boundary(lines, j + want, n)
                c = next_word_boundary(lines, e + CONT_TOKENS, n)
                if c >= n: sys.exit(f"{label}: corpus exhausted inside {depth} case {counters[depth]}")
                prompt_txt = src[offs[j]:offs[e]].decode("utf-8")
                cont_txt = src[offs[e]:offs[c]].decode("utf-8")
                got = count_chat(INSTR + prompt_txt)
                if abs(got - N) <= TOL * N: break
                want += N - got
            else:
                sys.exit(f"{label} {depth} case {counters[depth]}: {got} tokens after 4 attempts (want {N})")
            key = prompt_txt[:1024]
            if key in seen_prefix: sys.exit(f"{label}: duplicate 1024-char prefix at token {j}")
            seen_prefix.add(key)
            cid = f"long_{depth}_{counters[depth]:03d}"; counters[depth] += 1
            with open(os.path.join(OUT, "prompts", cid + ".txt"), "w", encoding="utf-8") as f: f.write(INSTR + prompt_txt)
            with open(os.path.join(OUT, "continuations", cid + ".txt"), "w", encoding="utf-8") as f: f.write(cont_txt)
            cases.append({"id": cid, "depth": depth, "nominal_prompt_tokens": N, "prompt_tokens": got,
                          "cont_tokens_source": c - e, "source": label, "token_start": j, "token_end": e,
                          "byte_offset": offs[j], "prompt_chars": len(prompt_txt), "cont_chars": len(cont_txt)})
            print(f"  {cid} {label} tokens {j}-{e} prompt={got} (nominal {N}) cont={c - e}", file=sys.stderr)
            j = c
    # manifest order: round-robin over depths so an interrupted run still covers every depth
    by_depth = {d: [c for c in cases if c["depth"] == d] for d in DEPTHS}
    rows = []
    while any(by_depth.values()):
        for d in DEPTHS:
            if by_depth[d]: rows.append(by_depth[d].pop(0))
    def line(c): return f"{c['id']}\t{REL}/prompts/{c['id']}.txt\t{REL}/continuations/{c['id']}.txt\t"
    with open(os.path.join(OUT, "manifest.tsv"), "w") as f:
        f.write("# id\tprompt_file\tcontinuation_file\tresponse_file\n"); f.write("\n".join(line(c) for c in rows) + "\n")
    with open(os.path.join(OUT, "manifest-smoke.tsv"), "w") as f:   # one case per depth: the first of each
        f.write("# id\tprompt_file\tcontinuation_file\tresponse_file\n"); f.write("\n".join(line(c) for c in rows[:len(DEPTHS)]) + "\n")
    with open(os.path.join(OUT, "provenance.json"), "w") as f: json.dump(cases, f, indent=1)
    print(f"total cases: {len(cases)} " + " ".join(f"{d}={counters[d]}" for d in DEPTHS), file=sys.stderr)
if __name__ == "__main__": main()
