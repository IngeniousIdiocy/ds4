#!/usr/bin/env python3
"""Build the 890-case equivalence manifest for the prefill Tier 2 gate (bench/FIDELITY.md).
Each case: prompt = a 'continue this passage' user message carrying a passage of 1,200-3,600
characters (~300-900 tokens); continuation = the next ~800 characters (~190-220 tokens) of the same
source. Sources are local public-domain / repo text; windows never overlap within a source, and no
two cases share a 256-token prefix. Provenance is recorded per case. Deterministic (seeded).
Corpora not tracked in the repository are read from MANIFEST_CORPUS_DIR (default
bench/fidelity/corpus): shakespeare.txt (Project Gutenberg, The Complete Works of William
Shakespeare) and deep-math.txt.  DS4_ROOT overrides the repository root."""
import os, random, re, sys, json
random.seed(20260903)
ROOT = os.environ.get("DS4_ROOT") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
OUT = os.path.join(ROOT, "bench/fidelity/manifest-1k")
CORPUS = os.environ.get("MANIFEST_CORPUS_DIR", os.path.join(ROOT, "bench/fidelity/corpus"))
SOURCES = [  # (label, path, share)
    ("shakespeare", os.path.join(CORPUS, "shakespeare.txt"), 520),
    ("promessi_sposi_it", os.path.join(ROOT, "speed-bench/promessi_sposi.txt"), 220),
    ("deep_math", os.path.join(CORPUS, "deep-math.txt"), 50),
    ("story_prompt", os.path.join(ROOT, "tests/long_context_story_prompt.txt"), 50),
    ("security_prompt", os.path.join(ROOT, "tests/long_context_security_prompt.txt"), 50),
    # A sixth source, "repo_docs" (110 cases cut from the repository's own Markdown, ids
    # case_0890-case_0999), was generated last and has been retired: the documents it
    # sampled were campaign working notes that are not part of the public tree.  Because
    # it came last in this list, the 890 cases above are reproduced unchanged by this
    # script (the final shuffle now acts on 890 rows, so manifest.tsv's row order may
    # differ from the tracked file); retained fidelity TSVs may still carry the retired ids.
]
def load(label, path):
    return open(path, encoding="utf-8", errors="replace").read()
def clean(t):
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"[ \t]+\n", "\n", t)
    return t
os.makedirs(os.path.join(OUT, "prompts"), exist_ok=True)
os.makedirs(os.path.join(OUT, "continuations"), exist_ok=True)
rows, prov, seen_prefix = [], [], set()
case = 0
for label, path, share in SOURCES:
    text = clean(load(label, path))
    n = len(text)
    # candidate windows: prompt len P in [1200,3600] chars, continuation C = 800 chars; non-overlapping
    pos, made, tries = 0, 0, 0
    while made < share and tries < share * 20:
        tries += 1
        P = random.choice([1200, 1600, 2000, 2400, 3000, 3600])
        C = 800
        if pos + P + C > n:
            pos = 0  # wrap once; the prefix check prevents duplicates
        if pos == 0:
            start = 0
        else:
            nl = text.find("\n", pos)
            start = nl + 1 if nl >= 0 else -1
        if start < 0 or start + P + C > n:
            pos = 0; continue
        prompt_txt = text[start:start + P]
        # cut prompt at a word boundary, continuation starts exactly after it
        cut = prompt_txt.rfind(" ")
        if cut < P // 2: pos = start + P + C; continue
        prompt_txt = prompt_txt[:cut]
        cont_txt = text[start + cut:start + cut + C]
        cut2 = cont_txt.rfind(" ")
        if cut2 < C // 2: pos = start + P + C; continue
        cont_txt = cont_txt[:cut2]
        key = prompt_txt[:1024]
        if key in seen_prefix or len(prompt_txt.strip()) < 600 or len(cont_txt.strip()) < 300:
            pos = start + P + C; continue
        seen_prefix.add(key)
        cid = f"case_{case:04d}"
        with open(os.path.join(OUT, "prompts", cid + ".txt"), "w", encoding="utf-8") as f:
            f.write("Continue the following text exactly in the same style. Output only the continuation.\n\n" + prompt_txt)
        with open(os.path.join(OUT, "continuations", cid + ".txt"), "w", encoding="utf-8") as f:
            f.write(cont_txt)
        rel = "bench/fidelity/manifest-1k"
        rows.append(f"{cid}\t{rel}/prompts/{cid}.txt\t{rel}/continuations/{cid}.txt\t")
        prov.append({"id": cid, "source": label, "offset": start, "prompt_chars": len(prompt_txt), "cont_chars": len(cont_txt)})
        case += 1; made += 1
        pos = start + P + C
    print(f"{label}: {made} cases (wanted {share}), source {n} chars", file=sys.stderr)
random.shuffle(rows)
with open(os.path.join(OUT, "manifest.tsv"), "w") as f:
    f.write("# id\tprompt_file\tcontinuation_file\tresponse_file\n")
    f.write("\n".join(rows) + "\n")
with open(os.path.join(OUT, "provenance.json"), "w") as f:
    json.dump(prov, f, indent=1)
print(f"total cases: {len(rows)}", file=sys.stderr)
