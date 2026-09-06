#!/usr/bin/env python3
"""Quick depth benchmark: 50k and 100k context, baseline / serial / dflash.

Contexts are agent-shaped filler rendered from a coding-agent session
transcript (QUICKBENCH_TRANSCRIPT, a JSONL file of user/assistant/tool
events); requests go through the real server path (:8095) with z.ai
bench-parity reasoning (glm-5.3-flash-reasoner alias = thinking high, server
sampling defaults). Decode t/s comes from the server log's decode-only
averages.

Safety: if QUICKBENCH_LAUNCHD_LABEL names a launchd service running a
production ds4-server, it is booted out first and restored at the end
regardless of outcome; a memory_pressure gate (>=40% free) runs between
model loads.

Environment (all optional; defaults are relative to the repository):
  QUICKBENCH_PROD_DIR      tree holding the deployed build and gguf/ (default: repo root)
  QUICKBENCH_BIN_DIR       candidate build dir (default: QUICKBENCH_PROD_DIR)
  QUICKBENCH_GGUF          model file (default: <prod>/gguf/GLM-5.3-Flash-Q4_K.gguf)
  QUICKBENCH_BASE_GGUF     pre-campaign model file for the `baseline` config (same default)
  QUICKBENCH_DFLASH_GGUF   DFlash2 drafter GGUF (default: <prod>/dflash2/GLM-5.3-Flash-DFlash2.gguf)
  QUICKBENCH_BASELINE_DIR  build dir of the pre-campaign commit for `baseline`
  QUICKBENCH_UPSTREAM_DIR  build dir of upstream for `upstream` / `upstream_mtp`
  QUICKBENCH_TRANSCRIPT    session JSONL used to build the corpus (default: bench/corpus/session.jsonl)
  QUICKBENCH_LAUNCHD_LABEL launchd label of a production server to boot out (default: none)
  QUICKBENCH_LAUNCHD_PLIST plist path used to restore it (required with the label)
  QUICKBENCH_BUCKETS       comma list of 50k,100k
"""
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request

BENCH = pathlib.Path(__file__).resolve().parent
REPO = BENCH.parent
PROD_DIR = pathlib.Path(os.environ.get("QUICKBENCH_PROD_DIR", str(REPO)))
TRANSCRIPT = pathlib.Path(os.environ.get(
    "QUICKBENCH_TRANSCRIPT", str(BENCH / "corpus/session.jsonl")))
BASE = "http://127.0.0.1:8095"
CHARS_PER_TOK = 3.6
BUCKETS = [("50k", 50_000), ("100k", 100_000)]
if os.environ.get("QUICKBENCH_BUCKETS"):
    _want = os.environ["QUICKBENCH_BUCKETS"].split(",")
    BUCKETS = [b for b in BUCKETS if b[0] in _want]
SEED = 12345
MAX_TOKENS = 1500
R_INC = 5
# Production launchd service to boot out during the run (e.g. a label
# "com.example.ds4-glm" -> "gui/<uid>/com.example.ds4-glm"); unset = none.
_LABEL = os.environ.get("QUICKBENCH_LAUNCHD_LABEL", "")
GLM_LABEL = ("gui/%d/%s" % (os.getuid(), _LABEL)) if _LABEL else ""
GLM_PLIST = os.environ.get("QUICKBENCH_LAUNCHD_PLIST", "")

PROD_GGUF = os.environ.get("QUICKBENCH_GGUF",
                           str(PROD_DIR / "gguf/GLM-5.3-Flash-Q4_K.gguf"))
BASE_GGUF = os.environ.get("QUICKBENCH_BASE_GGUF", PROD_GGUF)
DFLASH_GGUF = os.environ.get("QUICKBENCH_DFLASH_GGUF",
                             str(PROD_DIR / "dflash2/GLM-5.3-Flash-DFlash2.gguf"))
# QUICKBENCH_BIN_DIR: benchmark a candidate build (e.g. the eval worktree)
# instead of the deployed prod binary; the server needs that dir as cwd for
# its Metal shader sources.
CAND_DIR = pathlib.Path(os.environ.get("QUICKBENCH_BIN_DIR", str(PROD_DIR)))
PROD_BIN = str(CAND_DIR / "ds4-server")
BASELINE_DIR = os.environ.get("QUICKBENCH_BASELINE_DIR", "/tmp/ds4-glm-baseline")
BASE_BIN = BASELINE_DIR + "/ds4-server"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def mem_free_pct():
    out = subprocess.run(["memory_pressure"], capture_output=True,
                         text=True, timeout=30).stdout
    m = re.search(r"free percentage: (\d+)%", out)
    return int(m.group(1)) if m else 0


def wait_mem(min_pct=40):
    for _ in range(60):
        pct = mem_free_pct()
        if pct >= min_pct:
            log(f"memory free {pct}% (>= {min_pct}%), proceeding")
            return
        log(f"waiting for memory to settle: free {pct}%")
        time.sleep(5)
    raise RuntimeError("memory never settled")


def stop_servers():
    if GLM_LABEL:
        subprocess.run(["launchctl", "bootout", GLM_LABEL],
                       capture_output=True, timeout=30)
    for _ in range(120):
        r = subprocess.run(["pgrep", "-f", "ds4-server"],
                           capture_output=True, text=True)
        if r.returncode != 0:
            return
        time.sleep(1)
    raise RuntimeError("a ds4-server would not exit")


def wait_ready(timeout=900):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(BASE + "/v1/models", timeout=3) as r:
                if r.status == 200:
                    return
        except Exception:
            pass
        time.sleep(3)
    raise RuntimeError("bench server never became ready")


def post_chat(messages):
    body = {"model": "glm-5.3-flash-reasoner", "messages": messages,
            "max_tokens": MAX_TOKENS, "seed": SEED, "stream": False}
    req = urllib.request.Request(
        BASE + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=3600) as r:
        resp = json.load(r)
    wall = time.time() - t0
    msg = resp["choices"][0]["message"]
    text = (msg.get("content") or "")
    usage = resp.get("usage", {})
    return text, usage, wall


# ---------------------------------------------------------------- corpus

def transcript_blocks():
    """Ordered (role, text) blocks from the session transcript."""
    blocks = []
    if not TRANSCRIPT.exists():
        raise SystemExit(f"corpus transcript {TRANSCRIPT} not found; set "
                         "QUICKBENCH_TRANSCRIPT to a session JSONL")
    with open(TRANSCRIPT, encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                ev = json.loads(line)
            except ValueError:
                continue
            t = ev.get("type")
            d = ev.get("data") or {}
            if t == "user/message":
                parts = d.get("content") or []
                txt = "\n".join(p.get("text", "") for p in parts
                                if isinstance(p, dict))
                if txt.strip():
                    blocks.append(("user", txt))
            elif t == "assistant/message":
                parts = (d.get("message") or {}).get("content") or []
                out = []
                for p in parts:
                    if not isinstance(p, dict):
                        continue
                    if p.get("reasoning"):
                        out.append(p["reasoning"])
                    if p.get("text"):
                        out.append(p["text"])
                    if p.get("type") == "tool-call" or p.get("toolName"):
                        out.append("[tool call] %s(%s)" % (
                            p.get("toolName", "tool"),
                            json.dumps(p.get("args", {}))[:400]))
                txt = "\n".join(out)
                if txt.strip():
                    blocks.append(("assistant", txt))
            elif t == "tool/result":
                parts = (d.get("message") or {}).get("content") or []
                out = []
                for p in parts:
                    if isinstance(p, dict):
                        for q in p.get("content") or []:
                            if isinstance(q, dict) and q.get("text"):
                                out.append(q["text"])
                txt = "\n".join(out)
                if txt.strip():
                    blocks.append(("tool", txt))
    return blocks


def build_corpus():
    blocks = transcript_blocks()
    log(f"transcript blocks: {len(blocks)}")
    system = ("You are a coding agent working in a web-game repository "
              "(three.js third-person shooter). You edit files, run tests, "
              "and iterate on play-test feedback.")

    def render(target_tokens, start):
        """Alternating user/assistant messages from blocks[start:], sized
        to ~target_tokens. Tool results fold into user messages. Returns
        (messages, next_index)."""
        budget = int(target_tokens * CHARS_PER_TOK)
        msgs = [{"role": "system", "content": system}]
        used = len(system)
        pend_user = []
        i = start
        while used < budget and i < len(blocks):
            role, txt = blocks[i]
            txt = txt[:60_000]
            if role in ("user", "tool"):
                pend_user.append(txt)
            else:
                if pend_user:
                    joined = "\n\n".join(pend_user)
                    msgs.append({"role": "user", "content": joined})
                    used += len(joined)
                    pend_user = []
                else:
                    msgs.append({"role": "user", "content": "(continue)"})
                msgs.append({"role": "assistant", "content": txt})
                used += len(txt)
            i += 1
        # close with a real-feeling user ask so generation is agentic
        msgs.append({"role": "user", "content":
                     "Play-test feedback: aiming feels floaty on iPhone and "
                     "headshots on the large NPCs still miss high. Diagnose "
                     "the likely cause in the hit-scan code and propose the "
                     "exact fix, step by step."})
        return msgs, i

    corpora = {}
    idx = 0
    for name, target in BUCKETS:
        msgs, idx = render(target, idx)
        chars = sum(len(m["content"]) for m in msgs)
        corpora[name] = msgs
        log(f"bucket {name}: {len(msgs)} messages, {chars} chars "
            f"(~{int(chars / CHARS_PER_TOK)} tok est)")

    incs = [t[:400] for r, t in blocks if r == "tool" and len(t) > 200][:40]
    if len(incs) < R_INC * len(BUCKETS):
        incs += ["$ npm test\n... 42 passing, 1 failing: hitbox raycast "
                 "returns miss for scaled NPC head collider"] * 10
    return corpora, incs


# ---------------------------------------------------------------- configs

UPSTREAM_DIR = os.environ.get("QUICKBENCH_UPSTREAM_DIR", "/tmp/ds4-glm-upstream")


def config_cwd(name):
    if name == "baseline":
        return BASELINE_DIR
    if name in ("upstream", "upstream_mtp"):
        return UPSTREAM_DIR
    return str(CAND_DIR)


def config_cmd(name, kv_dir):
    common = ["--metal", "--host", "127.0.0.1", "--port", "8095",
              "-c", "409600", "--kv-disk-dir", kv_dir,
              "--kv-disk-space-mb", "8192"]
    if name == "baseline":
        return [BASE_BIN, "-m", BASE_GGUF] + common
    if name == "upstream":
        # upstream engine on OUR artifact: isolates engine work from quant
        return [UPSTREAM_DIR + "/ds4-server", "-m", PROD_GGUF] + common
    if name == "upstream_mtp":
        # upstream engine with the deployed speculative-session flags: does
        # the MTP session path (not fork code) explain deep-ctx cache reuse?
        return [UPSTREAM_DIR + "/ds4-server", "-m", PROD_GGUF] + common + \
            ["--mtp", "--mtp-exact-sampling"]
    if name == "serial_plain":
        # our engine, plain session path (no --mtp/--dflash flags at all)
        return [PROD_BIN, "-m", PROD_GGUF] + common
    cmd = [PROD_BIN, "-m", PROD_GGUF] + common + \
        ["--dflash", DFLASH_GGUF, "--mtp", "--mtp-exact-sampling"]
    return cmd


def run_config(name, corpora, incs, results_dir):
    logpath = results_dir / f"server-{name}.log"
    kv_dir = tempfile.mkdtemp(prefix=f"bench-kv-{name}-")
    env = dict(os.environ,
               DS4_METAL_MODEL_UNTRACKED="1",
               DS4_GLM53_MEMORY_CEILING_GB="280")
    if name == "serial":
        env["DS4_DFLASH_DISABLE"] = "1"
    stop_servers()
    wait_mem()
    log(f"[{name}] launching server")
    with open(logpath, "w") as lf:
        proc = subprocess.Popen(config_cmd(name, kv_dir),
                                stdout=lf, stderr=subprocess.STDOUT,
                                env=env, cwd=config_cwd(name))
    try:
        wait_ready()
        log(f"[{name}] server ready")
        rows = {}
        inc_i = 0
        for bname, _ in BUCKETS:
            msgs = list(corpora[bname])
            reqs = []
            for step in range(R_INC + 1):
                text, usage, wall = post_chat(msgs)
                reqs.append({"step": step, "usage": usage, "wall": wall,
                             "text_tail": text[-120:]})
                log(f"[{name}] {bname} step {step}: "
                    f"prompt={usage.get('prompt_tokens')} "
                    f"gen={usage.get('completion_tokens')} wall={wall:.1f}s")
                if text.strip():
                    msgs.append({"role": "assistant", "content": text})
                else:
                    # thinking-only turn: the server's live context (after
                    # its thinking-checkpoint rewind) holds the prompt with
                    # nothing appended, so sending an empty assistant turn
                    # would token-mismatch and force a full re-prefill.
                    log(f"[{name}] {bname} step {step}: thinking-only "
                        "reply; skipping assistant append")
                msgs.append({"role": "user", "content":
                             "Tool result:\n" + incs[inc_i % len(incs)] +
                             "\nContinue with the fix."})
                inc_i += 1
            rows[bname] = reqs
        return rows, logpath
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
        shutil.rmtree(kv_dir, ignore_errors=True)


# ---------------------------------------------------------------- scoring

def parse_server_log(logpath):
    """Sequential per-request records: (decode_avg_tps, prefill_span,
    gen_tokens, elapsed)."""
    reqs = []
    last_avg = None
    for line in open(logpath, encoding="utf-8", errors="replace"):
        m = re.search(r"chat ctx=(\d+)\.\.(\d+):(\d+) gen=(\d+).*"
                      r"decoding .*avg=([0-9.]+) t/s", line)
        if m:
            last_avg = float(m.group(5))
            continue
        m = re.search(r"chat ctx=(\d+)\.\.(\d+):(\d+) gen=(\d+) .*"
                      r"finish=(\S+) ([0-9.]+)s", line)
        if m:
            reqs.append({"span_a": int(m.group(1)),
                         "span_n": int(m.group(3)),
                         "gen": int(m.group(4)),
                         "decode_tps": last_avg,
                         "elapsed": float(m.group(6))})
            last_avg = None
    return reqs


def parse_prefill(logpath):
    """Cold-prefill avg t/s per request order (last prefill avg before each
    finish; None for requests with no prefill lines)."""
    out = []
    last_pf = None
    for line in open(logpath, encoding="utf-8", errors="replace"):
        m = re.search(r"prefill chunk .*avg=([0-9.]+) t/s", line)
        if m:
            last_pf = float(m.group(1))
            continue
        if re.search(r"finish=\S+ [0-9.]+s", line):
            out.append(last_pf)
            last_pf = None
    return out


LABELS = {
    "upstream": "Upstream main ab06d19, our GGUF (serial)",
    "upstream_mtp": "Upstream head + --mtp flags, our GGUF",
    "serial_plain": "Ours, plain serial",
    "baseline": "Pre-campaign (Aug 29 am, Q4_K)",
    "serial": "Ours, MTP-2 speculation (--mtp)",
    "dflash": "Ours, DFlash2 speculative",
}


def score(all_rows, results_dir):
    n_per_bucket = R_INC + 1
    out = ["# Depth benchmark — agent-shaped contexts, thinking high, "
           "seed %d\n" % SEED]
    for bidx, (bname, _) in enumerate(BUCKETS):
        out.append(f"\n## {bname} context\n")
        out.append("| config | decode t/s | prefill t/s |")
        out.append("|---|---|---|")
        for cname, (rows, logpath) in all_rows:
            parsed = parse_server_log(logpath)
            pf = parse_prefill(logpath)
            seg = parsed[bidx * n_per_bucket:(bidx + 1) * n_per_bucket]
            seg_pf = pf[bidx * n_per_bucket:(bidx + 1) * n_per_bucket]
            inc = [r for r in seg[1:] if r["decode_tps"]]
            hot = [r for r in inc if r["span_n"] < 5000]
            miss = len(inc) - len(hot)
            tps = sum(r["decode_tps"] for r in hot) / len(hot) if hot else 0
            cold_pf = seg_pf[0] if seg_pf and seg_pf[0] else 0
            note = f" ({miss} cache miss)" if miss else ""
            out.append(f"| {LABELS.get(cname, cname)} | {tps:.1f}{note} | "
                       f"{cold_pf:.0f} |")
    report = "\n".join(out) + "\n"
    (results_dir / "report.md").write_text(report)
    print("\n" + report)


def restore_production():
    stop_servers()
    if not (GLM_LABEL and GLM_PLIST):
        return
    subprocess.run(["launchctl", "bootstrap", "gui/%d" % os.getuid(),
                    GLM_PLIST], capture_output=True, timeout=30)
    log("production bootstrap requested")


def main():
    results_dir = BENCH / "results" / time.strftime("%Y%m%d-%H%M%S")
    results_dir.mkdir(parents=True)
    corpora, incs = build_corpus()
    (results_dir / "corpus-meta.json").write_text(json.dumps(
        {b: sum(len(m["content"]) for m in msgs)
         for b, msgs in corpora.items()}, indent=2))
    configs = sys.argv[1:] or ["baseline", "serial", "dflash"]
    all_rows = []
    try:
        for name in configs:
            rows, logpath = run_config(name, corpora, incs, results_dir)
            all_rows.append((name, (rows, logpath)))
            (results_dir / f"responses-{name}.json").write_text(
                json.dumps(rows, indent=2))
    finally:
        restore_production()
    score(all_rows, results_dir)
    log(f"done; results in {results_dir}")


if __name__ == "__main__":
    main()
