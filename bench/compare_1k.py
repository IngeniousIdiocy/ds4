#!/usr/bin/env python3
"""Prefill Tier 2 gate, criterion 2 (bench/FIDELITY.md): equivalence of a candidate scorer TSV
with its same-build control on a manifest, plus the cumulative-drift check on the 100-prompt set.
usage: compare_1k.py [--manifest M.tsv] CONTROL.tsv CANDIDATE.tsv [CONTROL_100.tsv CANDIDATE_100.tsv]
Default manifest: bench/fidelity/manifest-1k/manifest.tsv (the long manifest is passed explicitly).
Both arms must hold every manifest id exactly once with positive prompt/target counts that MATCH across the
arms per id and finite metrics;
anything else is INCOMPLETE (exit 2) - a partial run is never a smaller assessment.
Ids of the form <stratum>_<n> are also broken down per stratum (reported, not gating).
Exit 0 PASS, 1 FAIL, 2 incomplete or invalid input."""
import csv, math, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
MANIFEST_1K = os.path.join(HERE, "fidelity/manifest-1k/manifest.tsv")
MANIFEST_100 = os.path.join(HERE, "../gguf-tools/quality-testing/data/glm53-flash-openrouter-zai-fp8-100/manifest.tsv")
PIN_100 = 0.299642280      # clean upstream ab06d19 on the 100-prompt OpenRouter set (FIDELITY.md)
BUDGET_100 = 3e-4
class Incomplete(Exception): pass
def manifest_ids(path):
    ids = []
    with open(path, encoding="utf-8") as fp:
        for line in fp:
            if line.strip() and not line.startswith("#"): ids.append(line.split("\t")[0].strip())
    if len(set(ids)) != len(ids): raise Incomplete(f"{path}: duplicate ids in the manifest")
    return ids
def load(path, expected):
    """rows keyed by id: (avg_nll, target_tokens, nll, first_match, prompt_tokens); every expected id exactly once."""
    rows, want = {}, set(expected)
    with open(path, newline="", encoding="utf-8") as fp:
        for r in csv.DictReader(fp, delimiter="\t"):
            i = r["id"]
            if i not in want: raise Incomplete(f"{path}: id {i} is not in the manifest")
            if i in rows: raise Incomplete(f"{path}: id {i} appears more than once")
            tt, nll, avg, fm, pt = int(r["target_tokens"]), float(r["nll"]), float(r["avg_nll"]), int(r["first_match"]), int(r["prompt_tokens"])
            if tt <= 0: raise Incomplete(f"{path}: id {i} has target_tokens={tt}")
            if pt <= 0: raise Incomplete(f"{path}: id {i} has prompt_tokens={pt}")
            if not (math.isfinite(nll) and math.isfinite(avg)): raise Incomplete(f"{path}: id {i} has a non-finite nll")
            if fm not in (0, 1): raise Incomplete(f"{path}: id {i} has first_match={fm}")
            rows[i] = (avg, tt, nll, fm, pt)
    missing = [i for i in expected if i not in rows]
    if missing: raise Incomplete(f"{path}: {len(missing)} of {len(expected)} manifest ids missing (first: {missing[:3]})")
    return rows
def check_pair(a, b, pa, pb):
    """the two arms must have scored the same prompt and the same continuation for every id"""
    for i in a:
        if a[i][4] != b[i][4] or a[i][1] != b[i][1]:
            raise Incomplete(f"id {i}: prompt/target tokens {a[i][4]}/{a[i][1]} in {pa} vs {b[i][4]}/{b[i][1]} in {pb}")
def weighted(rows):
    tok = sum(v[1] for v in rows.values()); return sum(v[2] for v in rows.values()) / tok
def binom_two_sided(w, l):
    n = w + l
    if n == 0: return 1.0
    k = min(w, l); p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * p)
def paired(a, b, ids):
    d = [b[i][0] - a[i][0] for i in ids]
    n = len(d); mean = sum(d) / n
    sd = math.sqrt(sum((x - mean) ** 2 for x in d) / (n - 1)) if n > 1 else float("nan")
    se = sd / math.sqrt(n) if n > 1 else float("nan")
    wins = sum(1 for x in d if x < 0); losses = sum(1 for x in d if x > 0); ties = n - wins - losses
    fm_a = sum(a[i][3] for i in ids); fm_b = sum(b[i][3] for i in ids)
    return n, mean, se, wins, losses, ties, binom_two_sided(wins, losses), fm_a, fm_b
def mean_ok(mean, se):
    """criterion 2: |paired mean delta| <= 2 SE, applied directly. All-zero deltas (mean = SE = 0) are
    exact identity and PASS; a uniform nonzero shift (SE = 0, mean != 0) FAILS; nothing is divided."""
    if not (math.isfinite(mean) and math.isfinite(se)): return False
    if se == 0.0: return mean == 0.0
    return abs(mean) <= 2.0 * se
def ratio(mean, se): return f"{abs(mean) / se:.2f}" if se > 0 else "n/a"
def stratum(i): return i.rsplit("_", 1)[0] if "_" in i else i
def run(argv):
    man, man100, args = MANIFEST_1K, MANIFEST_100, []
    it = iter(argv)
    for x in it:
        if x == "--manifest": man = next(it)
        elif x == "--manifest-100": man100 = next(it)
        else: args.append(x)
    if len(args) not in (2, 4): print(__doc__); return 2
    try:
        ids = manifest_ids(man)
        a, b = load(args[0], ids), load(args[1], ids); check_pair(a, b, args[0], args[1])
        if len(args) == 4:
            ids100 = manifest_ids(man100); a100, b100 = load(args[2], ids100), load(args[3], ids100); check_pair(a100, b100, args[2], args[3])
    except (Incomplete, KeyError, ValueError) as e:
        print(f"CRITERION 2 VERDICT: INCOMPLETE ({e})"); return 2
    n, mean, se, w, l, t, p, fm_a, fm_b = paired(a, b, ids)
    ok1 = mean_ok(mean, se); ok2 = p > 0.05; ok3 = abs(fm_b - fm_a) <= 1
    print(f"{os.path.basename(os.path.dirname(man))}: cases={n} control_avg_nll={weighted(a):.9f} candidate_avg_nll={weighted(b):.9f}")
    print(f"  paired mean delta {mean:+.3e}  SE {se:.3e}  |delta|/SE {ratio(mean, se)}  -> {'PASS' if ok1 else 'FAIL'} (|delta| <= 2 SE)")
    print(f"  wins/losses/ties {w}/{l}/{t}  binomial p {p:.3f}  -> {'PASS' if ok2 else 'FAIL'} (> 0.05)")
    print(f"  first_match {fm_a} -> {fm_b} of {n}  -> {'PASS' if ok3 else 'FAIL'} (within +-1)")
    strata = sorted(set(stratum(i) for i in ids))
    if len(strata) > 1:
        for s in strata:
            sid = [i for i in ids if stratum(i) == s]
            sn, sm, sse, sw, sl, st, sp, sfa, sfb = paired(a, b, sid)
            print(f"  stratum {s}: n={sn} mean delta {sm:+.3e} SE {sse:.3e} |delta|/SE {ratio(sm, sse)} "
                  f"wins/losses/ties {sw}/{sl}/{st} p {sp:.3f} first_match {sfa} -> {sfb} (reported, not gating)")
    verdict = ok1 and ok2 and ok3
    if len(args) == 4:
        wa, wb = weighted(a100), weighted(b100)
        drift = wb - PIN_100
        ok4 = abs(drift) <= BUDGET_100
        print(f"100-prompt set: control {wa:.9f} candidate {wb:.9f} cumulative vs upstream pin {drift:+.3e} ({100*abs(drift)/BUDGET_100:.1f}% of budget) -> {'PASS' if ok4 else 'FAIL'}")
        verdict = verdict and ok4
    print("CRITERION 2 VERDICT:", "PASS" if verdict else "FAIL")
    return 0 if verdict else 1
if __name__ == "__main__": sys.exit(run(sys.argv[1:]))
