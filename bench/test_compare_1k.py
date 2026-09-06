#!/usr/bin/env python3
"""Unit tests for compare_1k.py: exact identity passes, partial/duplicate/invalid runs are INCOMPLETE,
first_match tolerance is +-1 at any n. Run: python3 bench/test_compare_1k.py"""
import io, os, sys, tempfile, unittest, contextlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compare_1k
HDR = "id\tprompt_tokens\ttarget_tokens\tnll\tavg_nll\tfirst_match\tgreedy_lcp\n"
class T(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp(); self.n = 0
    def path(self, name): return os.path.join(self.d, name)
    def manifest(self, ids):
        p = self.path(f"man{self.n}.tsv"); self.n += 1
        with open(p, "w") as f:
            f.write("# id\tprompt_file\tcontinuation_file\tresponse_file\n")
            for i in ids: f.write(f"{i}\tp\tc\t\n")
        return p
    def tsv(self, rows):
        """rows: list of (id, target_tokens, avg_nll, first_match[, prompt_tokens]); nll = avg * tokens"""
        p = self.path(f"arm{self.n}.tsv"); self.n += 1
        with open(p, "w") as f:
            f.write(HDR)
            for r in rows:
                i, tt, avg, fm = r[:4]; pt = r[4] if len(r) > 4 else 500
                f.write(f"{i}\t{pt}\t{tt}\t{avg * tt:.9f}\t{avg:.9f}\t{fm}\t0\n")
        return p
    def run_cmp(self, man, a, b):
        out = io.StringIO()
        with contextlib.redirect_stdout(out): rc = compare_1k.run(["--manifest", man, a, b])
        return rc, out.getvalue()
    def rows(self, ids, avg=1.0, fm=0): return [(i, 230, avg, fm) for i in ids]
    def test_exact_identity_passes(self):
        ids = [f"long_6k_{k:03d}" for k in range(6)] + [f"long_62k_{k:03d}" for k in range(6)]
        a = self.tsv(self.rows(ids)); b = self.tsv(self.rows(ids))
        rc, out = self.run_cmp(self.manifest(ids), a, b)
        self.assertEqual(rc, 0, out); self.assertIn("|delta|/SE n/a", out); self.assertIn("VERDICT: PASS", out)
        self.assertIn("stratum long_62k", out)
    def test_partial_run_is_incomplete(self):
        ids = [f"case_{k:04d}" for k in range(10)]
        a = self.tsv(self.rows(ids)); b = self.tsv(self.rows(ids[:-1]))
        rc, out = self.run_cmp(self.manifest(ids), a, b)
        self.assertEqual(rc, 2, out); self.assertIn("INCOMPLETE", out); self.assertIn("1 of 10 manifest ids missing", out)
    def test_duplicate_id_is_incomplete(self):
        ids = [f"case_{k:04d}" for k in range(4)]
        a = self.tsv(self.rows(ids)); b = self.tsv(self.rows(ids) + [(ids[0], 230, 1.0, 0)])
        rc, out = self.run_cmp(self.manifest(ids), a, b)
        self.assertEqual(rc, 2, out); self.assertIn("more than once", out)
    def test_zero_targets_and_nonfinite_are_incomplete(self):
        ids = [f"case_{k:04d}" for k in range(4)]
        a = self.tsv(self.rows(ids))
        rc, out = self.run_cmp(self.manifest(ids), a, self.tsv([(ids[0], 0, 1.0, 0)] + self.rows(ids[1:])))
        self.assertEqual(rc, 2, out); self.assertIn("target_tokens=0", out)
        rc, out = self.run_cmp(self.manifest(ids), a, self.tsv([(ids[0], 230, float("nan"), 0)] + self.rows(ids[1:])))
        self.assertEqual(rc, 2, out); self.assertIn("non-finite", out)
    def test_foreign_id_is_incomplete(self):
        ids = [f"case_{k:04d}" for k in range(4)]
        rc, out = self.run_cmp(self.manifest(ids), self.tsv(self.rows(ids)), self.tsv(self.rows(ids[:-1] + ["case_9999"])))
        self.assertEqual(rc, 2, out); self.assertIn("not in the manifest", out)
    def test_first_match_tolerance_is_one_at_n_1000(self):
        ids = [f"case_{k:04d}" for k in range(1000)]
        base = [(i, 230, 1.0 + (0.001 if k % 2 else -0.001), 0) for k, i in enumerate(ids)]
        cand = [(i, 230, 1.0, 0) for i in ids]   # symmetric +-1e-3 deltas: mean 0
        a = self.tsv(base)
        flip2 = [(i, tt, avg, 1 if k < 2 else fm) for k, (i, tt, avg, fm) in enumerate(cand)]
        rc, out = self.run_cmp(self.manifest(ids), a, self.tsv(flip2))
        self.assertEqual(rc, 1, out); self.assertIn("first_match 0 -> 2 of 1000  -> FAIL", out)
        flip1 = [(i, tt, avg, 1 if k < 1 else fm) for k, (i, tt, avg, fm) in enumerate(cand)]
        rc, out = self.run_cmp(self.manifest(ids), a, self.tsv(flip1))
        self.assertEqual(rc, 0, out); self.assertIn("first_match 0 -> 1 of 1000  -> PASS", out)
    def test_uniform_shift_fails_without_division(self):
        ids = [f"case_{k:04d}" for k in range(8)]
        rc, out = self.run_cmp(self.manifest(ids), self.tsv(self.rows(ids, 1.0)), self.tsv(self.rows(ids, 1.01)))
        self.assertEqual(rc, 1, out); self.assertIn("SE 0.000e+00  |delta|/SE n/a  -> FAIL", out)
    def test_unequal_lengths_are_incomplete(self):
        ids = ["case_0000", "case_0001"]   # 230 targets vs 1 target per case, equal avg nll
        a = self.tsv([(i, 230, 1.0, 0) for i in ids])
        rc, out = self.run_cmp(self.manifest(ids), a, self.tsv([(i, 1, 1.0, 0) for i in ids]))
        self.assertEqual(rc, 2, out); self.assertIn("INCOMPLETE", out); self.assertIn("prompt/target tokens 500/230", out)
        rc, out = self.run_cmp(self.manifest(ids), a, self.tsv([(i, 230, 1.0, 0, 501) for i in ids]))
        self.assertEqual(rc, 2, out); self.assertIn("vs 501/230", out)
        rc, out = self.run_cmp(self.manifest(ids), a, self.tsv([(i, 230, 1.0, 0, 500) for i in ids]))
        self.assertEqual(rc, 0, out)
    def test_mean_ok_direct(self):
        self.assertTrue(compare_1k.mean_ok(0.0, 0.0)); self.assertFalse(compare_1k.mean_ok(1e-3, 0.0))
        self.assertTrue(compare_1k.mean_ok(2e-4, 1e-4)); self.assertFalse(compare_1k.mean_ok(2.1e-4, 1e-4))
        self.assertFalse(compare_1k.mean_ok(float("nan"), float("nan"))); self.assertFalse(compare_1k.mean_ok(0.0, float("nan")))
if __name__ == "__main__": unittest.main()
