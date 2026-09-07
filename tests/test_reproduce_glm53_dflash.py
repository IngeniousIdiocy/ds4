#!/usr/bin/env python3
import importlib.util
import pathlib
import sys
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
DRIVER = ROOT / "bench" / "reproduce-glm53-dflash.py"
SPEC = importlib.util.spec_from_file_location("reproduce_glm53_dflash", DRIVER)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class ReproduceGlm53DflashTest(unittest.TestCase):
    def test_packaged_fixtures_match_declared_public_hashes(self):
        fixtures = MODULE.checked_fixtures()
        self.assertEqual({
            "json512", "prose512", "sql512", "middle-chat512",
            "middle-summary512", "middle-rag512", "middle-factual512",
            "middle-math512", "middle-code512",
        }, set(fixtures))
        self.assertEqual({512}, {item["max_generated_tokens"] for item in fixtures.values()})
        middle = [item for name, item in fixtures.items() if name.startswith("middle-")]
        self.assertEqual(6, len(middle))
        self.assertTrue(all(item["natural_eos"] for item in middle))
        self.assertTrue(all("Original" in item["provenance"] for item in middle))

    def test_environment_scrub_removes_all_inherited_ds4_and_mtl_flags(self):
        source = {
            "PATH": "/bin", "DS4_DFLASH_DISABLE_PROPOSER_HEAD_NT4": "1",
            "DS4_DFLASH_FAIL": "drafter_once", "MTL_DEBUG_LAYER": "1",
        }
        env = MODULE.scrubbed_environment({"DS4_DFLASH_STATS": "1"}, source)
        self.assertEqual("/bin", env["PATH"])
        self.assertEqual("1", env["DS4_DFLASH_STATS"])
        self.assertNotIn("DS4_DFLASH_DISABLE_PROPOSER_HEAD_NT4", env)
        self.assertNotIn("DS4_DFLASH_FAIL", env)
        self.assertNotIn("MTL_DEBUG_LAYER", env)

    def test_policy_is_explicit_and_nt4_is_opt_in(self):
        baseline = MODULE.policy_environment(True, False)
        self.assertEqual("0", baseline["DS4_DFLASH_ADAPTIVE"])
        self.assertEqual("1", baseline["DS4_DFLASH_RETRY"])
        self.assertEqual("1", baseline["DS4_DFLASH_SAVINGS_RETRY"])
        self.assertNotIn("DS4_DFLASH_PROPOSER_HEAD_NT4", baseline)
        nt4 = MODULE.policy_environment(True, True)
        self.assertEqual("1", nt4["DS4_DFLASH_PROPOSER_HEAD_NT4"])

    def test_public_speculative_requires_full_block_and_real_ack_alignment(self):
        parser = MODULE.load_evidence_parser(ROOT / "bench/dflash_adaptive_evidence.py")
        text = (ROOT / "tests/fixtures/dflash_adaptive/"
                "public_full_block_partial_eos.log").read_text()
        record = {"exit_code": 0}
        workload = {"tokens": 512, "natural_eos": True}
        MODULE.validate_attempt(record, text, parser, workload, "speculative", True)
        self.assertTrue(record["valid"], record["validation_errors"])
        self.assertEqual(record["engagement"]["receipt_source"],
                         "full-block request ACK accounting")
        self.assertEqual(record["generation_alignment"][0]["ack_consumed_tokens"], 3)

        historical = text.replace("profile=speculative", "profile=aggressive", 1)
        rejected = {"exit_code": 0}
        MODULE.validate_attempt(rejected, historical, parser, workload,
                                "speculative", True)
        self.assertFalse(rejected["valid"])


if __name__ == "__main__":
    unittest.main()
