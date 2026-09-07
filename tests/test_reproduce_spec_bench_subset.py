#!/usr/bin/env python3

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from unittest import mock
import unittest


ROOT = Path(__file__).resolve().parents[1]
DRIVER_PATH = ROOT / "bench/reproduce-spec-bench-subset.py"
PARSER_PATH = ROOT / "bench/dflash_adaptive_evidence.py"


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


driver = load(DRIVER_PATH, "reproduce_spec_bench_subset")
parser = load(PARSER_PATH, "dflash_adaptive_evidence_test")


class SpecBenchSubsetDriverTests(unittest.TestCase):
    def synthetic_dataset(self):
        lines = []
        qid = 1
        categories = ["writing"] * 10 + ["roleplay"] * 2
        for category in categories:
            row = {"question_id": qid, "category": category,
                   "turns": [f"question {qid} a", f"question {qid} b"]}
            lines.append((json.dumps(row) + "\n").encode())
            qid += 1
        for category in ("translation", "summarization", "qa", "math_reasoning", "rag"):
            for _ in range(12):
                row = {"question_id": qid, "category": category,
                       "turns": [f"question {qid}"]}
                lines.append((json.dumps(row) + "\n").encode())
                qid += 1
        return lines

    def test_selection_freezes_72_rows_and_84_turns_without_text(self):
        rows, manifest = driver.select_subset_lines(self.synthetic_dataset())
        self.assertEqual(len(rows), 72)
        self.assertEqual(sum(len(row["turns"]) for row in rows), 84)
        self.assertEqual(manifest["total_turn_requests"], 252)
        self.assertEqual([row["logical_category"] for row in rows[::12]],
                         list(driver.LOGICAL_CATEGORIES))
        encoded = json.dumps(manifest)
        self.assertNotIn("question 1 a", encoded)
        self.assertEqual(manifest["rows"][0]["turn_count"], 2)
        self.assertRegex(manifest["rows"][0]["turns"][0]["sha256"], r"^[0-9a-f]{64}$")

    def test_second_turn_includes_the_actual_first_answer(self):
        messages = driver.messages_for_turn(["first", "second"], ["answer"], 1)
        self.assertEqual(messages, [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "answer"},
            {"role": "user", "content": "second"},
        ])
        with self.assertRaises(driver.ReproductionError):
            driver.messages_for_turn(["first", "second"], [], 1)

    def test_preflight_smoke_keeps_subset_frozen_and_adds_factual_reuse(self):
        rows, manifest = driver.select_subset_lines(self.synthetic_dataset())
        smoke = driver.select_preflight_smoke(rows)
        self.assertEqual(len(rows), 72)
        self.assertEqual(manifest["total_turn_requests"], 252)
        self.assertEqual(len(smoke), 4)
        self.assertEqual(sum(len(row["turns"]) for row in smoke), 5)
        self.assertEqual([row["logical_category"] for row in smoke],
                         ["mt_bench", "qa", "preflight_factual_eos",
                          "preflight_factual_eos"])
        self.assertEqual(smoke[2]["turns"], smoke[3]["turns"])
        self.assertNotEqual(driver.unit_id(smoke[2], 0), driver.unit_id(smoke[3], 0))

    def test_idle_telemetry_precedes_server_and_requires_no_child(self):
        events = []
        class Lock:
            child = None
            def preflight(self):
                events.append("preflight")
        class Telemetry:
            @staticmethod
            def sample_idle_gpu(lock, run_dir, stem):
                events.append(("sample", lock.child, run_dir, stem))
                return {"status": "idle"}
        receipt = driver.sample_idle_before_server(Telemetry, Lock(), Path("mode"))
        self.assertEqual(receipt["status"], "idle")
        self.assertEqual(events, ["preflight", ("sample", None, Path("mode"), "pre-server")])

    def test_public_modes_bind_entry_gate_without_overriding_profile_defaults(self):
        serial = driver.mode_environment("serial", False, 3)
        conservative = driver.mode_environment("conservative", False, 3)
        speculative = driver.mode_environment("speculative", False, 3)
        self.assertEqual(serial, {"DS4_DFLASH_STATS": "1"})
        self.assertEqual(conservative["DS4_DFLASH_MIN_SERIAL_TOKENS"], "3")
        self.assertNotIn("DS4_DFLASH_MIN_SERIAL_TOKENS", speculative)
        for public in (conservative, speculative):
            self.assertNotIn("DS4_DFLASH_RETRY_TAX", public)
            self.assertNotIn("DS4_DFLASH_SAVINGS_RETRY", public)
            self.assertNotIn("DS4_DFLASH_LOSS_BUDGET", public)
        self.assertEqual(driver.mode_environment("speculative", True, 0),
                         {"DS4_DFLASH_STATS": "1"})
        self.assertEqual(driver.mode_environment("conservative", True, 0)[
            "DS4_DFLASH_PROPOSER_HEAD_NT4"], "1")

    def test_server_evidence_can_require_or_omit_cli_counters(self):
        cycle = (
            "ds4: dflash cycle drafted=3 accepted=2 rollback=tail ctx=4 "
            "draft=1.0ms verify=2.0ms heads=0.5ms tail=0.1ms serial=1.5ms "
            "cyc=3.6ms be=1.2\n"
        )
        with self.assertRaises(parser.EvidenceError):
            parser.parse_evidence(cycle)
        evidence = parser.parse_evidence(cycle, require_counters=False)
        self.assertEqual(len(evidence.cycle_receipts), 1)

    def test_conservative_committed_tau_uses_actual_ack_consumption(self):
        text = (ROOT / "tests/fixtures/dflash_adaptive/central_reference.log").read_text()
        result = driver.engagement(parser, text, "conservative")
        evidence = parser.parse_evidence(text, require_adaptive=False)
        request = evidence.requests[0]
        pairs = [(call, ack) for call, ack in zip(request.calls, request.acks)
                 if call["verified"] > 0]
        self.assertEqual(result["committed_tokens"], sum(ack["consumed"] for _, ack in pairs))
        self.assertEqual(result["accepted_draft_tokens"], sum(call["accepted"] for call, _ in pairs))
        self.assertIn("actual consumed", result["committed_metric_source"])

    def test_server_partial_eos_binds_visible_generation_to_ack_consumption(self):
        text = (ROOT / "tests/fixtures/dflash_adaptive/partial_eos.log").read_text()
        result = driver.engagement(parser, text, "conservative", 3, "stop")
        self.assertEqual(result["consumed_visible_rows"], 3)
        self.assertEqual(result["returned_target_rows"], 4)
        self.assertEqual(result["unconsumed_evaluated_suffix"], 1)
        with self.assertRaisesRegex(driver.ReproductionError, "visible generation"):
            driver.engagement(parser, text, "conservative", 4, "stop")
        with self.assertRaisesRegex(driver.ReproductionError, "natural stop"):
            driver.engagement(parser, text, "conservative", 3, "length")

    def test_speculative_committed_tau_is_labeled_conventional(self):
        cycle = (
            "ds4: dflash cycle drafted=3 accepted=2 rollback=tail ctx=4 "
            "draft=1.0ms verify=2.0ms heads=0.5ms tail=0.1ms serial=1.5ms "
            "cyc=3.6ms be=1.2\n"
        )
        result = driver.engagement(parser, cycle, "speculative",
                                   allow_blind_research=True)
        self.assertEqual(result["accepted_draft_tokens_per_verify"], 2)
        self.assertEqual(result["committed_tokens_per_verify"], 3)
        self.assertIn("conventional estimate", result["committed_metric_source"])

    def test_public_speculative_rejects_legacy_cycle_receipts(self):
        cycle = (
            "ds4: dflash cycle drafted=3 accepted=2 rollback=tail ctx=4 "
            "draft=1.0ms verify=2.0ms heads=0.5ms tail=0.1ms serial=1.5ms "
            "cyc=3.6ms be=1.2\n"
        )
        with self.assertRaisesRegex(driver.ReproductionError, "historical"):
            driver.engagement(parser, cycle, "speculative")

    def test_public_speculative_committed_tau_uses_actual_ack_consumption(self):
        text = (ROOT / "tests/fixtures/dflash_adaptive/"
                "public_full_block_partial_eos.log").read_text()
        result = driver.engagement(parser, text, "speculative", 3, "stop")
        self.assertEqual(result["receipt_source"], "full-block request ACK accounting")
        self.assertEqual(result["committed_tokens"], 2)
        self.assertEqual(result["unconsumed_evaluated_suffix"], 1)
        self.assertEqual(result["accepted_draft_tokens"], 2)
        self.assertEqual(result["verify_cycles"], 1)
        self.assertEqual(result["drafted_tokens"], 7)
        self.assertEqual(result["full_width_verify_cycles"], 1)
        checked = driver.engagement(parser, text, "speculative", 3, "stop",
                                    require_public_profile=True)
        self.assertEqual(checked["profile"], "speculative")

    def test_adaptive_ack_ledger_wins_when_low_level_cycle_timing_is_also_present(self):
        text = (ROOT / "tests/fixtures/dflash_adaptive/partial_eos.log").read_text()
        cycle = (
            "ds4: dflash cycle drafted=7 accepted=2 rollback=tail ctx=29 "
            "draft=20.188ms verify=70.039ms heads=3.389ms tail=0.217ms "
            "serial=0ms cyc=97.493ms be=3.0\n"
        )
        result = driver.engagement(parser, text + cycle, "conservative", 3, "stop")
        self.assertEqual(result["receipt_source"], "adaptive request ACK ledger")
        self.assertEqual(result["committed_tokens"], 2)

    def test_trace_exposes_end_to_end_and_decode_only_boundaries(self):
        trace = b"""===== request 7 now =====
prompt_tokens: 9
effective_prompt_tokens: 9
cached_tokens: 0
cache_source: none
max_tokens: 256
temperature: 0.000

--- rendered prompt ---
hello
--- generated text ---
world

--- parsed message ---
finish: stop
generated_tokens: 2
elapsed_sec: 0.500

===== end request 7 =====
"""
        log = b"""ds4-server: chat ctx=0..9:9 gen=2 decoding chunk=4.00 t/s avg=5.00 t/s 0.400s
ds4-server: chat ctx=0..9:9 gen=2 finish=stop 0.500s
"""
        response = {"finish_reason": "stop", "usage": {
            "prompt_tokens": 9, "completion_tokens": 2, "total_tokens": 11,
            "prompt_tokens_details": {"cached_tokens": 0},
        }}
        result = driver.validate_trace_and_log(trace, log, response)
        self.assertEqual(result["trace_id"], 7)
        self.assertEqual(result["server_elapsed_s"], .5)
        self.assertEqual(result["decode_elapsed_s"], .4)
        self.assertNotEqual(result["rendered_prompt"]["sha256"], "")

    def test_timestamped_production_server_receipts_are_parsed(self):
        # Exact receipt shape retained by the first smoke attempt.  The server's
        # production timestamp prefix must remain part of the accepted grammar.
        log = b"""0906 20:04:54 ds4-server: chat ctx=228..278:50 gen=250 decoding chunk=40.69 t/s avg=40.79 t/s 6.130s
0906 20:04:54 ds4-server: chat ctx=278..284:6 gen=256 decoding chunk=40.55 t/s avg=40.78 t/s 6.278s
0906 20:04:54 ds4-server: chat ctx=0..28:28 gen=256 finish=length 6.535s
"""
        progress = list(driver.DECODE_PROGRESS.finditer(log.decode()))
        final = driver.FINAL_LOG.search(log.decode())
        self.assertEqual(progress[-1].groupdict(), {"generated": "256", "elapsed": "6.278"})
        self.assertEqual(final.groupdict(), {
            "generated": "256", "finish": "length", "elapsed": "6.535"})

    def test_first_smoke_attempt_retained_artifacts_parse_when_available(self):
        retained = (Path.home() / "megakernel-refs/public-artifact/"
                    "ASTRAL-SPEC-BENCH-SMOKE-20260907T000500Z/serial")
        if not retained.exists():
            self.skipTest("retained local smoke evidence is not installed")
        trace = (retained / "mt_bench.00.q81.t0.trace.log").read_bytes()
        response = (retained / "mt_bench.00.q81.t0.response").read_bytes()
        log = (retained / "server.log").read_bytes()
        parsed_response = driver.response_fields(response)
        result = driver.validate_trace_and_log(trace, log, parsed_response)
        self.assertEqual(result["trace_id"], 1)
        self.assertEqual(result["server_elapsed_s"], 6.535)
        self.assertEqual(result["decode_elapsed_s"], 6.278)

    def test_elapsed_and_throughput_ratios_remain_separate(self):
        def row(mode, generated, elapsed, digest):
            return {
                "unit_id": "qa.00.q1.t0", "mode": mode, "valid": True,
                "content_bytes": 1, "content_sha256": digest,
                "usage": {"completion_tokens": generated},
                "rendered_prompt": {"sha256": "prompt"},
                "server_elapsed_s": elapsed, "decode_elapsed_s": elapsed,
                "client_wall_s": elapsed,
            }
        result = driver.paired_comparison(
            [row("serial", 10, 2.0, "a")], [row("conservative", 20, 3.0, "b")])
        self.assertAlmostEqual(result["matched_server_elapsed_s_ratio_serial_over_mode"], 2 / 3)
        self.assertAlmostEqual(
            result["matched_server_elapsed_s_throughput_ratio_mode_over_serial"], 4 / 3)
        self.assertEqual(result["byte_identical_outputs"], 0)

    def test_request_identity_rejects_prior_trace_with_same_token_counts(self):
        body = {"messages": [{"role": "user", "content": "new"}]}
        def trace(value):
            return ("--- raw request json ---\n" + json.dumps(value) +
                    "\n\n--- rendered prompt ---\nhello").encode()
        driver.validate_request_identity(trace(body), body)
        with self.assertRaises(driver.ReproductionError):
            driver.validate_request_identity(trace({"messages": []}), body)

    def test_interrupted_run_retains_completed_and_all_unattempted_units(self):
        rows, _ = driver.select_subset_lines(self.synthetic_dataset())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "serial").mkdir()
            name = driver.unit_id(rows[0], 0)
            saved = {"mode": "serial", "unit_id": name, "valid": True}
            (root / "serial" / f"{name}.json").write_text(json.dumps(saved))
            records = driver.complete_declared_records([], rows, root)
            self.assertEqual(len(records), 252)
            self.assertEqual(sum(row["valid"] for row in records), 1)
            self.assertEqual(sum(row.get("attempted") is False for row in records), 251)
            self.assertEqual(len(driver.complete_declared_records(records, rows, root)), 252)

    def test_interrupting_http_preserves_attempt_receipt(self):
        rows, _ = driver.select_subset_lines(self.synthetic_dataset())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            log = root / "server.log"
            log.write_text("")
            with mock.patch.object(driver, "post", side_effect=KeyboardInterrupt):
                record, _ = driver.attempt_one(None, root, rows[0], 0, [], "serial",
                                               1, log, root / "trace", 0, 1)
            self.assertFalse(record["valid"])
            self.assertTrue(record["interrupted"])
            self.assertTrue((root / (record["unit_id"] + ".json")).exists())

    def test_completed_http_uses_short_post_response_log_deadline(self):
        rows, _ = driver.select_subset_lines(self.synthetic_dataset())
        observed = []
        response = json.dumps({
            "choices": [{"message": {"content": "x"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }).encode()
        def fail_log(path, offset, timeout):
            observed.append(timeout)
            raise driver.ReproductionError("missing final receipt")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            log = root / "server.log"
            log.write_text("")
            with mock.patch.object(driver, "post", return_value=(200, response)), \
                    mock.patch.object(driver, "next_trace", return_value=(b"trace", 1)), \
                    mock.patch.object(driver, "complete_log_segment", side_effect=fail_log):
                record, _ = driver.attempt_one(None, root, rows[0], 0, [], "serial",
                                               1, log, root / "trace", 0, 600)
        self.assertEqual(observed, [driver.POST_RESPONSE_LOG_TIMEOUT])
        self.assertIn("missing final receipt", record["validation_errors"])

    def test_eos_is_allowed_but_hidden_context_length_truncation_is_invalid(self):
        def response(finish, count):
            return json.dumps({"choices": [{"message": {"content": "x"},
                                            "finish_reason": finish}],
                               "usage": {"prompt_tokens": 9, "completion_tokens": count,
                                         "total_tokens": count + 9}}).encode()
        self.assertTrue(driver.response_fields(response("stop", 2))["natural_eos"])
        for finish, count in (("length", 2), ("stop", 257)):
            with self.assertRaises(driver.ReproductionError):
                driver.response_fields(response(finish, count))

    def test_failed_http_attempts_keep_wall_times_without_inventing_token_counts(self):
        record = {"unit_id": "qa.00.q1.t0", "valid": False, "client_wall_s": 1.0}
        result = driver.paired_comparison([record], [record])
        self.assertFalse(result["comparison_valid"])
        self.assertEqual(result["matched_client_wall_s_ratio_serial_over_mode"], 1)
        self.assertIsNone(result["matched_client_wall_s_throughput_ratio_mode_over_serial"])

    def test_group_summary_preserves_nonapplicable_controller_metrics(self):
        def row(engagement):
            return {
                "logical_category": "qa", "question_id": 1, "valid": True,
                "natural_eos": True, "cached_tokens": 0,
                "usage": {"prompt_tokens": 2, "completion_tokens": 3},
                "server_elapsed_s": 1.0, "decode_elapsed_s": .5,
                "client_wall_s": 1.1, "engagement": engagement,
            }
        full_block = driver.group_summary([row({
            "cycles": 1, "verify_cycles": 1, "accepted_draft_tokens": 2,
            "committed_tokens": 3, "declines": None, "skips": None,
            "net_ms": 4.0,
        })])
        self.assertIsNone(full_block["declines"])
        self.assertIsNone(full_block["skips"])
        self.assertEqual(full_block["internal_net_ms"], 4.0)
        serial = driver.group_summary([row({
            "cycles": 0, "verify_cycles": 0, "accepted_draft_tokens": 0,
        })])
        self.assertIsNone(serial["declines"])
        self.assertIsNone(serial["skips"])
        self.assertIsNone(serial["internal_net_ms"])

    def test_group_summary_sums_available_controller_metrics(self):
        records = []
        for index, (declines, skips, net_ms) in enumerate(((1, 7, 2.5), (2, 9, -1.0))):
            records.append({
                "logical_category": "qa", "question_id": index, "valid": True,
                "natural_eos": True, "cached_tokens": 0,
                "usage": {"prompt_tokens": 2, "completion_tokens": 3},
                "server_elapsed_s": 1.0, "decode_elapsed_s": .5,
                "client_wall_s": 1.1,
                "engagement": {
                    "cycles": 1, "verify_cycles": 1,
                    "accepted_draft_tokens": 1, "committed_tokens": 2,
                    "declines": declines, "skips": skips, "net_ms": net_ms,
                },
            })
        summary = driver.group_summary(records)
        self.assertEqual(summary["declines"], 3)
        self.assertEqual(summary["skips"], 16)
        self.assertEqual(summary["internal_net_ms"], 1.5)

    def test_portable_driver_has_no_local_home_path(self):
        self.assertNotIn("/Users/", DRIVER_PATH.read_text())


if __name__ == "__main__":
    unittest.main()
