#!/usr/bin/env python3
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

from dflash_adaptive_evidence import (  # noqa: E402
    EvidenceError,
    parse_adaptive_request,
    parse_evidence,
    validate_adaptive_generation_alignment,
    validate_cached_horizon,
)


FIXTURES = ROOT / "tests" / "fixtures" / "dflash_adaptive"


class AdaptiveEvidenceTest(unittest.TestCase):
    def parse(self, name):
        return parse_adaptive_request((FIXTURES / name).read_text())

    def public_profile_partial_eos(self):
        lines = []
        call_index = 0
        for line in (FIXTURES / "partial_eos.log").read_text().splitlines():
            if line.startswith("ds4: dflash admission "):
                line += " profile=conservative min_serial_tokens=0 loss_meter=1"
            elif line.startswith("ds4: dflash adaptive "):
                before = 0 if call_index == 0 else 1
                line += f" entry=0 serial_consumed_before={before}"
                call_index += 1
            elif line.startswith("ds4: dflash adaptive_ack "):
                line += " serial_consumed=1 entry_serial_tokens=0"
            elif line.startswith("ds4: dflash adaptive_totals "):
                line += " serial_consumed=1 entry_serial_tokens=0"
            lines.append(line)
        return "\n".join(lines) + "\n"

    def test_favorable_active_path_and_growth(self):
        request = self.parse("favorable.log").requests[0]
        self.assertEqual(2, len(request.calls))
        self.assertEqual([3, 4], [call["chosen"] for call in request.calls])
        self.assertEqual(6, request.totals["accepted"])
        self.assertEqual(8, request.totals["consumed"])
        self.assertEqual(0, request.terminal_ack["consumed"])
        self.assertEqual(1, request.terminal_ack["done"])

    def test_unfavorable_still_exercises_confidence_gate(self):
        request = self.parse("unfavorable.log").requests[0]
        summary = request.summary()
        self.assertEqual(2, summary["verify_calls"])
        self.assertEqual(1, summary["confidence_truncations"])
        self.assertEqual(0, request.totals["accepted"])

    def test_confidence_zero_keeps_paid_draft_work_visible(self):
        request = self.parse("confidence_zero.log").requests[0]
        summary = request.summary()
        self.assertEqual(1, summary["confidence_checks"])
        self.assertEqual(1, summary["zero_verify_checks"])
        self.assertEqual(0, summary["zero_proposal_checks"])
        self.assertEqual(1.6, summary["draft_ms_on_zero_verify"])
        self.assertEqual(7, request.totals["drafted"])
        self.assertEqual(0, request.totals["verified"])

    def test_false_positive_then_error_invalidates_before_healthy_reuse(self):
        evidence = self.parse("failure_reuse.log")
        self.assertEqual(2, len(evidence.requests))
        failed, healthy = evidence.requests
        self.assertEqual(0, failed.calls[0]["accepted"])
        self.assertEqual("error", failed.calls[-1]["status"])
        self.assertEqual(1, failed.totals["invalid"])
        self.assertEqual(0, healthy.totals["invalid"])
        self.assertEqual(2, healthy.totals["accepted"])

    def test_early_stop_consumes_prefix_and_next_request_resets(self):
        evidence = self.parse("early_stop_reuse.log")
        stopped, reused = evidence.requests
        self.assertEqual(4, stopped.totals["returned"])
        self.assertEqual(1, stopped.totals["consumed"])
        self.assertEqual(500, stopped.calls[0]["pos"])
        self.assertEqual(600, reused.calls[0]["pos"])
        self.assertEqual("earlystop", evidence.cli_counters[0]["stop"])
        alignment = validate_adaptive_generation_alignment(evidence)
        self.assertEqual(3, alignment[0]["unconsumed_evaluated_suffix"])
        self.assertEqual(0, alignment[0]["final_unevaluated_output_tokens"])
        self.assertEqual(0, alignment[1]["unconsumed_evaluated_suffix"])

    def test_rejects_partial_eos_position_or_consumption_mismatch(self):
        text = (FIXTURES / "early_stop_reuse.log").read_text()
        evidence = parse_evidence(text.replace(
            "pos_final=504 committed_forward_positions=4",
            "pos_final=501 committed_forward_positions=1",
        ))
        with self.assertRaisesRegex(EvidenceError, "evaluated rows do not equal adaptive returned"):
            validate_adaptive_generation_alignment(evidence)
        evidence = parse_evidence(text.replace(
            "generated=1 requested=8", "generated=2 requested=8", 1,
        ))
        with self.assertRaisesRegex(EvidenceError, "natural-stop output"):
            validate_adaptive_generation_alignment(evidence)

    def test_actual_partial_eos_receipt_separates_visible_and_evaluated_rows(self):
        evidence = parse_evidence((FIXTURES / "partial_eos.log").read_text())
        alignment = validate_adaptive_generation_alignment(evidence)[0]
        self.assertEqual(3, alignment["generated_visible_tokens"])
        self.assertEqual(3, alignment["ack_consumed_tokens"])
        self.assertEqual(4, alignment["evaluated_forward_positions"])
        self.assertEqual(4, alignment["adaptive_returned_tokens"])
        self.assertEqual(1, alignment["unconsumed_evaluated_suffix"])

    def test_rejects_missing_done_totals(self):
        text = (FIXTURES / "favorable.log").read_text()
        text = "\n".join(
            line for line in text.splitlines()
            if "adaptive_totals" not in line
        )
        with self.assertRaisesRegex(EvidenceError, "missing adaptive_totals"):
            parse_adaptive_request(text)

    def test_rejects_call_after_error_done(self):
        text = (FIXTURES / "failure_reuse.log").read_text()
        marker = "ds4: dflash adaptive_totals done=1 calls=2"
        text = text.replace(marker, (FIXTURES / "unfavorable.log").read_text().splitlines()[1] + "\n" + marker, 1)
        with self.assertRaises(EvidenceError):
            parse_adaptive_request(text)

    def test_cached_horizon_requires_actual_restored_session_depth(self):
        valid = parse_evidence((FIXTURES / "cached_valid.log").read_text())
        ids = validate_cached_horizon(
            valid, pos_initial=299994, generated=24, evaluated=23,
        )
        self.assertEqual(24, len(ids))
        wrong = parse_evidence((FIXTURES / "cached_wrong_depth.log").read_text())
        with self.assertRaisesRegex(EvidenceError, "not restored"):
            validate_cached_horizon(
                wrong, pos_initial=299994, generated=24, evaluated=23,
            )

    def test_cached_speculative_full_horizon_keeps_counts_strict(self):
        text = (FIXTURES / "cached_speculative_valid.log").read_text()
        evidence = parse_evidence(text)
        ids = validate_cached_horizon(
            evidence, pos_initial=299994, generated=24, evaluated=24,
        )
        self.assertEqual(24, len(ids))
        with self.assertRaisesRegex(EvidenceError, "expected 24/23"):
            validate_cached_horizon(
                evidence, pos_initial=299994, generated=24, evaluated=23,
            )
        wrong_position = parse_evidence(text.replace("pos_final=300018", "pos_final=300017"))
        with self.assertRaisesRegex(EvidenceError, "final position"):
            validate_cached_horizon(
                wrong_position, pos_initial=299994, generated=24, evaluated=24,
            )
        lines = text.splitlines()
        lines[1] = lines[1].replace("count=24", "count=23", 1).rsplit(" ", 1)[0]
        wrong_ids = parse_evidence("\n".join(lines) + "\n")
        with self.assertRaisesRegex(EvidenceError, "token-id count"):
            validate_cached_horizon(
                wrong_ids, pos_initial=299994, generated=24, evaluated=24,
            )

    def test_retry_economics_counts_all_declines_and_skips(self):
        request = self.parse("retry_economics.log").requests[0]
        self.assertEqual(
            ["verify", "serial-calibration", "economic-decline", "retry-skip"],
            [call["operation"] for call in request.calls],
        )
        self.assertEqual(1, request.totals["economic_declines"])
        self.assertEqual(1, request.totals["skipped_steps"])
        self.assertEqual(1, request.totals["serial_calibrations"])
        self.assertEqual(-3.3, request.totals["net_ms"])
        self.assertEqual("min-serial", request.totals["reference_source"])
        self.assertEqual("measured", request.totals["floor_source"])

    def test_soft_probe_has_one_bounded_early_recovery(self):
        request = self.parse("retry_probe_recovery.log").requests[0]
        self.assertEqual(1, request.totals["probe_entries"])
        self.assertEqual(9, request.totals["probe_entry_consumed"])
        self.assertEqual(1, request.totals["recovery_used"])
        self.assertEqual(1, request.totals["recovery_checks"])
        self.assertEqual(0, request.totals["recovery_pending"])
        self.assertEqual([0, 0, 1], [call["recovery"] for call in request.calls])

    def test_terminal_zero_ack_accounts_eos_without_a_call(self):
        request = self.parse("retry_terminal_eos.log").requests[0]
        self.assertEqual([], request.calls)
        self.assertEqual(0, request.terminal_ack["consumed"])
        self.assertEqual(0, request.totals["calls"])
        self.assertEqual(0.06, request.totals["net_ms"])

    def test_savings_funded_retry_accounts_bank_and_draft_cost(self):
        request = self.parse("savings_funded.log").requests[0]
        funded = [call for call in request.calls if call["funded"]]
        self.assertEqual(1, len(funded))
        self.assertGreaterEqual(funded[0]["funding_bank_ms"],
                                funded[0]["funding_cost_ms"])
        self.assertEqual(1, request.totals["funded_retries"])
        self.assertEqual(1.5, request.totals["funded_draft_ms"])
        self.assertEqual(1, request.totals["nonescalating_declines"])
        self.assertEqual(12.2, request.totals["savings_ms"])
        self.assertEqual(1.74, request.totals["draft_estimate_ms"])

    def test_rejects_savings_counter_and_funding_regressions(self):
        text = (FIXTURES / "savings_funded.log").read_text()
        with self.assertRaisesRegex(EvidenceError, "invalid savings-funded retry"):
            parse_adaptive_request(text.replace(
                "funding_bank_ms=3.200000 funding_cost_ms=1.800000",
                "funding_bank_ms=1.700000 funding_cost_ms=1.800000",
            ))
        with self.assertRaisesRegex(EvidenceError, "funded draft time mismatch"):
            parse_adaptive_request(text.replace(
                "funded_draft_ms=1.500000 nonescalating_declines=1",
                "funded_draft_ms=1.400000 nonescalating_declines=1",
                1,
            ))
        with self.assertRaisesRegex(EvidenceError, "nonescalating declines exceed"):
            parse_adaptive_request(text.replace(
                "nonescalating_declines=1\n" +
                "ds4: CLI session counters:",
                "nonescalating_declines=2\n" +
                "ds4: CLI session counters:",
            ))

    def test_rejects_retry_schema_and_terminal_flag_regressions(self):
        text = (FIXTURES / "retry_probe_recovery.log").read_text()
        with self.assertRaisesRegex(EvidenceError, "missing fields: recovery"):
            parse_adaptive_request(text.replace(" recovery=0\n", "\n", 1))
        with self.assertRaisesRegex(EvidenceError, "recovery_checks"):
            parse_adaptive_request(text.replace("recovery_checks=1", "recovery_checks=2"))

        favorable = (FIXTURES / "favorable.log").read_text()
        terminal = favorable.replace(
            "ds4: dflash admission policy=confidence-prefix adaptive=1 n_min=1 n_max=7 n_start=3 p_min=0.750000 valid=1",
            "ds4: dflash admission policy=confidence-prefix adaptive=1 n_min=1 n_max=7 n_start=3 p_min=0.750000 valid=1 retry=1 retry_max=512 retry_tax=0.010000 loss_budget=0.030000 early_recovery=1",
        )
        # The old call records now deliberately lack the required retry fields.
        with self.assertRaisesRegex(EvidenceError, "missing fields"):
            parse_adaptive_request(terminal)

    def test_rejects_terminal_ack_that_changes_retry_state(self):
        text = (FIXTURES / "retry_economics.log").read_text()
        ack = (
            "ds4: dflash adaptive_ack consumed=0 done=1 returned=0 total_consumed=11 "
            "total_returned=11 total_ms=18.650000 invalid=0 net_ms=-3.350000 "
            "reference_ms=22.000000 probe_only=0 retry_remaining=98 "
            "reference_step_ms=2.000000 reference_source=min-serial loss_floor_ms=6.000000 "
            "floor_source=measured recovery_used=0 recovery_pending=0 recovery_checks=0\n"
        )
        text = text.replace("ds4: dflash adaptive_totals", ack + "ds4: dflash adaptive_totals")
        with self.assertRaisesRegex(EvidenceError, "terminal ACK changed retry_remaining"):
            parse_adaptive_request(text)

    def test_speculative_cycle_metrics_are_separate_from_adaptive_ledger(self):
        text = (
            "ds4: dflash cycle drafted=7 accepted=3 rollback=stepsnap ctx=85 "
            "draft=23.1ms verify=128.3ms heads=8.5ms tail=1.2ms serial=26.3ms "
            "cyc=187.4ms be=4.10\n"
            "ds4: CLI session counters: generated=4 requested=4 pos_initial=81 "
            "pos_final=84 committed_forward_positions=3 decode_s=0.187400000 "
            "resolved_mode=speculative stop_reason=limit\n"
        )
        evidence = parse_evidence(text)
        metrics = evidence.mode_metrics("speculative")
        self.assertEqual(1, metrics["cycles"])
        self.assertEqual(3, metrics["accepted_draft_tokens"])
        self.assertEqual(3.0, metrics["accepted_per_verify"])
        self.assertIsNone(metrics["net_ms"])
        self.assertEqual(metrics["receipt_source"], "legacy speculative cycle receipt")

    def test_public_speculative_metrics_prefer_adaptive_ack_ledger(self):
        evidence = self.parse("central_reference.log")
        metrics = evidence.mode_metrics("speculative")
        self.assertEqual(metrics["receipt_source"], "adaptive request ACK ledger")
        self.assertEqual(metrics["accepted_draft_tokens"],
                         evidence.mode_metrics("conservative")["accepted_draft_tokens"])

    def test_public_full_block_uses_exact_ack_and_full_width_receipts(self):
        evidence = self.parse("public_full_block_partial_eos.log")
        request = evidence.requests[0]
        self.assertEqual("full-block", request.policy["policy"])
        self.assertEqual("speculative", request.policy["profile"])
        verify = [call for call in request.calls if call["operation"] == "verify"]
        self.assertEqual(1, len(verify))
        self.assertEqual((7, 7, 7),
                         (verify[0]["chosen"], verify[0]["proposed"], verify[0]["verified"]))
        self.assertEqual(-1.0, verify[0]["confidence_first"])
        self.assertEqual(2, request.acks[-1]["consumed"])
        metrics = evidence.mode_metrics("speculative")
        self.assertEqual("full-block request ACK accounting", metrics["receipt_source"])
        self.assertEqual(7, metrics["drafted_tokens"])
        self.assertIsNone(metrics["declines"])
        aligned = validate_adaptive_generation_alignment(evidence)[0]
        self.assertEqual(1, aligned["unconsumed_evaluated_suffix"])
        self.assertEqual(3, aligned["ack_consumed_tokens"])

    def test_public_full_block_policy_is_exact(self):
        text = (FIXTURES / "public_full_block_partial_eos.log").read_text()
        for changed in (
            text.replace("n_start=7", "n_start=3", 1),
            text.replace("p_min=0.000000", "p_min=0.750000", 1),
            text.replace("retry=0", "retry=1", 1),
            text.replace("profile=speculative", "profile=aggressive", 1),
        ):
            with self.assertRaisesRegex(EvidenceError, "retry_max|full-block policy"):
                parse_adaptive_request(changed)
        for changed in (
            text.replace("verified=7 accepted=2", "verified=6 accepted=2", 1),
            text.replace("confidence_first=-1.000000", "confidence_first=0.900000", 1),
            text.replace("economic=0", "economic=1", 1),
        ):
            with self.assertRaisesRegex(
                    EvidenceError, "full-block|operation/flag|verify did not use"):
                parse_adaptive_request(changed)

    def test_public_profile_accounts_serial_entry_state_strictly(self):
        text = self.public_profile_partial_eos()
        evidence = parse_adaptive_request(text)
        request = evidence.requests[0]
        self.assertEqual(request.totals["serial_consumed"], 1)
        self.assertEqual(request.totals["entry_serial_tokens"], 0)
        self.assertEqual(evidence.mode_metrics("speculative")["profile"], "conservative")
        with self.assertRaisesRegex(EvidenceError, "serial_consumed_before mismatch"):
            parse_adaptive_request(text.replace(
                "entry=0 serial_consumed_before=1", "entry=0 serial_consumed_before=0", 1))
        with self.assertRaisesRegex(EvidenceError, "entry ACK accounting mismatch"):
            parse_adaptive_request(text.replace(
                "serial_consumed=1 entry_serial_tokens=0",
                "serial_consumed=0 entry_serial_tokens=0", 1))

    def test_aggressive_public_profile_disables_meter_and_savings(self):
        text = self.public_profile_partial_eos().replace(
            "profile=conservative min_serial_tokens=0 loss_meter=1",
            "profile=aggressive min_serial_tokens=0 loss_meter=0",
        ).replace("savings_retry=1", "savings_retry=0", 1)
        request = parse_adaptive_request(text).requests[0]
        self.assertEqual(request.policy["profile"], "aggressive")
        for bad in (
            text.replace("loss_meter=0", "loss_meter=1", 1),
            text.replace("savings_retry=0", "savings_retry=1", 1),
        ):
            with self.assertRaisesRegex(EvidenceError, "profile/"):
                parse_adaptive_request(bad)

    def test_aggressive_alignment_requires_matched_speculative_counter(self):
        text = self.public_profile_partial_eos().replace(
            "profile=conservative min_serial_tokens=0 loss_meter=1",
            "profile=aggressive min_serial_tokens=0 loss_meter=0",
        ).replace("savings_retry=1", "savings_retry=0", 1).replace(
            "resolved_mode=conservative", "resolved_mode=speculative")
        evidence = parse_adaptive_request(text)
        receipt = validate_adaptive_generation_alignment(evidence)[0]
        self.assertEqual(receipt["profile"], "aggressive")
        self.assertEqual(receipt["resolved_mode"], "speculative")
        with self.assertRaisesRegex(EvidenceError, "matched resolved_mode=speculative"):
            validate_adaptive_generation_alignment(parse_adaptive_request(
                text.replace("resolved_mode=speculative", "resolved_mode=conservative")))

    def test_retained_aggressive_chat_alignment_when_available(self):
        retained = (pathlib.Path.home() / "megakernel-refs/public-artifact/"
                    "ASTRAL-DFLASH-ADAPTIVE-20260907T002721Z/"
                    "middle-chat512.speculative.err")
        if not retained.exists():
            self.skipTest("retained aggressive chat evidence is not installed")
        evidence = parse_adaptive_request(retained.read_text(errors="replace"))
        receipt = validate_adaptive_generation_alignment(evidence)[0]
        self.assertEqual(receipt["profile"], "aggressive")
        self.assertEqual(receipt["resolved_mode"], "speculative")
        self.assertEqual(receipt["generated_visible_tokens"], 512)
        self.assertEqual(receipt["adaptive_returned_tokens"], 512)

    def test_public_entry_operation_contract_and_gate(self):
        import dflash_adaptive_evidence as module
        text = self.public_profile_partial_eos()
        line = next(value for value in text.splitlines()
                    if value.startswith(module.CALL_PREFIX))
        item = module._typed(module._kv(line, module.CALL_PREFIX, 1), 1)
        item.update(operation="entry-serial", entry=1, chosen=0, proposed=0,
                    candidate=0, verified=0, accepted=0, returned=1,
                    serial_calibration=0, skipped=0)
        module._validate_call(item, 1, True, True, True)
        item["serial_calibration"] = 1
        with self.assertRaisesRegex(EvidenceError,
                                    "operation/flag mismatch|inconsistent entry-serial"):
            module._validate_call(item, 1, True, True, True)

    def test_rejects_malformed_speculative_cycle(self):
        text = (
            "ds4: dflash cycle drafted=2 accepted=3 rollback=none ctx=9 "
            "draft=1ms verify=2ms heads=1ms tail=1ms serial=1ms cyc=6ms be=2\n"
            "ds4: GLM gen counters: n_generated=1 n_decode_eval=0 prompt_len=9 "
            "final_pos=10 ctx=8192 n_predict=1 decode_s=0.1 ms_per_eval=0 "
            "stop=predict_limit\n"
        )
        with self.assertRaisesRegex(EvidenceError, "accepted exceeds drafted"):
            parse_evidence(text)

    def test_central_reference_prices_actual_serial_and_fixed_verify_rows(self):
        request = self.parse("central_reference.log").requests[0]
        calibration = request.acks[1]
        self.assertEqual(2.0, calibration["reference_serial_added_ms"])
        self.assertEqual(8.0, calibration["bootstrap_added_ms"])
        reprice = request.acks[2]
        self.assertEqual(2.0, reprice["bootstrap_reprice_ms"])
        self.assertEqual(1.5, reprice["serial_central_ms"])
        final = request.acks[3]
        self.assertEqual(12.0, final["reference_verify_added_ms"])
        self.assertEqual(3.0, final["reference_serial_ms"])
        self.assertEqual(18.0, final["reference_verify_ms"])
        self.assertEqual("serial-actual+verify-median9", final["reference_source"])
        self.assertEqual("row-calibrated", request.acks[0]["floor_source"])
        self.assertEqual(0, request.terminal_ack["consumed"])
        self.assertEqual(0.0, request.terminal_ack["reference_verify_added_ms"])

    def test_rejects_central_reference_increment_and_component_regressions(self):
        text = (FIXTURES / "central_reference.log").read_text()
        with self.assertRaisesRegex(EvidenceError, "serial reference increment mismatch"):
            parse_adaptive_request(text.replace(
                "reference_serial_added_ms=2.000000",
                "reference_serial_added_ms=1.000000",
                1,
            ))
        with self.assertRaisesRegex(EvidenceError, "reference components do not sum"):
            parse_adaptive_request(text.replace(
                "reference_serial_ms=3.000000 reference_verify_ms=18.000000",
                "reference_serial_ms=2.000000 reference_verify_ms=18.000000",
                1,
            ))


if __name__ == "__main__":
    unittest.main()
