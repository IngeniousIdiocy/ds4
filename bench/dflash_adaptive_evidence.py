#!/usr/bin/env python3
"""Parse and validate DS4 adaptive DFlash request evidence.

The parser deliberately accepts unrelated stderr lines.  Lines carrying the
adaptive prefixes are strict: malformed, incomplete, or internally
inconsistent evidence is rejected instead of being silently ignored.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import re
from typing import Any, Dict, Iterable, List, Optional


POLICY_PREFIX = "ds4: dflash admission "
CALL_PREFIX = "ds4: dflash adaptive "
ACK_PREFIX = "ds4: dflash adaptive_ack "
TOTALS_PREFIX = "ds4: dflash adaptive_totals "

_INT_FIELDS = {
    "adaptive", "n_min", "n_max", "n_start", "pos", "chosen",
    "proposed", "verified", "accepted", "returned", "next",
    "bootstrap_rows", "consumed", "done", "total_consumed",
    "total_returned", "invalid", "valid", "done", "calls", "drafted",
    "retry", "retry_max", "early_recovery", "candidate", "skipped",
    "economic", "calibration", "serial_calibration", "retry_remaining",
    "limit", "probe_only", "recovery", "zero_prefix", "skipped_steps",
    "economic_declines", "losing_cycles", "calibrations",
    "serial_calibrations", "probe_entries", "probe_entry_consumed",
    "recovery_used", "recovery_pending", "recovery_checks",
    "savings_retry", "funded", "funded_retries", "nonescalating_declines",
    "serial_samples", "serial_window", "reference_unpriced_rows", "bootstrap_rows",
    "min_serial_tokens", "loss_meter", "entry", "serial_consumed_before",
    "serial_consumed", "entry_serial_tokens",
}
_FLOAT_FIELDS = {
    "p_min", "confidence_first", "confidence_min", "setup_ms",
    "refresh_ms", "draft_ms", "verify_ms", "heads_ms", "tail_ms",
    "serial_ms", "total_ms", "retry_tax", "loss_budget", "net_ms",
    "reference_ms", "net_percent", "reference_step_ms", "loss_floor_ms",
    "funding_bank_ms", "funding_cost_ms", "savings_ms", "draft_estimate_ms",
    "funded_draft_ms",
    "serial_min_ms", "serial_central_ms", "reference_call_step_ms",
    "reference_serial_added_ms", "reference_verify_added_ms", "bootstrap_added_ms",
    "bootstrap_reprice_ms", "reference_serial_ms", "reference_verify_ms",
    "bootstrap_repriced_ms", "bootstrap_step_ms",
}


class EvidenceError(ValueError):
    pass


def _kv(line: str, prefix: str, line_no: int) -> Dict[str, str]:
    if not line.startswith(prefix):
        raise EvidenceError(f"line {line_no}: missing prefix {prefix!r}")
    out: Dict[str, str] = {}
    for word in line[len(prefix):].strip().split():
        if "=" not in word:
            raise EvidenceError(f"line {line_no}: malformed field {word!r}")
        key, value = word.split("=", 1)
        if not key or not value or key in out:
            raise EvidenceError(f"line {line_no}: invalid field {word!r}")
        out[key] = value
    return out


def _typed(raw: Dict[str, str], line_no: int) -> Dict[str, Any]:
    out: Dict[str, Any] = dict(raw)
    for key, value in raw.items():
        try:
            if key in _INT_FIELDS:
                out[key] = int(value)
            elif key in _FLOAT_FIELDS:
                number = float(value)
                if not math.isfinite(number):
                    raise ValueError("non-finite")
                out[key] = number
        except ValueError as exc:
            raise EvidenceError(
                f"line {line_no}: invalid numeric {key}={value!r}"
            ) from exc
    return out


def _require(item: Dict[str, Any], required: Iterable[str], line_no: int) -> None:
    missing = sorted(set(required) - item.keys())
    if missing:
        raise EvidenceError(f"line {line_no}: missing fields: {', '.join(missing)}")


@dataclass
class AdaptiveRequest:
    policy: Dict[str, Any]
    policy_line: int
    calls: List[Dict[str, Any]] = field(default_factory=list)
    acks: List[Dict[str, Any]] = field(default_factory=list)
    terminal_ack: Optional[Dict[str, Any]] = None
    totals: Optional[Dict[str, Any]] = None

    def summary(self) -> Dict[str, Any]:
        successful = [c for c in self.calls if c["status"] == "ok"]
        verifies = [c for c in successful if c["verified"] > 0]
        confidence_checks = [
            c for c in successful if c["confidence_first"] >= 0.0
        ]
        result = {
            "policy": self.policy,
            "calls": len(self.calls),
            "verify_calls": len(verifies),
            "confidence_truncations": sum(
                c["verified"] < c["proposed"] for c in confidence_checks
            ),
            "confidence_checks": len(confidence_checks),
            "zero_proposal_checks": sum(c["proposed"] == 0 for c in confidence_checks),
            "zero_verify_checks": sum(c["verified"] == 0 for c in confidence_checks),
            "draft_ms_on_zero_verify": sum(
                c["draft_ms"] for c in confidence_checks if c["verified"] == 0
            ),
            "length_changes": sum(
                c["next"] != c["chosen"] for c in successful
            ),
            "error_calls": sum(c["status"] == "error" for c in self.calls),
            "final_done": bool(
                (self.terminal_ack and self.terminal_ack["done"] == 1) or
                (self.acks and self.acks[-1]["done"] == 1)
            ),
            "terminal_ack": self.terminal_ack,
            "totals": self.totals,
        }
        if "retry" in self.policy:
            result["retry"] = {
                key: self.totals[key] for key in (
                    "zero_prefix", "skipped_steps", "economic_declines",
                    "losing_cycles", "calibrations", "serial_calibrations",
                    "net_ms", "reference_ms", "net_percent", "probe_only",
                    "probe_entries", "probe_entry_consumed", "recovery_checks",
                )
            }
            if "savings_retry" in self.policy:
                result["retry"].update({
                    key: self.totals[key] for key in (
                        "savings_ms", "draft_estimate_ms", "funded_retries",
                        "funded_draft_ms", "nonescalating_declines",
                    )
                })
        return result


@dataclass
class Evidence:
    requests: List[AdaptiveRequest]
    cli_counters: List[Dict[str, Any]]
    token_id_receipts: List[Dict[str, Any]]
    cycle_receipts: List[Dict[str, Any]]

    def summary(self) -> Dict[str, Any]:
        return {
            "requests": [request.summary() for request in self.requests],
            "cli_counters": self.cli_counters,
            "token_id_receipts": self.token_id_receipts,
            "cycle_summary": {
                "cycles": len(self.cycle_receipts),
                "drafted": sum(item["drafted"] for item in self.cycle_receipts),
                "accepted": sum(item["accepted"] for item in self.cycle_receipts),
            },
        }

    def mode_metrics(self, mode: str) -> Dict[str, Any]:
        """Return comparable engagement fields without conflating ledger debt and slowdown."""
        if mode in {"conservative", "speculative"} and self.requests:
            calls = [call for request in self.requests for call in request.calls]
            totals = [request.totals for request in self.requests]
            verify_calls = [call for call in calls if call["verified"] > 0]
            accepted = sum(call["accepted"] for call in verify_calls)
            total_ms = sum(item["total_ms"] for item in totals)
            net_ms = sum(item.get("net_ms", 0.0) for item in totals)
            full_block = self.requests[0].policy.get("policy") == "full-block"
            return {
                "cycles": sum(call["chosen"] > 0 for call in calls),
                "verify_cycles": len(verify_calls),
                "accepted_draft_tokens": accepted,
                "accepted_per_verify": accepted / len(verify_calls) if verify_calls else None,
                "drafted_tokens": sum(call["chosen"] for call in calls),
                "declines": (None if full_block else sum(
                    item.get("zero_prefix", 0) + item.get("economic_declines", 0)
                    for item in totals)),
                "skips": (None if full_block else sum(
                    item.get("skipped_steps", 0) for item in totals)),
                "net_ms": net_ms if totals and "net_ms" in totals[0] else None,
                "net_percent_of_internal_actual": (
                    100.0 * net_ms / total_ms
                    if total_ms and totals and "net_ms" in totals[0] else None
                ),
                "ledger_scope": (
                    "internal counterfactual reference; compare paired complete-generation "
                    "timers for measured slowdown"
                ),
                "receipt_source": ("full-block request ACK accounting" if full_block else
                                   "adaptive request ACK ledger"),
                "policy": self.requests[0].policy.get("policy"),
                "profile": self.requests[0].policy.get("profile"),
                "serial_consumed": sum(item.get("serial_consumed", 0) for item in totals),
                "entry_serial_tokens": sum(
                    item.get("entry_serial_tokens", 0) for item in totals),
            }
        if mode == "speculative":
            accepted = sum(item["accepted"] for item in self.cycle_receipts)
            cycles = len(self.cycle_receipts)
            return {
                "cycles": cycles,
                "verify_cycles": cycles,
                "accepted_draft_tokens": accepted,
                "accepted_per_verify": accepted / cycles if cycles else None,
                "declines": None,
                "zero_accept_cycles": sum(item["accepted"] == 0
                                          for item in self.cycle_receipts),
                "skips": None,
                "net_ms": None,
                "net_percent_of_internal_actual": None,
                "ledger_scope": (
                    "legacy blind-research cycle receipts; no adaptive decline/skip/net ledger"
                ),
                "receipt_source": "legacy speculative cycle receipt",
            }
        if mode == "serial":
            return {
                "cycles": 0, "verify_cycles": 0, "accepted_draft_tokens": 0,
                "accepted_per_verify": None, "declines": 0, "skips": 0,
                "net_ms": None, "net_percent_of_internal_actual": None,
                "ledger_scope": "no DFlash cycle or adaptive ledger in serial mode",
            }
        raise EvidenceError(f"unsupported mode for metrics: {mode!r}")


_GLM_COUNTERS = re.compile(
    r"^ds4: GLM gen counters: n_generated=(?P<generated>\d+) "
    r"n_decode_eval=(?P<evaluated>\d+) prompt_len=(?P<prompt_len>\d+) "
    r"final_pos=(?P<final_pos>\d+) ctx=(?P<ctx>\d+) "
    r"n_predict=(?P<requested>\d+) decode_s=(?P<decode_s>[0-9.eE+-]+) "
    r"ms_per_eval=(?P<ms_per_eval>[0-9.eE+-]+) stop=(?P<stop>\S+)$"
)
_CLI_COUNTERS = re.compile(
    r"^ds4: CLI session counters: generated=(?P<generated>\d+) "
    r"requested=(?P<requested>\d+) pos_initial=(?P<pos_initial>\d+) "
    r"pos_final=(?P<final_pos>\d+) "
    r"committed_forward_positions=(?P<evaluated>-?\d+) "
    r"decode_s=(?P<decode_s>[0-9.eE+-]+) "
    r"resolved_mode=(?P<resolved_mode>\S+) stop_reason=(?P<stop>\S+)$"
)
_RUNFX_IDS = re.compile(
    r"^ds4: RUNFX_IDS tag=(?P<tag>\S+) count=(?P<count>\d+)(?P<ids>(?: \d+)*)$"
)
_DFLASH_CYCLE = re.compile(
    r"^ds4: dflash cycle drafted=(?P<drafted>\d+) accepted=(?P<accepted>\d+) "
    r"rollback=(?P<rollback>\S+) ctx=(?P<ctx>\d+) "
    r"draft=(?P<draft_ms>[0-9.eE+-]+)ms verify=(?P<verify_ms>[0-9.eE+-]+)ms "
    r"heads=(?P<heads_ms>[0-9.eE+-]+)ms tail=(?P<tail_ms>[0-9.eE+-]+)ms "
    r"serial=(?P<serial_ms>[0-9.eE+-]+)ms cyc=(?P<cycle_ms>[0-9.eE+-]+)ms "
    r"be=(?P<break_even>[0-9.eE+-]+)$"
)


def _parse_cycle(line: str, line_no: int) -> Optional[Dict[str, Any]]:
    if not line.startswith("ds4: dflash cycle "):
        return None
    match = _DFLASH_CYCLE.match(line)
    if not match:
        raise EvidenceError(f"line {line_no}: malformed DFlash cycle receipt")
    item: Dict[str, Any] = match.groupdict()
    for key in ("drafted", "accepted", "ctx"):
        item[key] = int(item[key])
    for key in ("draft_ms", "verify_ms", "heads_ms", "tail_ms", "serial_ms",
                "cycle_ms", "break_even"):
        item[key] = float(item[key])
        if not math.isfinite(item[key]) or item[key] < 0:
            raise EvidenceError(f"line {line_no}: invalid cycle {key}")
    if item["accepted"] > item["drafted"]:
        raise EvidenceError(f"line {line_no}: cycle accepted exceeds drafted")
    return item


def _parse_counter(line: str, line_no: int) -> Optional[Dict[str, Any]]:
    match = _GLM_COUNTERS.match(line) or _CLI_COUNTERS.match(line)
    if not match:
        return None
    item: Dict[str, Any] = match.groupdict()
    for key in ("generated", "evaluated", "prompt_len", "final_pos",
                "ctx", "requested", "pos_initial"):
        if item.get(key) is not None:
            item[key] = int(item[key])
    for key in ("decode_s", "ms_per_eval"):
        if item.get(key) is not None:
            item[key] = float(item[key])
            if not math.isfinite(item[key]):
                raise EvidenceError(f"line {line_no}: non-finite {key}")
    if item["decode_s"] <= 0 or item["generated"] < 0 or item["evaluated"] < 0:
        raise EvidenceError(f"line {line_no}: invalid generation counters")
    return item


def _validate_policy(item: Dict[str, Any], line_no: int) -> None:
    _require(item, ("policy", "adaptive", "n_min", "n_max", "n_start", "p_min", "valid"), line_no)
    if item["adaptive"] not in (0, 1):
        raise EvidenceError(f"line {line_no}: adaptive must be 0 or 1")
    if item["policy"] not in {"confidence-prefix", "full-block"}:
        raise EvidenceError(f"line {line_no}: unexpected policy {item['policy']!r}")
    if item["valid"] != 1:
        raise EvidenceError(f"line {line_no}: confidence configuration is invalid")
    if not (1 <= item["n_min"] <= item["n_start"] <= item["n_max"] <= 7):
        raise EvidenceError(f"line {line_no}: invalid adaptive length bounds")
    if not (0.0 <= item["p_min"] <= 1.0):
        raise EvidenceError(f"line {line_no}: p_min outside [0,1]")
    full_block = item["policy"] == "full-block"
    if "retry" in item:
        _require(item, ("retry", "retry_max", "retry_tax", "loss_budget",
                        "early_recovery"), line_no)
        if item["retry"] not in (0, 1) or item["early_recovery"] not in (0, 1):
            raise EvidenceError(f"line {line_no}: retry flags must be 0 or 1")
        if not ((full_block and item["retry_max"] == 0) or
                (not full_block and 1 <= item["retry_max"] <= 4096)):
            raise EvidenceError(f"line {line_no}: retry_max outside 1..4096")
        if not ((full_block and item["retry_tax"] == item["loss_budget"] == 0.0) or
                (not full_block and 0.001 <= item["retry_tax"] <= 0.1 and
                 0.001 <= item["loss_budget"] <= 0.25)):
            raise EvidenceError(f"line {line_no}: invalid retry economics")
        if "savings_retry" in item and item["savings_retry"] not in (0, 1):
            raise EvidenceError(f"line {line_no}: savings_retry must be 0 or 1")
    if "profile" in item:
        _require(item, ("profile", "min_serial_tokens", "loss_meter"), line_no)
        if item["profile"] not in {"conservative", "aggressive", "speculative"}:
            raise EvidenceError(f"line {line_no}: unsupported public profile")
        if not (0 <= item["min_serial_tokens"] <= 64):
            raise EvidenceError(f"line {line_no}: min_serial_tokens outside 0..64")
        if item["loss_meter"] not in (0, 1):
            raise EvidenceError(f"line {line_no}: loss_meter must be 0 or 1")
        expected_meter = int(item["profile"] == "conservative")
        if item["loss_meter"] != expected_meter:
            raise EvidenceError(f"line {line_no}: profile/loss_meter mismatch")
        if "savings_retry" in item:
            expected_savings = int(item["profile"] == "conservative")
            if item["savings_retry"] != expected_savings:
                raise EvidenceError(f"line {line_no}: profile/savings_retry mismatch")
    if full_block:
        expected = {
            "profile": "speculative", "adaptive": 0, "n_min": 1, "n_max": 7,
            "n_start": 7, "p_min": 0.0, "valid": 1, "retry": 0,
            "retry_max": 0, "retry_tax": 0.0, "loss_budget": 0.0,
            "early_recovery": 0, "savings_retry": 0, "min_serial_tokens": 0,
            "loss_meter": 0,
        }
        if any(item.get(key) != value for key, value in expected.items()):
            raise EvidenceError(f"line {line_no}: invalid public full-block policy")


def _validate_call(item: Dict[str, Any], line_no: int, retry_schema: bool,
                   savings_schema: bool, public_schema: bool = False) -> None:
    required = (
        "pos", "operation", "chosen", "proposed", "verified", "accepted",
        "returned", "next", "p_min", "confidence_first", "confidence_min",
        "bootstrap_rows", "setup_ms", "draft_ms", "verify_ms",
        "heads_ms", "tail_ms", "serial_ms", "total_ms", "status",
    )
    _require(item, required, line_no)
    if item["operation"] not in {
        "verify", "refresh", "serial", "draft-fallback", "draft-disabled",
        "retry-skip", "serial-calibration", "economic-decline",
        "entry-serial",
    }:
        raise EvidenceError(f"line {line_no}: unknown operation {item['operation']!r}")
    if item["status"] not in {"ok", "error"}:
        raise EvidenceError(f"line {line_no}: invalid status {item['status']!r}")
    for key in ("pos", "chosen", "proposed", "verified", "accepted", "returned",
                "next", "bootstrap_rows"):
        if item[key] < 0:
            raise EvidenceError(f"line {line_no}: negative {key}")
    for key in ("setup_ms", "draft_ms", "verify_ms", "heads_ms",
                "tail_ms", "serial_ms", "total_ms"):
        if item[key] < 0:
            raise EvidenceError(f"line {line_no}: negative {key}")
    if not (item["proposed"] <= item["chosen"] and
            item["verified"] <= item["proposed"] and
            item["accepted"] <= item["verified"]):
        raise EvidenceError(f"line {line_no}: inconsistent draft counts")
    if retry_schema:
        fields = ("candidate", "skipped", "economic", "calibration",
                  "serial_calibration", "retry_remaining", "limit",
                  "probe_only", "recovery")
        _require(item, fields, line_no)
        if any(item[k] not in (0, 1) for k in
               ("skipped", "economic", "calibration", "serial_calibration",
                "probe_only", "recovery")):
            raise EvidenceError(f"line {line_no}: invalid retry call flag")
        if item["candidate"] < 0 or item["candidate"] > item["proposed"]:
            raise EvidenceError(f"line {line_no}: candidate outside proposed prefix")
        if item["verified"] > item["candidate"]:
            raise EvidenceError(f"line {line_no}: verified exceeds candidate")
        if item["retry_remaining"] < 0 or item["limit"] < 0:
            raise EvidenceError(f"line {line_no}: negative retry state")
        op = item["operation"]
        if op == "retry-skip" and not (
                item["skipped"] == 1 and item["chosen"] == item["proposed"] ==
                item["candidate"] == item["verified"] == item["accepted"] == 0):
            raise EvidenceError(f"line {line_no}: inconsistent retry-skip")
        if op == "serial-calibration" and not (
                item["serial_calibration"] == 1 and item["chosen"] ==
                item["proposed"] == item["candidate"] == item["verified"] == 0):
            raise EvidenceError(f"line {line_no}: inconsistent serial calibration")
        if op == "economic-decline" and not (
                item["economic"] == 1 and item["candidate"] > 0 and
                item["verified"] == item["accepted"] == 0):
            raise EvidenceError(f"line {line_no}: inconsistent economic decline")
        if op == "verify" and not (item["candidate"] == item["verified"] and
                                    item["candidate"] > 0):
            raise EvidenceError(f"line {line_no}: verify did not use candidate prefix")
        if item["skipped"] != (op == "retry-skip") or \
                item["economic"] != (op == "economic-decline") or \
                item["serial_calibration"] != (op == "serial-calibration"):
            raise EvidenceError(f"line {line_no}: operation/flag mismatch")
        if item["recovery"] and item["chosen"] == 0:
            raise EvidenceError(f"line {line_no}: recovery did not attempt proposer")
    if public_schema:
        _require(item, ("entry", "serial_consumed_before"), line_no)
        if item["entry"] not in (0, 1) or item["serial_consumed_before"] < 0:
            raise EvidenceError(f"line {line_no}: invalid entry call fields")
        if item["entry"] != (item["operation"] == "entry-serial"):
            raise EvidenceError(f"line {line_no}: entry operation/flag mismatch")
        if item["entry"] and not (
                item["chosen"] == item["proposed"] == item["verified"] ==
                item["accepted"] == 0 and item["returned"] == 1 and
                item.get("serial_calibration") == 0 and item.get("skipped") == 0):
            raise EvidenceError(f"line {line_no}: inconsistent entry-serial")
    if savings_schema:
        _require(item, ("funded", "funding_bank_ms", "funding_cost_ms"), line_no)
        if item["funded"] not in (0, 1):
            raise EvidenceError(f"line {line_no}: funded must be 0 or 1")
        if item["funding_bank_ms"] < 0 or item["funding_cost_ms"] < 0:
            raise EvidenceError(f"line {line_no}: negative funding value")
        if item["funded"]:
            if not (item["chosen"] > 0 and item["retry_remaining"] > 0 and
                    item["skipped"] == 0 and item["funding_cost_ms"] > 0 and
                    item["funding_bank_ms"] + 2e-3 >= item["funding_cost_ms"]):
                raise EvidenceError(f"line {line_no}: invalid savings-funded retry")
        elif item["funding_bank_ms"] != 0.0 or item["funding_cost_ms"] != 0.0:
            raise EvidenceError(f"line {line_no}: unfunded call carries funding values")
    if item["status"] == "error":
        if item["returned"] != 0:
            raise EvidenceError(f"line {line_no}: error returned tokens")
    elif item["operation"] == "verify":
        if item["returned"] != item["accepted"] + 1:
            raise EvidenceError(f"line {line_no}: verify return is not accepted+anchor")
    elif item["returned"] != 1:
        raise EvidenceError(f"line {line_no}: fallback operation must return one anchor")


def _validate_full_block_call(item: Dict[str, Any], line_no: int) -> None:
    if item["p_min"] != 0.0 or item["confidence_first"] != -1.0 or \
            item["confidence_min"] != -1.0:
        raise EvidenceError(f"line {line_no}: full-block call used confidence")
    if item["operation"] not in {"verify", "refresh", "draft-disabled", "serial"}:
        raise EvidenceError(f"line {line_no}: full-block call used controller operation")
    for key in ("skipped", "economic", "calibration", "serial_calibration",
                "recovery", "funded", "entry"):
        if item.get(key) != 0:
            raise EvidenceError(f"line {line_no}: full-block call has nonzero {key}")
    if item.get("retry_remaining") != 0 or item.get("funding_bank_ms") != 0.0 or \
            item.get("funding_cost_ms") != 0.0:
        raise EvidenceError(f"line {line_no}: full-block call used retry funding")
    if item["status"] == "ok" and item["operation"] == "verify":
        if not (1 <= item["chosen"] <= 7 and item["chosen"] == item["proposed"] ==
                item["candidate"] == item["verified"]):
            raise EvidenceError(f"line {line_no}: full-block verify was not full proposed width")


def _validate_ack(item: Dict[str, Any], line_no: int, retry_schema: bool,
                  savings_schema: bool, public_schema: bool = False) -> None:
    _require(item, ("consumed", "done", "returned", "total_consumed",
                    "total_returned", "total_ms", "invalid"), line_no)
    for key in ("consumed", "returned", "total_consumed", "total_returned"):
        if item[key] < 0:
            raise EvidenceError(f"line {line_no}: negative {key}")
    if item["done"] not in (0, 1) or item["invalid"] not in (0, 1):
        raise EvidenceError(f"line {line_no}: done/invalid must be 0 or 1")
    if item["consumed"] > item["returned"]:
        raise EvidenceError(f"line {line_no}: caller consumed more than returned")
    if item["total_ms"] < 0:
        raise EvidenceError(f"line {line_no}: negative total_ms")
    if retry_schema:
        _require(item, ("net_ms", "reference_ms", "probe_only", "retry_remaining",
                        "reference_step_ms", "reference_source", "loss_floor_ms",
                        "floor_source", "recovery_used", "recovery_pending",
                        "recovery_checks"), line_no)
        _validate_meter(item, line_no)
    if savings_schema:
        _require(item, ("savings_ms", "draft_estimate_ms", "funded_retries",
                        "funded_draft_ms", "nonescalating_declines"), line_no)
        _validate_savings(item, line_no)
    if public_schema:
        _require(item, ("serial_consumed", "entry_serial_tokens"), line_no)
        if item["serial_consumed"] < 0 or item["entry_serial_tokens"] < 0:
            raise EvidenceError(f"line {line_no}: negative entry accounting")
        if item["entry_serial_tokens"] > item["serial_consumed"]:
            raise EvidenceError(f"line {line_no}: entry tokens exceed serial consumption")


def _validate_meter(item: Dict[str, Any], line_no: int, *, totals: bool = False) -> None:
    numeric = ["reference_ms", "reference_step_ms", "loss_floor_ms",
               "recovery_checks"]
    if not totals:
        numeric.append("retry_remaining")
    for key in numeric:
        if item[key] < 0:
            raise EvidenceError(f"line {line_no}: negative meter field {key}")
    for key in ("probe_only", "recovery_used", "recovery_pending"):
        if item[key] not in (0, 1):
            raise EvidenceError(f"line {line_no}: invalid meter flag {key}")
    if item["reference_source"] not in {
            "min-serial", "serial-actual+verify-median9"}:
        raise EvidenceError(f"line {line_no}: unsupported reference source")
    if item["floor_source"] not in {
            "measured", "row-estimate", "row-calibrated", "unavailable"}:
        raise EvidenceError(f"line {line_no}: unsupported floor source")
    if item["floor_source"] == "unavailable" and item["loss_floor_ms"] != 0.0:
        raise EvidenceError(f"line {line_no}: unavailable loss floor is nonzero")
    consumed = item["consumed"] if totals else item["total_consumed"]
    tolerance = max(2e-3, consumed * 1e-6)
    if abs(item["net_ms"] - (item["total_ms"] - item["reference_ms"])) > tolerance:
        raise EvidenceError(f"line {line_no}: net_ms does not equal total-reference")
    if item["reference_source"] == "min-serial":
        if abs(item["reference_ms"] -
               consumed * item["reference_step_ms"]) > tolerance:
            raise EvidenceError(f"line {line_no}: reference is not consumed*minimum serial")
    else:
        _validate_central_reference(item, line_no)
    if item["recovery_pending"] and not item["recovery_used"]:
        raise EvidenceError(f"line {line_no}: recovery pending without allowance")


def _validate_central_reference(item: Dict[str, Any], line_no: int) -> None:
    fields = (
        "serial_min_ms", "serial_central_ms", "serial_samples", "serial_window",
        "reference_call_step_ms", "reference_serial_added_ms",
        "reference_verify_added_ms", "bootstrap_added_ms", "bootstrap_reprice_ms",
        "reference_unpriced_rows", "reference_serial_ms", "reference_verify_ms",
        "bootstrap_repriced_ms", "bootstrap_rows", "bootstrap_step_ms",
    )
    _require(item, fields, line_no)
    for key in fields:
        if item[key] < 0:
            raise EvidenceError(f"line {line_no}: negative central reference field {key}")
    if item["serial_window"] != 9:
        raise EvidenceError(f"line {line_no}: serial_window must be 9")
    if item["serial_samples"] > item["serial_window"]:
        raise EvidenceError(f"line {line_no}: serial sample count exceeds window")
    if item["serial_samples"] == 0:
        if item["serial_min_ms"] != 0.0 or item["serial_central_ms"] != 0.0:
            raise EvidenceError(f"line {line_no}: serial estimate exists without samples")
    elif not (0.0 < item["serial_min_ms"] <= item["serial_central_ms"]):
        raise EvidenceError(f"line {line_no}: invalid min/central serial estimate")
    if abs(item["reference_step_ms"] - item["serial_central_ms"]) > 2e-3:
        raise EvidenceError(f"line {line_no}: reference step is not serial central")
    if abs(item["reference_ms"] -
           (item["reference_serial_ms"] + item["reference_verify_ms"])) > 2e-3:
        raise EvidenceError(f"line {line_no}: reference components do not sum")
    if item["bootstrap_reprice_ms"] > item["bootstrap_repriced_ms"] + 2e-3:
        raise EvidenceError(f"line {line_no}: bootstrap reprice exceeds cumulative amount")
    if item["bootstrap_rows"] == 0 and item["bootstrap_step_ms"] != 0.0:
        raise EvidenceError(f"line {line_no}: bootstrap price without rows")


def _validate_savings(item: Dict[str, Any], line_no: int) -> None:
    for key in ("savings_ms", "draft_estimate_ms", "funded_draft_ms",
                "funded_retries", "nonescalating_declines"):
        if item[key] < 0:
            raise EvidenceError(f"line {line_no}: negative savings field {key}")
    expected = max(0.0, -item["net_ms"])
    if abs(item["savings_ms"] - expected) > 2e-3:
        raise EvidenceError(f"line {line_no}: savings_ms is not max(0,-net_ms)")


def _validate_totals(item: Dict[str, Any], line_no: int, retry_schema: bool,
                     savings_schema: bool, public_schema: bool = False) -> None:
    required = (
        "done", "calls", "drafted", "proposed", "verified", "accepted", "returned", "consumed",
        "setup_ms", "refresh_ms", "draft_ms", "verify_ms", "heads_ms",
        "tail_ms", "total_ms", "invalid",
    )
    _require(item, required, line_no)
    for key in required:
        if key == "invalid":
            continue
        if item[key] < 0:
            raise EvidenceError(f"line {line_no}: negative total {key}")
    if item["invalid"] not in (0, 1):
        raise EvidenceError(f"line {line_no}: invalid must be 0 or 1")
    if item["done"] != 1:
        raise EvidenceError(f"line {line_no}: totals must have done=1")
    if retry_schema:
        _require(item, ("zero_prefix", "skipped_steps", "economic_declines",
                        "losing_cycles", "calibrations", "serial_calibrations",
                        "net_ms", "reference_ms", "net_percent", "probe_only",
                        "probe_entries", "probe_entry_consumed", "reference_step_ms",
                        "reference_source", "loss_floor_ms", "floor_source",
                        "recovery_used", "recovery_pending", "recovery_checks"), line_no)
        for key in ("zero_prefix", "skipped_steps", "economic_declines",
                    "losing_cycles", "calibrations", "serial_calibrations",
                    "probe_entries", "probe_entry_consumed"):
            if item[key] < 0:
                raise EvidenceError(f"line {line_no}: negative total {key}")
        _validate_meter(item, line_no, totals=True)
        expected_percent = 100.0 * item["net_ms"] / item["total_ms"] if item["total_ms"] else 0.0
        if abs(item["net_percent"] - expected_percent) > 2e-3:
            raise EvidenceError(f"line {line_no}: net_percent mismatch")
    if savings_schema:
        _require(item, ("savings_ms", "draft_estimate_ms", "funded_retries",
                        "funded_draft_ms", "nonescalating_declines"), line_no)
        _validate_savings(item, line_no)
    if public_schema:
        _require(item, ("serial_consumed", "entry_serial_tokens"), line_no)
        if not (0 <= item["entry_serial_tokens"] <= item["serial_consumed"] <=
                item["consumed"]):
            raise EvidenceError(f"line {line_no}: invalid total entry accounting")


def _finish_request(request: AdaptiveRequest) -> None:
    if len(request.calls) != len(request.acks):
        raise EvidenceError(
            f"request at line {request.policy_line}: {len(request.calls)} calls but "
            f"{len(request.acks)} ACKs"
        )
    if request.totals is None:
        raise EvidenceError(f"request at line {request.policy_line}: missing adaptive_totals")
    final_ack = request.terminal_ack or (request.acks[-1] if request.acks else None)
    if final_ack is None or final_ack["done"] != 1:
        raise EvidenceError(f"request at line {request.policy_line}: final ACK is not done")
    cumulative_returned = 0
    cumulative_consumed = 0
    previous_total_ms = 0.0
    retry_schema = "retry" in request.policy
    savings_schema = "savings_retry" in request.policy
    public_schema = "profile" in request.policy
    central_schema = retry_schema and final_ack["reference_source"] == \
        "serial-actual+verify-median9"
    funded_retries = 0
    funded_draft_ms = 0.0
    previous_nonescalating = 0
    draft_ema_ms = 0.0
    draft_estimate_ms = 0.0
    reference_serial_ms = 0.0
    reference_verify_ms = 0.0
    bootstrap_repriced_ms = 0.0
    serial_consumed = 0
    entry_serial_tokens = 0
    for index, (call, ack) in enumerate(zip(request.calls, request.acks), 1):
        if public_schema:
            if call["serial_consumed_before"] != serial_consumed:
                raise EvidenceError(
                    f"request line {request.policy_line} call {index}: "
                    "serial_consumed_before mismatch"
                )
            if call["chosen"] > 0 and serial_consumed < request.policy["min_serial_tokens"]:
                raise EvidenceError(
                    f"request line {request.policy_line} call {index}: proposal before entry gate"
                )
            if call["entry"] and serial_consumed >= request.policy["min_serial_tokens"]:
                raise EvidenceError(
                    f"request line {request.policy_line} call {index}: entry after gate"
                )
        if ack["returned"] != call["returned"]:
            raise EvidenceError(f"request line {request.policy_line} call {index}: returned mismatch")
        if call["status"] == "error":
            if ack["invalid"] != 1 or ack["done"] != 1 or ack["consumed"] != 0:
                raise EvidenceError(
                    f"request line {request.policy_line} call {index}: unsafe error ACK"
                )
        elif ack["invalid"] != 0:
            raise EvidenceError(
                f"request line {request.policy_line} call {index}: successful call invalidated"
            )
        if index < len(request.acks) and ack["done"]:
            raise EvidenceError(
                f"request line {request.policy_line}: call follows done ACK"
            )
        if call["status"] == "ok" and not (
            request.policy["n_min"] <= call["next"] <= request.policy["n_max"]
        ):
            raise EvidenceError(
                f"request line {request.policy_line} call {index}: next outside policy bounds"
            )
        if abs(call["p_min"] - request.policy["p_min"]) > 5e-7:
            raise EvidenceError(
                f"request line {request.policy_line} call {index}: p_min changed"
            )
        cumulative_returned += call["returned"]
        cumulative_consumed += ack["consumed"]
        if public_schema:
            if call["chosen"] == 0 and call["verified"] == 0:
                serial_consumed += ack["consumed"]
            if call["entry"]:
                entry_serial_tokens += ack["consumed"]
            if ack["serial_consumed"] != serial_consumed or \
                    ack["entry_serial_tokens"] != entry_serial_tokens:
                raise EvidenceError(
                    f"request line {request.policy_line} call {index}: "
                    "entry ACK accounting mismatch"
                )
        if ack["total_returned"] != cumulative_returned or \
                ack["total_consumed"] != cumulative_consumed:
            raise EvidenceError(
                f"request line {request.policy_line} call {index}: cumulative ACK mismatch"
            )
        if ack["total_ms"] < previous_total_ms:
            raise EvidenceError(
                f"request line {request.policy_line} call {index}: total_ms moved backwards"
            )
        previous_total_ms = ack["total_ms"]
        if savings_schema:
            if call["chosen"] and call["draft_ms"] > 0.0:
                draft_ema_ms = (0.8 * draft_ema_ms + 0.2 * call["draft_ms"]
                                if draft_ema_ms > 0.0 else call["draft_ms"])
                draft_estimate_ms = max(call["draft_ms"], draft_ema_ms)
            funded_retries += call["funded"]
            if call["funded"]:
                funded_draft_ms += call["draft_ms"]
            if ack["funded_retries"] != funded_retries:
                raise EvidenceError(
                    f"request line {request.policy_line} call {index}: funded retry count mismatch"
                )
            if abs(ack["funded_draft_ms"] - funded_draft_ms) > 2e-3:
                raise EvidenceError(
                    f"request line {request.policy_line} call {index}: funded draft time mismatch"
                )
            if abs(ack["draft_estimate_ms"] - draft_estimate_ms) > 2e-3:
                raise EvidenceError(
                    f"request line {request.policy_line} call {index}: draft estimate mismatch"
                )
            if ack["nonescalating_declines"] < previous_nonescalating:
                raise EvidenceError(
                    f"request line {request.policy_line} call {index}: nonescalating count moved backwards"
                )
            previous_nonescalating = ack["nonescalating_declines"]
        if central_schema:
            expected_serial_added = (
                call["serial_ms"] * ack["consumed"] / call["returned"]
                if ack["consumed"] and call["serial_ms"] > 0.0 and call["returned"] else 0.0
            )
            expected_verify_added = (
                ack["consumed"] * ack["reference_call_step_ms"]
                if ack["consumed"] and call["serial_ms"] == 0.0 and
                ack["reference_call_step_ms"] > 0.0 else 0.0
            )
            if abs(ack["reference_serial_added_ms"] - expected_serial_added) > 2e-3:
                raise EvidenceError(
                    f"request line {request.policy_line} call {index}: serial reference increment mismatch"
                )
            if abs(ack["reference_verify_added_ms"] - expected_verify_added) > 2e-3:
                raise EvidenceError(
                    f"request line {request.policy_line} call {index}: verify reference increment mismatch"
                )
            reference_serial_ms += ack["reference_serial_added_ms"]
            reference_verify_ms += (
                ack["reference_verify_added_ms"] + ack["bootstrap_added_ms"] -
                ack["bootstrap_reprice_ms"]
            )
            bootstrap_repriced_ms += ack["bootstrap_reprice_ms"]
            if abs(ack["reference_serial_ms"] - reference_serial_ms) > 2e-3:
                raise EvidenceError(
                    f"request line {request.policy_line} call {index}: cumulative serial reference mismatch"
                )
            if abs(ack["reference_verify_ms"] - reference_verify_ms) > 2e-3:
                raise EvidenceError(
                    f"request line {request.policy_line} call {index}: cumulative verify reference mismatch"
                )
            if abs(ack["bootstrap_repriced_ms"] - bootstrap_repriced_ms) > 2e-3:
                raise EvidenceError(
                    f"request line {request.policy_line} call {index}: cumulative bootstrap reprice mismatch"
                )
    if request.terminal_ack is not None:
        ack = request.terminal_ack
        if ack["consumed"] != 0 or ack["returned"] != 0 or ack["done"] != 1:
            raise EvidenceError(
                f"request line {request.policy_line}: invalid terminal ACK"
            )
        if ack["total_returned"] != cumulative_returned or \
                ack["total_consumed"] != cumulative_consumed:
            raise EvidenceError(
                f"request line {request.policy_line}: terminal cumulative ACK mismatch"
            )
        if ack["total_ms"] < previous_total_ms:
            raise EvidenceError(
                f"request line {request.policy_line}: terminal total_ms moved backwards"
            )
        if retry_schema and request.acks:
            prior = request.acks[-1]
            for key in ("total_consumed", "total_returned", "reference_ms",
                        "probe_only", "retry_remaining", "reference_step_ms",
                        "reference_source", "loss_floor_ms", "floor_source",
                        "recovery_used", "recovery_pending", "recovery_checks"):
                tolerance = 2e-3 if isinstance(ack[key], float) else 0
                if (abs(ack[key] - prior[key]) > tolerance if tolerance else
                        ack[key] != prior[key]):
                    raise EvidenceError(
                        f"request line {request.policy_line}: terminal ACK changed {key}"
                    )
        if public_schema and request.acks:
            prior = request.acks[-1]
            for key in ("serial_consumed", "entry_serial_tokens"):
                if ack[key] != prior[key]:
                    raise EvidenceError(
                        f"request line {request.policy_line}: terminal ACK changed {key}"
                    )
        if savings_schema and request.acks:
            prior = request.acks[-1]
            for key in ("draft_estimate_ms", "funded_retries", "funded_draft_ms",
                        "nonescalating_declines"):
                tolerance = 2e-3 if isinstance(ack[key], float) else 0
                if (abs(ack[key] - prior[key]) > tolerance if tolerance else
                        ack[key] != prior[key]):
                    raise EvidenceError(
                        f"request line {request.policy_line}: terminal ACK changed {key}"
                    )
        if central_schema:
            for key in ("reference_serial_added_ms", "reference_verify_added_ms",
                        "bootstrap_added_ms", "bootstrap_reprice_ms"):
                if ack[key] != 0.0:
                    raise EvidenceError(
                        f"request line {request.policy_line}: terminal ACK changed {key}"
                    )
            if request.acks:
                prior = request.acks[-1]
                for key in ("serial_min_ms", "serial_central_ms", "serial_samples",
                            "serial_window", "reference_unpriced_rows",
                            "reference_serial_ms", "reference_verify_ms",
                            "bootstrap_repriced_ms", "bootstrap_rows",
                            "bootstrap_step_ms"):
                    tolerance = 2e-3 if isinstance(ack[key], float) else 0
                    if (abs(ack[key] - prior[key]) > tolerance if tolerance else
                            ack[key] != prior[key]):
                        raise EvidenceError(
                            f"request line {request.policy_line}: terminal ACK changed {key}"
                        )
    totals = request.totals
    sums = {
        "done": 1,
        "calls": len(request.calls),
        "drafted": sum(c["chosen"] for c in request.calls),
        "proposed": sum(c["proposed"] for c in request.calls),
        "verified": sum(c["verified"] for c in request.calls),
        "accepted": sum(c["accepted"] for c in request.calls),
        "returned": sum(c["returned"] for c in request.calls),
        "consumed": sum(a["consumed"] for a in request.acks),
    }
    for key, expected in sums.items():
        if totals[key] != expected:
            raise EvidenceError(
                f"request line {request.policy_line}: total {key}={totals[key]} "
                f"but calls sum to {expected}"
            )
    if retry_schema:
        derived = {
            "zero_prefix": sum(c["chosen"] != 0 and c["candidate"] == 0
                               for c in request.calls),
            "skipped_steps": sum(c["skipped"] and a["consumed"] > 0
                                 for c, a in zip(request.calls, request.acks)),
            "economic_declines": sum(c["economic"] for c in request.calls),
            "calibrations": sum(c["calibration"] for c in request.calls),
            "serial_calibrations": sum(c["serial_calibration"] for c in request.calls),
            "recovery_checks": sum(c["recovery"] for c in request.calls),
        }
        for key, expected in derived.items():
            if totals[key] != expected:
                raise EvidenceError(
                    f"request line {request.policy_line}: total {key}={totals[key]} "
                    f"but calls imply {expected}"
                )
        if totals["recovery_checks"] > 1:
            raise EvidenceError(f"request line {request.policy_line}: repeated early recovery")
        if totals["probe_entry_consumed"] > totals["consumed"]:
            raise EvidenceError(f"request line {request.policy_line}: probe entry past consumption")
    if public_schema:
        if totals["serial_consumed"] != serial_consumed or \
                totals["entry_serial_tokens"] != entry_serial_tokens:
            raise EvidenceError(
                f"request line {request.policy_line}: total entry accounting mismatch"
            )
        if totals["entry_serial_tokens"] > request.policy["min_serial_tokens"]:
            raise EvidenceError(
                f"request line {request.policy_line}: entry tokens exceeded configured gate"
            )
    if savings_schema:
        if totals["funded_retries"] != funded_retries:
            raise EvidenceError(
                f"request line {request.policy_line}: total funded retries mismatch"
            )
        if abs(totals["funded_draft_ms"] - funded_draft_ms) > 2e-3:
            raise EvidenceError(
                f"request line {request.policy_line}: total funded draft time mismatch"
            )
        if abs(totals["draft_estimate_ms"] - draft_estimate_ms) > 2e-3:
            raise EvidenceError(
                f"request line {request.policy_line}: total draft estimate mismatch"
            )
        if totals["nonescalating_declines"] > (
                totals["zero_prefix"] + totals["economic_declines"]):
            raise EvidenceError(
                f"request line {request.policy_line}: nonescalating declines exceed declines"
            )
        if request.policy["savings_retry"] == 0 and (
                totals["funded_retries"] or totals["nonescalating_declines"]):
            raise EvidenceError(
                f"request line {request.policy_line}: disabled savings retry changed counters"
            )
    timer_sums = {
        "setup_ms": sum(c["setup_ms"] for c in request.calls),
        "refresh_ms": sum(c["serial_ms"] for c in request.calls),
        "draft_ms": sum(c["draft_ms"] for c in request.calls),
        "verify_ms": sum(c["verify_ms"] for c in request.calls),
        "heads_ms": sum(c["heads_ms"] for c in request.calls),
        "tail_ms": sum(c["tail_ms"] for c in request.calls),
    }
    tolerance = max(1e-5, len(request.calls) * 1e-6 + 1e-5)
    for key, expected in timer_sums.items():
        if abs(totals[key] - expected) > tolerance:
            raise EvidenceError(
                f"request line {request.policy_line}: total {key}={totals[key]:.6f} "
                f"but calls sum to {expected:.6f}"
            )
    if totals["invalid"] != final_ack["invalid"]:
        raise EvidenceError(f"request line {request.policy_line}: invalid total mismatch")
    if totals["total_ms"] + 1e-6 < sum(c["total_ms"] for c in request.calls):
        raise EvidenceError(f"request line {request.policy_line}: total_ms omits call work")
    if retry_schema:
        for key in ("reference_ms", "probe_only",
                    "reference_step_ms", "reference_source", "loss_floor_ms",
                    "floor_source", "recovery_used", "recovery_pending",
                    "recovery_checks"):
            tolerance = 2e-3 if isinstance(totals[key], float) else 0
            if (abs(totals[key] - final_ack[key]) > tolerance if tolerance else
                    totals[key] != final_ack[key]):
                raise EvidenceError(
                    f"request line {request.policy_line}: totals/ACK mismatch for {key}"
                )
    if public_schema:
        for key in ("serial_consumed", "entry_serial_tokens"):
            if totals[key] != final_ack[key]:
                raise EvidenceError(
                    f"request line {request.policy_line}: totals/ACK mismatch for {key}"
                )
    if central_schema:
        for key in ("serial_min_ms", "serial_central_ms", "serial_samples",
                    "serial_window", "reference_call_step_ms",
                    "reference_serial_added_ms", "reference_verify_added_ms",
                    "bootstrap_added_ms", "bootstrap_reprice_ms",
                    "reference_unpriced_rows", "reference_serial_ms",
                    "reference_verify_ms", "bootstrap_repriced_ms", "bootstrap_rows",
                    "bootstrap_step_ms"):
            tolerance = 2e-3 if isinstance(totals[key], float) else 0
            if (abs(totals[key] - final_ack[key]) > tolerance if tolerance else
                    totals[key] != final_ack[key]):
                raise EvidenceError(
                    f"request line {request.policy_line}: totals/ACK mismatch for {key}"
                )
    if savings_schema:
        for key in ("draft_estimate_ms", "funded_retries", "funded_draft_ms",
                    "nonescalating_declines"):
            tolerance = 2e-3 if isinstance(totals[key], float) else 0
            if (abs(totals[key] - final_ack[key]) > tolerance if tolerance else
                    totals[key] != final_ack[key]):
                raise EvidenceError(
                    f"request line {request.policy_line}: totals/ACK mismatch for {key}"
                )


def parse_evidence(text: str, *, require_adaptive: Optional[bool] = None,
                   require_counters: bool = True) -> Evidence:
    requests: List[AdaptiveRequest] = []
    current: Optional[AdaptiveRequest] = None
    counters: List[Dict[str, Any]] = []
    token_id_receipts: List[Dict[str, Any]] = []
    cycle_receipts: List[Dict[str, Any]] = []
    pending_call = False

    for line_no, raw_line in enumerate(text.splitlines(), 1):
        line = raw_line.strip()
        cycle = _parse_cycle(line, line_no)
        if cycle is not None:
            cycle_receipts.append(cycle)
            continue
        counter = _parse_counter(line, line_no)
        if counter is not None:
            counters.append(counter)
            continue
        ids_match = _RUNFX_IDS.match(line)
        if ids_match:
            ids = [int(value) for value in ids_match.group("ids").split()]
            count = int(ids_match.group("count"))
            if len(ids) != count:
                raise EvidenceError(
                    f"line {line_no}: token-id receipt count={count} but has {len(ids)} ids"
                )
            token_id_receipts.append({
                "tag": ids_match.group("tag"), "count": count, "ids": ids,
            })
            continue
        if line.startswith(POLICY_PREFIX):
            if current is not None:
                _finish_request(current)
            item = _typed(_kv(line, POLICY_PREFIX, line_no), line_no)
            _validate_policy(item, line_no)
            if require_adaptive is not None and bool(item["adaptive"]) != require_adaptive:
                raise EvidenceError(
                    f"line {line_no}: adaptive={item['adaptive']} does not match required "
                    f"adaptive={int(require_adaptive)}"
                )
            current = AdaptiveRequest(item, line_no)
            requests.append(current)
            pending_call = False
        elif line.startswith(CALL_PREFIX):
            if current is None:
                raise EvidenceError(f"line {line_no}: adaptive call before policy")
            if current.totals is not None or pending_call or current.terminal_ack is not None:
                raise EvidenceError(f"line {line_no}: call after totals or before ACK")
            item = _typed(_kv(line, CALL_PREFIX, line_no), line_no)
            _validate_call(item, line_no, "retry" in current.policy,
                           "savings_retry" in current.policy,
                           "profile" in current.policy)
            if current.policy.get("policy") == "full-block":
                _validate_full_block_call(item, line_no)
            current.calls.append(item)
            pending_call = True
        elif line.startswith(ACK_PREFIX):
            if current is None or current.totals is not None:
                raise EvidenceError(f"line {line_no}: ACK outside request")
            item = _typed(_kv(line, ACK_PREFIX, line_no), line_no)
            _validate_ack(item, line_no, "retry" in current.policy,
                          "savings_retry" in current.policy,
                          "profile" in current.policy)
            if pending_call:
                current.acks.append(item)
                pending_call = False
            elif (current.terminal_ack is None and item["consumed"] == 0 and
                  item["returned"] == 0 and item["done"] == 1):
                current.terminal_ack = item
            else:
                raise EvidenceError(f"line {line_no}: ACK without pending call")
        elif line.startswith(TOTALS_PREFIX):
            if current is None or pending_call or current.totals is not None:
                raise EvidenceError(f"line {line_no}: totals at invalid boundary")
            item = _typed(_kv(line, TOTALS_PREFIX, line_no), line_no)
            _validate_totals(item, line_no, "retry" in current.policy,
                             "savings_retry" in current.policy,
                             "profile" in current.policy)
            current.totals = item

    if current is not None:
        _finish_request(current)
    if require_counters and not counters:
        raise EvidenceError("missing complete-generation counters")
    return Evidence(requests, counters, token_id_receipts, cycle_receipts)


def validate_adaptive_generation_alignment(evidence: Evidence) -> List[Dict[str, Any]]:
    """Bind adaptive returned/consumed rows to native CLI position/output counters.

    DFlash evaluates every returned target row before the caller scans that block
    for EOS.  Therefore an EOS-truncated suffix advances the session position but
    is not visible output and is not ACK-consumed.  A limit-stopped CLI request
    may print one final sampled token without evaluating it.
    """
    if len(evidence.requests) != len(evidence.cli_counters):
        raise EvidenceError(
            f"adaptive/counter request count mismatch: {len(evidence.requests)} != "
            f"{len(evidence.cli_counters)}"
        )
    receipts: List[Dict[str, Any]] = []
    for index, (request, counter) in enumerate(
            zip(evidence.requests, evidence.cli_counters), 1):
        profile = request.policy.get("profile")
        expected_mode = "speculative" if profile in {"aggressive", "speculative"} else "conservative"
        if counter.get("resolved_mode") != expected_mode or \
                counter.get("pos_initial") is None:
            raise EvidenceError(
                f"request {index}: adaptive profile={profile or 'historical-conservative'} "
                f"needs matched resolved_mode={expected_mode} CLI counters")
        totals = request.totals
        generated, evaluated = counter["generated"], counter["evaluated"]
        consumed, returned = totals["consumed"], totals["returned"]
        if counter["final_pos"] - counter["pos_initial"] != evaluated:
            raise EvidenceError(f"request {index}: CLI position delta does not equal evaluated rows")
        if evaluated != returned:
            raise EvidenceError(f"request {index}: evaluated rows do not equal adaptive returned rows")
        if returned < consumed:
            raise EvidenceError(f"request {index}: adaptive returned rows precede consumed rows")
        if counter["stop"] in {"eos", "earlystop", "early_stop"}:
            if generated != consumed:
                raise EvidenceError(f"request {index}: natural-stop output does not equal ACK consumption")
            final_unevaluated = 0
        elif counter["stop"] in {"limit", "predict_limit"}:
            final_unevaluated = generated - consumed
            if final_unevaluated not in (0, 1):
                raise EvidenceError(
                    f"request {index}: limit stop has {final_unevaluated} unevaluated output rows"
                )
            if returned != consumed:
                raise EvidenceError(f"request {index}: limit stop has an unconsumed returned suffix")
        elif counter["stop"] == "cancel":
            final_unevaluated = generated - consumed
            if final_unevaluated < 0 or final_unevaluated > 1:
                raise EvidenceError(f"request {index}: cancelled output/consumption mismatch")
        else:
            raise EvidenceError(f"request {index}: unsupported CLI stop {counter['stop']!r}")
        receipts.append({
            "request": index,
            "generated_visible_tokens": generated,
            "ack_consumed_tokens": consumed,
            "evaluated_forward_positions": evaluated,
            "adaptive_returned_tokens": returned,
            "unconsumed_evaluated_suffix": returned - consumed,
            "final_unevaluated_output_tokens": final_unevaluated,
            "stop": counter["stop"],
            "profile": profile,
            "resolved_mode": counter["resolved_mode"],
        })
    return receipts


def validate_cached_horizon(evidence: Evidence, *, pos_initial: int,
                            generated: int, evaluated: int) -> List[int]:
    """Require that a cached fixed-horizon arm actually used the session path."""
    if len(evidence.cli_counters) != 1:
        raise EvidenceError(
            f"cached horizon needs one counter, got {len(evidence.cli_counters)}"
        )
    counter = evidence.cli_counters[0]
    if counter.get("pos_initial") != pos_initial:
        raise EvidenceError(
            f"cached payload was not restored: pos_initial={counter.get('pos_initial')!r}, "
            f"expected {pos_initial}"
        )
    if counter["generated"] != generated or counter["evaluated"] != evaluated:
        raise EvidenceError(
            f"cached horizon counts are generated={counter['generated']} "
            f"evaluated={counter['evaluated']}, expected {generated}/{evaluated}"
        )
    if counter["final_pos"] - counter["pos_initial"] != evaluated:
        raise EvidenceError("cached final position does not match evaluated positions")
    receipts = [item for item in evidence.token_id_receipts
                if item["tag"] == "cli-session-loop-forced"]
    if len(receipts) != 1:
        raise EvidenceError(
            f"cached horizon needs one session-loop token-id receipt, got {len(receipts)}"
        )
    if receipts[0]["count"] != generated:
        raise EvidenceError("cached token-id count does not match generated tokens")
    return receipts[0]["ids"]


def parse_adaptive_request(text: str) -> Evidence:
    """Parse one conservative controller or public full-block accounting request."""
    evidence = parse_evidence(text)
    if not evidence.requests:
        raise EvidenceError("missing adaptive active-path evidence")
    return evidence
