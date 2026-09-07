#!/usr/bin/env python3
"""Run the frozen 12-row/category Spec-Bench subset through one DS4 server per mode.

The upstream dataset stays in a caller-supplied checkout.  This driver records
its exact revision and hashes, retains every request/response/trace/log segment,
and never retries or replaces a failed turn.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple
import urllib.error
import urllib.request


SPEC_BENCH_COMMIT = "fd2c1cd7d2201ef71db4c5f4e455008f017967bf"
QUESTION_SHA256 = "4b6d33e79484f9841c487ee87d1cf6aa8c6066f61d5d482ff09e5a007fafdf04"
SOURCE_LICENSE_METADATA_SHA256 = "63d401ffff8c1cc8244fb3e59139e1620ac40ced0b7450773d18ead119885353"
PUBLISHED_MODEL_SHA256 = "828f413cae8ceee74796606814295e00e04c1c7b151aca9a3c5c75db32cb73e0"
PUBLISHED_MODEL_BYTES = 185299232064
PUBLISHED_DRAFT_SHA256 = "a4bbfbd9e5db62ea31c5cde0bab38a4f9be11a8005dfd078f0d455bb630d66a9"
PUBLISHED_DRAFT_BYTES = 2342595168
ROWS_PER_CATEGORY = 12
MAX_GENERATED_TOKENS = 256
POST_RESPONSE_LOG_TIMEOUT = 5
MODES = ("serial", "conservative", "speculative")
LOGICAL_CATEGORIES = (
    "mt_bench", "translation", "summarization", "qa", "math_reasoning", "rag",
)
MT_BENCH_CATEGORIES = {
    "writing", "roleplay", "reasoning", "math", "coding", "extraction", "stem",
    "humanities",
}
PARSER_PATH = Path(__file__).with_name("dflash_adaptive_evidence.py")
FACTUAL_FIXTURE = Path(__file__).parent / "fixtures/glm53-dflash/middle-factual512.txt"
TRACE_SEGMENT = re.compile(
    rb"^===== request (?P<id>[0-9]+) .*? =====\n.*?^===== end request (?P=id) =====\n?",
    re.MULTILINE | re.DOTALL,
)
DECODE_PROGRESS = re.compile(
    r"^(?:[0-9]{4} [0-9]{2}:[0-9]{2}:[0-9]{2} )?"
    r"ds4-server: chat .*? gen=(?P<generated>[0-9]+).*? decoding .*? "
    r"(?P<elapsed>[0-9]+(?:\.[0-9]+)?)s$", re.MULTILINE,
)
FINAL_LOG = re.compile(
    r"^(?:[0-9]{4} [0-9]{2}:[0-9]{2}:[0-9]{2} )?"
    r"ds4-server: chat .*? gen=(?P<generated>[0-9]+).*? finish=(?P<finish>\S+) "
    r"(?P<elapsed>[0-9]+(?:\.[0-9]+)?)s$", re.MULTILINE,
)


class ReproductionError(RuntimeError):
    pass


def require(value: Any, message: str) -> None:
    if not value:
        raise ReproductionError(message)


def interrupted_signal(signum: int, frame: Any) -> None:
    raise KeyboardInterrupt(f"signal {signum}")


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def checked_digest(value: str, label: str) -> str:
    require(bool(re.fullmatch(r"[0-9a-f]{64}", value)), f"invalid {label}")
    return value


def run_text(argv: List[str], cwd: Path) -> str:
    return subprocess.run(
        argv, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        check=True,
    ).stdout.strip()


def git_identity(root: Path, allow_dirty: bool) -> Dict[str, Any]:
    head = run_text(["git", "rev-parse", "HEAD"], root)
    status = run_text(["git", "status", "--porcelain", "--untracked-files=no"], root)
    require(allow_dirty or not status, f"tracked source changes in {root}")
    return {"path": str(root), "head": head, "tracked_dirty": bool(status)}


def immutable_identity(path: Path, published_sha256: str, expected_bytes: int,
                       label: str) -> Dict[str, Any]:
    stat = path.stat()
    require(stat.st_size == expected_bytes,
            f"{label} size {stat.st_size}, expected {expected_bytes}")
    return {
        "path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
        "device": stat.st_dev, "inode": stat.st_ino,
        "published_sha256": published_sha256,
        "identity_method": (
            "published immutable digest recorded; local size checked and exact stat recorded; "
            "large artifact not rehashed"
        ),
    }


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, f"cannot load module {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_parser(path: Path):
    return load_module(path, "dflash_adaptive_evidence")


def logical_category(row: Dict[str, Any]) -> Optional[str]:
    category = row.get("category")
    if category in MT_BENCH_CATEGORIES:
        return "mt_bench"
    return category if category in LOGICAL_CATEGORIES else None


def select_subset_lines(raw_lines: List[bytes]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    selected: Dict[str, List[Dict[str, Any]]] = {name: [] for name in LOGICAL_CATEGORIES}
    for global_index, raw in enumerate(raw_lines):
        row = json.loads(raw)
        category = logical_category(row)
        if category is None or len(selected[category]) >= ROWS_PER_CATEGORY:
            continue
        require(isinstance(row.get("turns"), list) and row["turns"],
                f"row {global_index} has no turns")
        require(all(isinstance(turn, str) and turn for turn in row["turns"]),
                f"row {global_index} has an invalid turn")
        selected[category].append({
            "logical_category": category,
            "source_category": row["category"],
            "category_index": len(selected[category]),
            "global_index": global_index,
            "question_id": row["question_id"],
            "turns": row["turns"],
            "source_line_sha256": bytes_sha256(raw),
        })
    for category, rows in selected.items():
        require(len(rows) == ROWS_PER_CATEGORY,
                f"category {category} has only {len(rows)} selected rows")
    ordered = [row for category in LOGICAL_CATEGORIES for row in selected[category]]
    request_units = sum(len(row["turns"]) for row in ordered)
    require(len(ordered) == 72 and request_units == 84,
            f"pinned subset shape changed: rows={len(ordered)} turns={request_units}")
    manifest_rows = []
    for row in ordered:
        manifest_rows.append({
            key: value for key, value in row.items() if key != "turns"
        } | {
            "turn_count": len(row["turns"]),
            "turns": [
                {"turn_index": index, "bytes": len(text.encode()),
                 "sha256": bytes_sha256(text.encode())}
                for index, text in enumerate(row["turns"])
            ],
        })
    manifest = {
        "schema": 1,
        "upstream": "https://github.com/hemingkx/Spec-Bench",
        "commit": SPEC_BENCH_COMMIT,
        "question_file": "data/spec_bench/question.jsonl",
        "question_file_sha256": QUESTION_SHA256,
        "selection": (
            "stable file order; first 12 question rows in each of mt_bench, translation, "
            "summarization, qa, math_reasoning, rag"
        ),
        "conversation_semantics": (
            "turns execute sequentially; each generated assistant answer is included before "
            "the next source turn, matching upstream evaluation/eval.py"
        ),
        "question_rows": 72,
        "turn_requests_per_mode": 84,
        "modes": list(MODES),
        "total_turn_requests": 252,
        "max_generated_tokens_per_turn": MAX_GENERATED_TOKENS,
        "temperature": 0.0,
        "natural_eos": True,
        "rows": manifest_rows,
        "dataset_text_vendored": False,
    }
    return ordered, manifest


def load_subset(spec_root: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    question_file = spec_root / "data/spec_bench/question.jsonl"
    require(question_file.is_file(), f"missing {question_file}")
    require(sha256(question_file) == QUESTION_SHA256,
            "Spec-Bench question file differs from the pinned dataset")
    require(run_text(["git", "rev-parse", "HEAD"], spec_root) == SPEC_BENCH_COMMIT,
            "Spec-Bench checkout is not at the pinned commit")
    return select_subset_lines(question_file.read_bytes().splitlines(keepends=True))


def select_preflight_smoke(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Choose a fixed two-turn and single-turn row without changing the full subset."""
    mt = next((row for row in rows
               if row["logical_category"] == "mt_bench" and len(row["turns"]) == 2), None)
    single = next((row for row in rows
                   if row["logical_category"] == "qa" and len(row["turns"]) == 1), None)
    require(mt is not None and single is not None, "pinned preflight smoke rows are unavailable")
    require(FACTUAL_FIXTURE.is_file(), f"missing smoke fixture {FACTUAL_FIXTURE}")
    factual = FACTUAL_FIXTURE.read_text()
    factual_sha = bytes_sha256(factual.encode())
    extra = [
        {
            "logical_category": "preflight_factual_eos",
            "source_category": "original-factual-fixture",
            "category_index": index,
            "global_index": -1,
            "question_id": f"middle-factual512-{index + 1}",
            "turns": [factual],
            "source_line_sha256": factual_sha,
        }
        for index in range(2)
    ]
    return [mt, single, *extra]


def messages_for_turn(turns: List[str], answers: List[str], turn_index: int) -> List[Dict[str, str]]:
    require(turn_index <= len(answers), "missing prior answer for conversation turn")
    messages: List[Dict[str, str]] = []
    for index in range(turn_index):
        messages.extend((
            {"role": "user", "content": turns[index]},
            {"role": "assistant", "content": answers[index]},
        ))
    messages.append({"role": "user", "content": turns[turn_index]})
    return messages


def unit_id(row: Dict[str, Any], turn_index: int) -> str:
    return f"{row['logical_category']}.{row['category_index']:02d}.q{row['question_id']}.t{turn_index}"


def scrubbed_environment(overrides: Dict[str, str]) -> Dict[str, str]:
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("DS4_", "MTL_"))}
    env.update(overrides)
    return env


def mode_environment(mode: str, proposer_head_nt4: bool,
                     min_serial_tokens: int) -> Dict[str, str]:
    env = {"DS4_DFLASH_STATS": "1"}
    if mode == "conservative" and min_serial_tokens:
        # Zero is the public immediate-entry default and is deliberately left
        # unset.  A nonzero value is an explicit diagnostic override.
        env["DS4_DFLASH_MIN_SERIAL_TOKENS"] = str(min_serial_tokens)
    if mode == "conservative" and proposer_head_nt4:
        env["DS4_DFLASH_PROPOSER_HEAD_NT4"] = "1"
    return env


class GpuLock:
    def __init__(self, script: Optional[Path], timeout: int):
        self.script = script
        self.timeout = timeout
        self.token: Optional[str] = None
        self.child: Optional[subprocess.Popen] = None

    def acquire(self) -> None:
        if self.script is None:
            return
        result = subprocess.run(
            [str(self.script), "acquire", "spec-bench-subset", str(self.timeout)],
            env={**os.environ, "DRIVER_PID": str(os.getpid())}, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        require(result.returncode == 0 and result.stdout.strip(),
                result.stderr.strip() or "GPU lock acquire failed")
        self.token = result.stdout.strip()

    def preflight(self) -> None:
        if self.script is None:
            return
        require(self.token is not None, "GPU lock was not acquired")
        require(subprocess.run([str(self.script), "preflight", self.token]).returncode == 0,
                "GPU lock preflight failed")

    def release(self) -> Dict[str, Any]:
        if self.child is not None and self.child.poll() is None:
            try:
                terminate(self.child)
            except Exception as error:
                return {"required": self.script is not None, "returncode": 1,
                        "error": f"child cleanup failed; lock retained: {error}"}
        if self.script is None or self.token is None:
            return {"required": False}
        result = subprocess.run(
            [str(self.script), "release", self.token], text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        receipt = {"required": True, "returncode": result.returncode,
                   "output": result.stdout, "token": self.token}
        if result.returncode == 0:
            self.token = None
        return receipt


def terminate(process: subprocess.Popen) -> int:
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=15)
        except ProcessLookupError:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=10)
    return int(process.returncode)


def sample_idle_before_server(telemetry: Any, lock: GpuLock,
                              mode_dir: Path) -> Dict[str, Any]:
    lock.preflight()
    require(lock.child is None, "idle telemetry cannot run with a live model child")
    return telemetry.sample_idle_gpu(lock, mode_dir, "pre-server")


def wait_ready(port: int, process: subprocess.Popen, timeout: int) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        require(process.poll() is None, "server exited during startup")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=2) as reply:
                if reply.status == 200:
                    return
        except (OSError, urllib.error.URLError):
            pass
        time.sleep(1)
    raise ReproductionError("server readiness timeout")


def post(port: int, body: Dict[str, Any], timeout: int) -> Tuple[int, bytes]:
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(body, ensure_ascii=False).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as reply:
            return reply.status, reply.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()


def next_trace(path: Path, offset: int, timeout: int) -> Tuple[bytes, int]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            with path.open("rb") as stream:
                stream.seek(offset)
                chunk = stream.read()
            match = TRACE_SEGMENT.search(chunk)
            if match:
                return match.group(0), offset + match.end()
        time.sleep(.02)
    raise ReproductionError("complete server trace segment timeout")


def complete_log_segment(path: Path, offset: int, timeout: int) -> bytes:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with path.open("rb") as stream:
            stream.seek(offset)
            chunk = stream.read()
        if FINAL_LOG.search(chunk.decode(errors="replace")):
            time.sleep(.02)
            with path.open("rb") as stream:
                stream.seek(offset)
                return stream.read()
        time.sleep(.02)
    raise ReproductionError("final server log receipt timeout")


def trace_value(text: str, name: str) -> str:
    match = re.search(rf"^{re.escape(name)}:\s*(.*?)\s*$", text, re.MULTILINE)
    require(match is not None, f"trace missing {name}")
    return match.group(1)


def rendered_prompt_identity(trace: str) -> Dict[str, Any]:
    match = re.search(
        r"^--- rendered prompt ---\n(.*?)\n--- generated text ---$", trace,
        re.MULTILINE | re.DOTALL,
    )
    require(match is not None, "trace missing rendered prompt boundary")
    raw = match.group(1).encode()
    return {"bytes": len(raw), "sha256": bytes_sha256(raw)}


def validate_request_identity(trace_raw: bytes, body: Dict[str, Any]) -> None:
    trace = trace_raw.decode()
    match = re.search(r"^--- raw request json ---\n(.*?)\n--- rendered prompt ---$",
                      trace, re.MULTILINE | re.DOTALL)
    require(match is not None, "trace missing raw request identity")
    require(json.loads(match.group(1)) == body, "trace belongs to a different HTTP request")


def response_fields(raw: bytes) -> Dict[str, Any]:
    data = json.loads(raw)
    require(isinstance(data.get("choices"), list) and len(data["choices"]) == 1,
            "response must contain one choice")
    choice = data["choices"][0]
    content = choice.get("message", {}).get("content")
    usage = data.get("usage")
    require(isinstance(content, str), "response content is not text")
    require(isinstance(usage, dict), "response has no usage")
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        require(type(usage.get(key)) is int and usage[key] >= 0, f"invalid usage {key}")
    require(usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"],
            "usage total is not prompt+completion")
    finish = choice.get("finish_reason")
    require(finish in {"stop", "length"}, f"unexpected finish_reason {finish!r}")
    require(usage["completion_tokens"] <= MAX_GENERATED_TOKENS,
            "response exceeds declared generation cap")
    require(finish != "length" or usage["completion_tokens"] == MAX_GENERATED_TOKENS,
            "length stop before declared generation cap (possible context exhaustion)")
    content_raw = content.encode()
    return {
        "content": content, "content_bytes": len(content_raw),
        "content_sha256": bytes_sha256(content_raw), "finish_reason": finish,
        "natural_eos": finish == "stop", "usage": usage,
    }


def engagement(parser: Any, log_text: str, mode: str,
               generated_tokens: Optional[int] = None,
               finish_reason: Optional[str] = None,
               allow_blind_research: bool = False,
               require_public_profile: bool = False) -> Dict[str, Any]:
    evidence = parser.parse_evidence(log_text, require_adaptive=False,
                                     require_counters=False)
    if mode == "serial":
        require(not evidence.requests and not evidence.cycle_receipts,
                "serial request entered a DFlash evidence path")
        return {"cycles": 0, "verify_cycles": 0, "accepted_draft_tokens": 0,
                "accepted_draft_tokens_per_verify": None,
                "committed_tokens_per_verify": None,
                "committed_metric_source": "not-applicable"}
    if mode == "speculative" and not evidence.requests:
        require(allow_blind_research,
                "historical speculative request emitted cycle-only receipts")
        cycles = len(evidence.cycle_receipts)
        accepted = sum(item["accepted"] for item in evidence.cycle_receipts)
        return {
            "cycles": cycles, "verify_cycles": cycles,
            "accepted_draft_tokens": accepted,
            "accepted_draft_tokens_per_verify": accepted / cycles if cycles else None,
            "committed_tokens_per_verify": ((accepted + cycles) / cycles if cycles else None),
            "committed_metric_source": (
                "conventional estimate: accepted draft tokens plus one anchor per cycle; "
                "historical cycle-only receipts do not expose actual EOS-truncated consumption"
            ),
            "zero_accept_cycles": sum(item["accepted"] == 0
                                      for item in evidence.cycle_receipts),
            "receipt_source": "historical speculative cycle-only receipt",
        }
    require(len(evidence.requests) == 1,
            f"{mode} request needs one adaptive ledger")
    request = evidence.requests[0]
    if require_public_profile:
        expected_profile = "speculative" if mode == "speculative" else "conservative"
        require(request.policy.get("profile") == expected_profile,
                f"{mode} request missing {expected_profile} public profile receipt")
        expected_policy = "full-block" if mode == "speculative" else "confidence-prefix"
        require(request.policy.get("policy") == expected_policy,
                f"{mode} request missing policy={expected_policy} receipt")
    pairs = [(call, ack) for call, ack in zip(request.calls, request.acks)
             if call["verified"] > 0]
    full_width_pairs = [
        (call, ack) for call, ack in pairs
        if call["chosen"] == call["proposed"] == call.get("candidate") ==
           call["verified"] == 7 and call["confidence_first"] ==
           call["confidence_min"] == -1.0
    ]
    accepted = sum(call["accepted"] for call, _ in pairs)
    committed = sum(ack["consumed"] for _, ack in pairs)
    totals = request.totals
    if generated_tokens is not None:
        require(totals["consumed"] == generated_tokens,
                "server visible generation does not equal adaptive ACK consumption")
    unconsumed_suffix = totals["returned"] - totals["consumed"]
    require(unconsumed_suffix >= 0, "adaptive consumption exceeds returned rows")
    if unconsumed_suffix:
        require(finish_reason == "stop",
                "unconsumed evaluated suffix without natural stop")
    return {
        "cycles": sum(call["chosen"] > 0 for call in request.calls),
        "verify_cycles": len(pairs), "accepted_draft_tokens": accepted,
        "drafted_tokens": sum(call["chosen"] for call in request.calls),
        "full_width_verify_cycles": len(full_width_pairs),
        "accepted_draft_tokens_per_verify": accepted / len(pairs) if pairs else None,
        "committed_tokens": committed,
        "committed_tokens_per_verify": committed / len(pairs) if pairs else None,
        "committed_metric_source": "actual consumed count from each verified call's ACK",
        "receipt_source": ("full-block request ACK accounting"
                           if request.policy.get("policy") == "full-block"
                           else "adaptive request ACK ledger"),
        "profile": request.policy.get("profile"),
        "serial_consumed": totals.get("serial_consumed"),
        "entry_serial_tokens": totals.get("entry_serial_tokens"),
        "returned_target_rows": totals["returned"],
        "consumed_visible_rows": totals["consumed"],
        "unconsumed_evaluated_suffix": unconsumed_suffix,
        "declines": (None if request.policy.get("policy") == "full-block" else
                     totals.get("zero_prefix", 0) + totals.get("economic_declines", 0)),
        "skips": (None if request.policy.get("policy") == "full-block" else
                  totals.get("skipped_steps", 0)),
        "net_ms": totals.get("net_ms"),
        "net_percent_of_internal_actual": totals.get("net_percent"),
        "ledger_scope": (
            "runtime internal counterfactual; paired elapsed ratios are the measured comparison"
        ),
        "component_ms": {key: totals[key] for key in (
            "setup_ms", "refresh_ms", "draft_ms", "verify_ms", "heads_ms",
            "tail_ms", "total_ms")},
        "evidence": evidence.summary(),
    }


def validate_trace_and_log(trace_raw: bytes, log_raw: bytes,
                           response: Dict[str, Any]) -> Dict[str, Any]:
    trace = trace_raw.decode(errors="replace")
    log = log_raw.decode(errors="replace")
    usage = response["usage"]
    generated = int(trace_value(trace, "generated_tokens"))
    prompt_tokens = int(trace_value(trace, "effective_prompt_tokens"))
    cached = int(trace_value(trace, "cached_tokens"))
    require(0 <= cached <= prompt_tokens, "invalid cached-token count")
    require(usage.get("prompt_tokens_details", {}).get("cached_tokens") == cached,
            "trace/usage cached-token count mismatch")
    elapsed = float(trace_value(trace, "elapsed_sec"))
    require(math.isfinite(elapsed) and elapsed > 0, "invalid server elapsed timer")
    require(generated == usage["completion_tokens"], "trace/usage completion count mismatch")
    require(prompt_tokens == usage["prompt_tokens"], "trace/usage prompt count mismatch")
    require(int(trace_value(trace, "max_tokens")) == MAX_GENERATED_TOKENS,
            "server trace max_tokens changed")
    require(float(trace_value(trace, "temperature")) == 0.0,
            "server trace is not greedy")
    require(trace_value(trace, "finish") == response["finish_reason"],
            "trace/response finish mismatch")
    progress = list(DECODE_PROGRESS.finditer(log))
    decode_elapsed = None
    decode_source = "unavailable: zero generated tokens have no final decode-progress receipt"
    if generated:
        require(bool(progress), "missing final decode-progress receipt")
        final = progress[-1]
        require(int(final.group("generated")) == generated,
                "decode-progress generated count mismatch")
        decode_elapsed = float(final.group("elapsed"))
        require(math.isfinite(decode_elapsed) and decode_elapsed > 0,
                "invalid decode-progress timer")
        decode_source = (
            "final server decode-progress receipt; starts immediately after prefill completion"
        )
    return {
        "server_elapsed_s": elapsed,
        "server_elapsed_source": (
            "trace elapsed_sec; begins before session sync/prefill and ends before response publish"
        ),
        "decode_elapsed_s": decode_elapsed,
        "decode_elapsed_source": decode_source,
        "rendered_prompt": rendered_prompt_identity(trace),
        "trace_id": int(TRACE_SEGMENT.search(trace_raw).group("id")),
        "cached_tokens": cached,
        "cache_source": trace_value(trace, "cache_source"),
        "canonical_prompt_tokens": int(trace_value(trace, "prompt_tokens")),
    }


def server_command(server: Path, model: Path, draft: Path, mode: str, port: int,
                   context: int, trace: Path) -> List[str]:
    return [
        str(server), "-m", str(model), "--dflash", str(draft),
        "--dflash-mode", mode, "--metal", "--host", "127.0.0.1",
        "--port", str(port), "--ctx", str(context), "--tokens",
        str(MAX_GENERATED_TOKENS), "--trace", str(trace),
    ]


def attempt_one(parser: Any, mode_dir: Path, row: Dict[str, Any], turn_index: int,
                answers: List[str], mode: str, port: int, log_path: Path,
                trace_path: Path, trace_offset: int, timeout: int) -> Tuple[Dict[str, Any], int]:
    name = unit_id(row, turn_index)
    stem = mode_dir / name
    record: Dict[str, Any] = {
        "unit_id": name, "mode": mode, "logical_category": row["logical_category"],
        "source_category": row["source_category"], "category_index": row["category_index"],
        "global_index": row["global_index"], "question_id": row["question_id"],
        "turn_index": turn_index, "start_utc": utc_now(), "valid": False,
        "validation_errors": [],
    }
    try:
        messages = messages_for_turn(row["turns"], answers, turn_index)
    except Exception as error:
        record["validation_errors"].append(f"dependent turn unavailable: {error}")
        record["attempted"] = False
        record["complete_utc"] = utc_now()
        Path(f"{stem}.json").write_text(json.dumps(record, indent=2) + "\n")
        return record, trace_offset
    body = {
        "model": "glm-5.3-flash", "messages": messages, "temperature": 0.0,
        "think": False, "stream": False, "max_tokens": MAX_GENERATED_TOKENS,
        "seed": 424242,
    }
    request_raw = (json.dumps(body, ensure_ascii=False, indent=2) + "\n").encode()
    Path(f"{stem}.request.json").write_bytes(request_raw)
    record.update({"attempted": True, "request_bytes": len(request_raw),
                   "request_sha256": bytes_sha256(request_raw)})
    log_offset = log_path.stat().st_size
    started = time.monotonic()
    status = None
    response_raw = b""
    trace_raw = b""
    log_raw = b""
    try:
        status, response_raw = post(port, body, timeout)
        record["client_wall_s"] = time.monotonic() - started
        trace_raw, trace_offset = next_trace(trace_path, trace_offset, timeout)
        # The HTTP response already completed.  A missing final receipt is a local
        # attribution failure, not another potentially long model request.
        log_raw = complete_log_segment(
            log_path, log_offset, min(timeout, POST_RESPONSE_LOG_TIMEOUT))
        response = response_fields(response_raw)
        record.update(response)
        validate_request_identity(trace_raw, body)
        record.update(validate_trace_and_log(trace_raw, log_raw, response))
        require(turn_index != 0 or record["cached_tokens"] == 0,
                "unrelated question reused a cached prompt prefix")
        record["engagement"] = engagement(
            parser, log_raw.decode(errors="replace"), mode,
            response["usage"]["completion_tokens"], response["finish_reason"],
            require_public_profile=mode in {"conservative", "speculative"},
        )
        generated = response["usage"]["completion_tokens"]
        record["server_end_to_end_tps"] = generated / record["server_elapsed_s"]
        record["decode_only_tps"] = (
            generated / record["decode_elapsed_s"] if record["decode_elapsed_s"] else None
        )
        record["client_wall_tps"] = generated / record["client_wall_s"]
        require(status == 200, f"HTTP status {status}")
    except (Exception, KeyboardInterrupt) as error:
        if isinstance(error, KeyboardInterrupt):
            record["interrupted"] = True
        record.setdefault("client_wall_s", time.monotonic() - started)
        record["validation_errors"].append(str(error) or type(error).__name__)
    record.update({
        "http_status": status, "complete_utc": utc_now(),
        "response_bytes": len(response_raw), "response_sha256": bytes_sha256(response_raw),
        "trace_bytes": len(trace_raw), "trace_sha256": bytes_sha256(trace_raw),
        "server_log_bytes": len(log_raw), "server_log_sha256": bytes_sha256(log_raw),
    })
    record["valid"] = not record["validation_errors"]
    Path(f"{stem}.response").write_bytes(response_raw)
    Path(f"{stem}.trace.log").write_bytes(trace_raw)
    Path(f"{stem}.server.log").write_bytes(log_raw)
    Path(f"{stem}.status").write_text(f"{status}\n")
    Path(f"{stem}.json").write_text(json.dumps(record, indent=2) + "\n")
    return record, trace_offset


def run_mode(parser: Any, output: Path, rows: List[Dict[str, Any]], mode: str,
             server: Path, source_root: Path, model: Path, draft: Path, port: int,
             context: int, start_timeout: int, request_timeout: int,
             proposer_head_nt4: bool, min_serial_tokens: int, lock: GpuLock,
             telemetry: Optional[Any] = None) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    mode_dir = output / mode
    mode_dir.mkdir()
    log_path, trace_path = mode_dir / "server.log", mode_dir / "trace.log"
    command = server_command(server, model, draft, mode, port, context, trace_path)
    overrides = mode_environment(mode, proposer_head_nt4, min_serial_tokens)
    env = scrubbed_environment(overrides)
    process = None
    process_started = None
    records: List[Dict[str, Any]] = []
    lifecycle: Dict[str, Any] = {
        "mode": mode, "argv": command, "environment_overrides": overrides,
        "environment_scrubbed_prefixes": ["DS4_", "MTL_"], "start_utc": utc_now(),
    }
    try:
        if telemetry is not None:
            lifecycle["idle_gpu"] = sample_idle_before_server(telemetry, lock, mode_dir)
        else:
            lock.preflight()
        started = time.monotonic()
        process_started = started
        with log_path.open("wb") as log_stream:
            process = subprocess.Popen(
                command, cwd=source_root, env=env, stdout=log_stream, stderr=log_stream,
                start_new_session=True,
            )
            lock.child = process
            lifecycle["pid"] = process.pid
            wait_ready(port, process, start_timeout)
            lifecycle["startup_wall_s"] = time.monotonic() - started
            trace_offset = 0
            for row in rows:
                answers: List[str] = []
                for turn_index in range(len(row["turns"])):
                    record, trace_offset = attempt_one(
                        parser, mode_dir, row, turn_index, answers, mode, port, log_path,
                        trace_path, trace_offset, request_timeout,
                    )
                    records.append(record)
                    if record.get("interrupted"):
                        raise KeyboardInterrupt
                    if record["valid"] and isinstance(record.get("content"), str):
                        answers.append(record["content"])
                    with (mode_dir / "progress.jsonl").open("a") as progress:
                        progress.write(json.dumps({
                            "unit_id": record["unit_id"], "valid": record["valid"],
                            "complete_utc": record["complete_utc"],
                        }) + "\n")
            lifecycle["request_phase_wall_s"] = time.monotonic() - started - lifecycle["startup_wall_s"]
    except (Exception, KeyboardInterrupt) as error:
        lifecycle["error"] = repr(error)
        lifecycle["interrupted"] = isinstance(error, KeyboardInterrupt)
    finally:
        if process is not None:
            lifecycle["server_returncode"] = terminate(process)
        if process_started is not None:
            lifecycle["process_wall_s_including_startup_and_shutdown"] = (
                time.monotonic() - process_started
            )
        lock.child = None
        lifecycle["complete_utc"] = utc_now()
        lifecycle["records"] = len(records)
        lifecycle["valid_records"] = sum(record["valid"] for record in records)
        (mode_dir / "server-lifecycle.json").write_text(json.dumps(lifecycle, indent=2) + "\n")
    return records, lifecycle


def complete_declared_records(records: List[Dict[str, Any]], rows: List[Dict[str, Any]],
                              output: Path) -> List[Dict[str, Any]]:
    """Keep unattempted work visible after startup failure or interruption."""
    seen = {(record["mode"], record["unit_id"]) for record in records}
    for mode in MODES:
        for row in rows:
            for turn_index in range(len(row["turns"])):
                name = unit_id(row, turn_index)
                if (mode, name) in seen:
                    continue
                saved = output / mode / f"{name}.json"
                if saved.is_file():
                    record = json.loads(saved.read_text())
                    require(record["unit_id"] == name and record["mode"] == mode,
                            "retained attempt identity mismatch")
                    records.append(record)
                    continue
                record = {key: row[key] for key in (
                    "logical_category", "source_category", "category_index",
                    "global_index", "question_id")}
                record.update(unit_id=name, mode=mode, turn_index=turn_index,
                              attempted=False, valid=False, complete_utc=utc_now(),
                              validation_errors=["not attempted: mode failed or run interrupted"])
                records.append(record)
                (output / mode).mkdir(exist_ok=True)
                (output / mode / f"{name}.json").write_text(json.dumps(record, indent=2) + "\n")
    return records


def sum_optional(records: Iterable[Dict[str, Any]], key: str) -> Optional[float]:
    values = [record.get(key) for record in records]
    return sum(values) if values and all(value is not None for value in values) else None


def mean_question_tps(records: List[Dict[str, Any]], timer: str) -> Optional[float]:
    questions: Dict[Tuple[str, Any], List[Dict[str, Any]]] = {}
    for record in records:
        questions.setdefault(
            (record["logical_category"], record["question_id"]), []
        ).append(record)
    rates = []
    for turns in questions.values():
        if not all(turn.get(timer) is not None and turn.get("usage") for turn in turns):
            return None
        elapsed = sum(turn[timer] for turn in turns)
        generated = sum(turn["usage"]["completion_tokens"] for turn in turns)
        if elapsed <= 0:
            return None
        rates.append(generated / elapsed)
    return sum(rates) / len(rates) if rates else None


def group_summary(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    generated = sum(record.get("usage", {}).get("completion_tokens", 0) for record in records)
    prompt = sum(record.get("usage", {}).get("prompt_tokens", 0) for record in records)
    server_elapsed = sum_optional(records, "server_elapsed_s")
    decode_elapsed = sum_optional(records, "decode_elapsed_s")
    client_wall = sum_optional(records, "client_wall_s")
    engagement_rows = [record.get("engagement", {}) for record in records]
    verify_cycles = sum(item.get("verify_cycles", 0) for item in engagement_rows)
    accepted = sum(item.get("accepted_draft_tokens", 0) for item in engagement_rows)
    has_actual_committed = any("committed_tokens" in item for item in engagement_rows)
    committed_actual = sum(item.get("committed_tokens", 0) for item in engagement_rows)
    conventional_committed = sum(
        item.get("accepted_draft_tokens", 0) + item.get("verify_cycles", 0)
        for item in engagement_rows
    )
    committed_source = {item.get("committed_metric_source") for item in engagement_rows
                        if item.get("committed_metric_source")}
    return {
        "turn_requests": len(records), "valid": bool(records) and all(r["valid"] for r in records),
        "valid_turn_requests": sum(r["valid"] for r in records),
        "natural_eos_turns": sum(bool(r.get("natural_eos")) for r in records),
        "prompt_tokens": prompt, "generated_tokens": generated,
        "cached_prompt_tokens": sum(record.get("cached_tokens", 0) for record in records),
        "server_elapsed_s": server_elapsed, "decode_elapsed_s": decode_elapsed,
        "client_wall_s": client_wall,
        "server_end_to_end_tps": generated / server_elapsed if server_elapsed else None,
        "decode_only_tps": generated / decode_elapsed if decode_elapsed else None,
        "client_wall_tps": generated / client_wall if client_wall else None,
        "mean_question_server_end_to_end_tps": mean_question_tps(records, "server_elapsed_s"),
        "mean_question_decode_only_tps": mean_question_tps(records, "decode_elapsed_s"),
        "mean_question_client_wall_tps": mean_question_tps(records, "client_wall_s"),
        "cycles": sum(item.get("cycles", 0) for item in engagement_rows),
        "verify_cycles": verify_cycles, "accepted_draft_tokens": accepted,
        "accepted_draft_tokens_per_verify": accepted / verify_cycles if verify_cycles else None,
        "committed_tokens_actual": committed_actual if has_actual_committed else None,
        "committed_tokens_per_verify_actual": (
            committed_actual / verify_cycles if verify_cycles and has_actual_committed else None
        ),
        "committed_tokens_conventional_estimate": (
            conventional_committed if verify_cycles and not has_actual_committed else None
        ),
        "committed_tokens_per_verify_conventional_estimate": (
            conventional_committed / verify_cycles
            if verify_cycles and not has_actual_committed else None
        ),
        "committed_metric_sources": sorted(committed_source),
        # These controller fields are intentionally absent for serial and
        # explicitly non-applicable for public full-block speculative mode.
        # Preserve that distinction instead of coercing missing/None to zero.
        "declines": sum_optional(engagement_rows, "declines"),
        "skips": sum_optional(engagement_rows, "skips"),
        "internal_net_ms": sum_optional(engagement_rows, "net_ms"),
    }


def paired_comparison(serial: List[Dict[str, Any]], candidate: List[Dict[str, Any]]) -> Dict[str, Any]:
    left = {row["unit_id"]: row for row in serial}
    right = {row["unit_id"]: row for row in candidate}
    ids = sorted(set(left) & set(right))
    outputs = []
    for name in ids:
        a, b = left[name], right[name]
        outputs.append({
            "unit_id": name,
            "output_bytes_equal": (a.get("content_bytes") is not None and
                                   a.get("content_bytes") == b.get("content_bytes")),
            "output_sha256_equal": (a.get("content_sha256") is not None and
                                    a.get("content_sha256") == b.get("content_sha256")),
            "generated_tokens_equal": (
                a.get("usage", {}).get("completion_tokens") is not None and
                a.get("usage", {}).get("completion_tokens") ==
                b.get("usage", {}).get("completion_tokens")
            ),
            "rendered_prompt_equal": (
                a.get("rendered_prompt", {}).get("sha256") is not None and
                a.get("rendered_prompt", {}).get("sha256") ==
                b.get("rendered_prompt", {}).get("sha256")
            ),
        })
    result: Dict[str, Any] = {
        "matched_turn_requests": len(ids),
        "all_declared_units_matched": len(ids) == len(serial) == len(candidate),
        "comparison_valid": (len(ids) == len(serial) == len(candidate) and
                             all(row["valid"] for row in serial + candidate)),
        "byte_identical_outputs": sum(item["output_bytes_equal"] and
                                      item["output_sha256_equal"] for item in outputs),
        "generated_count_matches": sum(item["generated_tokens_equal"] for item in outputs),
        "rendered_prompt_matches": sum(item["rendered_prompt_equal"] for item in outputs),
        "output_equality": outputs,
    }
    for timer in ("server_elapsed_s", "decode_elapsed_s", "client_wall_s"):
        pairs = [(left[name].get(timer), right[name].get(timer)) for name in ids]
        if pairs and all(a is not None and b is not None for a, b in pairs):
            serial_total = sum(a for a, _ in pairs)
            mode_total = sum(b for _, b in pairs)
            have_counts = all(
                row.get("usage", {}).get("completion_tokens") is not None
                for name in ids for row in (left[name], right[name]))
            serial_tokens = (sum(left[name]["usage"]["completion_tokens"] for name in ids)
                             if have_counts else None)
            mode_tokens = (sum(right[name]["usage"]["completion_tokens"] for name in ids)
                           if have_counts else None)
            result[f"matched_{timer}_serial_total"] = serial_total
            result[f"matched_{timer}_mode_total"] = mode_total
            result[f"matched_{timer}_ratio_serial_over_mode"] = (
                serial_total / mode_total if mode_total else None
            )
            result[f"matched_{timer}_throughput_ratio_mode_over_serial"] = (
                (mode_tokens / mode_total) / (serial_tokens / serial_total)
                if mode_total and serial_total and serial_tokens else None
            )
    return result


def build_report(records: List[Dict[str, Any]], lifecycles: List[Dict[str, Any]],
                 expected_turn_requests: int, categories: Iterable[str],
                 run_scope: str, min_serial_tokens: int = 3) -> Dict[str, Any]:
    by_mode = {mode: [r for r in records if r["mode"] == mode] for mode in MODES}
    groups: Dict[str, Any] = {"overall": {}}
    for category in categories:
        groups[category] = {}
    for group in groups:
        for mode in MODES:
            rows = by_mode[mode] if group == "overall" else [
                r for r in by_mode[mode] if r["logical_category"] == group]
            groups[group][mode] = group_summary(rows)
        serial = by_mode["serial"] if group == "overall" else [
            r for r in by_mode["serial"] if r["logical_category"] == group]
        groups[group]["comparisons"] = {}
        for mode in ("conservative", "speculative"):
            candidate = by_mode[mode] if group == "overall" else [
                r for r in by_mode[mode] if r["logical_category"] == group]
            groups[group]["comparisons"][f"{mode}_vs_serial"] = paired_comparison(
                serial, candidate)
    report = {
        "schema": 1, "complete_utc": utc_now(), "run_scope": run_scope,
        "preflight_smoke_only": run_scope == "preflight-smoke",
        "declared_subset_claim_eligible": run_scope == "frozen-72-question-subset",
        "expected_turn_requests": expected_turn_requests,
        "observed_turn_records": len(records),
        "passed": (len(records) == expected_turn_requests and
                   all(r["valid"] for r in records) and
                   all("error" not in item for item in lifecycles)),
        "timing_definitions": {
            "server_elapsed_s": "native trace timer spanning prefill and decode",
            "decode_elapsed_s": "native final decode-progress timer after prefill",
            "client_wall_s": "client monotonic POST duration including transport and response handling",
            "ratio": "matched serial total elapsed divided by candidate total elapsed",
            "throughput_ratio": (
                "candidate generated-tokens/elapsed divided by serial generated-tokens/elapsed; "
                "reported separately because natural-EOS lengths may differ"
            ),
            "mean_question_tps": (
                "arithmetic mean across selected question rows of summed turn tokens divided "
                "by summed turn time, matching the aggregation shape of upstream speed.py; "
                "native response token counts and each declared timing boundary are used"
            ),
        },
        "tau_definitions": {
            "accepted_draft_tokens_per_verify": (
                "sum runtime accepted draft tokens divided by actual verify/cycle receipts"
            ),
            "committed_tokens_per_verify_conservative": (
                "sum real ACK consumed counts paired to verified calls divided by those calls; "
                "includes committed anchor tokens and respects terminal partial consumption"
            ),
            "committed_tokens_per_verify_speculative": (
                "real frontend ACK consumption paired to public full-block verify calls; "
                "conventional accepted+anchor estimate only for retained historical "
                "cycle-only receipts"
            ),
        },
        "groups": groups, "lifecycles": lifecycles,
    }
    if run_scope == "preflight-smoke":
        reuse: Dict[str, Any] = {}
        smoke_ok = True
        for mode in MODES:
            pair = [r for r in by_mode[mode]
                    if r["logical_category"] == "preflight_factual_eos"]
            item = {
                "request_count": len(pair),
                "both_valid": len(pair) == 2 and all(r["valid"] for r in pair),
                "both_natural_eos": len(pair) == 2 and all(r.get("natural_eos") for r in pair),
                "both_generated_three": len(pair) == 2 and all(
                    r.get("usage", {}).get("completion_tokens") == 3 for r in pair
                ),
                "outputs_identical": len(pair) == 2 and
                    pair[0].get("content_sha256") is not None and
                    pair[0].get("content_sha256") == pair[1].get("content_sha256"),
                "consecutive_trace_ids": len(pair) == 2 and
                    pair[1].get("trace_id") == pair[0].get("trace_id", -2) + 1,
            }
            if mode in {"conservative", "speculative"} and len(pair) == 2:
                if min_serial_tokens == 0 or mode == "speculative":
                    label = ("public_full_block_verified_partial_eos"
                             if mode == "speculative" else
                             "public_immediate_entry_verified_partial_eos")
                    item[label] = all(
                        r.get("engagement", {}).get("verify_cycles", 0) > 0 and
                        r.get("engagement", {}).get("unconsumed_evaluated_suffix", 0) > 0
                        for r in pair
                    )
                    if mode == "speculative":
                        item["public_full_block_width7_engagement"] = all(
                            r.get("engagement", {}).get("full_width_verify_cycles", 0) > 0
                            for r in pair
                        )
                else:
                    item["default_entry_gate_prevented_short_response_verify"] = all(
                        r.get("engagement", {}).get("verify_cycles", 0) == 0 and
                        r.get("engagement", {}).get("entry_serial_tokens") == 3
                        for r in pair
                    )
            reuse[mode] = item
            smoke_ok = smoke_ok and all(value for key, value in item.items()
                                        if key != "request_count") and item["request_count"] == 2
        report["preflight_factual_eos_reuse"] = reuse
        report["passed"] = report["passed"] and smoke_ok
    return report


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--draft-model", type=Path, required=True)
    parser.add_argument("--spec-bench-root", type=Path, required=True)
    parser.add_argument(
        "--source-license-metadata", type=Path, required=True,
        help="reviewed JSON metadata for the six source-dataset license scopes",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-published-sha256", default=PUBLISHED_MODEL_SHA256)
    parser.add_argument("--model-expected-bytes", type=int, default=PUBLISHED_MODEL_BYTES)
    parser.add_argument("--draft-published-sha256", default=PUBLISHED_DRAFT_SHA256)
    parser.add_argument("--draft-expected-bytes", type=int, default=PUBLISHED_DRAFT_BYTES)
    parser.add_argument("--gpu-lock-script", type=Path)
    parser.add_argument("--gpu-lock-timeout", type=int, default=0)
    parser.add_argument("--i-own-exclusive-gpu", action="store_true")
    parser.add_argument("--port", type=int, default=8099)
    parser.add_argument("--context", type=int, default=32768)
    parser.add_argument("--server-start-timeout", type=int, default=300)
    parser.add_argument("--request-timeout", type=int, default=600)
    parser.add_argument("--proposer-head-nt4", action="store_true")
    parser.add_argument(
        "--min-serial-tokens", type=int, choices=(0, 3), default=0,
        help=("leave the public immediate-entry default unset with 0; 3 is an explicit "
              "diagnostic serial-entry gate for conservative mode only"),
    )
    parser.add_argument(
        "--telemetry-helper", type=Path,
        help=("optional Python module exposing sample_idle_gpu; preflight smoke live runs "
              "require it and sample only before each server starts"),
    )
    parser.add_argument(
        "--preflight-smoke", action="store_true",
        help=("run pinned q81 (two turns), q321 (one turn), and two immediate "
              "three-token factual EOS/reuse requests across all modes; functional smoke "
              "only, never a subset performance claim"),
    )
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    source_root, server = args.source_root.resolve(), args.server.resolve()
    model, draft = args.model.resolve(), args.draft_model.resolve()
    spec_root, output = args.spec_bench_root.resolve(), args.output.resolve()
    license_metadata = args.source_license_metadata.resolve()
    lock_script = args.gpu_lock_script.resolve() if args.gpu_lock_script else None
    telemetry_path = args.telemetry_helper.resolve() if args.telemetry_helper else None
    require(not output.exists(), f"output already exists: {output}")
    require(args.context >= 4096, "context is too small for the benchmark")
    require(args.port > 0 and args.port + len(MODES) < 65536, "invalid port")
    require(args.gpu_lock_timeout >= 0, "negative lock timeout")
    if not args.dry_run and lock_script is None:
        require(args.i_own_exclusive_gpu,
                "live run without --gpu-lock-script requires --i-own-exclusive-gpu")
    if args.preflight_smoke and not args.dry_run:
        require(telemetry_path is not None,
                "live preflight smoke requires --telemetry-helper")
    for path in (source_root, server, model, draft, spec_root, PARSER_PATH,
                 license_metadata):
        require(path.exists(), f"missing required path {path}")
    require(sha256(license_metadata) == SOURCE_LICENSE_METADATA_SHA256,
            "source-dataset license metadata differs from reviewed receipt")
    if lock_script:
        require(lock_script.is_file(), f"missing lock script {lock_script}")
    if telemetry_path:
        require(telemetry_path.is_file(), f"missing telemetry helper {telemetry_path}")
    frozen_rows, manifest = load_subset(spec_root)
    run_scope = ("preflight-smoke" if args.preflight_smoke else
                 "frozen-72-question-subset")
    rows = select_preflight_smoke(frozen_rows) if args.preflight_smoke else frozen_rows
    turns_per_mode = sum(len(row["turns"]) for row in rows)
    expected_turn_requests = turns_per_mode * len(MODES)
    run_categories = tuple(dict.fromkeys(row["logical_category"] for row in rows))
    source = git_identity(source_root, args.allow_dirty)
    dataset_git = git_identity(spec_root, False)
    require(dataset_git["head"] == SPEC_BENCH_COMMIT, "dataset revision changed")
    model_digest = checked_digest(args.model_published_sha256, "model digest")
    draft_digest = checked_digest(args.draft_published_sha256, "draft digest")
    output.mkdir(parents=True)
    identities = {
        "source": source,
        "server": {"path": str(server), "bytes": server.stat().st_size,
                   "sha256": sha256(server)},
        "driver": {"path": str(Path(__file__).resolve()),
                   "sha256": sha256(Path(__file__).resolve())},
        "evidence_parser": {"path": str(PARSER_PATH), "sha256": sha256(PARSER_PATH)},
        "model": immutable_identity(model, model_digest, args.model_expected_bytes, "model"),
        "draft_model": immutable_identity(draft, draft_digest, args.draft_expected_bytes,
                                           "draft model"),
        "spec_bench": dataset_git | {
            "question_file_sha256": QUESTION_SHA256,
            "license_sha256": sha256(spec_root / "LICENSE"),
            "readme_sha256": sha256(spec_root / "Readme.md"),
            "eval_loader_sha256": sha256(spec_root / "evaluation/eval.py"),
            "speed_report_sha256": sha256(spec_root / "evaluation/speed.py"),
            "repository_url": "https://github.com/hemingkx/Spec-Bench",
            "license": "Apache-2.0 (upstream repository LICENSE only)",
            "dataset_license_scope": (
                "Repository license does not establish the licenses of every source dataset; "
                "source text is external and retains upstream attribution and terms."
            ),
        },
        "source_dataset_license_metadata": {
            "path": str(license_metadata), "bytes": license_metadata.stat().st_size,
            "sha256": SOURCE_LICENSE_METADATA_SHA256,
            "scope": (
                "reviewed category-source metadata and unresolved lineage; copied into the "
                "run receipt without treating it as a blanket text license"
            ),
        },
    }
    if telemetry_path:
        identities["idle_telemetry_helper"] = {
            "path": str(telemetry_path), "bytes": telemetry_path.stat().st_size,
            "sha256": sha256(telemetry_path),
            "sampling_contract": (
                "helper default: settle three seconds, then preserve three ioreg samples "
                "at offsets 0, 1.5, and 3 seconds"
            ),
        }
    policy = {
        mode: mode_environment(mode, args.proposer_head_nt4, args.min_serial_tokens)
        for mode in MODES
    }
    plan = {
        "schema": 1, "declared_utc": utc_now(), "mode_order": list(MODES),
        "one_server_process_per_mode": True, "model_loaded_once_per_mode": True,
        "prompt_cache_policy": (
            "disk cache unconfigured; first turn of every independent question requires zero "
            "cached tokens; subsequent turns may reuse same-mode conversation state"
        ),
        "multiturn_prompt_policy": (
            "each mode includes its own prior answer; later rendered prompts can differ "
            "across modes and their identities are compared, not assumed equal"
        ),
        "run_scope": run_scope,
        "preflight_smoke_only": args.preflight_smoke,
        "declared_subset_claim_eligible": not args.preflight_smoke,
        "question_rows_per_category": (ROWS_PER_CATEGORY if not args.preflight_smoke else None),
        "question_rows": len(rows), "turn_requests_per_mode": turns_per_mode,
        "total_turn_requests": expected_turn_requests,
        "run_categories": list(run_categories),
        "max_generated_tokens_per_turn": MAX_GENERATED_TOKENS,
        "natural_eos": True, "temperature": 0.0, "seed": 424242,
        "context": args.context, "base_port": args.port,
        "mode_policy_environment": policy,
        "min_serial_tokens_control": args.min_serial_tokens,
        "min_serial_tokens_source": (
            "selected binary public default; no environment override"
            if args.min_serial_tokens == 0 else
            "explicit diagnostic DS4_DFLASH_MIN_SERIAL_TOKENS override"
        ),
        "policy_defaults_source": (
            "selected binary; driver sets stats and the declared startup gate only"
        ),
        "environment_scrubbed_prefixes": ["DS4_", "MTL_"],
        "failure_policy": (
            "one declared attempt per turn; preserve errors and continue independent rows; "
            "never retry or select a favorable attempt"
        ),
        "comparison_policy": (
            "record every per-turn output equality, generated count, prompt identity, complete "
            "elapsed ratio, and throughput ratio; divergent natural-EOS lengths are retained"
        ),
        "metric_scope": (
            "functional preflight smoke only; no subset performance claim"
            if args.preflight_smoke else
            "bounded Spec-Bench subset; not a full published Spec-Bench score"
        ),
        "idle_telemetry_helper": str(telemetry_path) if telemetry_path else None,
        "gpu_lock": str(lock_script) if lock_script else None,
        "gpu_ownership": "shared lock" if lock_script else "caller attested exclusive ownership",
        "dry_run": args.dry_run,
    }
    (output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    (output / "identity.json").write_text(json.dumps(identities, indent=2) + "\n")
    (output / "subset-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if args.preflight_smoke:
        smoke_manifest = {
            "schema": 1,
            "scope": "functional preflight smoke; no subset performance claim",
            "source_manifest": "subset-manifest.json",
            "modes": list(MODES),
            "question_rows": len(rows),
            "turn_requests_per_mode": turns_per_mode,
            "total_turn_requests": expected_turn_requests,
            "max_generated_tokens_per_turn": MAX_GENERATED_TOKENS,
            "temperature": 0.0,
            "min_serial_tokens_control": args.min_serial_tokens,
            "natural_eos": True,
            "rows": [
                {
                    "logical_category": row["logical_category"],
                    "source_category": row["source_category"],
                    "category_index": row["category_index"],
                    "global_index": row["global_index"],
                    "question_id": row["question_id"],
                    "turn_count": len(row["turns"]),
                    "source_line_sha256": row["source_line_sha256"],
                    "turns": [
                        {"turn_index": index, "bytes": len(text.encode()),
                         "sha256": bytes_sha256(text.encode())}
                        for index, text in enumerate(row["turns"])
                    ],
                }
                for row in rows
            ],
            "dataset_text_vendored": False,
        }
        (output / "smoke-manifest.json").write_text(
            json.dumps(smoke_manifest, indent=2) + "\n"
        )
    shutil.copyfile(Path(__file__).resolve(), output / "driver.py")
    shutil.copyfile(PARSER_PATH, output / "dflash_adaptive_evidence.py")
    shutil.copyfile(license_metadata, output / "source-dataset-licenses.json")
    if lock_script:
        shutil.copyfile(lock_script, output / "gpu-lock-script.snapshot")
    if telemetry_path:
        shutil.copyfile(telemetry_path, output / "telemetry-helper.snapshot.py")
    if args.dry_run:
        (output / "driver.rc").write_text("0\n")
        print(output)
        return 0
    parser = load_parser(PARSER_PATH)
    telemetry = load_module(telemetry_path, "dflash_idle_telemetry_helper") \
        if telemetry_path else None
    lock = GpuLock(lock_script, args.gpu_lock_timeout)
    records: List[Dict[str, Any]] = []
    lifecycles: List[Dict[str, Any]] = []
    report: Dict[str, Any] = {"passed": False}
    previous_sigterm = signal.signal(signal.SIGTERM, interrupted_signal)
    try:
        lock.acquire()
        if lock.token:
            (output / "lock-token.txt").write_text(lock.token + "\n")
        for index, mode in enumerate(MODES):
            mode_records, lifecycle = run_mode(
                parser, output, rows, mode, server, source_root, model, draft,
                args.port + index, args.context, args.server_start_timeout,
                args.request_timeout, args.proposer_head_nt4,
                args.min_serial_tokens, lock,
                telemetry,
            )
            records.extend(mode_records)
            lifecycles.append(lifecycle)
            if lifecycle.get("interrupted"):
                break
        complete_declared_records(records, rows, output)
        report = build_report(
            records, lifecycles, expected_turn_requests, run_categories, run_scope,
            args.min_serial_tokens,
        )
    except (Exception, KeyboardInterrupt) as error:
        report = {"passed": False, "error": repr(error), "complete_utc": utc_now(),
                  "observed_turn_records": len(records), "lifecycles": lifecycles}
    finally:
        report["lock_release"] = lock.release()
        signal.signal(signal.SIGTERM, previous_sigterm)
        complete_declared_records(records, rows, output)
        report["observed_turn_records"] = len(records)
        if report["lock_release"].get("returncode", 0) != 0:
            report["passed"] = False
        (output / "attempts.jsonl").write_text(
            "".join(json.dumps(record) + "\n" for record in records)
        )
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        exit_code = 0 if report.get("passed") else 1
        (output / "driver.rc").write_text(f"{exit_code}\n")
        print(output)
    return exit_code


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ReproductionError as error:
        print(f"Spec-Bench subset driver: {error}", file=sys.stderr)
        raise SystemExit(2)
