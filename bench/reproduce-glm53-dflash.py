#!/usr/bin/env python3
"""Reproduce fixed-horizon GLM-5.3 DFlash serial/conservative comparisons.

The caller supplies every artifact and output path.  The driver runs no build,
scrubs inherited DS4_/MTL_ settings, retains every raw attempt, and reports only
the native complete-generation timer.  Pass --gpu-lock-script for the shared
lock protocol.  Without one, --i-own-exclusive-gpu is required explicitly.

The published model digests are recorded but the large files are not rehashed.
Their local size and full stat identity are checked and recorded honestly.
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
from typing import Any, Dict, Iterable, List, Optional


HERE = Path(__file__).resolve().parent
FIXTURE_MANIFEST = HERE / "fixtures" / "glm53-dflash" / "manifest.json"
EVIDENCE_PARSER = HERE / "dflash_adaptive_evidence.py"
PUBLISHED_MODEL_SHA256 = "828f413cae8ceee74796606814295e00e04c1c7b151aca9a3c5c75db32cb73e0"
PUBLISHED_MODEL_BYTES = 185299232064
PUBLISHED_DRAFT_SHA256 = "a4bbfbd9e5db62ea31c5cde0bab38a4f9be11a8005dfd078f0d455bb630d66a9"
PUBLISHED_DRAFT_BYTES = 2342595168
SOURCE_FILES = (
    "ds4.c", "ds4_dflash_adaptive.h", "ds4_dflash_glm.inc", "metal/dflash2.metal",
)


class ReproductionError(RuntimeError):
    pass


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scrubbed_environment(overrides: Dict[str, str], source: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    base = os.environ if source is None else source
    env = {
        key: value for key, value in base.items()
        if not key.startswith("DS4_") and not key.startswith("MTL_")
    }
    env.update(overrides)
    return env


def load_evidence_parser(path: Path):
    spec = importlib.util.spec_from_file_location("dflash_adaptive_evidence", path)
    if spec is None or spec.loader is None:
        raise ReproductionError(f"cannot import evidence parser: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def checked_fixtures(manifest_path: Path = FIXTURE_MANIFEST) -> Dict[str, Dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text())
    result: Dict[str, Dict[str, Any]] = {}
    for entry in manifest["workloads"]:
        item = dict(entry)
        path = (manifest_path.parent / entry["file"]).resolve()
        actual = sha256(path)
        if actual != entry["sha256"]:
            raise ReproductionError(
                f"fixture {entry['name']} hash mismatch: {actual} != {entry['sha256']}"
            )
        item["prompt"] = path
        result[entry["name"]] = item
    return result


def stat_identity(path: Path, *, published_sha256: Optional[str], expected_bytes: Optional[int],
                  kind: str) -> Dict[str, Any]:
    stat = path.stat()
    if expected_bytes is not None and stat.st_size != expected_bytes:
        raise ReproductionError(
            f"{kind} byte size {stat.st_size} does not match declared {expected_bytes}"
        )
    return {
        "path": str(path),
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "device": stat.st_dev,
        "inode": stat.st_ino,
        "published_sha256": published_sha256,
        "sha256_verified_locally": False,
        "identity_method": (
            ("published digest recorded; " if published_sha256 else "no published digest supplied; ") +
            "local byte size and exact stat identity checked; large artifact not rehashed"
        ),
    }


def checked_digest(value: Optional[str], label: str) -> Optional[str]:
    if value is None:
        return None
    if not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ReproductionError(f"{label} must be a lowercase SHA-256 digest")
    return value


def run_git(source_root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(source_root), *args], text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if result.returncode:
        raise ReproductionError(result.stderr.strip() or f"git {' '.join(args)} failed")
    return result.stdout


def source_identity(source_root: Path, allow_dirty: bool) -> Dict[str, Any]:
    head = run_git(source_root, "rev-parse", "HEAD").strip()
    status = run_git(source_root, "status", "--porcelain", "--untracked-files=all")
    if status and not allow_dirty:
        raise ReproductionError("source tree is dirty; commit it or pass --allow-dirty")
    files: Dict[str, Any] = {}
    for relative in SOURCE_FILES:
        path = source_root / relative
        if path.exists():
            files[relative] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    if not files:
        raise ReproductionError("source tree contains none of the declared DFlash sources")
    diff = run_git(source_root, "diff", "--binary", "HEAD") if status else ""
    return {
        "path": str(source_root), "git_head": head, "status_porcelain": status,
        "tracked_diff_sha256": hashlib.sha256(diff.encode()).hexdigest() if status else None,
        "source_files": files,
    }


class GpuLock:
    def __init__(self, script: Optional[Path], timeout: int):
        self.script = script
        self.timeout = timeout
        self.token: Optional[str] = None
        self.child: Optional[subprocess.Popen] = None
        self.release_error: Optional[str] = None

    def acquire(self) -> None:
        if self.script is None:
            return
        env = dict(os.environ)
        env["DRIVER_PID"] = str(os.getpid())
        result = subprocess.run(
            [str(self.script), "acquire", "reproduce-glm53-dflash", str(self.timeout)],
            env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        if result.returncode:
            raise ReproductionError(result.stderr.strip() or "GPU lock acquire failed")
        self.token = result.stdout.strip()
        if not self.token:
            raise ReproductionError("GPU lock returned no ownership token")

    def preflight(self) -> None:
        if self.script is None:
            return
        if not self.token:
            raise ReproductionError("GPU lock preflight without ownership")
        result = subprocess.run(
            [str(self.script), "preflight", self.token], text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        if result.returncode:
            raise ReproductionError(result.stderr.strip() or "GPU lock preflight failed")

    def terminate_child(self) -> None:
        if self.child is None or self.child.poll() is not None:
            return
        try:
            os.killpg(self.child.pid, signal.SIGTERM)
            self.child.wait(timeout=20)
        except ProcessLookupError:
            pass
        except subprocess.TimeoutExpired:
            try:
                os.killpg(self.child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            self.child.wait()

    def release(self) -> None:
        self.terminate_child()
        if self.script is None or not self.token:
            return
        result = subprocess.run(
            [str(self.script), "release", self.token], text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        if result.returncode:
            self.release_error = result.stdout.strip() or "GPU lock release failed"
        else:
            self.token = None


def policy_environment(savings_retry: bool, proposer_head_nt4: bool) -> Dict[str, str]:
    env = {
        "DS4_DFLASH_ADAPTIVE": "0",
        "DS4_DFLASH_P_MIN": "0.75",
        "DS4_DFLASH_MIN_DRAFT": "1",
        "DS4_DFLASH_MAX_DRAFT": "7",
        "DS4_DFLASH_START_DRAFT": "3",
        "DS4_DFLASH_RETRY": "1",
        "DS4_DFLASH_RETRY_MAX": "512",
        "DS4_DFLASH_RETRY_TAX": "0.01",
        "DS4_DFLASH_LOSS_BUDGET": "0.03",
        "DS4_DFLASH_EARLY_RECOVERY": "1",
        "DS4_DFLASH_SAVINGS_RETRY": "1" if savings_retry else "0",
    }
    if proposer_head_nt4:
        env["DS4_DFLASH_PROPOSER_HEAD_NT4"] = "1"
    return env


def command(binary: Path, model: Path, draft: Path, workload: Dict[str, Any], arm: str) -> List[str]:
    return [
        str(binary), "-m", str(model), "--dflash", str(draft),
        "--dflash-mode", arm, "--metal", "--nothink",
        "-c", str(workload["context"]), "-n", str(workload["tokens"]),
        "--temp", "0", "--prompt-file", str(workload["prompt"]),
    ]


def validate_attempt(record: Dict[str, Any], stderr_text: str, parser: Any,
                     workload: Dict[str, Any], arm: str, savings_retry: bool) -> None:
    errors: List[str] = []
    try:
        evidence = parser.parse_evidence(stderr_text, require_adaptive=False)
        if arm == "speculative":
            if len(evidence.requests) != 1:
                raise parser.EvidenceError(
                    "public speculative attempt needs one accounting request")
            request = evidence.requests[0]
            if request.policy.get("policy") != "full-block" or \
                    request.policy.get("profile") != "speculative":
                raise parser.EvidenceError(
                    "public speculative attempt did not use policy=full-block "
                    "profile=speculative")
            if not any(call["chosen"] == call["proposed"] ==
                       call.get("candidate") == call["verified"] == 7
                       for call in request.calls):
                raise parser.EvidenceError(
                    "public speculative attempt has no width-seven verification")
        if len(evidence.cli_counters) != 1:
            raise parser.EvidenceError(
                f"expected one complete-generation counter, got {len(evidence.cli_counters)}"
            )
        counter = evidence.cli_counters[0]
        record["native_generation_counters"] = counter
        generated = counter["generated"]
        if not (1 <= generated <= workload["tokens"]):
            errors.append(f"generated {generated} outside declared maximum")
        natural_stop = workload.get("natural_eos") and generated < workload["tokens"]
        # Serial/direct loops use the two ordinary final-token boundary forms.
        # A DFlash natural stop can evaluate a returned suffix past the visible EOS;
        # validate that case below against exact returned and ACK-consumed receipts.
        allowed_evaluated = {generated - 1, generated}
        if not evidence.requests and counter["evaluated"] not in allowed_evaluated:
            errors.append(
                f"evaluated {counter['evaluated']} not in expected {sorted(allowed_evaluated)}"
            )
        if counter["requested"] != workload["tokens"]:
            errors.append("native requested count does not match declared horizon")
        expected_stops = {"eos", "earlystop", "early_stop"} if natural_stop else {
            "limit", "predict_limit"
        }
        if counter["stop"] not in expected_stops:
            errors.append(
                f"generation stopped as {counter['stop']!r}, expected {sorted(expected_stops)}"
            )
        if counter.get("pos_initial") is not None:
            if counter["final_pos"] - counter["pos_initial"] != counter["evaluated"]:
                errors.append("native session position delta does not match evaluated count")
        elif counter.get("prompt_len") is not None:
            if counter["final_pos"] - counter["prompt_len"] != counter["generated"]:
                errors.append("native direct position delta does not match generated count")
        if counter.get("resolved_mode") and counter["resolved_mode"] != arm:
            errors.append("resolved mode does not match selected arm")
        if arm == "conservative":
            if len(evidence.requests) != 1:
                errors.append(
                    f"expected one conservative request receipt, got {len(evidence.requests)}"
                )
            for request in evidence.requests:
                policy = request.policy
                expected_policy = {
                    "adaptive": 0, "retry": 1,
                    "savings_retry": int(savings_retry), "n_min": 1,
                    "n_max": 7, "n_start": 3,
                }
                for key, expected in expected_policy.items():
                    if policy.get(key) != expected:
                        errors.append(f"policy {key}={policy.get(key)!r}, expected {expected}")
                if abs(policy["p_min"] - 0.75) > 5e-7:
                    errors.append("policy p_min changed")
        if arm in {"conservative", "speculative"} and evidence.requests:
            record["generation_alignment"] = parser.validate_adaptive_generation_alignment(
                evidence
            )
        if workload.get("cached"):
            record["token_ids"] = parser.validate_cached_horizon(
                evidence, pos_initial=workload["pos_initial"],
                generated=workload["tokens"], evaluated=workload["expected_evaluated"],
            )
        record["evidence"] = evidence.summary()
        record["engagement"] = evidence.mode_metrics(arm)
        if counter["decode_s"] <= 0 or not math.isfinite(counter["decode_s"]):
            errors.append("native complete-generation timer is invalid")
        else:
            record["complete_generation_tps"] = counter["generated"] / counter["decode_s"]
    except Exception as exc:
        errors.append(str(exc))
    if record["exit_code"] != 0:
        errors.append(f"process exit code {record['exit_code']}")
    record["validation_errors"] = errors
    record["valid"] = not errors


def run_attempt(lock: GpuLock, parser: Any, output: Path, source_root: Path,
                binary: Path, model: Path, draft: Path, workload: Dict[str, Any],
                arm: str, savings_retry: bool, proposer_head_nt4: bool) -> Dict[str, Any]:
    name = workload["name"]
    stem = f"{name}.{arm}"
    stdout_path = output / f"{stem}.out"
    stderr_path = output / f"{stem}.err"
    rc_path = output / f"{stem}.rc"
    overrides = {"DS4_GLM_GEN_COUNTERS": "1", "DS4_DFLASH_STATS": "1"}
    if arm == "conservative":
        overrides.update(policy_environment(savings_retry, proposer_head_nt4))
    if workload.get("cached"):
        overrides.update({
            "DS4_GLM_LOAD_PAYLOAD": str(workload["payload"]),
            "DS4_CLI_FORCE_SESSION": "1",
            "DS4_GLM_IGNORE_EOS": "1",
        })
    argv = command(binary, model, draft, workload, arm)
    record: Dict[str, Any] = {
        "workload": name, "arm": arm, "argv": argv,
        "environment_overrides": overrides,
        "environment_scrubbed_prefixes": ["DS4_", "MTL_"],
        "start_utc": utc_now(), "scope": workload["scope"],
    }
    if workload.get("category"):
        record["category"] = workload["category"]
        record["category_shape"] = workload["category_shape"]
    lock.preflight()
    started = time.monotonic()
    rc = 125
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        try:
            lock.child = subprocess.Popen(
                argv, cwd=source_root, env=scrubbed_environment(overrides),
                stdout=stdout, stderr=stderr, start_new_session=True,
            )
            rc = lock.child.wait()
        except Exception as exc:
            stderr.write((f"driver process error: {exc}\n").encode())
        finally:
            lock.child = None
    rc_path.write_text(f"{rc}\n")
    record.update({
        "exit_code": rc, "complete_utc": utc_now(),
        "process_wall_s": time.monotonic() - started,
        "stdout_bytes": stdout_path.stat().st_size,
        "stderr_bytes": stderr_path.stat().st_size,
        "stdout_sha256": sha256(stdout_path),
        "stderr_sha256": sha256(stderr_path),
    })
    validate_attempt(
        record, stderr_path.read_text(errors="replace"), parser,
        workload, arm, savings_retry,
    )
    (output / f"{stem}.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def comparisons(records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_name: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for record in records:
        by_name.setdefault(record["workload"], {})[record["arm"]] = record
    result: List[Dict[str, Any]] = []
    for name, arms in by_name.items():
        serial = arms.get("serial")
        valid = bool(serial and len(arms) >= 2 and all(item["valid"] for item in arms.values()))
        item: Dict[str, Any] = {
            "workload": name, "comparison_valid": valid,
            "modes": {},
            "quality_scope": (
                "Byte equality of greedy output only. Natural EOS and generated counts are "
                "reported per mode; limit-stopped structured outputs may be incomplete."
            ),
        }
        representative = serial or next(iter(arms.values()))
        if representative.get("category"):
            item["category"] = representative["category"]
            item["category_shape"] = representative["category_shape"]
        for mode, record in arms.items():
            mode_item = {
                "valid": record["valid"],
                "generated": record.get("native_generation_counters", {}).get("generated"),
                "complete_generation_tps": record.get("complete_generation_tps"),
                "engagement": record.get("engagement"),
                "output_identical_to_serial": bool(
                    serial and record["stdout_bytes"] == serial["stdout_bytes"] and
                    record["stdout_sha256"] == serial["stdout_sha256"]
                ),
            }
            if valid and mode != "serial":
                mode_item["throughput_change_vs_serial_percent"] = 100.0 * (
                    record["complete_generation_tps"] /
                    serial["complete_generation_tps"] - 1.0
                )
            if record.get("token_ids") is not None and serial:
                mode_item["token_ids_identical_to_serial"] = (
                    record["token_ids"] == serial.get("token_ids")
                )
            item["modes"][mode] = mode_item
        result.append(item)
    return result


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--draft-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True,
                        help="new directory; existing evidence is never overwritten")
    parser.add_argument(
        "--workloads",
        default=(
            "json512,prose512,sql512,middle-chat512,middle-summary512,"
            "middle-rag512,middle-factual512,middle-math512,middle-code512"
        ),
    )
    parser.add_argument("--model-published-sha256", default=PUBLISHED_MODEL_SHA256)
    parser.add_argument("--model-expected-bytes", type=int, default=PUBLISHED_MODEL_BYTES)
    parser.add_argument("--draft-published-sha256", default=PUBLISHED_DRAFT_SHA256)
    parser.add_argument("--draft-expected-bytes", type=int, default=PUBLISHED_DRAFT_BYTES)
    parser.add_argument("--gpu-lock-script", type=Path)
    parser.add_argument("--gpu-lock-timeout", type=int, default=0)
    parser.add_argument("--i-own-exclusive-gpu", action="store_true",
                        help="required for a live run when --gpu-lock-script is omitted")
    parser.add_argument("--proposer-head-nt4", action="store_true")
    parser.add_argument("--disable-savings-retry", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--cached-payload", type=Path)
    parser.add_argument("--cached-context", type=int)
    parser.add_argument("--cached-tokens", type=int)
    parser.add_argument("--cached-pos-initial", type=int)
    parser.add_argument("--cached-evaluated", type=int)
    parser.add_argument("--cached-payload-expected-bytes", type=int)
    parser.add_argument("--cached-payload-published-sha256")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    binary = args.binary.resolve()
    source_root = args.source_root.resolve()
    model = args.model.resolve()
    draft = args.draft_model.resolve()
    output = args.output.resolve()
    lock_script = args.gpu_lock_script.resolve() if args.gpu_lock_script else None
    if output.exists():
        raise ReproductionError(f"output already exists: {output}")
    if not args.dry_run and lock_script is None and not args.i_own_exclusive_gpu:
        raise ReproductionError(
            "live run without --gpu-lock-script requires --i-own-exclusive-gpu"
        )
    for path in (binary, source_root, model, draft, EVIDENCE_PARSER, FIXTURE_MANIFEST):
        if not path.exists():
            raise ReproductionError(f"required path does not exist: {path}")
    if lock_script is not None and not lock_script.exists():
        raise ReproductionError(f"GPU lock script does not exist: {lock_script}")
    if args.gpu_lock_timeout < 0:
        raise ReproductionError("GPU lock timeout must be nonnegative")

    model_sha = checked_digest(args.model_published_sha256, "model published digest")
    draft_sha = checked_digest(args.draft_published_sha256, "draft published digest")
    payload_sha = checked_digest(
        args.cached_payload_published_sha256, "cached payload published digest"
    )
    fixtures = checked_fixtures()
    names = [name.strip() for name in args.workloads.split(",") if name.strip()]
    if not names or len(set(names)) != len(names):
        raise ReproductionError("workloads must be a nonempty unique comma-separated list")
    unknown = sorted(set(names) - fixtures.keys())
    if unknown:
        raise ReproductionError(f"unknown workload(s): {', '.join(unknown)}")
    workloads = []
    for name in names:
        item = dict(fixtures[name])
        item.update({
            "tokens": item["max_generated_tokens"],
            "expected_evaluated": item["max_generated_tokens"] - 1,
            "arms": (["serial", "conservative", "speculative"]
                     if name.startswith("middle-") else ["serial", "conservative"]),
            "scope": (
                "predeclared original category-inspired coverage with natural EOS; "
                "not an actual Spec-Bench score"
                if name.startswith("middle-") else
                "predeclared complete 512-token greedy horizon; retain regardless of result"
            ),
        })
        workloads.append(item)

    cached_values = (
        args.cached_context, args.cached_tokens,
        args.cached_pos_initial, args.cached_evaluated,
    )
    if args.cached_payload is not None:
        if any(value is None for value in cached_values):
            raise ReproductionError(
                "cached payload requires --cached-context, --cached-tokens, "
                "--cached-pos-initial, and --cached-evaluated"
            )
        payload = args.cached_payload.resolve()
        if not payload.exists():
            raise ReproductionError(f"cached payload does not exist: {payload}")
        if args.cached_tokens <= 0 or args.cached_evaluated != args.cached_tokens - 1:
            raise ReproductionError("cached evaluated count must equal tokens-1")
        cached_prompt = output.parent / f".{output.name}.cached-prompt.txt"
        workloads.append({
            "name": "cached", "prompt": cached_prompt,
            "context": args.cached_context, "tokens": args.cached_tokens,
            "expected_evaluated": args.cached_evaluated,
            "pos_initial": args.cached_pos_initial, "payload": payload, "cached": True,
            "arms": ["serial", "conservative"],
            "scope": (
                "fixed restored-prefix horizon with EOS ignored; payload is external and not "
                "included; strict initial/evaluated/generated/token-ID receipts required"
            ),
        })
    elif any(value is not None for value in cached_values) or payload_sha is not None:
        raise ReproductionError("cached options require --cached-payload")

    source = source_identity(source_root, args.allow_dirty)
    identities = {
        "source": source,
        "binary": {"path": str(binary), "bytes": binary.stat().st_size, "sha256": sha256(binary)},
        "driver": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
        "evidence_parser": {"path": str(EVIDENCE_PARSER), "sha256": sha256(EVIDENCE_PARSER)},
        "fixture_manifest": {"path": str(FIXTURE_MANIFEST), "sha256": sha256(FIXTURE_MANIFEST)},
        "model": stat_identity(
            model, published_sha256=model_sha, expected_bytes=args.model_expected_bytes,
            kind="model",
        ),
        "draft_model": stat_identity(
            draft, published_sha256=draft_sha, expected_bytes=args.draft_expected_bytes,
            kind="draft model",
        ),
        "prompts": {
            item["name"]: {"path": str(item["prompt"]), "bytes": item["prompt"].stat().st_size,
                           "sha256": sha256(item["prompt"])}
            for item in workloads if not item.get("cached")
        },
    }
    if args.cached_payload is not None:
        identities["cached_payload"] = stat_identity(
            args.cached_payload.resolve(), published_sha256=payload_sha,
            expected_bytes=args.cached_payload_expected_bytes, kind="cached payload",
        )

    output.mkdir(parents=True, exist_ok=False)
    if args.cached_payload is not None:
        cached_prompt = output / "cached-prompt.txt"
        cached_prompt.write_text("x\n")
        workloads[-1]["prompt"] = cached_prompt
        identities["prompts"]["cached"] = {
            "path": str(cached_prompt), "bytes": cached_prompt.stat().st_size,
            "sha256": sha256(cached_prompt),
        }
    policy = policy_environment(not args.disable_savings_retry, args.proposer_head_nt4)
    plan = {
        "schema": 1, "declared_utc": utc_now(),
        "order": [[item["name"], arm] for item in workloads for arm in item["arms"]],
        "workloads": [
            {key: str(value) if isinstance(value, Path) else value
             for key, value in item.items()} for item in workloads
        ],
        "arm_policy": (
            "packaged endpoint fixtures run serial/conservative; six original middle "
            "fixtures run serial/conservative/speculative"
        ),
        "conservative_policy_environment": policy,
        "environment_scrubbed_prefixes": ["DS4_", "MTL_"],
        "gpu_lock": str(lock_script) if lock_script else None,
        "gpu_ownership": "shared lock" if lock_script else "caller attested exclusive ownership",
        "comparison_policy": (
            "all attempts retained in declared order; complete native generation timer only; "
            "no favorable subwindow or attempt selection"
        ),
        "large_artifact_hash_policy": (
            "record published digest and verify local byte size/stat; do not rehash model, "
            "draft model, or optional cached payload"
        ),
        "dry_run": args.dry_run,
    }
    (output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    (output / "identity.json").write_text(json.dumps(identities, indent=2) + "\n")
    shutil.copyfile(Path(__file__).resolve(), output / "driver.py")
    shutil.copyfile(EVIDENCE_PARSER, output / "dflash_adaptive_evidence.py")
    shutil.copytree(FIXTURE_MANIFEST.parent, output / "fixtures")
    if lock_script:
        shutil.copyfile(lock_script, output / "gpu-lock-script.snapshot")
    if args.dry_run:
        print(output)
        return 0

    parser = load_evidence_parser(EVIDENCE_PARSER)
    lock = GpuLock(lock_script, args.gpu_lock_timeout)
    records: List[Dict[str, Any]] = []
    overall_rc = 0
    prior_handlers: Dict[int, Any] = {}

    def interrupted(signum, _frame):
        lock.terminate_child()
        raise KeyboardInterrupt(f"signal {signum}")

    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        prior_handlers[sig] = signal.signal(sig, interrupted)
    try:
        lock.acquire()
        if lock.token:
            (output / "gpu-lock-token.txt").write_text(lock.token + "\n")
        for workload in workloads:
            for arm in workload["arms"]:
                record = run_attempt(
                    lock, parser, output, source_root, binary, model, draft,
                    workload, arm, not args.disable_savings_retry,
                    args.proposer_head_nt4,
                )
                records.append(record)
                if not record["valid"]:
                    overall_rc = 1
                print(
                    f"completed workload={workload['name']} arm={arm} "
                    f"rc={record['exit_code']} valid={int(record['valid'])} "
                    f"tps={record.get('complete_generation_tps', 'n/a')}",
                    flush=True,
                )
    except KeyboardInterrupt:
        overall_rc = 130
    except Exception as exc:
        overall_rc = 1
        plan["driver_error"] = str(exc)
    finally:
        lock.release()
        for sig, handler in prior_handlers.items():
            signal.signal(sig, handler)
        if lock.release_error:
            overall_rc = 1
            plan["lock_release_error"] = lock.release_error

    report = {
        **plan, "complete_utc": utc_now(), "attempts": records,
        "comparisons": comparisons(records), "exit_code": overall_rc,
        "lock_released": lock.token is None,
        "raw_receipts_preserved": True,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (output / "driver.rc").write_text(f"{overall_rc}\n")
    print(output)
    return overall_rc


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ReproductionError as exc:
        print(f"reproducer: {exc}", file=sys.stderr)
        raise SystemExit(2)
