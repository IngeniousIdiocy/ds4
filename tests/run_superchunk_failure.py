#!/usr/bin/env python3
"""Root-scheduled failure/re-prime screen; --self-test uses no GPU or server.

The caller owns an existing gpulock token. This script only starts/stops its
own server, and deliberately does not release the caller's lock.
"""
import argparse
import hashlib
import json
import os
import pathlib
import re
import socket
import subprocess
import time
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
COMPARATOR_VERSION = 2
FIRST_CHUNK = 4096
GROUP_TOKENS = 32768
FAULT_POSITION = FIRST_CHUNK + GROUP_TOKENS
FAULT = re.compile(r"TEST_SC_ROUTED_FAILURE group=(\d+) pos=(\d+) tokens=(\d+) "
                   r"layer=(\d+) routed=(\d+) banks=(\d+) refused=(\d+)")
CALLER = re.compile(r"TEST_SC_FAILURE_CALLER session=(\S+) status=(-?\d+) "
                    r"checkpoint=(\d+) valid=(\d+)")
PRIME = re.compile(r"TEST_SC_REPRIME session=(\S+) prior_len=(\d+) "
                   r"prior_valid=(\d+) prompt=(\d+)")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def verify(fault_log, recovery_log, before, after):
    fault = FAULT.findall(fault_log)
    caller = CALLER.findall(fault_log)
    require(len(fault) == len(caller) == 1, "need exactly one routed fault and invalid caller receipt")
    group, pos, tokens, layer, routed, banks, refused = map(int, fault[0])
    require((group, pos, tokens, routed, banks, refused) ==
            (2, FAULT_POSITION, GROUP_TOKENS, 8, 8, 0),
            "fault did not follow the ordinary 4096-token chunk, one full group and eight banked dispatches")
    require("pos=4096..36863" in fault_log and
            "status=1 (done) tokens=32768 chunks=4 routed_layers=42 bank_expanded=42" in fault_log,
            "first banked group did not complete after the ordinary initial chunk")
    session, status, checkpoint, valid = caller[0]
    require((int(status), int(checkpoint), int(valid)) == (-1, FAULT_POSITION, 0),
            "routed failure did not invalidate the nonempty checkpoint")
    require(fault_log.index("TEST_SC_ROUTED_FAILURE") < fault_log.index("TEST_SC_FAILURE_CALLER"),
            "caller receipt precedes the fault")
    prime = PRIME.findall(recovery_log)
    require(any(s == session and int(n) == FAULT_POSITION and int(v) == 0
                for s, n, v, _ in prime),
            "same session did not re-prime from invalid state")
    require(not FAULT.search(recovery_log), "fault repeated during recovery")
    require(isinstance(before, str) and before, "fresh baseline assistant content is empty")
    require(before.encode() == after.encode(), "recovered assistant content differs from fresh baseline")
    return {"comparator_version": COMPARATOR_VERSION, "ordinary_prefix": FIRST_CHUNK,
            "group": group, "checkpoint": pos, "routed": routed, "banks": banks,
            "layer": layer, "same_session": session, "content_bytes": len(before.encode())}


def self_test():
    # Frontier and summary fields from the retained 538c37c runtime receipt.
    f = ("GLM SUPER-CHUNK prefill ENGAGED: pos=4096..36863\n"
         "GLM SUPER-CHUNK done: status=1 (done) tokens=32768 chunks=4 routed_layers=42 bank_expanded=42\n"
         "TEST_SC_ROUTED_FAILURE group=2 pos=36864 tokens=32768 layer=10 routed=8 banks=8 refused=0\n"
         "TEST_SC_FAILURE_CALLER session=0x123 status=-1 checkpoint=36864 valid=0\n")
    r = "TEST_SC_REPRIME session=0x123 prior_len=36864 prior_valid=0 prompt=14\n"
    verify(f, r, "healthy", "healthy")
    controls = [
        (f.replace("pos=36864", "pos=0"), r, "healthy", "healthy"),
        (f.replace("pos=36864", "pos=32768"), r, "healthy", "healthy"),
        (f.replace("checkpoint=36864", "checkpoint=32768"), r, "healthy", "healthy"),
        (f.replace("status=1 (done)", "status=-1 (failed)"), r, "healthy", "healthy"),
        (f.replace("pos=4096..36863", "pos=0..32767"), r, "healthy", "healthy"),
        (f.replace("banks=8", "banks=7"), r, "healthy", "healthy"),
        (f.replace("valid=0", "valid=1"), r, "healthy", "healthy"),
        (f, r.replace("0x123", "0x456"), "healthy", "healthy"),
        (f, r.replace("prior_len=36864", "prior_len=0"), "healthy", "healthy"),
        (f, r, "healthy", "Healthy"),
        (f, r, "", ""),
    ]
    for args in controls:
        try:
            verify(*args)
        except RuntimeError:
            continue
        raise RuntimeError("comparator accepted a planted bad receipt")
    c = (ROOT / "ds4.c").read_text()
    stripped = re.sub(r"#if defined\(DS4_TEST_GLM_SC_FAIL_GROUP\)\n.*?#endif\n", "", c, flags=re.S)
    require("TEST_SC_" not in stripped and "test_sc_group" not in stripped,
            "fault logic or receipts escaped the dedicated compile-time guard")
    print(f"PASS failure comparator v{COMPARATOR_VERSION}: valid recovery, {len(controls)} planted refusals, production guards")


def reanalyze(directory):
    """Read preserved artifacts; never rewrite the original failed result."""
    original = json.loads((directory / "result.json").read_text())
    baseline = json.loads((directory / "baseline.response").read_text())
    fault = json.loads((directory / "fault.response").read_text())
    recovery = json.loads((directory / "recovery.response").read_text())
    require(original.get("server_returncode") == 0, "server did not exit cleanly")
    require("error" in fault, "fault response does not contain an error")
    for response in (baseline, recovery):
        require(response["usage"]["prompt_tokens_details"]["cached_tokens"] == 0,
                "baseline or recovery reused cached prompt tokens")
    result = verify((directory / "fault.log").read_text(),
                    (directory / "recovery.log").read_text(),
                    baseline["choices"][0]["message"]["content"],
                    recovery["choices"][0]["message"]["content"])
    result.update({"passed": True, "reanalyzed_from": str(directory.resolve()),
                   "original_result": original, "server_returncode": 0})
    print(json.dumps(result, indent=2))


def request(port, body=None):
    url = f"http://127.0.0.1:{port}/v1/" + ("models" if body is None else "chat/completions")
    req = urllib.request.Request(url, data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=900 if body else 2) as reply:
            return reply.status, reply.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--receipts", type=pathlib.Path, help="CPU-only reanalysis of a retained vehicle directory")
    parser.add_argument("--server", type=pathlib.Path, default=ROOT / "tests/ds4_server_sc_failure")
    parser.add_argument("--model", type=pathlib.Path)
    parser.add_argument("--fault-prompt", type=pathlib.Path, help="native ~62k-token prompt; padded by default")
    parser.add_argument("--padding", type=int, default=4096, help="append this many ' 17' units")
    parser.add_argument("--out", type=pathlib.Path)
    parser.add_argument("--port", type=int, default=8099)
    parser.add_argument("--lock-token")
    parser.add_argument("--lock-script", type=pathlib.Path,
                        help="GPU-lock helper; required for the runtime screen")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.receipts:
        reanalyze(args.receipts)
        return
    require(args.model and args.fault_prompt and args.out and args.lock_token and args.lock_script,
            "--model, --fault-prompt, --out, --lock-token and --lock-script are required")
    require(0 <= args.padding <= 8192, "padding must be 0..8192")
    require(args.server.is_file() and args.model.is_file(), "missing test server or model")
    require(b"TEST_SC_ROUTED_FAILURE" in args.server.read_bytes(), "server lacks the compile-only hook")
    subprocess.run([str(args.lock_script), "preflight", args.lock_token], check=True)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", args.port))  # refuse an existing listener
    args.out.mkdir(parents=True, exist_ok=False)
    baseline = "Reply with exactly one word: healthy."
    fault = args.fault_prompt.read_text() + "\nFailure fixture padding:" + " 17" * args.padding
    (args.out / "fault-prompt.txt").write_text(fault)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("DS4_", "MTL_"))}
    env.update({
        "DS4_METAL_MODEL_UNTRACKED": "1", "DS4_GLM53_MEMORY_CEILING_GB": "280",
        "DS4_GLM53_PREFILL_CHUNK": "8192", "DS4_GLM_ENABLE_SCORER_XREDUCE": "1",
        "DS4_GLM_ENABLE_EXPERT_BANK": "1", "DS4_GLM_EXPERT_BANK_FUSED_LAYER_CB": "1",
        "DS4_GLM_EXPERT_BANK_PIPELINED_LAYERS": "1", "DS4_GLM_EXPERT_BANK_TIMING": "1",
        "DS4_GLM_EXPERT_BANK_MIN_TOKENS": "32768", "DS4_GLM_EXPERT_BANK_SUPERCHUNK": "32768",
        "DS4_GLM_DISABLE_ROUTER_SPLITK_B4": "1",
    })
    command = [str(args.server.resolve()), "-m", str(args.model.resolve()), "--metal",
               "--dflash-mode", "serial", "--ctx", "81920", "--tokens", "16",
               "--host", "127.0.0.1", "--port", str(args.port)]
    # No --kv-disk-dir, no batched sessions: persistent cache is disabled and
    # every request runs sequentially on the one default worker/session.
    identity = {"command": command, "environment": {k: v for k, v in env.items() if k.startswith("DS4_")},
                "server_sha256": hashlib.sha256(args.server.read_bytes()).hexdigest(),
                "fault_prompt_sha256": hashlib.sha256(fault.encode()).hexdigest(),
                "model_path": str(args.model.resolve()), "model_bytes": args.model.stat().st_size}
    (args.out / "identity.json").write_text(json.dumps(identity, indent=2))
    (args.out / "driver.py").write_bytes(pathlib.Path(__file__).read_bytes())
    result = {"passed": False}
    log_path = args.out / "server.err"
    with log_path.open("wb") as log:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=log)
        (args.out / "server.pid").write_text(str(process.pid))
        try:
            deadline = time.monotonic() + 300
            while True:
                require(process.poll() is None, "server exited during startup")
                try:
                    if request(args.port)[0] == 200:
                        break
                except (OSError, urllib.error.URLError):
                    pass
                require(time.monotonic() < deadline, "startup timed out")
                time.sleep(1)

            def run(name, text):
                start = log_path.stat().st_size
                body = {"model": "glm-5.3-flash", "messages": [{"role": "user", "content": text}],
                        "temperature": 0, "think": False, "stream": False, "max_tokens": 16}
                status, raw = request(args.port, body)
                (args.out / (name + ".response")).write_bytes(raw)
                segment = log_path.read_bytes()[start:].decode(errors="replace")
                (args.out / (name + ".log")).write_text(segment)
                return status, json.loads(raw), segment

            status_a, a, a_log = run("baseline", baseline)
            require(status_a == 200 and "GLM SUPER-CHUNK prefill ENGAGED" not in a_log,
                    "fresh short baseline failed or unexpectedly consumed a group")
            status_b, b, b_log = run("fault", fault)
            require(status_b >= 400, "fault request did not return an HTTP failure")
            status_c, c, c_log = run("recovery", baseline)
            require(status_c == 200, "recovery request failed")
            before = a["choices"][0]["message"]["content"]
            after = c["choices"][0]["message"]["content"]
            result.update(verify(b_log, c_log, before, after))
            result.update({"passed": True, "statuses": [status_a, status_b, status_c],
                           "baseline_content": before, "recovered_content": after})
        except Exception as error:
            result["error"] = repr(error)
        finally:
            process.terminate()  # only the server this driver created
            try:
                process.wait(timeout=120)
                result["server_returncode"] = process.returncode
                if process.returncode != 0:
                    result.update({"passed": False, "error": "owned server did not shut down cleanly"})
            except subprocess.TimeoutExpired:
                result.update({"passed": False, "error": "owned server failed to stop; lock remains owned",
                               "live_server_pid": process.pid})
            (args.out / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    require(result["passed"], "failure recovery screen failed; see receipts")


if __name__ == "__main__":
    main()
