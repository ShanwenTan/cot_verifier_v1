"""Wait for available GPUs and label supervised CoT by Qwen-to-Blender execution."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, FIRST_COMPLETED, wait
from datetime import datetime, timezone
from pathlib import Path
from queue import Queue
from threading import BoundedSemaphore
import argparse
import csv
import fcntl
import hashlib
import json
import os
import re
import signal
import socket
import subprocess
import time
import urllib.request
import uuid

from cot_text_smoke import attempt

MODEL_NAME = "qwen3_8_27b_native_cot_label"
FAIL = {"empty_model_code", "python_syntax_error", "blender_error", "invalid_mesh", "empty_render"}
SEQUENCE = Path("/data/wangshuo/code/wuxiao-clean-cot-comparison-training")
VLLM = Path("/data/wangshuo/nvme0/wangshuo/sft-training/checkpoints/vllm-venv/bin/vllm")
MAX_CONTEXT = 32768


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def save(path, data):
    stage = path.with_suffix(".tmp")
    stage.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    stage.replace(path)


def inspect_inputs(args):
    cases, ids, manifest = [], set(), {}
    for split, file in (("train", args.train), ("val", args.val)):
        digest, failures, image_counts = hashlib.sha256(), 0, Counter()
        start = len(cases)
        with file.open("rb") as entries:
            for line_no, line in enumerate(entries, 1):
                digest.update(line)
                if not line.strip():
                    continue
                row = json.loads(line)
                system, user, answer = row["messages"]
                identity = row["id"]
                if [m["role"] for m in (system, user, answer)] != ["system", "user", "assistant"]:
                    raise ValueError(f"Unexpected roles in {file}:{line_no}")
                if not isinstance(identity, str) or not identity or identity in ids:
                    raise ValueError(f"Duplicate or missing ID in {file}:{line_no}: {identity}")
                ids.add(identity)
                cot = answer.get("reasoning_content")
                problem, images = None, []
                if not isinstance(cot, str) or not cot.strip():
                    problem = "missing_supervised_cot"
                if isinstance(user["content"], list):
                    for item in user["content"]:
                        if item.get("type") == "image":
                            path = (args.media_root / item["path"]).resolve()
                            if not path.is_relative_to(args.media_root) or not path.is_file():
                                problem = f"input_media_unavailable: {item['path']}"
                            images.append(path)
                        elif item.get("type") != "text":
                            problem = "unsupported_input_block"
                elif not isinstance(user["content"], str):
                    problem = "unsupported_input_format"
                if len(images) > 4:
                    problem = "image_count_exceeds_server_limit"
                failures += bool(problem)
                image_counts[len(images)] += 1
                answer.pop("content", None)
                cases.append((split, line_no, row, problem))
        manifest[split] = {"path": str(file), "sha256": digest.hexdigest(), "rows": len(cases) - start,
                           "image_counts": {str(k): v for k, v in image_counts.items()}, "input_issues": failures}
    return cases, manifest


def nvidia_rows(fields, app=False):
    kind = "compute-apps" if app else "gpu"
    text = subprocess.check_output(["nvidia-smi", f"--query-{kind}={fields}",
                                    "--format=csv,noheader,nounits"], text=True, timeout=15)
    return [list(map(str.strip, row)) for row in csv.reader(text.splitlines()) if row]


def gpu_snapshot():
    active = Counter(row[0] for row in nvidia_rows("gpu_uuid,pid", app=True))
    devices = nvidia_rows("index,uuid,memory.used,memory.total,utilization.gpu")
    return [{"index": int(i), "uuid": u, "used_mib": int(used), "total_mib": int(total),
             "util": int(util), "compute_processes": active[u],
             "idle": int(total)-int(used) >= 64000 and int(used) <= 512 and int(util) <= 5 and not active[u]}
            for i, u, used, total, util in devices]


def wait_for_gpus(args, report):
    previous = set()
    while True:
        try:
            snapshot = gpu_snapshot()
            idle = {card["index"] for card in snapshot if card["idle"]}
            confirmed = idle & previous
            if len(confirmed) >= args.min_gpus:
                return [d for d in snapshot if d["index"] in sorted(confirmed)[:args.max_gpus]]
            report("waiting_for_gpus", gpus=snapshot, idle_count=len(idle), confirmed_count=len(confirmed))
            previous = idle
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            previous = set()
            report("waiting_for_gpus", gpu_probe_error=str(exc))
        time.sleep(args.poll_seconds)


def port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def launch_servers(args, selected, logs, eager=False):
    servers = []
    try:
        for card in selected:
            address = port()
            command = [str(args.vllm), "serve", str(args.checkpoint), "--host", "127.0.0.1",
                       "--port", str(address), "--served-model-name", MODEL_NAME, "--dtype", "bfloat16",
                       "--max-model-len", str(MAX_CONTEXT), "--gpu-memory-utilization", "0.85",
                       "--limit-mm-per-prompt", '{"image": 4}', "--reasoning-parser", "qwen3",
                       "--generation-config", "vllm", "--max-num-seqs", str(args.requests_per_gpu)]
            if eager:
                command.append("--enforce-eager")
            logfile = logs / f"vllm-gpu{card['index']}-{uuid.uuid4().hex[:8]}.log"
            with logfile.open("wb") as log:
                process = subprocess.Popen(command, env={**os.environ, "CUDA_VISIBLE_DEVICES": card["uuid"]},
                                           stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                           start_new_session=True)
            servers.append((card["index"], f"http://127.0.0.1:{address}/v1", process, str(logfile)))
    except BaseException:
        stop_servers(servers)
        raise
    return servers


def ready_servers(servers, checkpoint):
    pending = servers[:]
    ready = []
    deadline = time.monotonic()+540
    while pending and time.monotonic() < deadline:
        for server in pending[:]:
            _, endpoint, process, _ = server
            if process.poll() is not None:
                pending.remove(server)
                continue
            try:
                with urllib.request.urlopen(endpoint+"/models", timeout=2) as response:
                    models = json.load(response)["data"]
                if any(m["id"] == MODEL_NAME and m["root"] == str(checkpoint) for m in models):
                    pending.remove(server)
                    ready.append(server)
            except (OSError, ValueError, KeyError):
                pass
        if pending:
            time.sleep(3)
    return ready


def stop_servers(servers):
    for _, _, process, _ in servers:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    for _, _, process, _ in servers:
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()


def classify(item, servers, slots, renders, args):
    split, line, row, input_problem = item
    identity = row["id"]
    record = {"id": identity, "split": split, "line": line, "asset": row.get("asset"),
              "task": row.get("task"), "model": MODEL_NAME, "cot_quality": "inconclusive",
              "attempts": [], "processed_utc": utc_now()}
    if input_problem:
        return {**record, "status": "input_error", "detail": input_problem}
    root = args.out / "cases" / f"{split}-{line:05d}"
    budget = args.max_tokens
    for _ in range(3):
        if not any(server[2].poll() is None for server in servers):
            return {**record, "status": "all_replicas_unavailable"}
        index, endpoint, process, log = slots.get()
        folder = root / uuid.uuid4().hex
        try:
            if process.poll() is not None:
                continue
            result = attempt(row, folder, endpoint, MODEL_NAME, args.media_root, args.sandbox,
                             budget, inference_timeout=650, render_timeout=120, render_slots=renders,
                             reject_truncation=True, quiet=True)
        except (OSError, ValueError, KeyError) as exc:
            result = {"status": "pipeline_error", "detail": str(exc)}
        finally:
            slots.put((index, endpoint, process, log))
        reason = result["status"]
        record["attempts"].append({"status": reason, "max_tokens": budget, "gpu": index,
                                   "evidence": str(folder.relative_to(args.out))})
        if reason == "output_limit":
            prompt_tokens = (result.get("usage") or {}).get("prompt_tokens", MAX_CONTEXT)
            increased = min(args.retry_tokens, MAX_CONTEXT-prompt_tokens-32)
            if increased > budget and len(record["attempts"]) < 3:
                budget = increased
                continue
        if reason == "inference_failed" and result.get("http_status") == 400:
            prompt_size = re.search(r"([\d,]+)\s+(?:input|prompt)\s+tokens", result.get("detail", ""), re.I)
            if prompt_size:
                reduced = MAX_CONTEXT-int(prompt_size[1].replace(",", ""))-32
                if 0 < reduced < budget and len(record["attempts"]) < 3:
                    budget = reduced
                    continue
        if reason == "inference_failed" and result.get("http_status", 503) in {408,429,500,502,503,504} and len(record["attempts"]) < 3:
            time.sleep(3)
            continue
        quality = "pass" if reason == "pass" else "fail" if reason in FAIL else "inconclusive"
        return {**result, **record, "cot_quality": quality, "status": reason}
    return {**record, "status": "inference_retries_exhausted"}


def result_counts(results):
    labels = Counter(entry["cot_quality"] for entry in results.values())
    reasons = Counter(entry["status"] for entry in results.values())
    return {"processed": len(results), "labels": dict(labels), "reasons": dict(reasons)}


def grade(cases, results, tags, servers, args, report):
    workers = len(servers)*args.requests_per_gpu
    renders = BoundedSemaphore(args.render_workers)
    slots = Queue()
    for server in servers:
        for _ in range(args.requests_per_gpu):
            slots.put(server)
    remaining = iter(item for item in cases if item[2]["id"] not in results)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        def submit():
            item = next(remaining, None)
            return pool.submit(classify, item, servers, slots, renders, args) if item else None
        pending = {job for _ in range(workers) if (job := submit())}
        while pending:
            finished, pending = wait(pending, timeout=30, return_when=FIRST_COMPLETED)
            for job in finished:
                record = job.result()
                if record["status"] != "all_replicas_unavailable":
                    tags.write(json.dumps(record, ensure_ascii=False) + "\n")
                    tags.flush()
                    results[record["id"]] = record
                if any(server[2].poll() is None for server in servers):
                    if new := submit():
                        pending.add(new)
            report("labeling", gpus=[s[0] for s in servers if s[2].poll() is None],
                   in_flight=len(pending), **result_counts(results))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--sandbox", required=True, type=Path)
    p.add_argument("--train", type=Path, default=SEQUENCE/"dataset/original_train.jsonl")
    p.add_argument("--val", type=Path, default=SEQUENCE/"dataset/original_val.jsonl")
    p.add_argument("--media-root", type=Path, default=Path.home()/"img2_3D")
    p.add_argument("--checkpoint", type=Path, default=Path.home()/"models/Qwen/Qwen3.8-27B")
    p.add_argument("--vllm", type=Path, default=VLLM)
    p.add_argument("--min-gpus", type=int, default=4)
    p.add_argument("--max-gpus", type=int, default=8)
    p.add_argument("--poll-seconds", type=int, default=30)
    p.add_argument("--requests-per-gpu", type=int, default=12)
    p.add_argument("--render-workers", type=int, default=24)
    p.add_argument("--max-tokens", type=int, default=8192)
    p.add_argument("--retry-tokens", type=int, default=16384)
    p.add_argument("--preflight", action="store_true")
    args = p.parse_args()
    if not (1 <= args.min_gpus <= args.max_gpus <= 8 and args.poll_seconds > 0
            and args.requests_per_gpu > 0 and args.render_workers > 0
            and 0 < args.max_tokens < args.retry_tokens < MAX_CONTEXT):
        p.error("Invalid GPU count, concurrency or token budget")
    for field in ("out","sandbox","train","val","media_root","checkpoint","vllm"):
        setattr(args, field, getattr(args, field).resolve())
    for required in (args.vllm, args.checkpoint/"config.json",
                     args.sandbox/"isolated_blender.sh", args.sandbox/"batch_blender_worker.py"):
        if not required.is_file():
            p.error(f"Missing runtime file: {required}")
    args.out.mkdir(parents=True, exist_ok=True)
    lock = (args.out/"controller.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    cases, manifest = inspect_inputs(args)
    options = {"dataset": manifest, "total": len(cases), "checkpoint": str(args.checkpoint),
               "media_root": str(args.media_root), "max_tokens": args.max_tokens,
               "retry_tokens": args.retry_tokens, "enable_thinking": False,
               "min_gpus": args.min_gpus, "max_gpus": args.max_gpus,
               "requests_per_gpu": args.requests_per_gpu, "render_workers": args.render_workers}
    provenance = args.out/"inputs.json"
    if provenance.exists() and json.loads(provenance.read_bytes())["dataset"] != manifest:
        p.error("Dataset changed since this output directory was initialized")
    save(provenance, options)

    def report(phase, **fields):
        save(args.out/"status.json", {"phase": phase, "updated_utc": utc_now(), "pid": os.getpid(),
                                     "total": len(cases), **fields})
    if args.preflight:
        print(json.dumps(options, ensure_ascii=False, indent=2))
        return

    logs = args.out/"logs"
    logs.mkdir(exist_ok=True)
    tags_path = args.out/"tags.jsonl"
    results = {}
    with tags_path.open("a+", encoding="utf-8") as tags:
        tags.seek(0)
        valid_end = 0
        for line in iter(tags.readline, ""):
            if not line.endswith("\n"):
                tags.seek(valid_end)
                tags.truncate()
                break
            entry = json.loads(line)
            results[entry["id"]] = entry
            valid_end = tags.tell()
        failed_starts = 0
        try:
            while len(results) < len(cases):
                def report_progress(phase, **fields):
                    report(phase, **result_counts(results), **fields)
                selected = wait_for_gpus(args, report_progress)
                servers = launch_servers(args, selected, logs, eager=bool(failed_starts))
                try:
                    report_progress("starting_model", gpus=[s[0] for s in servers], model_logs=[s[3] for s in servers])
                    ready = ready_servers(servers, args.checkpoint)
                    if len(ready) >= args.min_gpus:
                        before = len(results)
                        grade(cases, results, tags, ready, args, report)
                        failed_starts = 0 if len(results) > before else failed_starts+1
                    else:
                        failed_starts += 1
                finally:
                    stop_servers(servers)
                if len(results) < len(cases) and failed_starts >= 3:
                    raise RuntimeError("The required number of healthy vLLM replicas could not be sustained")
                if len(results) < len(cases):
                    report_progress("retrying_replicas", failed_starts=failed_starts)
                    time.sleep(args.poll_seconds)
            summary = {**result_counts(results), "by_split": {
                split: dict(Counter(r["cot_quality"] for r in results.values() if r["split"] == split))
                for split in manifest}, "dataset":manifest,
                "pass_rate_among_decided":sum(r["cot_quality"]=="pass" for r in results.values()) /
                max(1, sum(r["cot_quality"] in {"pass","fail"} for r in results.values())),
                "completed_utc":utc_now()}
            save(args.out/"summary.json", summary)
            report("completed", **result_counts(results))
        except BaseException as error:
            report("stopped" if isinstance(error, KeyboardInterrupt) else "error",
                   detail=str(error), **result_counts(results))
            raise


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, signal.default_int_handler)
    main()
