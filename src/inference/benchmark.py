"""Resumable fixed-output-length generation benchmarks on validation prompts."""
import argparse
import csv
import itertools
import json
from pathlib import Path
import random
import statistics
import time
import torch
from src.inference.runtime import (checked_manifest, resolve_adapter, choose_device,
    load_model, autocast, write_json, read_json, Session, release, synchronize,
    generation_config, identity)
from src.utils.helpers import load_yaml


def select_prompts(path, longest, batch_size, seed, eos_id):
    """Reservoir sample eligible validation documents; never read test prompts."""
    rng = random.Random(seed)
    selected = []
    eligible = 0
    with Path(path).open(encoding="utf-8") as file:
        for line in file:
            doc = json.loads(line)
            tokens = doc["input_ids"][:longest]
            if len(tokens) < longest or eos_id in tokens:
                continue
            eligible += 1
            item = {"id": doc["id"], "source": doc["source"], "input_ids": tokens}
            if len(selected) < batch_size:
                selected.append(item)
            else:
                index = rng.randrange(eligible)
                if index < batch_size:
                    selected[index] = item
    if len(selected) < batch_size:
        raise ValueError("Not enough eligible validation documents for benchmark prompts")
    return selected


def measured_generation(model, tokenizer, ids, precision, new_tokens):
    device = ids.device
    mask = torch.ones_like(ids)
    gc = generation_config(tokenizer, new_tokens, fixed_length=True)
    synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    with torch.inference_mode(), autocast(device, precision):
        output = model.generate(input_ids=ids, attention_mask=mask,
                                generation_config=gc, use_model_defaults=False)
    synchronize(device)
    elapsed = time.perf_counter() - started
    count = output.shape[0] * (output.shape[1] - ids.shape[1])
    if count != ids.shape[0] * new_tokens:
        raise ValueError("Generation did not produce the fixed token budget")
    memory = {
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None,
        "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20 if device.type == "cuda" else None,
    }
    return {"seconds": elapsed, "generated_tokens": count, **memory}


def summarize_measurements(samples, batch_size):
    seconds = [s["seconds"] for s in samples]
    total = sum(seconds)
    return {
        "latency_mean_ms": statistics.mean(seconds) * 1000,
        "latency_median_ms": statistics.median(seconds) * 1000,
        "latency_stdev_ms": statistics.stdev(seconds) * 1000 if len(seconds) > 1 else 0.0,
        "generated_tokens_per_sec": sum(s["generated_tokens"] for s in samples) / total,
        "sequences_per_sec": len(samples) * batch_size / total,
        "per_sequence_tokens_per_sec": sum(s["generated_tokens"] for s in samples) / total / batch_size,
        "peak_allocated_mib": max((s["peak_allocated_mib"] for s in samples if s["peak_allocated_mib"] is not None), default=None),
        "peak_reserved_mib": max((s["peak_reserved_mib"] for s in samples if s["peak_reserved_mib"] is not None), default=None),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project-dir", type=Path, required=True)
    p.add_argument("--config", type=Path, default=Path("configs/benchmark_config.yaml"))
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = p.parse_args()
    cfg = load_yaml(args.config)
    for key in ("max_new_tokens", "warmup_runs", "measured_runs"):
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if not cfg["variants"] or len(set(cfg["variants"])) != len(cfg["variants"]):
        raise ValueError("Provide unique nonempty variants")
    for key in ("batch_sizes", "prompt_lengths"):
        if not cfg[key] or any(type(x) is not int or x < 1 for x in cfg[key]):
            raise ValueError(f"Invalid {key}")
    if not cfg["precisions"]:
        raise ValueError("Provide at least one precision")
    device = choose_device(args.device)
    for precision in cfg["precisions"]:
        with autocast(device, precision):
            pass
    data, manifest = checked_manifest(args.project_dir, ["validation"])
    variants = {}
    details = {}
    for name in cfg["variants"]:
        if name == "baseline":
            variants[name], details[name] = None, {"lora_rank": 0}
        else:
            variants[name], details[name] = resolve_adapter(args.project_dir, name, manifest)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(data / "tokenizer", local_files_only=True)
    prompts = select_prompts(data / "validation.jsonl", max(cfg["prompt_lengths"]),
                             max(cfg["batch_sizes"]), cfg["seed"], tokenizer.eos_token_id)
    spec = {"stage": "inference-benchmark", "settings": cfg, "variants": details,
            "model_name": manifest["model_name"], "model_revision": manifest["model_revision"],
            "validation_sha256": manifest["output_file_sha256"]["validation.jsonl"],
            "prompts": prompts, "weight_dtype": "float32", "attention": "sdpa",
            "adapter_mode": "unmerged", "fp16_mode": "autocast", "eos_stopping": False}
    output = args.output_dir or args.project_dir / "results/inference_benchmarks"
    session = Session(output, spec, device, args.resume)
    failed = True
    rows = []
    try:
        for name, adapter in variants.items():
            combinations = list(itertools.product(cfg["precisions"], cfg["batch_sizes"], cfg["prompt_lengths"]))
            model = None
            for precision, batch, length in combinations:
                case = {"variant": name, "precision": precision, "batch_size": batch,
                        "prompt_length": length, "new_tokens": cfg["max_new_tokens"]}
                destination = output / "cases" / (identity(case) + ".json")
                if destination.exists():
                    saved = read_json(destination)
                    rows.append(saved["summary"])
                    continue
                if model is None:
                    model, tokenizer = load_model(data, manifest, device, adapter)
                if length + cfg["max_new_tokens"] > model.config.max_position_embeddings:
                    raise ValueError("Requested generation exceeds context limit")
                ids = torch.tensor([p["input_ids"][:length] for p in prompts[:batch]], device=device)
                samples = []
                try:
                    # Fresh allocator cache per case; warmups then establish steady state.
                    release()
                    for _ in range(cfg["warmup_runs"]):
                        measured_generation(model, tokenizer, ids, precision, cfg["max_new_tokens"])
                    for _ in range(cfg["measured_runs"]):
                        samples.append(measured_generation(model, tokenizer, ids, precision, cfg["max_new_tokens"]))
                    row = {**case, "status": "ok", **summarize_measurements(samples, batch)}
                except torch.cuda.OutOfMemoryError:
                    row = {**case, "status": "oom"}
                del ids
                release()
                write_json(destination, {"summary": row, "measurements": samples,
                                         "execution_id": session.run.id})
                rows.append(row)
                session.run.log({"benchmark/case_index": len(rows),
                                 **{f"benchmark/{k}": v for k, v in row.items() if isinstance(v, (int, float))}})
                print(row, flush=True)
            del model
            release()
        keys = list(dict.fromkeys(k for row in rows for k in row))
        with (output / "summary.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
        # Table columns are unioned so failed/OOM cases retain their status.
        session.table("benchmark/comparison", [{k: row.get(k) for k in keys} for row in rows])
        write_json(output / "summary.json", {"specification": spec, "rows": rows,
                                             "wandb_url": session.run.url})
        print(f"PASS: {len(rows)} benchmark cases recorded in {output}", flush=True)
        failed = False
    finally:
        session.finish(failed)


if __name__ == "__main__":
    main()
