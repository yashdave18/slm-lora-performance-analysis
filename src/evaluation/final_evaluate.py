"""Evaluate the fixed selected adapter and its base on untouched test data."""
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from src.data.batching import CausalTextDataset, CausalCollator
from src.evaluation.metrics import MetricAccumulator
from src.inference.runtime import (checked_manifest, resolve_adapter, choose_device,
    load_model, autocast, write_json, read_json, Session, release)
from src.utils.helpers import load_yaml


def evaluate(model, dataset, pad_id, batch_size, bounds, device, precision):
    def collate(examples):
        return CausalCollator(pad_id)(examples), [e["source"] for e in examples]
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate)
    total = MetricAccumulator(bounds)
    sources = {}
    from src.evaluation.metrics import causal_nll
    model.eval()
    with torch.inference_mode():
        for index, (batch, names) in enumerate(loader, 1):
            batch = {k: v.to(device) for k, v in batch.items()}
            with autocast(device, precision):
                output = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False)
            losses, counts = causal_nll(output.logits, batch["labels"])
            total.update(losses, counts)
            for row, name in enumerate(names):
                accumulator = sources.setdefault(name, MetricAccumulator(bounds))
                accumulator.update(losses[row:row+1], counts[row:row+1])
            if index % 200 == 0 or index == len(loader):
                print(f"Test evaluation {index}/{len(loader)} batches", flush=True)
    result = total.compute()
    if result["n_tokens"] != dataset.n_targets:
        raise ValueError("Evaluation did not cover every prediction target")
    return {"metrics": result, "by_dataset": {k: v.compute() for k, v in sources.items()}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project-dir", type=Path, required=True)
    p.add_argument("--config", type=Path, default=Path("configs/final_evaluation.yaml"))
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = p.parse_args()
    cfg = load_yaml(args.config)
    if cfg["split"] != "test":
        raise ValueError("This command is the final test evaluation")
    if cfg["batch_size"] < 1 or cfg["max_sequence_length"] < 2:
        raise ValueError("Invalid batch or sequence length")
    device = choose_device(args.device)
    with autocast(device, cfg["precision"]):
        pass
    data, manifest = checked_manifest(args.project_dir, ["test"])
    adapter, adapter_info = resolve_adapter(args.project_dir, cfg["selected_run"], manifest)
    if adapter_info["checkpoint"] != cfg["expected_checkpoint"]:
        raise ValueError("Selected checkpoint changed; freeze selection before test evaluation")
    spec = {"stage": "final-test", "settings": cfg, "adapter": adapter_info,
            "model_name": manifest["model_name"], "model_revision": manifest["model_revision"],
            "test_sha256": manifest["output_file_sha256"]["test.jsonl"],
            "weight_dtype": "float32", "attention": "sdpa", "adapter_mode": "unmerged"}
    output = args.output_dir or args.project_dir / "results/final_test"
    dataset = CausalTextDataset(data / "test.jsonl", cfg["max_sequence_length"])
    expected = manifest["splits"]["test"]["tokens_including_eos"] - manifest["splits"]["test"]["documents"]
    if dataset.n_targets != expected:
        raise ValueError("Test manifest target count mismatch")
    session = Session(output, spec, device, args.resume)
    failed = True
    results = {}
    try:
        for label, path in (("baseline", None), ("selected", adapter)):
            destination = output / f"{label}.json"
            if destination.exists():
                result = read_json(destination)
            else:
                model, tokenizer = load_model(data, manifest, device, path)
                if cfg["max_sequence_length"] > model.config.max_position_embeddings:
                    raise ValueError("Sequence length exceeds model context")
                result = evaluate(model, dataset, manifest["pad_token_id"], cfg["batch_size"],
                                  cfg["length_bucket_upper_bounds"], device, cfg["precision"])
                write_json(destination, result)
                del model, tokenizer
                release()
            results[label] = result
            session.run.log({f"test/{label}/{k}": v for k, v in result["metrics"].items()
                             if k != "ppl_by_length_bucket"})
            session.table(f"test/{label}/by_dataset", [
                {"dataset": name, **{k: v for k, v in metrics.items() if k != "ppl_by_length_bucket"}}
                for name, metrics in result["by_dataset"].items()])
            session.table(f"test/{label}/by_length", [
                {"bucket": name, **metrics} for name, metrics in result["metrics"]["ppl_by_length_bucket"].items()])
            print(label, result["metrics"], flush=True)
        base_ppl = results["baseline"]["metrics"]["ppl"]
        selected_ppl = results["selected"]["metrics"]["ppl"]
        summary = {"specification": spec, "results": results,
                   "delta_ppl": selected_ppl - base_ppl,
                   "ppl_reduction_percent": 100 * (base_ppl - selected_ppl) / base_ppl,
                   "wandb_url": session.run.url}
        session.run.log({"test/delta_ppl": summary["delta_ppl"],
                         "test/ppl_reduction_percent": summary["ppl_reduction_percent"]})
        write_json(output / "summary.json", summary)
        print(f"PASS: final test results saved to {output}", flush=True)
        failed = False
    finally:
        session.finish(failed)


if __name__ == "__main__":
    main()
