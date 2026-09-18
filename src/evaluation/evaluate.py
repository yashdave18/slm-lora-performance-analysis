"""Evaluate the pretrained baseline on the validation split."""

import argparse
from importlib.metadata import version
import json
from pathlib import Path
import platform

import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
)
import wandb

from src.data.batching import (
    CausalTextDataset,
    CausalCollator,
    file_sha256,
)
from src.evaluation.metrics import causal_nll, MetricAccumulator
from src.utils.helpers import load_configs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    config = load_configs()
    evaluation = config["evaluation"]
    tracking = config["training"]["wandb"]

    if evaluation["split"] != "validation":
        raise ValueError("This baseline stage evaluates validation only.")

    metrics_path = args.output_dir / "metrics.json"

    if metrics_path.exists():
        raise FileExistsError(
            "Baseline results already exist. Use a new output directory."
        )

    if not torch.cuda.is_available():
        raise RuntimeError("Enable the Colab GPU before running evaluation.")

    set_seed(config["training"]["seed"])
    device = torch.device("cuda")

    manifest = json.loads(
        (args.data_dir / "tokenization_manifest.json").read_text(
            encoding="utf-8"
        )
    )

    revision = manifest["model_revision"]
    model_name = manifest["model_name"]

    if (
        config["model"]["revision"] != revision
        or config["model"]["name"] != model_name
    ):
        raise ValueError("Model configuration differs from tokenization.")

    data_path = args.data_dir / "validation.jsonl"
    expected_hash = manifest["output_file_sha256"]["validation.jsonl"]

    if file_sha256(data_path) != expected_hash:
        raise ValueError("Validation data checksum mismatch.")

    dataset = CausalTextDataset(
        data_path,
        max_length=evaluation["max_sequence_length"],
    )

    loader = DataLoader(
        dataset,
        batch_size=evaluation["batch_size"],
        shuffle=False,
        collate_fn=CausalCollator(manifest["pad_token_id"]),
        num_workers=0,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.data_dir / "tokenizer",
        local_files_only=True,
    )

    print(f"Loading base model: {model_name}", flush=True)

    # Keep stored weights in FP32; use FP16 autocast for GPU computation.
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=revision,
        torch_dtype=torch.float32,
        trust_remote_code=False,
    ).to(device)

    if evaluation["max_sequence_length"] > model.config.max_position_embeddings:
        raise ValueError("Sequence length exceeds the model context limit.")

    model.requires_grad_(False)
    model.eval()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    run_config = {
        **config,
        "stage": "base-model-validation",
        "adapters_attached": False,
        "resolved_model_revision": revision,
        "validation_sha256": expected_hash,
        "effective_precision": "fp16_autocast",
        "stored_weight_dtype": "float32",
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "transformers": version("transformers"),
            "wandb": version("wandb"),
        },
    }

    # Override the starter LoRA config to reflect this actual run.
    run_config["model"] = {
        **config["model"],
        "lora": {**config["model"]["lora"], "enabled": False},
    }

    run = wandb.init(
        project=tracking["project"],
        entity=tracking["entity"],
        mode=tracking["mode"],
        name="pythia-410m-baseline",
        job_type="evaluation",
        config=run_config,
        tags=[
            "baseline",
            "pythia-410m",
            "mixture-11",
            "fp16-autocast",
            f"seq-{evaluation['max_sequence_length']}",
            f"batch-{evaluation['batch_size']}",
        ],
    )

    failed = True

    try:
        accumulator = MetricAccumulator(
            evaluation["length_bucket_upper_bounds"]
        )

        with torch.inference_mode():
            for step, batch in enumerate(loader, start=1):
                batch = {
                    key: value.to(device)
                    for key, value in batch.items()
                }

                with torch.autocast("cuda", dtype=torch.float16):
                    output = model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        use_cache=False,
                    )

                sequence_nll, counts = causal_nll(
                    output.logits,
                    batch["labels"],
                )
                accumulator.update(sequence_nll, counts)

                if step % 100 == 0 or step == len(loader):
                    print(
                        f"Evaluated {step}/{len(loader)} batches",
                        flush=True,
                    )

                del output, sequence_nll, counts, batch

        metrics = accumulator.compute()

        if metrics["n_tokens"] != dataset.n_targets:
            raise ValueError("Evaluation did not cover all prediction targets.")

        payload = {
            f"val/{key}": value
            for key, value in metrics.items()
            if key != "ppl_by_length_bucket"
        }

        bucket_table = wandb.Table(
            columns=["length_bucket", "ppl", "n_tokens", "n_sequences"]
        )

        for name, bucket in metrics["ppl_by_length_bucket"].items():
            bucket_table.add_data(
                name,
                bucket["ppl"],
                bucket["n_tokens"],
                bucket["n_sequences"],
            )

        examples = []
        example_table = wandb.Table(columns=["prompt", "continuation"])

        with torch.inference_mode():
            for prompt in evaluation["generation"]["prompts"]:
                inputs = tokenizer(
                    prompt,
                    return_tensors="pt",
                ).to(device)

                with torch.autocast("cuda", dtype=torch.float16):
                    generated = model.generate(
                        **inputs,
                        max_new_tokens=evaluation["generation"]["max_new_tokens"],
                        do_sample=evaluation["generation"]["do_sample"],
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )

                continuation = tokenizer.decode(
                    generated[0, inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True,
                )

                examples.append({
                    "prompt": prompt,
                    "continuation": continuation,
                })
                example_table.add_data(prompt, continuation)

        payload["val/ppl_by_length_bucket"] = bucket_table
        payload["val/examples"] = example_table

        # A baseline compared with itself has zero delta.
        payload["val/delta_ppl"] = 0.0
        payload["val/checkpoint_rankings"] = wandb.Table(
            columns=["checkpoint", "token_loss", "ppl"],
            data=[["base", metrics["token_loss"], metrics["ppl"]]],
        )

        run.log(payload)

        result = {
            "stage": "baseline",
            "run_url": run.url,
            "configuration": run_config,
            "metrics": metrics,
            "examples": examples,
        }

        metrics_path.write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )

        print("\nBaseline validation metrics:")
        for key in ("token_loss", "sequence_loss", "ppl", "bpt"):
            print(f"{key}: {metrics[key]:.6f}")

        print(f"Results: {metrics_path}")
        print(f"W&B: {run.url}")
        failed = False

    finally:
        run.finish(exit_code=1 if failed else 0)


if __name__ == "__main__":
    main()