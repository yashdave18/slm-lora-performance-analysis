"""Command-line entry point for controlled LoRA experiments."""
import argparse
import copy
from pathlib import Path
from src.utils.helpers import load_configs, load_yaml
from src.training.trainer import train


def merge_known(base, override, prefix=""):
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key not in result:
            raise ValueError(f"Unknown configuration key: {prefix}{key}")
        if isinstance(result[key], dict):
            if not isinstance(value, dict):
                raise ValueError(f"Expected mapping: {prefix}{key}")
            result[key] = merge_known(result[key], value, prefix + key + ".")
        else:
            result[key] = value
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--experiment", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", choices=["cpu", "cuda"])
    args = parser.parse_args()
    config = load_configs()
    config["training"].setdefault("early_stopping", {
        "enabled": False, "patience": 5, "min_delta": 0.001,
    })
    if args.experiment:
        experiment = load_yaml(args.experiment)
        if experiment.get("mode") != "train":
            raise ValueError("Use the evaluation entry point for baseline experiments.")
        overrides = {k: v for k, v in experiment.items() if k not in ("name", "mode")}
        config = merge_known(config, overrides)
        config["experiment_name"] = experiment["name"]
    else:
        config["experiment_name"] = "lora-default"
    config["smoke"] = args.smoke
    if args.smoke:
        config["training"].update({
            "max_steps": 3, "log_every_steps": 1,
            "evaluate_every_steps": 3, "save_every_steps": 1,
        })
        config["evaluation"]["generation"]["max_new_tokens"] = 16
    train(
        config, args.data_dir, args.output_dir, args.baseline,
        resume=args.resume, device_name=args.device,
    )


if __name__ == "__main__":
    main()
