"""Lightweight baseline configuration validation, independent of GPU libraries."""
from src.utils.helpers import load_configs, load_yaml


def merge_known(base, overrides):
    for key, value in overrides.items():
        if key not in base:
            raise ValueError(f"Unknown configuration key: {key}")
        if isinstance(base[key], dict):
            if not isinstance(value, dict):
                raise ValueError(f"Expected a mapping: {key}")
            merge_known(base[key], value)
        else:
            base[key] = value
    return base


def baseline_config(experiment_path=None):
    config = load_configs()
    config["model"]["lora"]["enabled"] = False
    config["experiment_name"] = "pythia-410m-baseline"
    if experiment_path is not None:
        experiment = load_yaml(experiment_path)
        if experiment.get("mode") != "evaluate":
            raise ValueError("Baseline experiment mode must be evaluate.")
        overrides = {k: v for k, v in experiment.items() if k not in ("name", "mode")}
        config = merge_known(config, overrides)
        config["experiment_name"] = experiment["name"]
    if config["model"]["lora"]["enabled"]:
        raise ValueError("Baseline evaluation requires LoRA disabled.")
    if config["training"]["precision"] != "fp16":
        raise ValueError("This baseline evaluator currently uses FP16 autocast only.")
    if config["evaluation"]["split"] != "validation":
        raise ValueError("Baseline model selection uses validation only.")
    return config

