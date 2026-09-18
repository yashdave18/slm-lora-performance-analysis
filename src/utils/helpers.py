"""Configuration loading utilities."""

from pathlib import Path
import json
import math

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONFIG_FILES = {
    "data": "dataset_config.yaml",
    "model": "model_config.yaml",
    "training": "training_config.yaml",
    "evaluation": "evaluation_config.yaml",
}


def load_yaml(path):
    """Read a YAML file and require a mapping at its root."""
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        content = yaml.safe_load(file)

    if not isinstance(content, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")

    return content


def load_configs(config_dir=None):
    """Load and perform basic checks on the four base configurations."""
    config_dir = (
        Path(config_dir)
        if config_dir is not None
        else PROJECT_ROOT / "configs"
    )

    configs = {}

    for section, filename in CONFIG_FILES.items():
        content = load_yaml(config_dir / filename)

        if set(content) != {section}:
            raise ValueError(
                f"{filename} must contain exactly one root key: {section}"
            )

        if not isinstance(content[section], dict) or not content[section]:
            raise ValueError(f"{filename}: {section} must be a nonempty mapping")

        configs[section] = content[section]

    fractions = configs["data"]["splitting"]
    values = [
        fractions["train_fraction"],
        fractions["validation_fraction"],
        fractions["test_fraction"],
    ]

    if any(not 0 < value < 1 for value in values):
        raise ValueError("Each split fraction must be between 0 and 1.")

    if not math.isclose(sum(values), 1.0, abs_tol=1e-9):
        raise ValueError("Split fractions must sum to 1.")

    datasets = configs["data"]["datasets"]

    if not 10 <= len(datasets) <= 15:
        raise ValueError("The assignment requires 10–15 datasets.")

    identities = [
        (dataset["name"], dataset.get("subset"))
        for dataset in datasets
    ]

    if len(identities) != len(set(identities)):
        raise ValueError("Duplicate dataset entries found.")

    for dataset in datasets:
        fields = dataset["text_fields"]
        if not isinstance(fields, list) or not fields:
            raise ValueError(
                f"{dataset['name']}: text_fields must be a nonempty list."
            )

    return configs


if __name__ == "__main__":
    config = load_configs()
    print(json.dumps(config, indent=2))
    print("\nPASS: all four configurations loaded.")
