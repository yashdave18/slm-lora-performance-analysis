import pytest
from src.evaluation.configuration import baseline_config
from src.utils.helpers import PROJECT_ROOT


def test_baseline_defaults_disable_adapters():
    config = baseline_config()
    assert config["model"]["lora"]["enabled"] is False


def test_baseline_experiment_loads():
    config = baseline_config(PROJECT_ROOT / "experiments/experiment_01_baseline.yaml")
    assert config["evaluation"]["max_sequence_length"] == 256
    assert config["experiment_name"] == "pythia-410m-baseline"


@pytest.mark.parametrize("body", [
    "mode: train\nname: invalid\n",
    "mode: evaluate\nname: invalid\nmodel:\n  lora:\n    enabled: true\n",
    "mode: evaluate\nname: invalid\ntraining:\n  precision: fp32\n",
    "mode: evaluate\nname: invalid\nevaluation:\n  split: test\n",
])
def test_invalid_baseline_experiment_rejected(tmp_path, body):
    path = tmp_path / "invalid.yaml"
    path.write_text(body)
    with pytest.raises(ValueError):
        baseline_config(path)
