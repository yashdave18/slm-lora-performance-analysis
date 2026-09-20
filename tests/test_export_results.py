import importlib.util
import json
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location(
    "export_results", Path(__file__).resolve().parents[1] / "scripts/export_results.py"
)
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)


def test_export_excludes_weights_and_records_real_metrics(tmp_path):
    project = tmp_path / "project"
    run = project / "runs/lora-r8-long"
    run.mkdir(parents=True)
    record = {"step": 100, "metrics": {"token_loss": 3.0, "ppl": 20.0}, "smoke": False}
    (run / "validation-000100.json").write_text(json.dumps(record))
    (run / "summary.json").write_text(json.dumps({"completed_steps": 100}))
    weights = run / "checkpoints/step-000100/adapter"
    weights.mkdir(parents=True)
    (weights / "adapter_model.safetensors").write_bytes(b"weights")
    output = tmp_path / "export"
    exporter.export_results(project, output)
    assert json.loads((output / "runs/lora-r8-long/validation-000100.json").read_text()) == record
    assert not list(output.rglob("*.safetensors"))
    assert "lora-r8-long,100,3.0,20.0" in (output / "validation_history.csv").read_text()
    inventory = json.loads((output / "artifact_inventory.json").read_text())
    assert len(inventory) == 2
    with pytest.raises(FileExistsError):
        exporter.export_results(project, output)


def test_empty_source_does_not_create_export(tmp_path):
    output = tmp_path / "export"
    with pytest.raises(FileNotFoundError):
        exporter.export_results(tmp_path, output)
    assert not output.exists()
