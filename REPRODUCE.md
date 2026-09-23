# Reproduction workflow

Run from the repository root. The saved results do not require GPU reruns.

## Verify existing artifacts (CPU)

```bash
python -m pip install -r requirements-analysis.txt
python scripts/verify_export.py
python scripts/analyze_results.py
```

## Environment for fresh experiments

Use a Colab GPU runtime for model evaluation/training. Mount Drive and authenticate
W&B interactively. Preserve Colab's CUDA-enabled PyTorch installation.

```bash
python -m pip install -r requirements.txt
python -m pytest -q
```

Original preparation recorded datasets 4.8.5, huggingface-hub 0.36.2 and PyYAML
6.0.2. Match these when recreating data (install before running preparation):

```bash
python -m pip install datasets==4.8.5 huggingface-hub==0.36.2 PyYAML==6.0.2
```

Final evaluation used Python 3.13.15, PyTorch 2.11.0+cu128, Transformers 4.57.6,
PEFT 0.18.1, W&B 0.28.1 and a T4. Current Colab defaults can differ; consult the
archived per-run environments. Matching seeds alone does not guarantee bitwise
CUDA reproducibility.

## Prepare data and verify against the original sample

Use a NEW project directory so existing data and checkpoints are preserved.
For these commands the new Drive directory is `slm-lora-reproduction`.

```bash
python -m src.data.dataset_loader --output-dir /content/drive/MyDrive/slm-lora-reproduction/data/processed
```

Source SHAs and Parquet shard lists are pinned in dataset_config.yaml. Compare
new `train.jsonl`, `validation.jsonl` and `test.jsonl` SHA256 values with
`file_sha256` in `results/tables/data_manifest.json` before comparing runs. The
manifest itself will differ because it now includes explicit source pins.
The following Python cell performs the prepared-file comparison:

```python
from pathlib import Path
import hashlib, json
expected = json.loads(Path("results/tables/data_manifest.json").read_text())["file_sha256"]
prepared = Path("/content/drive/MyDrive/slm-lora-reproduction/data/processed")
for name, digest in expected.items():
    actual = hashlib.sha256((prepared / name).read_bytes()).hexdigest()
    assert actual == digest, f"Prepared data differs: {name}"
print("PASS: original prepared split checksums match")
```

```bash
python -m src.data.tokenization --input-dir /content/drive/MyDrive/slm-lora-reproduction/data/processed --output-dir /content/drive/MyDrive/slm-lora-reproduction/data/tokenized
python -m src.data.batching --data-dir /content/drive/MyDrive/slm-lora-reproduction/data/tokenized
```

## Baseline and three independent training runs

```bash
python -m src.evaluation.evaluate --experiment experiments/experiment_01_baseline.yaml --data-dir /content/drive/MyDrive/slm-lora-reproduction/data/tokenized --output-dir /content/drive/MyDrive/slm-lora-reproduction/results/baseline
python -m src.training.train --experiment experiments/experiment_05_lora_r8_long.yaml --data-dir /content/drive/MyDrive/slm-lora-reproduction/data/tokenized --output-dir /content/drive/MyDrive/slm-lora-reproduction/runs/lora-r8-long --baseline /content/drive/MyDrive/slm-lora-reproduction/results/baseline/metrics.json
python -m src.training.train --experiment experiments/experiment_06_lora_r16_long.yaml --data-dir /content/drive/MyDrive/slm-lora-reproduction/data/tokenized --output-dir /content/drive/MyDrive/slm-lora-reproduction/runs/lora-r16-long --baseline /content/drive/MyDrive/slm-lora-reproduction/results/baseline/metrics.json
python -m src.training.train --experiment experiments/experiment_08_lora_r16_lr5e4.yaml --data-dir /content/drive/MyDrive/slm-lora-reproduction/data/tokenized --output-dir /content/drive/MyDrive/slm-lora-reproduction/runs/lora-r16-lr5e4 --baseline /content/drive/MyDrive/slm-lora-reproduction/results/baseline/metrics.json
```

Resume an interrupted training run by appending `--resume` to its identical
command. Do not resume a completed run or alter its resolved configuration.

## Final evaluation and benchmarking

```bash
python -m src.evaluation.final_evaluate --project-dir /content/drive/MyDrive/slm-lora-reproduction
python -m src.inference.benchmark --project-dir /content/drive/MyDrive/slm-lora-reproduction
```

Final evaluation intentionally checks the locked validation checkpoint selection.
If a fresh run selects a different checkpoint because of numerical differences,
inspect validation results and explicitly record a new selection before opening
test data. Never modify selection in response to test scores. Fresh runs are new
experiments and should not replace the archived original measurements.

See FINAL_STAGE_GUIDE.md for final-stage recovery and timing semantics.
