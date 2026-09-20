# Pythia-410M LoRA: fine-tuning, evaluation and performance analysis

Reproducible next-token training with PEFT/LoRA on a curated mixture of 11 Hugging Face datasets. Training runs use a Colab Tesla T4 and persistent Google Drive checkpoints.

## Current status

Implemented: dataset preparation, tokenization, document-preserving chunking, baseline validation, LoRA training, W&B logging, checkpoint recovery, and resumable early stopping.

Completed runs reported on 20 September 2026:

| Experiment | Optimizer steps | Validation perplexity |
|---|---:|---:|
| Pretrained baseline | 0 | 22.310750 |
| Rank 8, short schedule | 300 | 20.7145 |
| Rank 8, long schedule, best checkpoint | 2800 | 19.482983618515064 |
| Rank 8, long schedule, final checkpoint | 3072 | 19.4840 |

The long run completed approximately three epochs, stopped at its maximum budget, and selected step 2800 by minimum validation token loss (2.969541449654233). Best validation perplexity is 12.67% below baseline. The curve plateaued under a decaying learning rate; this does not prove an optimal learning rate or global convergence. These values are transcribed from execution output; use exported JSON artifacts for full precision and provenance.

Rank-16 long-run configuration is provided; completion has not yet been reported. Test-set evaluation, inference benchmarks, additional batch/precision/length comparisons, final figures and the LaTeX report/PDF remain outstanding.

## Setup

Use a virtual environment locally. In Colab, retain its CUDA-enabled PyTorch installation. Install a suitable PyTorch build first if it is absent; it is intentionally not replaced by requirements.txt.

```bash
python -m pip install -r requirements.txt
python -m pytest -q
```

Authenticate W&B interactively (`wandb.login()` in Colab). Never commit API keys.

## Run training

Run from the repository root after mounting Drive and preparing the verified data. The existing data need not be regenerated.

```bash
python -m src.training.train \
  --experiment experiments/experiment_06_lora_r16_long.yaml \
  --data-dir /content/drive/MyDrive/slm-lora-performance-analysis/data/tokenized \
  --output-dir /content/drive/MyDrive/slm-lora-performance-analysis/runs/lora-r16-long \
  --baseline /content/drive/MyDrive/slm-lora-performance-analysis/results/baseline/metrics.json
```

To recover an interrupted run, repeat its exact command with `--resume`. Do not change the configuration, reuse a completed output directory for fresh training, or resume rank 16 from rank 8.

[Training guide](TRAINING_GUIDE.md) covers the schedule, stopping rule and checkpoint format. [Documentation](documentation.md) records the data, metric definitions, experiment status and remaining work.

## Repository

- `configs/`: shared defaults; experiment YAMLs override them.
- `src/data/`: preparation, tokenization and batching.
- `src/training/`: LoRA updates and resumable checkpoints.
- `src/evaluation/`: validation metrics and base-model evaluator.
- `src/utils/`: configuration, seeds and logging.
- `scripts/export_results.py`: copies small real artifacts for version control.
- `notebooks/analysis.ipynb`: inspects exported validation results without a GPU.
- `results/`: public metadata and exported metrics; no weights or full datasets.
- `src/inference/`, `scripts/benchmark.sh`, `report/`: remaining implementation/report work.

[W&B project](https://wandb.ai/daveyash1218-dwarkadas-j-sanghvi-college-of-engineering/slm-lora-performance-analysis)
