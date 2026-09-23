# Workflow and experiment documentation

## Project overview

This project fine-tunes `EleutherAI/pythia-410m` using PEFT/LoRA on a curated mixture of 11 Hugging Face text datasets.

The workflow includes dataset preparation, baseline evaluation, LoRA training, checkpoint recovery, validation-based model selection, held-out test evaluation, inference benchmarking, and reproducible analysis.

- Model revision: `9879c9b5f8bea9051dcb0e68dff21493d67e9d4f`
- LoRA target modules: `query_key_value`
- Training hardware: Google Colab Tesla T4
- Experiment tracking: Weights & Biases
- Persistent data and checkpoint storage: Google Drive

Base-model weights remain frozen during LoRA training.

## Data and reproducibility

All sources below use their original `train` split. We construct our own training, validation, and test partitions from sampled documents.

Only the configured natural-language fields are used. Classification labels and question-answer targets are excluded.

| Hugging Face dataset | Subset | Text fields |
|---|---|---|
| Salesforce/wikitext | wikitext-2-raw-v1 | text |
| roneneldan/TinyStories | default | text |
| fancyzhx/ag_news | default | text |
| stanfordnlp/imdb | default | text |
| Yelp/yelp_review_full | default | text |
| fancyzhx/amazon_polarity | default | title, content |
| abisee/cnn_dailymail | 3.0.0 | article |
| EdinburghNLP/xsum | default | document |
| rajpurkar/squad | default | context |
| dair-ai/emotion | split | text |
| cornell-movie-review-data/rotten_tomatoes | default | text |

### Preprocessing and sampling

Preparation normalizes Unicode and whitespace, retains documents of 40–20,000 characters, and deduplicates retained canonical text.

Seeded hash assignment uses probabilities of 80/10/10 for training, validation, and test. Per-source caps subsequently retain 1,000 training, 100 validation, and 100 test documents. Therefore, the final retained proportions are not exactly 80/10/10.

Streaming with a bounded shuffle buffer provides practical sampling rather than uniform sampling of each complete source. We use sampled documents, not the entire Hugging Face datasets.

No reasoning-oriented dataset was selected, but individual passages are not semantically screened for all incidental reasoning content.

The committed `results/tables/data_manifest.json` records source revisions, processing counts, and prepared-file checksums.

### Tokenization and chunking

Tokenization preserves complete documents, appends EOS, and saves the tokenizer and a tokenization checksum manifest.

At sequence length 256, chunking uses one overlapping context token so that each original next-token target is counted once. Chunks never combine different documents.

Padding labels are ignored by position; genuine EOS targets remain included.

| Split | Documents | Tokenized tokens | Sequences at length 256 | Prediction targets |
|---|---:|---:|---:|---:|
| Train | 11,000 | 2,478,818 | 16,381 | 2,467,818 |
| Validation | 1,100 | 247,245 | 1,634 | 246,145 |
| Test | 1,100 | 245,860 | 1,629 | 244,760 |

Exact duplicate checks do not establish the absence of near-duplicates or overlap with model pretraining data. Dataset redistribution rights must be reviewed before distributing source text; the repository stores metadata and small experimental artifacts.

## Experiment status

| Configuration | Status | Best validation perplexity |
|---|---|---:|
| Pretrained baseline | Completed | 22.310750 |
| Rank 8, 300-step preliminary schedule | Completed; excluded from main comparison | 20.714505 |
| Rank 8, LR 1e-4, long | Completed 3,072 steps; best checkpoint at 2,800 | 19.482984 |
| Rank 16, LR 1e-4, long | Early stopping at 3,000; best checkpoint at 3,000 | 19.282668 |
| Rank 16, LR 5e-4, long | Completed 3,072 steps; best checkpoint at 3,072 | 19.093474 |
| Short rank-16 configuration | Not reported as executed | — |
| Sequence-1024 training | Not executed; unused template removed | — |

An experiment configuration file is not evidence of an executed experiment.

The short and long rank-8 runs used separate schedules. The long run started as a fresh experiment rather than continuing the preliminary 300-step run.

## Baseline evaluation

The completed baseline does not need to be rerun. To reproduce it in a new output directory:

```bash
python -m src.evaluation.evaluate \
  --experiment experiments/experiment_01_baseline.yaml \
  --data-dir DATA_TOKENIZED \
  --output-dir NEW_BASELINE_OUTPUT
```

Replace `DATA_TOKENIZED` and `NEW_BASELINE_OUTPUT` with the appropriate paths.

This evaluator supports the base model, validation split, and FP16 autocast on CUDA. The experiment override is applied and validated; unsupported settings are rejected.

For saved-adapter test evaluation, use the separate `src.evaluation.final_evaluate` module described in `documentation.md`.

## Training and checkpoint selection

### Main training settings

| Setting | Value |
|---|---|
| LoRA rank | 8 or 16 |
| LoRA alpha | 16 for rank 8; 32 for rank 16 |
| LoRA dropout | 0.05 |
| Target modules | query_key_value |
| Optimizer | AdamW |
| Peak learning rate | 1e-4 or 5e-4 |
| Weight decay | 0.01 |
| Scheduler | Cosine |
| Warmup ratio | 0.05 |
| Maximum optimizer steps | 3,072 |
| Batch size | 2 |
| Gradient accumulation | 8 |
| Effective batch size | 16 chunks per optimizer update |
| Maximum sequence length | 256 |
| Precision | FP16 autocast with FP32 stored weights |
| Gradient checkpointing | Enabled |
| Maximum gradient norm | 1.0 |
| Seed | 42 |

Cosine scheduling follows 154 warmup steps: `ceil(0.05 * 3072)`.

The learning-rate comparison favors `5e-4` among the tested rank-16 settings. The warmup ratio was held fixed rather than tuned. These experiments do not establish a globally optimal learning rate or LoRA rank.

### Validation and stopping

Validation and checkpoint saving occur every 100 optimizer steps and at the final step.

Early stopping allows five successive evaluations without a decrease of at least `0.001` from the last significant best validation token loss. Small decreases can accumulate to a significant improvement.

Checkpoint selection separately uses the absolute lowest validation token loss, regardless of the early-stopping `min_delta`. Consequently, a run can stop while still making improvements smaller than the stopping threshold.

Both checkpoint rankings and the early-stopping state survive recovery.

### Selected checkpoints

Paths below are relative to the project directory on Google Drive.

| Experiment | Best adapter path |
|---|---|
| Rank 8, LR 1e-4 | `runs/lora-r8-long/checkpoints/step-002800/adapter` |
| Rank 16, LR 1e-4 | `runs/lora-r16-long/checkpoints/step-003000/adapter` |
| Rank 16, LR 5e-4 | `runs/lora-r16-lr5e4/checkpoints/step-003072/adapter` |

The latest complete checkpoint supports recovery; the best validation checkpoint supports downstream evaluation.

The final selected configuration is rank 16 with peak learning rate `5e-4`. Selection was completed using validation results before held-out test evaluation.

## Metrics and timing

### Language-model metrics

- **Token loss:** summed natural-log negative log-likelihood divided by valid prediction targets.
- **Sequence loss:** summed negative log-likelihood divided by chunk count; this depends on chunk length.
- **Perplexity:** `exp(token_loss)`.
- **Bits per token:** `token_loss / ln(2)`.
- **Length buckets:** groups based on valid target counts.

Evaluation aggregates negative log-likelihood across all valid targets rather than averaging batch losses. Aggregate perplexity is derived from aggregate token loss, not from an arithmetic mean of per-dataset perplexities.

### Training metrics

Training throughput counts valid prediction targets per optimizer-update time, including batch preparation and gradient accumulation but excluding validation, logging, and checkpoint saving.

Training throughput and inference generation throughput measure different workloads.

Gradient norm is logged before clipping. W&B also records learning rate, AMP retries, EMA token loss, configuration metadata, checkpoint paths, validation length buckets, examples, and checkpoint rankings.

## Recovery and environment

Google Drive holds prepared data and complete checkpoint folders.

After a Colab reset:

1. Mount Drive.
2. Restore the repository and dependencies.
3. Authenticate W&B.
4. Repeat the identical training command with `--resume`.

Resume restores adapters, optimizer, scheduler, scaler, random-number-generator states, data cursor, EMA, checkpoint rankings, and stopping state. Work since the last complete checkpoint may repeat.

Only trusted checkpoint files should be loaded.

Each resumed execution has its own W&B run segment. Combine segments by optimizer step when analyzing a complete training trajectory. Local JSONL histories may contain repeated steps after a retry.

The latest training `environment.json` is overwritten on resume. Consult the corresponding W&B segments when investigating environment changes across executions.

GPU operations may be nondeterministic despite seeding; bitwise-identical CUDA recovery is not claimed.

Colab disconnections and GPU-quota interruptions are environment interruptions, not evidence that a hyperparameter configuration failed numerically.

## Held-out test results

The selected rank-16 adapter and pretrained base were evaluated on the same held-out test targets.

| Metric | Pretrained base | Selected adapter |
|---|---:|---:|
| Token loss | 3.140413 | 2.978518 |
| Sequence loss | 471.852339 | 447.527333 |
| Perplexity | 23.113408 | 19.658658 |
| Bits per token | 4.530658 | 4.297093 |
| Prediction targets | 244,760 | 244,760 |
| Sequences | 1,629 | 1,629 |

The selected adapter reduces test perplexity by **14.95%**. Perplexity improves on all 11 source datasets.

These are next-token prediction results, including for datasets originally designed for classification. They do not measure classification accuracy.

Original results, per-dataset metrics, length-bucket metrics, and evaluation metadata are stored in `results/final_test/`.

## Inference benchmarks

All **72 benchmark cases** completed successfully.

| Dimension | Values |
|---|---|
| Model variants | Base and the three main validation-selected adapters |
| Precision modes | FP32 and FP16 autocast |
| Batch sizes | 1, 2, 4 |
| Prompt lengths | 64, 128, 256 tokens |
| Generated length | 64 tokens per sequence |
| Warmup runs | 2 per case |
| Measured runs | 5 per case |
| Total measured generations | 360 batch-generation calls |

### Measurement protocol

All variants retain FP32 stored weights. FP16 refers to autocast computation, not a model loaded entirely with FP16 weights.

Adapters remain unmerged. SDPA and KV caching are enabled, and TF32 is disabled.

Timed generation uses greedy decoding with EOS stopping disabled to keep output length fixed. Full-generation latency includes prefill and decoding but excludes model loading, tokenization, device transfer, warmup, logging, and disk writes.

Generated-token throughput is total generated tokens divided by total measured duration. It does not include prompt tokens.

Memory measurements are peak PyTorch allocated and reserved memory, not total GPU-process memory. Time to first token was not measured.

### Interpretation

Batching increases aggregate throughput in these measurements. FP16 autocast is slower than FP32 for this implementation and workload.

These findings should not be generalized to fully FP16-loaded models, merged adapters, other GPUs, or other workloads.

Prompt lengths vary at inference. No comparison of models trained at different sequence lengths was performed.

Error bars show standard deviations across five timing repetitions, not uncertainty across independent training seeds. Only four validation-derived prompt documents are used, limiting workload diversity.

Original summaries and individual case measurements are stored in `results/inference_benchmarks/`.

### Recorded final-stage environment

- Python: 3.13.15
- PyTorch: 2.11.0+cu128
- Transformers: 4.57.6
- PEFT: 0.18.1
- W&B: 0.28.1
- CUDA: 12.8
- GPU: Tesla T4

Exact environment records and the source commit are included in the original JSON artifacts. These final-stage versions should not be assumed to describe every earlier training segment.

## Archived training artifacts

The verified export is archived at:

`results/exports-submission-20260923-050201/`

It contains:

- Baseline evaluation metrics.
- Prepared-data and tokenization manifests.
- Training histories for the preliminary rank-8 run and all three main experiments.
- Resolved configurations and recorded environments.
- Run summaries, checkpoint rankings, and validation results.
- A combined validation-history CSV.
- Final-stage evaluation and benchmark artifacts.
- `artifact_inventory.json`, containing artifact SHA256 checksums.

All **205 artifact checksums** in the supplied export were verified. The three main training summaries match the reported checkpoint selections and validation scores.

The export excludes model weights, optimizer checkpoints, raw datasets, credentials, and W&B service files.

### Creating another export

Run from the repository root in Colab:

```bash
python scripts/export_results.py \
  --project-dir /content/drive/MyDrive/slm-lora-performance-analysis \
  --output-dir results/exports
```

Use a fresh destination for each export. Missing completion summaries are reported rather than invented. Review exported files before committing them.

## Reproducing the analysis

No GPU or W&B login is required for analysis of the saved results.

From the repository root:

```bash
python -m pip install -r requirements-analysis.txt
python scripts/analyze_results.py
```

The script verifies all 72 benchmark case identities, 360 measured batch-generation calls, derived timing and throughput metrics, test counts, and aggregate scores.

It generates:

- CSV tables in `results/tables/`.
- PNG and PDF figures in `results/figures/`.
- An artifact checksum inventory in `results/tables/verification.json`.

`notebooks/analysis.ipynb` runs the same analysis interactively.

The analysis script verifies final-test and inference artifacts. The separate training export has its own checksum inventory.

## Research report

The compiled report is available at:

[`report/report.pdf`](report/report.pdf)

Editable LaTeX source, bibliography, and figure assets are stored in `report/`.

To rebuild locally, run these commands from `report/`:

```bash
pdflatex report.tex
bibtex report
pdflatex report.tex
pdflatex report.tex
```

`report_overleaf.zip` contains a self-contained Overleaf project.

Editable Overleaf project:
[Open and edit the report](https://www.overleaf.com/7148523494hzbjtzmtdsnx#6800e9)

## W&B project and run links

Resumed-run links may contain only one segment of a training history. Use the archived local histories and associated W&B segments for the complete trajectory.

- [W&B project](https://wandb.ai/daveyash1218-dwarkadas-j-sanghvi-college-of-engineering/slm-lora-performance-analysis)
- [Baseline validation](https://wandb.ai/daveyash1218-dwarkadas-j-sanghvi-college-of-engineering/slm-lora-performance-analysis/runs/8f521k0q)
- [Rank 8, preliminary 300-step run](https://wandb.ai/daveyash1218-dwarkadas-j-sanghvi-college-of-engineering/slm-lora-performance-analysis/runs/r6ja3elz)
- [Rank 8 long, resumed segment](https://wandb.ai/daveyash1218-dwarkadas-j-sanghvi-college-of-engineering/slm-lora-performance-analysis/runs/e8dob4x9)
- [Rank 16, LR 1e-4, resumed segment](https://wandb.ai/daveyash1218-dwarkadas-j-sanghvi-college-of-engineering/slm-lora-performance-analysis/runs/y4mz6akb)
- [Rank 16, LR 5e-4, resumed segment](https://wandb.ai/daveyash1218-dwarkadas-j-sanghvi-college-of-engineering/slm-lora-performance-analysis/runs/ge2ub3kg)
- [Final test evaluation](https://wandb.ai/daveyash1218-dwarkadas-j-sanghvi-college-of-engineering/slm-lora-performance-analysis/runs/xtodi84b)
- [Inference benchmark](https://wandb.ai/daveyash1218-dwarkadas-j-sanghvi-college-of-engineering/slm-lora-performance-analysis/runs/b5fiqzmr)

## Limitations

- Main training comparisons use one seed.
- Equal document caps do not produce equal token contributions across datasets.
- Exact deduplication does not eliminate all near-duplicates or pretraining contamination.
- A late validation plateau under cosine decay does not prove global convergence.
- Hyperparameter comparisons cover only the configurations reported above.
- Inference measurements use a small prompt sample and shared Colab hardware.
- Precision comparisons retain FP32 stored weights.
- Merged-adapter performance and time to first token were not measured.
- Sequence-length comparisons concern inference prompts, not training configurations.

## Submission access

- Repository: [slm-lora-performance-analysis](https://github.com/yashdave18/slm-lora-performance-analysis)
- Compiled report: `report/report.pdf`
- Editable report source: `report/`
- Editable Overleaf project: linked above
- Training archive: `results/exports-submission-20260923-050201/`
- Test results: `results/final_test/`
- Inference measurements: `results/inference_benchmarks/`
- Analysis notebook: `notebooks/analysis.ipynb`
- Experiment tracking: W&B links above

Before submission, confirm that the exported folder and updated documentation are committed and pushed, and that the reviewer can access the repository, W&B results, and editable Overleaf project.
## Final integrity and reproduction notes

The data loader now uses immutable Parquet revisions and file lists in
`configs/dataset_config.yaml`, copied from the original data manifest. Existing
resolved training configs remain unchanged. See [REPRODUCE.md](REPRODUCE.md)
for preparation, checksum verification, tokenization and experiment commands.

Git had normalized two exported CSV files from CRLF to LF. The original bytes
were restored, and `.gitattributes` now preserves all archive bytes. The original
inventory is unchanged. Run `python scripts/verify_export.py` and, after staging,
`python scripts/verify_export.py --staged` to check all 205 original hashes.

## Operational reference

### Training recovery details

Repeat the identical command with --resume, retaining --smoke for smoke runs.
Restore covers adapter weights, optimizer, scheduler, AMP scaler, RNGs, batch cursor, EMA, rankings, early-stopping counter, and global step. Work after the latest checkpoint is replayed. Keep the same software/hardware stack.
Each resumed execution is a new W&B segment in the same group, with its resumed_from field. Use the custom step axis. Local history may contain repeated steps from a failed segment.
Before the first checkpoint, recover with a new output directory after addressing the error.
Completed runs reject resume. Changed budgets/configurations require a new experiment.
Partial/stale checkpoint directories encountered during replay are preserved as recovered-*.
Only load your own training_state.pt files; Python/NumPy RNG serialization requires pickle loading.

### Training output layout

Each run contains resolved_config.json, environment.json, history.jsonl, validation-XXXXXX.json, best_checkpoint.json, summary.json, and checkpoints.
Each checkpoint includes adapter/, tokenizer/, training_state.pt and COMPLETE.json.
checkpoints/latest.json points to the latest completed checkpoint.
Exceptions are recorded in failure-*.json.
Keep checkpoints on Drive and later copy only small summaries/plots into Git.

### Training test coverage

The training tests cover these behaviors using tiny local models:
1. Sampler restoration across epochs.
2. Unequal-length accumulation matches a combined token-weighted batch.
3. Only LoRA parameters change.
4. Adapter reload plus optimizer/scheduler/RNG restoration reproduces the next update.
5. Incomplete checkpoints are rejected.
6. The actual runner trains, logs offline, validates, generates and resumes after a simulated disconnect; final adapters match uninterrupted execution.

The T4 smoke test and short/long rank-8 training have completed. The training and early-stopping suite passed 10 tests before the long run. CPU recovery tests cover interrupted early stopping as well as optimizer state. GPU attention nondeterminism may cause small numerical differences. See documentation.md for actual run results and remaining work.

### GPU smoke benchmark

```bash
python -m src.inference.benchmark --project-dir /content/drive/MyDrive/slm-lora-performance-analysis --config configs/benchmark_smoke.yaml --output-dir /content/drive/MyDrive/slm-lora-performance-analysis/results/benchmark_smoke
```

Two small cases (base and selected adapter), eight generated tokens each, one warmup and two measurements. This verifies CUDA/PEFT generation compatibility without opening the test split. These timings are diagnostic, not final benchmark results. Inspect that both status values are `ok`.

### Evaluation and benchmark recovery

Restore Drive, code, dependencies and W&B login after a reset. Repeat the interrupted command with `--resume`, keeping settings and output directory unchanged. Each new execution gets its own W&B segment and environment file.

Test evaluation restarts an interrupted model evaluation, but skips a model whose JSON was atomically completed. Benchmark recovery skips finished cases and repeats the interrupted case. Configuration/weight/data fingerprints and core GPU/software environment are checked; a different GPU or library stack requires a separate output directory to avoid mixing measurements. Recovery does not load optimizer pickle state.

### Free-form generation

```bash
python -m src.inference.generate --project-dir /content/drive/MyDrive/slm-lora-performance-analysis --run lora-r16-lr5e4 --prompt "The small library opened its doors" --max-new-tokens 64
```

Use `--run baseline` for the original model. Generated text is an example, not a measured factuality or reasoning score.

### Final submission checks

Verify original export checksums with `python scripts/verify_export.py`.
After staging changes, also run `python scripts/verify_export.py --staged`.
Keep the report source, compiled PDF and Overleaf project synchronized.
Confirm reviewer access to GitHub, W&B and the editable Overleaf project.
