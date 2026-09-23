# Workflow and experiment documentation

## Data and reproducibility

Model: `EleutherAI/pythia-410m`, revision `9879c9b5f8bea9051dcb0e68dff21493d67e9d4f`. LoRA targets `query_key_value`; base weights remain frozen.

All sources below use their train split. Classification labels and QA reasoning are not training targets: only the configured text fields are used.

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

Preparation normalizes text (including Unicode and whitespace), retains documents of 40–20,000 characters, and deduplicates retained canonical text. Seeded hash assignment uses probabilities 80/10/10, then per-source caps of 1000/100/100 yield 11,000 train, 1,100 validation and 1,100 test documents. Thus the retained split proportions are not exactly 80/10/10. Streaming with a bounded shuffle buffer is practical sampling, not uniform sampling of each full source.

The committed `results/tables/data_manifest.json` records source revisions, processing counts and prepared-file checksums. Tokenization preserves full documents, appends EOS, and saves the tokenizer and a second checksum manifest. At sequence length 256, chunking uses one overlapping context token so each original next-token target is counted once; it never combines different documents. Labels at padding positions are ignored; genuine EOS targets remain included.

| Split | Documents | Tokenized tokens | Sequences at length 256 | Prediction targets |
|---|---:|---:|---:|---:|
| Train | 11000 | 2478818 | 16381 | 2467818 |
| Validation | 1100 | 247245 | 1634 | 246145 |
| Test | 1100 | 245860 | 1629 | 244760 |

Exact document/token duplicate checks do not establish absence of near-duplicates or pretraining contamination. Record this limitation. Dataset redistribution rights must be reviewed before distributing source text; this repository stores metadata and small experimental artifacts.

## Experiment status (21 September 2026)

| Configuration | Status | Best validation perplexity |
|---|---|---:|
| Baseline | Completed | 22.310750 |
| Rank 8, 300-step preliminary schedule | Completed, excluded from main comparison | 20.7145 |
| Rank 8, LR 1e-4, long | Completed 3072 steps; best 2800 | 19.482984 |
| Rank 16, LR 1e-4, long | Early stopping at 3000; best 3000 | 19.282668 |
| Rank 16, LR 5e-4, long | Completed 3072 steps; best 3072 | 19.093474 |
| Short rank-16 configuration | Not reported as executed | — |
| Sequence-1024 training placeholder | Not executed | — |

The selected higher-LR adapter was frozen before test evaluation. Test perplexity is 19.658658 versus baseline 23.113408: a 14.95% reduction over 244,760 targets. All 11 source datasets improve. All 72 inference cases completed successfully. Original artifacts are in `results/final_test/` and `results/inference_benchmarks/`.

## Baseline command

The completed baseline does not need to be rerun. For reproduction in a new output directory:

```bash
python -m src.evaluation.evaluate --experiment experiments/experiment_01_baseline.yaml --data-dir DATA_TOKENIZED --output-dir NEW_BASELINE_OUTPUT
```

This evaluator currently supports the base model, validation split and FP16 autocast on CUDA. For saved-adapter test evaluation use the separate `src.evaluation.final_evaluate` module described in `FINAL_STAGE_GUIDE.md`. The baseline experiment override is applied and validated; unsupported settings are rejected.

## Training and checkpoint selection

Long experiments: rank 8/alpha 16 or rank 16/alpha 32; dropout 0.05; AdamW; peak learning rate 1e-4 or 5e-4; weight decay 0.01; FP16 autocast with FP32 stored weights; batch 2; accumulation 8; sequence length 256; seed 42. Gradient checkpointing is enabled. Cosine scheduling follows 154 warmup steps (ceil(0.05 * 3072)). The learning rate comparison favors 5e-4 among tested settings; the warmup ratio was held fixed, not tuned.

Validation and checkpointing occur every 100 steps and at the final step. Early stopping allows five successive evaluations without a decrease of at least 0.001 from the last significant best token loss. Small decreases can accumulate to a significant improvement. This counter survives recovery. Checkpoint selection separately uses the absolute lowest validation loss, regardless of min_delta.

The rank-8 long run stopped at max_steps, not early stopping. Its best adapter is `runs/lora-r8-long/checkpoints/step-002800/adapter` on Drive. The last checkpoint is for recovery; the best checkpoint is for downstream evaluation. Reserve test data until configuration selection is complete.

## Metrics and timing

Token loss is summed natural-log NLL divided by valid prediction targets. Sequence loss is summed NLL divided by the number of chunks and depends on chunk length. Perplexity is exp(token_loss); bits per token is token_loss / ln(2). Validation aggregates all targets rather than averaging batch losses. Length buckets use valid target counts.

Training throughput counts valid prediction targets per optimizer-update time, including batch preparation and gradient accumulation but excluding validation, logging and checkpoint saving. It is not inference generation throughput. Gradient norm is logged before clipping. W&B logs learning rate used, AMP retries, EMA loss, checkpoint paths, validation bucket tables, examples and rankings.

## Recovery and environment

Drive holds data and complete checkpoint folders. After a Colab reset, mount Drive, restore repository/dependencies, authenticate W&B, then repeat the identical training command with --resume. Restore only trusted checkpoint files. Resume restores adapters, optimizer, scheduler, scaler, RNGs, data cursor, EMA, rankings and stopping state. Work since the last complete checkpoint may repeat.

Each resumed execution has its own W&B run segment. Combine segments by optimizer step; the local JSONL may contain repeated steps after a retry. Recorded environment metadata describes each execution; the latest environment.json is overwritten on resume, so consult both W&B segments when documenting a runtime change. GPU attention may be nondeterministic despite seeding; bitwise identical CUDA recovery is not claimed.

## Artifact export

```bash
python scripts/export_results.py --project-dir /content/drive/MyDrive/slm-lora-performance-analysis --output-dir results/exports
```

Use a new destination for each export. The script copies original manifests, baseline metrics, run summaries, resolved configurations, environment files, validation results, history and failure records. It excludes checkpoints, raw data, credentials and W&B service files, derives a validation CSV and records SHA256 checksums of copied artifacts. A missing completion summary is reported, not invented. Review small artifacts before committing them.

## Run links

- [Project](https://wandb.ai/daveyash1218-dwarkadas-j-sanghvi-college-of-engineering/slm-lora-performance-analysis)
- [Baseline](https://wandb.ai/daveyash1218-dwarkadas-j-sanghvi-college-of-engineering/slm-lora-performance-analysis/runs/8f521k0q)
- [Rank 8, 300 steps](https://wandb.ai/daveyash1218-dwarkadas-j-sanghvi-college-of-engineering/slm-lora-performance-analysis/runs/r6ja3elz)
- [Rank 8 long, resumed segment](https://wandb.ai/daveyash1218-dwarkadas-j-sanghvi-college-of-engineering/slm-lora-performance-analysis/runs/e8dob4x9)

## Final analysis and report

```bash
python -m pip install -r requirements-analysis.txt
python scripts/analyze_results.py
```

This verifies all 72 case identities, 360 measured generations, derived timing/throughput metrics, test counts and aggregate scores. It writes reproducible CSV tables, PNG/PDF figures and a checksum inventory. `notebooks/analysis.ipynb` runs the same workflow without a GPU.

Read `report/report.pdf` for methods, results and limitations. Rebuild from `report/` using `pdflatex report.tex`, `bibtex report`, then `pdflatex report.tex` twice. `report_overleaf.zip` contains a self-contained Overleaf project.

Editable Overleaf link: **PENDING — add your project's editable sharing link after import.**

### Benchmark interpretation

All variants use FP32 stored weights. FP16 is autocast, adapters are unmerged, output length is fixed at 64 tokens, and full-generation latency includes prefill and decoding. Two warmups and five measurements are used for each combination of four variants, two precisions, three batch sizes and three prompt lengths. Results show greater aggregate throughput with batching and slower FP16 autocast in this workload. Do not generalize to fully FP16-loaded or merged models. Prompt lengths vary at inference; no sequence-length training comparison was performed. Error bars are repeat standard deviations, not training-seed confidence intervals.

Final runs recorded Python 3.13.15, PyTorch 2.11.0+cu128, Transformers 4.57.6, PEFT 0.18.1 and W&B 0.28.1 on a T4. Exact environments and source commit are in the original JSON artifacts.

### Submission items still requiring user action

1. Import `report_overleaf.zip`, enable editable sharing and paste the link above.
2. Export and commit small original training histories/resolved configurations from Drive with `scripts/export_results.py`, using a fresh output directory. Final evaluation/benchmark artifacts do not replace training histories.
3. Confirm repository and W&B result visibility for the reviewer; commit this report and analysis update.

Final W&B runs: [test evaluation](https://wandb.ai/daveyash1218-dwarkadas-j-sanghvi-college-of-engineering/slm-lora-performance-analysis/runs/xtodi84b), [inference benchmark](https://wandb.ai/daveyash1218-dwarkadas-j-sanghvi-college-of-engineering/slm-lora-performance-analysis/runs/b5fiqzmr).
