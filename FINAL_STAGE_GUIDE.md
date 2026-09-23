# Final test evaluation and inference benchmarking

## What this update implements

- `src/inference/runtime.py`: pinned base/adapter loading, artifact identities, W&B sessions, environment capture and guarded recovery.
- `src/evaluation/final_evaluate.py`: base and frozen selected adapter on the entire test split; overall, per-dataset and length-bucket token-weighted metrics.
- `src/inference/benchmark.py`: fixed-length greedy generation across four variants, two precisions, three batch sizes and three prompt lengths (72 cases).
- `src/inference/generate.py`: prompt-based generation CLI for the base or validation-selected adapter.
- `scripts/export_results.py`: extended to include the higher-LR run and final-stage JSON/CSV artifacts.
- Separate YAML settings, shell launchers and offline tiny-model tests.

The training modules and completed run configurations are not changed. Six final-stage tests passed locally on CPU with Transformers 4.57.6 and PEFT 0.18.1. Tests exercise real tiny GPT-NeoX/PEFT models, but no T4 timing or actual test scores are claimed until Colab executes the commands. Existing requirements suffice; preserve Colab's CUDA PyTorch.

## Install and verify

Extract the delivered ZIP into the repository root; it contains full file paths, not a nested project folder. Back up or review existing changed files before replacement. Commit the extracted files and push, then pull from Colab.

```bash
python -m pytest -q tests/test_final_stage.py
```

## 1. GPU smoke benchmark (validation prompts only)

```bash
python -m src.inference.benchmark --project-dir /content/drive/MyDrive/slm-lora-performance-analysis --config configs/benchmark_smoke.yaml --output-dir /content/drive/MyDrive/slm-lora-performance-analysis/results/benchmark_smoke
```

Two small cases (base and selected adapter), eight generated tokens each, one warmup and two measurements. This verifies CUDA/PEFT generation compatibility without opening the test split. These timings are diagnostic, not final benchmark results. Inspect that both status values are `ok`.

## 2. Final test

```bash
python -m src.evaluation.final_evaluate --project-dir /content/drive/MyDrive/slm-lora-performance-analysis
```

Selection is frozen in `configs/final_evaluation.yaml`: `lora-r16-lr5e4`, best checkpoint `step-003072`. The command rejects changed checkpoint selection, mismatched model revisions, inconsistent run summaries, incompatible training/validation fingerprints or test-file checksum mismatches. It never trains, optimizes, ranks alternatives using test data or changes checkpoints.

Base and selected model are each evaluated from scratch with FP32 stored weights and FP16 autocast at max sequence length 256, batch 2. It verifies full next-token target coverage (expected 244,760 on this corpus). It writes `results/final_test/baseline.json`, `selected.json`, `summary.json` and session/environment metadata on Drive. Compare base and adapted TEST scores to each other, not to validation scores as if they measured the same examples. Per-dataset aggregation uses source identities carried alongside tensor batches; documents can produce multiple chunks.

Do not adjust hyperparameters based on these test results. Checkpoint selection was completed on validation data.

## 3. Full inference benchmark

```bash
python -m src.inference.benchmark --project-dir /content/drive/MyDrive/slm-lora-performance-analysis
```

`configs/benchmark_config.yaml` compares:

- Base model, best rank-8/1e-4, best rank-16/1e-4, best rank-16/5e-4.
- FP32 computation versus FP16 autocast; stored weights remain FP32 in both. This is NOT a comparison of full FP16 versus full FP32 weight storage.
- Batch sizes 1, 2 and 4; prompt lengths 64, 128 and 256; exactly 64 newly generated tokens per sequence.
- Two warmups and five measured runs per case. Adapters remain unmerged; attention uses SDPA, KV caching is enabled and TF32 is disabled. Merged-adapter performance is not measured.

Prompts are deterministically sampled from eligible validation documents using seed 42. The same token prefixes are reused across all models and precisions; shorter lengths truncate those same prompts and smaller batches use subsets. The prompts and their IDs are recorded in session metadata. This small workload samples performance, not representative generation quality across every domain. Five repeats give a basic variability estimate; no p95 accuracy is claimed.

Greedy generation disables EOS termination during timed cases to hold output length fixed. It may generate past a natural stopping point; free-form example generation uses normal EOS termination instead.

The CUDA-synchronized wall-clock timer surrounds `model.generate`, including prefill and decoding. It excludes model loading, tokenization, device transfer, warmup, W&B logging and disk writes. It is full batch completion latency, not time-to-first-token or server/network latency.

Generated tokens/sec = total generated tokens across repeats divided by total elapsed seconds. Sequences/sec uses the same aggregate duration; per-sequence tokens/sec divides aggregate token throughput by batch size. Peak CUDA allocated and reserved memory include the model, input, KV cache and temporary tensors; these are PyTorch allocator statistics, not total GPU/process memory. CPU checks report memory as null.

Individual case JSONs include every repeat. `summary.csv` and `summary.json` are written at completion in `results/inference_benchmarks` on Drive. OOM cases are explicitly recorded with status `oom`, not silently omitted or automatically reconfigured. Unexpected errors stop the command; completed cases remain recoverable.

## Recovery

Restore Drive, code, dependencies and W&B login after a reset. Repeat the interrupted command with `--resume`, keeping settings and output directory unchanged. Each new execution gets its own W&B segment and environment file.

Test evaluation restarts an interrupted model evaluation, but skips a model whose JSON was atomically completed. Benchmark recovery skips finished cases and repeats the interrupted case. Configuration/weight/data fingerprints and core GPU/software environment are checked; a different GPU or library stack requires a separate output directory to avoid mixing measurements. Recovery does not load optimizer pickle state.

## Free-form generation

```bash
python -m src.inference.generate --project-dir /content/drive/MyDrive/slm-lora-performance-analysis --run lora-r16-lr5e4 --prompt "The small library opened its doors" --max-new-tokens 64
```

Use `--run baseline` for the original model. Generated text is an example, not a measured factuality or reasoning score.

## Export and next steps

```bash
python scripts/export_results.py --project-dir /content/drive/MyDrive/slm-lora-performance-analysis --output-dir results/exports-final
```

Use a fresh export destination so earlier result snapshots are preserved. The exporter includes the new final-stage artifacts and higher-LR training history; it omits model weights, optimizer states and raw datasets. W&B tracks evaluation metrics/tables and benchmark comparison tables. Baseline/adapter results are not fabricated in this update.

Final test evaluation, all 72 inference cases, figures, notebook, report/PDF and
training-history export are complete. See documentation.md for results and links.
Use the commands above only to reproduce experiments, with fresh output directories.
