# LoRA training stage

This update implements training, validation, W&B logging, local results, and resumable adapter checkpoints. Inference benchmarking, final test evaluation, dataset-level analysis, and the final report remain later stages.

## Files
- src/training/train.py: CLI and experiment overrides.
- src/training/trainer.py: model, token-normalized optimization, validation, generation.
- src/training/checkpoints.py: checkpoint writing and completion markers.
- src/utils/logger.py: W&B plus local JSONL logging.
- src/utils/seed.py: seeding and random-state capture/restore.
- tests/test_training.py: six offline tests using tiny random GPT-NeoX.
- experiments/experiment_02_lora_r8.yaml and experiment_03_lora_r16.yaml: rank comparisons.
- scripts/train.sh: shell entry point.

## Run order
1. Install requirements in Colab, preserving the installed PyTorch.
2. Run python -m pytest tests/test_training.py -q.
3. Run --smoke: three optimizer updates, checkpoint every step, evaluation on two validation sequences per source. A matched base score is recomputed on that subset. These scores are diagnostic only.
4. Start a fresh rank-8 run without --smoke, supplying --baseline with the original baseline metrics.json.
5. Compare rank-16 in a separate output directory with identical data, seed, precision, lengths, effective batch size, and update budget.

Required CLI arguments:
- --data-dir: your Drive data/tokenized directory.
- --output-dir: a NEW Drive run directory.
- --experiment: experiments/experiment_02_lora_r8.yaml.
- --baseline: your Drive results/baseline/metrics.json (full runs).
- --smoke: optional three-step diagnostic run.
- --resume: optional recovery using identical settings and output directory.

## Budget
300 optimizer updates at batch size 2 and accumulation 8 consume approximately 4,800 of 16,381 training sequences, about 0.293 epochs. This is an initial fixed-budget experiment, not convergence or a complete epoch. Record tokens_seen and fractional_epoch. Agree on a larger fixed budget before the rank comparisons if the learning curves justify it.

## Recovery
Repeat the identical command with --resume, retaining --smoke for smoke runs.
Restore covers adapter weights, optimizer, scheduler, AMP scaler, RNGs, batch cursor, EMA, rankings, and global step. Work after the latest checkpoint is replayed. Keep the same software/hardware stack.
Each resumed execution is a new W&B segment in the same group, with its resumed_from field. Use the custom step axis. Local history may contain repeated steps from a failed segment.
Before the first checkpoint, recover with a new output directory after addressing the error.
Completed runs reject resume. Changed budgets/configurations require a new experiment.
Partial/stale checkpoint directories encountered during replay are preserved as recovered-*.
Only load your own training_state.pt files; Python/NumPy RNG serialization requires pickle loading.

## Metrics
Training loss and speed describe the logged optimizer update, not the whole logging interval.
- token_loss: total NLL / valid targets across the entire accumulation window.
- sequence_loss: total NLL / number of sequences.
- grad_norm: unscaled, before clipping.
- throughput_tokens_per_sec: valid targets / update seconds.
- step_time includes batch construction and optimization, excludes validation, logging, generation and checkpoint writing.
- train/lr: learning rate used by that update.
- train/amp_retries: reduced-scale retries of the same data and dropout state.
- val/delta_ppl: adapted perplexity minus matched base perplexity; negative means improvement.
- Length buckets use valid target length.
- best_checkpoint.json selects the best trained adapter; it may still be worse than the base.

Checksums and pinned model revisions are verified. Near-duplicate excerpts and pretraining overlap are not measured here. Test data is never opened by training.

## Precision
Stored weights are FP32; FP16/BF16 use autocast. T4 uses FP16 with gradient scaling; unsupported BF16 is rejected. CPU requires FP32.
Full comparisons require matching baseline precision, validation length, revision and data checksum. Your existing baseline uses FP16 autocast and length 256.

## Outputs
Each run contains resolved_config.json, environment.json, history.jsonl, validation-XXXXXX.json, best_checkpoint.json, summary.json, and checkpoints.
Each checkpoint includes adapter/, tokenizer/, training_state.pt and COMPLETE.json.
checkpoints/latest.json points to the latest completed checkpoint.
Exceptions are recorded in failure-*.json.
Keep checkpoints on Drive and later copy only small summaries/plots into Git.

## Verification
Six CPU tests passed with Transformers 4.57.6, PEFT 0.18.1, W&B 0.28.1 and PyTorch 2.14.0:
1. Sampler restoration across epochs.
2. Unequal-length accumulation matches a combined token-weighted batch.
3. Only LoRA parameters change.
4. Adapter reload plus optimizer/scheduler/RNG restoration reproduces the next update.
5. Incomplete checkpoints are rejected.
6. The actual runner trains, logs offline, validates, generates and resumes after a simulated disconnect; final adapters match uninterrupted execution.

Pythia-410M on a T4, CUDA mixed precision, memory and quality must still be checked in your Colab smoke run. No training results are fabricated.
