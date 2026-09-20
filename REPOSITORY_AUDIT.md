# Repository audit — 20 September 2026

Reviewed GitHub main at e3b8358add0c4534404258a118892e5a0a0a4163 (Add resumable early stopping and longer rank-8 experiment).

## Already committed and current

All four shared configs; data preparation/preprocessing/tokenization/batching; metrics; training CLI/trainer/checkpoints; logging/seeding/helpers; data/batching/metrics/training/early-stopping tests; dependency requirements; short r8/r16 configs; long r8 config; data_manifest.json. Training files match the tested longer-training update. They are not replaced by this package.

## Changes in this package

- New long rank-16 configuration (3072 steps, rank 16, alpha 32).
- Baseline YAML filled in and actually supported by --experiment in the baseline evaluator.
- Lightweight configuration validation and tests; unsupported baseline precision/split/adapters rejected.
- Functional evaluate.sh wrapper.
- Result export script and tests, with a whitelist excluding weights and raw datasets.
- README, documentation and training guide updated with the reported completed runs, actual workflow and limitations.
- Basic executable analysis notebook for exported results, without GPU dependencies.
- Ignore top-level runs and transport ZIP archives.

Eight targeted configuration/export tests passed. Changed Python files and notebook code cells compile; YAML parses. GPU baseline inference was not rerun. The trainer was not modified or retested by this package.

## Real artifacts still to transfer from Drive

Baseline metrics.json; short and long rank-8 summary/config/environment/history/validation results; best_checkpoint.json; tokenization manifest. Use scripts/export_results.py; do not substitute fabricated JSON for original artifacts. Export captures additional rank-16 artifacts if present and notes missing completion summaries. No original Drive artifact was accessible in this audit.

## Remaining implementation, not missing uploads of completed code

- src/inference/generate.py and scripts/benchmark.sh: placeholders.
- experiments/experiment_04_seq1024.yaml: empty template; not runnable.
- tests/test_model.py: placeholder; actual tiny-model training tests live in test_training.py.
- Adapter/test evaluation and final multi-configuration benchmarks: not implemented yet.
- report/report.tex and references.bib: templates; report.pdf absent; Overleaf link pending.
- Final plots and inference tables: absent. Analysis notebook in this package is an initial results inspection notebook.

Retain experiment_03_lora_r16.yaml as the short-budget configuration; its presence does not claim execution. Shared training defaults remain at 300 steps; the long experiment overrides them. Do not modify resolved settings of a run being resumed.

The repository is ready for the next experiment after these updates and artifact export, but the full assignment is not yet complete.
