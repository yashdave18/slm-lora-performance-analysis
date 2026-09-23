# Final repository audit

This document supersedes the interim audit dated 20 September 2026.

Completed: data preparation for 11 sources, pinned Pythia-410M, baseline evaluation,
three main LoRA comparisons, recovery, W&B logging, held-out test evaluation,
72 inference benchmark cases, analysis tables/figures, notebook, LaTeX/PDF report,
editable Overleaf link and archived training histories.

## Integrity and reproducibility fixes

Two archived CSV files had been normalized by Git from CRLF to LF. The original
205-file inventory was correct for the uploaded export. Original export bytes
have been restored without changing the inventory or measured results.
`.gitattributes` disables text conversion for this archive. Verify the working
tree and staged Git bytes using `scripts/verify_export.py` and `--staged`.

Dataset configuration now pins the original Parquet conversion SHAs and shard
lists. The loader rejects mutable/missing revisions and mismatched shard lists.
This changes future preparation; archived resolved training configs stay original.
Dataset recreation still requires recorded library versions, accessible upstream
files and comparison with original prepared-file checksums. Exact GPU bitwise
reproducibility is not claimed.

Empty sequence-1024 and model-test templates are removed. Existing tiny-model
training/recovery tests remain. No sequence-length training run is claimed;
length comparisons concern inference prompts.

## Validation scope

Final result analysis checks 72 cases, 360 timed batch generations and 244,760 test
targets. The cleanup has offline tests for pinned data sources. It does not rerun
GPU training or claim reviewer access to W&B/Overleaf. See `REPRODUCE.md` for commands.
