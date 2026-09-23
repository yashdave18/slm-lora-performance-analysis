# Final submission checks

The initial installation checklist is superseded by this page.

1. Run `python scripts/verify_export.py` and `python scripts/analyze_results.py`.
2. After staging, run `python scripts/verify_export.py --staged`.
3. Synchronize the corrected report source with the existing Overleaf project
   (keep its sharing link), and confirm the reviewer can view W&B results.
4. Commit/push the report, source, analysis and archive changes.
5. Submit the GitHub URL and compiled PDF. No extra GPU experiment is needed.

See REPRODUCE.md for a fresh-environment workflow. Do not commit local installers,
transport ZIPs other than report_overleaf.zip, raw datasets or checkpoints.
