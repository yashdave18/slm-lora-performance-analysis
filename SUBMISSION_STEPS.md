# Apply this update

1. Extract this ZIP into the existing repository root; review replacement README, documentation and analysis notebook. It does not change training code or configurations.
2. Run `python -m pip install -r requirements-analysis.txt` and `python scripts/analyze_results.py` in the VS Code terminal. No Colab GPU is needed.
3. Upload `report_overleaf.zip` as a new Overleaf project. Share it with an editable link and insert that link in `documentation.md`.
4. Export original training logs/resolved configs from mounted Drive in Colab:

```python
%cd /content/slm-lora-performance-analysis
!python scripts/export_results.py --project-dir /content/drive/MyDrive/slm-lora-performance-analysis --output-dir results/exports-submission
```

Use a new export directory if that one already exists. Download the export and place it in the local repository. It excludes checkpoints and raw datasets. Review it before committing.

5. In VS Code's terminal:

```powershell
git status
git add README.md documentation.md SUBMISSION_STEPS.md requirements-analysis.txt scripts/analyze_results.py notebooks/analysis.ipynb report report_overleaf.zip results
git commit -m "Add verified final results, reproducible figures and research report"
git pull --rebase origin main
git push
```

If rebase reports conflicts, stop and resolve them before pushing. No force push is needed.

6. Check that the public GitHub report opens and the reviewer can see the W&B project and editable Overleaf link.
