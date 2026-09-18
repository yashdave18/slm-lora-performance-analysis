from pathlib import Path
import json

# Save this script inside your local Git repository folder.
ROOT = Path(__file__).resolve().parent

FOLDERS = [
    "configs",
    "src/data",
    "src/training",
    "src/evaluation",
    "src/inference",
    "src/utils",
    "scripts",
    "experiments",
    "results/figures",
    "results/tables",
    "results/benchmarks",
    "checkpoints",
    "report",
    "notebooks",
    "tests",
]

PYTHON_FILES = [
    "src/__init__.py",
    "src/data/__init__.py",
    "src/data/dataset_loader.py",
    "src/data/preprocessing.py",
    "src/training/__init__.py",
    "src/training/trainer.py",
    "src/training/train.py",
    "src/evaluation/__init__.py",
    "src/evaluation/evaluate.py",
    "src/evaluation/metrics.py",
    "src/inference/__init__.py",
    "src/inference/generate.py",
    "src/utils/__init__.py",
    "src/utils/logger.py",
    "src/utils/seed.py",
    "src/utils/helpers.py",
    "tests/test_data.py",
    "tests/test_model.py",
    "tests/test_metrics.py",
]

YAML_FILES = [
    "configs/dataset_config.yaml",
    "configs/model_config.yaml",
    "configs/training_config.yaml",
    "configs/evaluation_config.yaml",
    "experiments/experiment_01_baseline.yaml",
    "experiments/experiment_02_lora_r8.yaml",
    "experiments/experiment_03_lora_r16.yaml",
    "experiments/experiment_04_seq1024.yaml",
]

FILES = {
    ".gitignore": """\
# Python
__pycache__/
*.py[cod]
*.egg-info/
.pytest_cache/
.venv/
venv/
build/
dist/

# Credentials
.env
.env.*
!.env.example
.netrc

# Local data and training outputs
/data/
/outputs/
/cache/
wandb/
checkpoints/*
!checkpoints/.gitkeep
*.pt
*.pth
*.ckpt
*.safetensors
pytorch_model*.bin

# Notebook and editor files
.ipynb_checkpoints/
.vscode/
.idea/
.DS_Store

# LaTeX temporary files; compiled PDFs remain trackable
*.aux
*.log
*.out
*.toc
*.bbl
*.blg
*.fls
*.fdb_latexmk
*.synctex.gz
""",
    "README.md": """\
# SLM Fine-Tuning and Evaluation

LoRA fine-tuning, evaluation, and performance analysis of a small
autoregressive language model.

## Status

Repository scaffold only. Training, evaluation, and inference
implementations will be added next.

## Organization

- `configs/`: dataset, model, training, and evaluation configurations
- `src/`: Python implementation modules
- `scripts/`: command-line launch scripts
- `experiments/`: experiment configurations
- `results/`: figures, tables, and benchmark summaries
- `checkpoints/`: local checkpoints, excluded from Git
- `report/`: LaTeX report and references
- `notebooks/`: analysis notebook
- `tests/`: implementation tests
""",
    "documentation.md": """\
# Documentation

## Implementation status

Scaffold only; no experiments have been run through this repository.

## Planned workflow

1. Configure and prepare datasets.
2. Evaluate the pretrained baseline.
3. Train LoRA adapters with W&B tracking.
4. Evaluate checkpoints and benchmark inference.
5. Analyze results and compile the report.

## Experiment links

- W&B project: pending
- Editable Overleaf link: pending

## Storage

Keep credentials, datasets, model weights, and checkpoints out of Git.
Commit source code, configurations, small result files, and the final PDF.
""",
    "requirements.txt": (
        "# Add validated dependency versions during environment setup.\n"
    ),
    "pyproject.toml": """\
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "slm-finetuning-evaluation"
version = "0.1.0"
description = "SLM fine-tuning, evaluation, and performance analysis"
readme = "README.md"
requires-python = ">=3.10"

[tool.setuptools.packages.find]
where = ["."]
include = ["src", "src.*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
""",
    "report/report.tex": r"""\documentclass[11pt]{article}
\usepackage[a4paper,margin=1in]{geometry}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{hyperref}

\title{SLM Fine-Tuning, Evaluation and Performance Analysis}
\author{Yash Dave}
\date{\today}

\begin{document}
\maketitle

\begin{abstract}
Report template. Experiments and results are pending.
\end{abstract}

\section{Methodology}
% Describe the experimental design.

\section{Dataset Overview}
% Document sources, preprocessing, splits, and token counts.

\section{Model Architecture}
% Describe the selected model.

\section{LoRA Configuration}
% Specify adapter targets, rank, alpha, and dropout.

\section{Training Setup}
% Record hardware, software, optimizer, and training settings.

\section{Results}
% Add actual W&B plots and evaluation results.

\section{Inference Performance}
% Report measured latency, throughput, and memory usage.

\section{Analysis and Discussion}
% Discuss findings, failures, and limitations.

\bibliographystyle{plain}
\bibliography{references}

\end{document}
""",
    "report/references.bib": "% Add verified bibliography entries here.\n",
}

for filename in PYTHON_FILES:
    FILES[filename] = '"""Module template: implementation pending."""\n'

for filename in YAML_FILES:
    FILES[filename] = "# Configuration template: values pending.\n{}\n"

for name in ["train", "evaluate", "benchmark"]:
    FILES[f"scripts/{name}.sh"] = (
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f'echo "{name}: implementation pending." >&2\n'
        "exit 1\n"
    )

# Git does not track empty folders, so keep placeholders in these.
for folder in [
    "results/figures",
    "results/tables",
    "results/benchmarks",
    "checkpoints",
]:
    FILES[f"{folder}/.gitkeep"] = ""

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Experiment Analysis\n",
                "\n",
                "Add analysis after collecting actual experiment results.\n",
            ],
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

FILES["notebooks/analysis.ipynb"] = json.dumps(notebook, indent=2) + "\n"


def main():
    for folder in FOLDERS:
        (ROOT / folder).mkdir(parents=True, exist_ok=True)

    created = 0
    skipped = 0

    for relative_path, content in FILES.items():
        target = ROOT / relative_path

        try:
            # Exclusive creation protects existing files.
            with target.open("x", encoding="utf-8", newline="\n") as file:
                file.write(content)
            print(f"Created: {relative_path}")
            created += 1
        except FileExistsError:
            print(f"Kept existing: {relative_path}")
            skipped += 1

    print(f"\nDone: {created} files created, {skipped} existing files kept.")
    print(f"Project directory: {ROOT}")
    print("report.pdf will be generated when the LaTeX report is compiled.")
    print("Files are created locally; commit and push them to upload to GitHub.")


if __name__ == "__main__":
    main()