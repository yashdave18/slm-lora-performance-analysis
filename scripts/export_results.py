"""Copy small experiment artifacts from Drive for a Git commit; never weights/data.
Run: python scripts/export_results.py --project-dir /content/drive/MyDrive/slm-lora-performance-analysis
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil


def export_results(project, output):
    project, output = Path(project), Path(output)
    if not project.is_dir():
        raise FileNotFoundError(project)
    if output.exists():
        raise FileExistsError(f"Use a new output directory: {output}")
    sources = []
    def include(path, relative):
        if path.is_file():
            sources.append((path, Path(relative)))
    include(project / "results/baseline/metrics.json", "baseline/metrics.json")
    include(project / "data/processed/manifest.json", "data/manifest.json")
    include(project / "data/tokenized/tokenization_manifest.json", "data/tokenization_manifest.json")
    for name in ("lora-r8", "lora-r8-long", "lora-r16", "lora-r16-long", "lora-r16-lr5e4"):
        run = project / "runs" / name
        if not run.is_dir():
            continue
        for filename in ("summary.json", "best_checkpoint.json", "resolved_config.json",
                         "environment.json", "history.jsonl"):
            include(run / filename, f"runs/{name}/{filename}")
        for pattern in ("validation-*.json", "failure-*.json"):
            for path in sorted(run.glob(pattern)):
                include(path, f"runs/{name}/{path.name}")
    # Final-stage artifacts: include only known JSON/CSV files, never W&B caches.
    for name in ("final_test", "inference_benchmarks", "benchmark_smoke"):
        folder = project / "results" / name
        for pattern in ("*.json", "*.csv", "cases/*.json"):
            for path in sorted(folder.glob(pattern)):
                include(path, Path("final_stage") / name / path.relative_to(folder))
    if not sources:
        raise FileNotFoundError("No experiment artifacts found; check --project-dir")
    output.mkdir(parents=True)
    inventory, rows = [], []
    for source, relative in sources:
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        inventory.append({"path": relative.as_posix(), "sha256": hashlib.sha256(target.read_bytes()).hexdigest()})
        if source.name.startswith("validation-"):
            record = json.loads(target.read_text())
            if not record.get("smoke", False):
                rows.append({"run": relative.parts[1], "step": record["step"],
                             "token_loss": record["metrics"]["token_loss"],
                             "ppl": record["metrics"]["ppl"]})
    with (output / "validation_history.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["run", "step", "token_loss", "ppl"])
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: (r["run"], r["step"])))
    (output / "artifact_inventory.json").write_text(json.dumps(inventory, indent=2) + "\n")
    print(f"Exported {len(inventory)} original files to {output}")
    if not (output / "baseline/metrics.json").exists():
        print("NOTE: baseline metrics were not found.")
    for name in ("lora-r8", "lora-r8-long", "lora-r16-long"):
        if not (output / "runs" / name / "summary.json").exists():
            print(f"NOTE: {name} has no exported completion summary.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/exports"))
    args = parser.parse_args()
    export_results(args.project_dir, args.output_dir)
