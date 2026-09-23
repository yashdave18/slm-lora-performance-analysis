"""Prepare bounded, deduplicated text datasets from Hugging Face."""

import argparse
from contextlib import ExitStack
import hashlib
from importlib.metadata import version
import json
import re
from pathlib import Path
import shutil
import tempfile

from datasets import load_dataset
from huggingface_hub import HfApi

from src.data.preprocessing import (
    assign_split,
    clean_text,
    document_id,
)
from src.utils.helpers import PROJECT_ROOT, load_configs


SPLITS = ("train", "validation", "test")


def open_source(spec, seed, buffer_size):
    """Resolve immutable Parquet files and create a shuffled stream.

    Returns the stream and metadata identifying the exact source files.
    Missing datasets, subsets, or splits stop preparation explicitly.
    """
    api = HfApi()
    name = spec["name"]
    source_split = spec["source_split"]

    # Use the exact conversion commit recorded during the original preparation.
    pinned = spec.get("parquet_revision")
    if not isinstance(pinned, str) or not re.fullmatch(r"[0-9a-f]{40}", pinned):
        raise ValueError(f"{name}: parquet_revision must be an immutable 40-character SHA")
    files = spec.get("parquet_files")
    if (not isinstance(files, list) or not files
            or any(not isinstance(path, str) for path in files)
            or len(files) != len(set(files))):
        raise ValueError(f"{name}: provide the original unique parquet_files list")
    info = api.dataset_info(name, revision=pinned)
    revision = info.sha
    if revision != pinned:
        raise ValueError(f"{name}: resolved revision differs from the pinned SHA")

    paths = sorted(
        path
        for path in api.list_repo_files(
            name,
            repo_type="dataset",
            revision=revision,
        )
        if path.endswith(".parquet")
    )

    # HF conversion layout: <subset>/<split>/<shard>.parquet
    available = sorted({
        path.split("/")[0]
        for path in paths
        if len(path.split("/")) >= 3
        and path.split("/")[1] == source_split
    })

    subset = spec.get("subset")

    if subset is None:
        if len(available) != 1:
            raise ValueError(
                f"{name}: specify a subset explicitly. "
                f"Available subsets for '{source_split}': {available}"
            )
        subset = available[0]

    prefix = f"{subset}/{source_split}/"
    selected = [path for path in paths if path.startswith(prefix)]

    if not selected:
        raise ValueError(
            f"{name}: no Parquet files for {subset}/{source_split}. "
            f"Available subsets: {available}"
        )

    if selected != sorted(files):
        raise ValueError(f"{name}: pinned Parquet file list differs from repository contents")

    urls = [
        f"https://huggingface.co/datasets/{name}"
        f"/resolve/{revision}/{path}"
        for path in selected
    ]

    stream = load_dataset(
        "parquet",
        data_files={"source": urls},
        split="source",
        streaming=True,
    )

    stream = stream.shuffle(
        seed=seed,
        buffer_size=buffer_size,
    )

    metadata = {
        "name": name,
        "subset": subset,
        "source_split": source_split,
        "text_fields": spec["text_fields"],
        "parquet_revision": revision,
        "parquet_files": selected,
    }

    return stream, metadata


def file_sha256(path):
    """Calculate a checksum without reading a whole file into memory."""
    digest = hashlib.sha256()

    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def prepare_data(config, output_dir, smoke=False, max_rows=50000):
    """Write JSONL splits and a manifest only if all sources succeed."""
    if max_rows <= 0:
        raise ValueError("max_rows must be positive.")

    output_dir = Path(output_dir).resolve()

    if output_dir.exists():
        raise FileExistsError(
            f"{output_dir} already exists. "
            "Use a new --output-dir to preserve previous results."
        )

    if not config["preprocessing"].get("deduplicate", True):
        raise ValueError(
            "Deduplication must remain enabled for this experiment."
        )

    limits = (
        {"train": 10, "validation": 3, "test": 3}
        if smoke
        else dict(config["limits_per_dataset"])
    )

    if set(limits) != set(SPLITS):
        raise ValueError("Limits must specify train, validation, and test.")

    if any(type(value) is not int or value <= 0 for value in limits.values()):
        raise ValueError("Document limits must be positive integers.")

    buffer_size = 100 if smoke else 1000
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    # Write into a temporary directory. Failed preparation does not
    # leave an apparently complete dataset at the requested location.
    temporary_dir = Path(tempfile.mkdtemp(
        prefix="preparing-",
        dir=output_dir.parent,
    ))

    seen = set()
    split_ids = {split: set() for split in SPLITS}
    totals = {split: 0 for split in SPLITS}

    manifest = {
        "status": "preparing",
        "smoke_test": smoke,
        "data_config": config,
        "effective_limits_per_dataset": limits,
        "max_rows_scanned_per_dataset": max_rows,
        "shuffle_buffer_size": buffer_size,
        "sampling": "bounded buffered shuffle; not a uniform full-corpus sample",
        "versions": {
            package: version(package)
            for package in ("datasets", "huggingface-hub", "PyYAML")
        },
        "sources": [],
    }

    try:
        with ExitStack() as stack:
            writers = {
                split: stack.enter_context(
                    (temporary_dir / f"{split}.jsonl").open(
                        "w", encoding="utf-8", newline="\n"
                    )
                )
                for split in SPLITS
            }

            for spec in config["datasets"]:
                print(f"\nPreparing: {spec['name']}", flush=True)

                try:
                    stream, metadata = open_source(
                        spec,
                        seed=config["seed"],
                        buffer_size=buffer_size,
                    )

                    counts = {split: 0 for split in SPLITS}
                    stats = {
                        "scanned": 0,
                        "rejected_by_length": 0,
                        "duplicate_of_retained_document": 0,
                        "skipped_full_split": 0,
                    }

                    for row in stream.take(max_rows):
                        stats["scanned"] += 1

                        text = clean_text(
                            row,
                            spec["text_fields"],
                            config["preprocessing"],
                        )

                        if text is None:
                            stats["rejected_by_length"] += 1
                            continue

                        doc_id = document_id(text)

                        if doc_id in seen:
                            stats["duplicate_of_retained_document"] += 1
                            continue

                        split = assign_split(
                            doc_id,
                            config["seed"],
                            config["splitting"],
                        )

                        if counts[split] >= limits[split]:
                            stats["skipped_full_split"] += 1
                            continue

                        record = {
                            "id": doc_id,
                            "source": spec["name"],
                            "text": text,
                        }

                        writers[split].write(
                            json.dumps(record, ensure_ascii=False) + "\n"
                        )

                        seen.add(doc_id)
                        split_ids[split].add(doc_id)
                        counts[split] += 1
                        totals[split] += 1

                        if all(
                            counts[key] >= limits[key]
                            for key in SPLITS
                        ):
                            break

                    if any(counts[split] == 0 for split in SPLITS):
                        raise ValueError(
                            f"At least one split is empty: {counts}. "
                            "Inspect the source or increase --max-rows."
                        )

                    metadata["counts"] = counts
                    metadata["filtering"] = stats
                    metadata["targets_met"] = all(
                        counts[key] == limits[key]
                        for key in SPLITS
                    )
                    manifest["sources"].append(metadata)

                    print(f"Retained: {counts}", flush=True)

                    if not metadata["targets_met"]:
                        print(
                            "NOTE: source ended or scan limit was reached "
                            "before all document caps were filled.",
                            flush=True,
                        )

                except Exception as error:
                    raise RuntimeError(
                        f"Preparation failed for {spec['name']}: {error}"
                    ) from error

        # Check retained canonical document identities across splits.
        for first, second in [
            ("train", "validation"),
            ("train", "test"),
            ("validation", "test"),
        ]:
            if split_ids[first] & split_ids[second]:
                raise RuntimeError(
                    f"Document overlap detected: {first}/{second}"
                )

        manifest["status"] = "complete"
        manifest["totals"] = totals
        manifest["file_sha256"] = {
            f"{split}.jsonl": file_sha256(
                temporary_dir / f"{split}.jsonl"
            )
            for split in SPLITS
        }

        (temporary_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n",
            encoding="utf-8",
        )

        # Both paths are on the same filesystem.
        temporary_dir.rename(output_dir)

        print(f"\nPASS: saved prepared data to {output_dir}")
        print(f"Total retained documents: {totals}")
        return manifest

    finally:
        if temporary_dir.exists():
            shutil.rmtree(temporary_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare text datasets for causal language modeling."
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Retain only 10/3/3 documents per source.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=50000,
        help="Maximum rows examined per source after buffered shuffling.",
    )
    args = parser.parse_args()

    config = load_configs()["data"]

    output_dir = args.output_dir

    if output_dir is None:
        output_dir = (
            PROJECT_ROOT / "data" / "smoke"
            if args.smoke
            else PROJECT_ROOT / config["output_dir"]
        )

    prepare_data(
        config,
        output_dir,
        smoke=args.smoke,
        max_rows=args.max_rows,
    )


if __name__ == "__main__":
    main()