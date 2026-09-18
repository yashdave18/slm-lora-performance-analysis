"""Tokenize prepared documents and record reproducibility metadata."""

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import shutil
import tempfile

from huggingface_hub import model_info
from transformers import AutoTokenizer

from src.utils.helpers import load_configs


SPLITS = ("train", "validation", "test")


def file_sha256(path):
    digest = hashlib.sha256()

    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def tokenize_data(input_dir, output_dir, model_config):
    input_dir = Path(input_dir).resolve()
    output_dir = Path(output_dir).resolve()

    if output_dir.exists():
        raise FileExistsError(
            f"{output_dir} already exists. "
            "Use a new output directory to preserve previous results."
        )

    source_manifest = json.loads(
        (input_dir / "manifest.json").read_text(encoding="utf-8")
    )

    # Verify prepared inputs before tokenizing.
    for split in SPLITS:
        filename = f"{split}.jsonl"
        expected = source_manifest["file_sha256"][filename]

        if file_sha256(input_dir / filename) != expected:
            raise ValueError(f"Input checksum mismatch: {filename}")

    model_name = model_config["name"]

    # Resolve the configured model revision to an immutable commit.
    revision = model_info(
        model_name,
        revision=model_config["revision"],
    ).sha

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        revision=revision,
        trust_remote_code=False,
    )

    if tokenizer.eos_token_id is None:
        raise ValueError("The tokenizer must define an EOS token.")

    # Padding will be applied to batches during training.
    tokenizer.pad_token = tokenizer.eos_token

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(tempfile.mkdtemp(
        prefix="tokenizing-",
        dir=output_dir.parent,
    ))

    metadata = {
        "status": "preparing",
        "model_name": model_name,
        "model_revision": revision,
        "tokenizer_class": type(tokenizer).__name__,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "add_special_tokens": False,
        "append_eos": True,
        "truncation": False,
        "padding": False,
        "input_file_sha256": source_manifest["file_sha256"],
        "versions": {
            package: version(package)
            for package in ("transformers", "huggingface-hub")
        },
        "splits": {},
    }

    # Detect exact tokenized-document overlap across splits.
    # Shorter matching excerpts require a separate chunk-level check.
    token_hash_splits = {}

    try:
        tokenizer.save_pretrained(temporary_dir / "tokenizer")

        for split in SPLITS:
            stats = {
                "documents": 0,
                "tokens_including_eos": 0,
                "max_document_tokens": 0,
                "by_source": {},
            }

            source_path = input_dir / f"{split}.jsonl"
            target_path = temporary_dir / f"{split}.jsonl"

            with (
                source_path.open("r", encoding="utf-8") as reader,
                target_path.open(
                    "w", encoding="utf-8", newline="\n"
                ) as writer,
            ):
                for line in reader:
                    if not line.strip():
                        continue

                    document = json.loads(line)

                    token_ids = tokenizer(
                        document["text"],
                        add_special_tokens=False,
                        truncation=False,
                        padding=False,
                        return_attention_mask=False,
                        return_token_type_ids=False,
                        verbose=False,
                    )["input_ids"]

                    if not token_ids:
                        raise ValueError(
                            f"Empty tokenization for {document['id']}"
                        )

                    token_ids.append(tokenizer.eos_token_id)

                    token_hash = hashlib.sha256(
                        json.dumps(
                            token_ids, separators=(",", ":")
                        ).encode("utf-8")
                    ).hexdigest()

                    previous_split = token_hash_splits.get(token_hash)

                    if previous_split is not None and previous_split != split:
                        raise ValueError(
                            "Exact tokenized-document overlap between "
                            f"{previous_split} and {split}: {document['id']}"
                        )

                    token_hash_splits[token_hash] = split

                    record = {
                        "id": document["id"],
                        "source": document["source"],
                        "input_ids": token_ids,
                    }
                    writer.write(json.dumps(record) + "\n")

                    length = len(token_ids)
                    stats["documents"] += 1
                    stats["tokens_including_eos"] += length
                    stats["max_document_tokens"] = max(
                        stats["max_document_tokens"], length
                    )

                    source_stats = stats["by_source"].setdefault(
                        document["source"],
                        {"documents": 0, "tokens_including_eos": 0},
                    )
                    source_stats["documents"] += 1
                    source_stats["tokens_including_eos"] += length

                    if stats["documents"] % 1000 == 0:
                        print(
                            f"{split}: {stats['documents']:,} documents",
                            flush=True,
                        )

            expected_count = source_manifest["totals"][split]

            if stats["documents"] != expected_count:
                raise ValueError(f"Document count mismatch in {split}")

            stats["mean_document_tokens"] = (
                stats["tokens_including_eos"] / stats["documents"]
            )
            metadata["splits"][split] = stats

        metadata["output_file_sha256"] = {
            f"{split}.jsonl": file_sha256(
                temporary_dir / f"{split}.jsonl"
            )
            for split in SPLITS
        }
        metadata["status"] = "complete"

        (temporary_dir / "tokenization_manifest.json").write_text(
            json.dumps(metadata, indent=2) + "\n",
            encoding="utf-8",
        )

        temporary_dir.rename(output_dir)

        print(f"\nPASS: tokenized data saved to {output_dir}")
        print(f"Model revision: {revision}")

        for split, stats in metadata["splits"].items():
            print(
                f"{split}: {stats['documents']:,} documents | "
                f"{stats['tokens_including_eos']:,} tokens"
            )

    finally:
        if temporary_dir.exists():
            shutil.rmtree(temporary_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    model_config = load_configs()["model"]

    tokenize_data(
        args.input_dir,
        args.output_dir,
        model_config,
    )


if __name__ == "__main__":
    main()