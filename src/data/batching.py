"""Document chunking and dynamic padding for causal language modeling."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

from src.utils.helpers import load_configs


def chunk_spans(length, max_length):
    """Return spans covering every next-token target exactly once.

    Adjacent chunks share one context token.
    Documents need at least two tokens to contribute a prediction target.
    """
    if max_length < 2:
        raise ValueError("max_length must be at least 2.")

    if length < 2:
        return []

    return [
        (start, min(start + max_length, length))
        for start in range(0, length - 1, max_length - 1)
    ]


def file_sha256(path):
    digest = hashlib.sha256()

    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


class CausalTextDataset(Dataset):
    """Expose tokenized documents as bounded training sequences."""

    def __init__(self, path, max_length):
        self.documents = []
        self.spans = []
        self.n_targets = 0

        seen_ids = set()

        with Path(path).open("r", encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue

                document = json.loads(line)
                doc_id = document["id"]
                token_ids = document["input_ids"]

                if doc_id in seen_ids:
                    raise ValueError(f"Duplicate document ID: {doc_id}")

                if (
                    not isinstance(token_ids, list)
                    or len(token_ids) < 2
                    or any(type(token) is not int or token < 0
                           for token in token_ids)
                ):
                    raise ValueError(f"Invalid token IDs: {doc_id}")

                seen_ids.add(doc_id)
                document_index = len(self.documents)
                self.documents.append(document)

                for start, end in chunk_spans(len(token_ids), max_length):
                    self.spans.append((document_index, start, end))
                    self.n_targets += end - start - 1

        if not self.spans:
            raise ValueError(f"No usable sequences in {path}")

    def __len__(self):
        return len(self.spans)

    def __getitem__(self, index):
        document_index, start, end = self.spans[index]
        document = self.documents[document_index]

        return {
            "input_ids": document["input_ids"][start:end],
            "document_id": document["id"],
            "source": document["source"],
        }


class CausalCollator:
    """Right-pad batches and mask only padding positions in labels."""

    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, examples):
        if not examples:
            raise ValueError("Cannot collate an empty batch.")

        lengths = [len(example["input_ids"]) for example in examples]

        if min(lengths) < 2:
            raise ValueError("Each sequence must contain at least two tokens.")

        batch_size = len(examples)
        width = max(lengths)

        input_ids = torch.full(
            (batch_size, width),
            self.pad_token_id,
            dtype=torch.long,
        )
        attention_mask = torch.zeros_like(input_ids)

        for index, example in enumerate(examples):
            length = lengths[index]
            input_ids[index, :length] = torch.tensor(
                example["input_ids"],
                dtype=torch.long,
            )
            attention_mask[index, :length] = 1

        labels = input_ids.clone()

        # EOS may also be the padding token.
        # Mask by position, never by token ID.
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    args = parser.parse_args()

    config = load_configs()
    manifest = json.loads(
        (args.data_dir / "tokenization_manifest.json").read_text(
            encoding="utf-8"
        )
    )

    if manifest["model_revision"] != config["model"]["revision"]:
        raise ValueError(
            "Pin model_config.yaml to the tokenization manifest revision."
        )

    collator = CausalCollator(manifest["pad_token_id"])

    for split in ("train", "validation", "test"):
        path = args.data_dir / f"{split}.jsonl"

        if file_sha256(path) != manifest["output_file_sha256"][path.name]:
            raise ValueError(f"Tokenized file checksum mismatch: {path.name}")

        section = "training" if split == "train" else "evaluation"

        dataset = CausalTextDataset(
            path,
            max_length=config[section]["max_sequence_length"],
        )

        # Every document's first token provides context but is not a target.
        expected_targets = (
            manifest["splits"][split]["tokens_including_eos"]
            - manifest["splits"][split]["documents"]
        )

        if dataset.n_targets != expected_targets:
            raise ValueError(f"Prediction-target count mismatch: {split}")

        loader = DataLoader(
            dataset,
            batch_size=config[section]["batch_size"],
            shuffle=False,
            collate_fn=collator,
            num_workers=0,
        )

        batch = next(iter(loader))

        # Causal models compare logits[:-1] against labels[1:].
        valid_targets = (batch["labels"][:, 1:] != -100).sum().item()

        print(
            f"{split}: {len(dataset):,} sequences | "
            f"{dataset.n_targets:,} prediction targets"
        )
        print(
            f"  First batch: {tuple(batch['input_ids'].shape)} | "
            f"{valid_targets} valid targets"
        )

    print("\nPASS: chunking, checksums, and batch construction verified.")


if __name__ == "__main__":
    main()