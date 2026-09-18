"""Tests for causal sequence chunking and padding."""

import pytest

from src.data.batching import chunk_spans, CausalCollator


@pytest.mark.parametrize(
    "length,max_length",
    [
        (2, 256),
        (255, 256),
        (256, 256),
        (257, 256),
        (511, 256),
        (512, 256),
        (1025, 256),
    ],
)
def test_prediction_targets_are_covered_once(length, max_length):
    spans = chunk_spans(length, max_length)

    target_positions = [
        position
        for start, end in spans
        for position in range(start + 1, end)
    ]

    assert target_positions == list(range(1, length))
    assert all(2 <= end - start <= max_length for start, end in spans)


def test_short_document_has_no_targets():
    assert chunk_spans(1, 256) == []


def test_invalid_sequence_length():
    with pytest.raises(ValueError):
        chunk_spans(10, 1)


def test_padding_does_not_mask_real_eos():
    # Token 0 is both EOS and padding.
    collator = CausalCollator(pad_token_id=0)

    batch = collator([
        {"input_ids": [10, 20, 0]},
        {"input_ids": [30, 0]},
    ])

    assert batch["input_ids"].tolist() == [
        [10, 20, 0],
        [30, 0, 0],
    ]
    assert batch["attention_mask"].tolist() == [
        [1, 1, 1],
        [1, 1, 0],
    ]
    assert batch["labels"].tolist() == [
        [10, 20, 0],
        [30, 0, -100],
    ]

    assert (batch["labels"][:, 1:] != -100).sum().item() == 3