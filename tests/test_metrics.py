"""Tests for causal loss and corpus-level aggregation."""

import math

import pytest
import torch

from src.evaluation.metrics import causal_nll, MetricAccumulator


def test_uniform_predictions_and_padding():
    # Uniform predictions over five tokens should give perplexity 5.
    logits = torch.zeros(2, 4, 5)

    labels = torch.tensor([
        [1, 2, 3, 4],
        [1, 0, -100, -100],
    ])

    nll, counts = causal_nll(logits, labels)

    assert counts.tolist() == [3, 1]
    assert nll.tolist() == pytest.approx(
        [3 * math.log(5), math.log(5)],
        rel=1e-6,
    )


def test_causal_shift():
    logits = torch.full((1, 3, 3), -20.0)

    # Position 0 predicts token 1; position 1 predicts token 2.
    logits[0, 0, 1] = 20.0
    logits[0, 1, 2] = 20.0

    nll, counts = causal_nll(
        logits,
        torch.tensor([[0, 1, 2]]),
    )

    assert counts.item() == 2
    assert nll.item() < 1e-6


def test_token_weighted_aggregation():
    accumulator = MetricAccumulator([2, 4])

    # Different lengths ensure we do not average sequence means.
    accumulator.update(
        torch.tensor([6.0, 1.0]),
        torch.tensor([3, 1]),
    )

    result = accumulator.compute()

    assert result["token_loss"] == pytest.approx(7 / 4)
    assert result["sequence_loss"] == pytest.approx(7 / 2)
    assert result["ppl"] == pytest.approx(math.exp(7 / 4))
    assert result["n_tokens"] == 4
    assert result["n_sequences"] == 2

    assert result["ppl_by_length_bucket"]["3-4"]["token_loss"] == 2
    assert result["ppl_by_length_bucket"]["1-2"]["token_loss"] == 1