"""Token-weighted metrics for causal language modeling."""

import math

import torch
import torch.nn.functional as F


def causal_nll(logits, labels):
    """Return summed negative log likelihood and target count per sequence.

    Position t predicts the token at position t+1.
    Labels equal to -100 are excluded from the loss.
    """
    targets = labels[:, 1:].contiguous()

    losses = F.cross_entropy(
        logits[:, :-1, :].float().reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).reshape_as(targets)

    sequence_nll = losses.sum(dim=1)
    target_counts = (targets != -100).sum(dim=1)

    return sequence_nll, target_counts


def summarize(nll_sum, n_tokens, n_sequences):
    """Calculate metrics from totals, not averages of batch averages."""
    if n_tokens == 0 or n_sequences == 0:
        raise ValueError("Cannot calculate metrics without valid targets.")

    token_loss = nll_sum / n_tokens

    return {
        "token_loss": token_loss,
        # Mean summed NLL per sequence; this depends on sequence length.
        "sequence_loss": nll_sum / n_sequences,
        "ppl": math.exp(token_loss),
        "bpt": token_loss / math.log(2),
        "n_tokens": n_tokens,
        "n_sequences": n_sequences,
    }


class MetricAccumulator:
    """Accumulate corpus totals and length-bucket statistics."""

    def __init__(self, bucket_upper_bounds):
        self.bounds = sorted(set(bucket_upper_bounds))
        self.nll_sum = 0.0
        self.n_tokens = 0
        self.n_sequences = 0
        self.buckets = {}

    def update(self, sequence_nll, target_counts):
        losses = sequence_nll.detach().double().cpu().tolist()
        counts = target_counts.detach().cpu().tolist()

        for loss, count in zip(losses, counts):
            if count == 0:
                continue

            if not math.isfinite(loss):
                raise ValueError("Non-finite evaluation loss encountered.")

            self.nll_sum += loss
            self.n_tokens += count
            self.n_sequences += 1

            lower = 1
            bucket_name = None

            for upper in self.bounds:
                if count <= upper:
                    bucket_name = f"{lower}-{upper}"
                    break
                lower = upper + 1

            if bucket_name is None:
                bucket_name = f"{lower}+"

            bucket = self.buckets.setdefault(
                bucket_name,
                {"nll_sum": 0.0, "n_tokens": 0, "n_sequences": 0},
            )
            bucket["nll_sum"] += loss
            bucket["n_tokens"] += count
            bucket["n_sequences"] += 1

    def compute(self):
        result = summarize(
            self.nll_sum,
            self.n_tokens,
            self.n_sequences,
        )
        result["ppl_by_length_bucket"] = {
            name: summarize(**totals)
            for name, totals in self.buckets.items()
        }
        return result