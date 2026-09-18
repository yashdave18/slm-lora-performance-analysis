"""Text cleaning and deterministic document splitting."""

import hashlib
import math
import re
import unicodedata
from collections.abc import Mapping


def clean_text(row, text_fields, settings):
    """Extract configured text fields and clean one document.

    Returns None when the document fails the length filter.
    Missing columns or non-string values raise errors rather than
    silently producing incorrect training data.
    """
    parts = []

    for field in text_fields:
        if field not in row:
            raise KeyError(f"Required text field is missing: {field}")

        value = row[field]

        if not isinstance(value, str):
            raise TypeError(
                f"Expected text in '{field}', got {type(value).__name__}"
            )

        parts.append(value)

    text = "\n".join(parts)
    text = unicodedata.normalize("NFKC", text)

    # Normalize common HTML line breaks in review datasets.
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)

    if settings.get("normalize_whitespace", True):
        text = re.sub(r"\s+", " ", text)

    text = text.strip()

    minimum = settings.get("min_characters", 40)
    maximum = settings.get("max_characters", 20000)

    if not text or not minimum <= len(text) <= maximum:
        return None

    return text


def document_id(text):
    """Create a stable identity for deduplication and split assignment.

    Case and whitespace differences share the same identity.
    The actual training text retains its original capitalization.
    """
    canonical = unicodedata.normalize("NFKC", text)
    canonical = " ".join(canonical.casefold().split())

    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def assign_split(doc_id, seed, fractions):
    """Assign a document to a reproducible train/validation/test split.

    The assignment depends on content and seed, not dataset order.
    Identical document IDs therefore always receive the same split.
    """
    if not isinstance(fractions, Mapping):
        raise TypeError("Split fractions must be a mapping.")

    train = fractions["train_fraction"]
    validation = fractions["validation_fraction"]
    test = fractions["test_fraction"]

    values = [train, validation, test]

    if any(not 0 < value < 1 for value in values):
        raise ValueError("Each split fraction must be between 0 and 1.")

    if not math.isclose(sum(values), 1.0, abs_tol=1e-9):
        raise ValueError("Split fractions must sum to 1.")

    digest = hashlib.sha256(
        f"{seed}:{doc_id}".encode("utf-8")
    ).digest()

    # Uniform deterministic integer in [0, 2**64).
    value = int.from_bytes(digest[:8], "big")
    scale = 2**64

    if value < int(train * scale):
        return "train"

    if value < int((train + validation) * scale):
        return "validation"

    return "test"
