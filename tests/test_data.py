"""Tests for text preprocessing and document splitting."""

import pytest

from src.data.preprocessing import (
    assign_split,
    clean_text,
    document_id,
)


SETTINGS = {
    "normalize_whitespace": True,
    "min_characters": 5,
    "max_characters": 100,
}

FRACTIONS = {
    "train_fraction": 0.8,
    "validation_fraction": 0.1,
    "test_fraction": 0.1,
}


def test_clean_text():
    result = clean_text(
        {"text": "  Hello<br />   world!  "},
        ["text"],
        SETTINGS,
    )
    assert result == "Hello world!"


def test_multiple_text_fields():
    result = clean_text(
        {"title": "Great book", "content": "I enjoyed reading it."},
        ["title", "content"],
        SETTINGS,
    )
    assert result == "Great book I enjoyed reading it."


def test_length_filters():
    assert clean_text({"text": "Hi"}, ["text"], SETTINGS) is None
    assert clean_text({"text": "x" * 101}, ["text"], SETTINGS) is None


def test_missing_field_is_not_silently_ignored():
    with pytest.raises(KeyError):
        clean_text({"label": 1}, ["text"], SETTINGS)


def test_non_string_field_is_rejected():
    with pytest.raises(TypeError):
        clean_text({"text": 123}, ["text"], SETTINGS)


def test_duplicate_identity():
    assert document_id("Hello WORLD") == document_id(" hello   world ")
    assert document_id("Hello world") != document_id("Different document")


def test_duplicates_receive_same_split():
    first = document_id("Hello WORLD")
    second = document_id(" hello   world ")

    assert assign_split(first, 42, FRACTIONS) == assign_split(
        second, 42, FRACTIONS
    )


def test_split_assignment_does_not_depend_on_order():
    ids = [document_id(f"Document number {i}") for i in range(100)]

    forward = {
        doc_id: assign_split(doc_id, 42, FRACTIONS)
        for doc_id in ids
    }
    reverse = {
        doc_id: assign_split(doc_id, 42, FRACTIONS)
        for doc_id in reversed(ids)
    }

    assert forward == reverse
    assert set(forward.values()) <= {"train", "validation", "test"}


def test_invalid_split_fractions():
    invalid = {
        "train_fraction": 0.8,
        "validation_fraction": 0.2,
        "test_fraction": 0.2,
    }

    with pytest.raises(ValueError):
        assign_split(document_id("Example"), 42, invalid)
