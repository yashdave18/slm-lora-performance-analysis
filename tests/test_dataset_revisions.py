"""The loader must never silently move to new Hub data."""
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from src.data import dataset_loader as loader

SHA = "a" * 40
SPEC = {"name": "test/corpus", "source_split": "train", "subset": "default",
        "text_fields": ["text"], "parquet_revision": SHA,
        "parquet_files": ["default/train/0000.parquet"]}

@pytest.fixture
def source(monkeypatch):
    api = Mock()
    api.dataset_info.return_value = SimpleNamespace(sha=SHA)
    api.list_repo_files.return_value = SPEC["parquet_files"]
    stream = Mock()
    stream.shuffle.return_value = stream
    load = Mock(return_value=stream)
    monkeypatch.setattr(loader, "HfApi", lambda: api)
    monkeypatch.setattr(loader, "load_dataset", load)
    return api, load

def test_pinned_source_is_used(source):
    api, load = source
    _, metadata = loader.open_source(SPEC, 42, 1000)
    api.dataset_info.assert_called_once_with("test/corpus", revision=SHA)
    assert load.call_args.kwargs["data_files"]["source"] == [
        f"https://huggingface.co/datasets/test/corpus/resolve/{SHA}/default/train/0000.parquet"]
    assert metadata["parquet_revision"] == SHA

@pytest.mark.parametrize("revision", [None, "main", "refs/convert/parquet"])
def test_mutable_or_missing_revision_rejected(source, revision):
    with pytest.raises(ValueError, match="immutable"):
        loader.open_source({**SPEC, "parquet_revision": revision}, 42, 1000)
    source[1].assert_not_called()

def test_changed_revision_rejected(source):
    source[0].dataset_info.return_value = SimpleNamespace(sha="b" * 40)
    with pytest.raises(ValueError, match="resolved revision"):
        loader.open_source(SPEC, 42, 1000)
    source[1].assert_not_called()

def test_changed_shards_rejected(source):
    source[0].list_repo_files.return_value += ["default/train/0001.parquet"]
    with pytest.raises(ValueError, match="file list"):
        loader.open_source({**SPEC, "parquet_files": ["default/train/0000.parquet"]}, 42, 1000)
    source[1].assert_not_called()
