"""Offline tests: adapter updates, token weighting, checkpoints, and batch cursors."""
import copy
import json
import pytest
import torch
from peft import PeftModel
from transformers import GPTNeoXConfig, GPTNeoXForCausalLM, get_scheduler
from src.data.batching import CausalCollator
from src.training.trainer import attach_lora, BatchStream, optimizer_update
from src.training.checkpoints import save_checkpoint, latest_checkpoint, read_state
from src.utils.seed import seed_everything, restore_rng

torch.set_num_threads(1)


def tiny_model(dropout=0.0):
    seed_everything(7)
    base = GPTNeoXForCausalLM(GPTNeoXConfig(
        vocab_size=17, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2,
        max_position_embeddings=32, hidden_dropout=0.0,
        attention_dropout=0.0, bos_token_id=0, eos_token_id=0,
    ))
    original = copy.deepcopy(base)
    model = attach_lora(base, {
        "rank": 2, "alpha": 4, "dropout": dropout,
        "target_modules": ["query_key_value"], "bias": "none",
    })
    return model, original


def toy_data():
    return [{"input_ids": [2, 3, 4, 0]}, {"input_ids": [5, 0]},
            {"input_ids": [6, 7, 8, 9, 0]}, {"input_ids": [2, 9, 0]},
            {"input_ids": [4, 5, 6, 0]}]


def scaler():
    return torch.amp.GradScaler("cuda", enabled=False)


def test_stream_resume_across_epoch_boundary():
    data, collator = toy_data(), CausalCollator(0)
    stream = BatchStream(data, collator, 2, 42)
    for _ in range(3):
        stream.next()
    saved = stream.state_dict()
    expected = [stream.next()["input_ids"] for _ in range(4)]
    restored = BatchStream(data, collator, 2, 42)
    restored.load_state_dict(saved)
    for batch in expected:
        assert torch.equal(batch, restored.next()["input_ids"])


def test_accumulation_matches_combined_token_objective():
    model, _ = tiny_model()
    second = copy.deepcopy(model)
    collator = CausalCollator(0)
    examples = toy_data()[:2]
    opt1 = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.1)
    opt2 = torch.optim.SGD([p for p in second.parameters() if p.requires_grad], lr=0.1)
    first = optimizer_update(model, [collator([x]) for x in examples],
                             opt1, scaler(), torch.device("cpu"), "fp32", 1000)
    combined = optimizer_update(second, [collator(examples)],
                                opt2, scaler(), torch.device("cpu"), "fp32", 1000)
    assert first["n_tokens"] == 4
    assert first["token_loss"] == pytest.approx(combined["token_loss"], abs=1e-6)
    for p1, p2 in zip(model.parameters(), second.parameters()):
        assert torch.allclose(p1, p2, atol=2e-6, rtol=1e-5)


def test_only_adapters_change():
    model, _ = tiny_model()
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.01)
    optimizer_update(model, [CausalCollator(0)(toy_data()[:2])],
                     opt, scaler(), torch.device("cpu"), "fp32", 1.0)
    changed = []
    for name, parameter in model.named_parameters():
        if not torch.equal(before[name], parameter):
            changed.append(name)
            assert "lora_" in name
        if "lora_" not in name:
            assert not parameter.requires_grad
    assert changed


class DummyTokenizer:
    def save_pretrained(self, destination):
        destination.mkdir(parents=True)
        (destination / "tokenizer.json").write_text("{}")


def test_checkpoint_reload_and_optimizer_resume(tmp_path):
    model, base = tiny_model(dropout=0.1)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.01)
    schedule = get_scheduler("cosine", opt, num_warmup_steps=0, num_training_steps=3)
    scale = scaler()
    stream = BatchStream(toy_data(), CausalCollator(0), 2, 42)
    optimizer_update(model, [stream.next()], opt, scale, torch.device("cpu"), "fp32", 1)
    schedule.step()
    checkpoint = save_checkpoint(
        tmp_path, model, DummyTokenizer(), opt, schedule, scale,
        {"step": 1, "stream": stream.state_dict()},
    )
    assert latest_checkpoint(tmp_path) == checkpoint
    state = read_state(checkpoint)

    probe = CausalCollator(0)(toy_data()[:1])
    model.eval()
    with torch.no_grad():
        expected = model(input_ids=probe["input_ids"]).logits
    loaded = PeftModel.from_pretrained(copy.deepcopy(base), checkpoint / "adapter", is_trainable=True)
    loaded.eval()
    with torch.no_grad():
        actual = loaded(input_ids=probe["input_ids"]).logits
    assert torch.allclose(expected, actual, atol=1e-6)

    # Same next optimization step after restoring optimizer, scheduler, sampler and RNG.
    opt2 = torch.optim.AdamW([p for p in loaded.parameters() if p.requires_grad], lr=0.01)
    schedule2 = get_scheduler("cosine", opt2, num_warmup_steps=0, num_training_steps=3)
    opt2.load_state_dict(state["optimizer"])
    schedule2.load_state_dict(state["scheduler"])
    scale2 = scaler()
    scale2.load_state_dict(state["scaler"])
    stream2 = BatchStream(toy_data(), CausalCollator(0), 2, 42)
    stream2.load_state_dict(state["stream"])

    model.train()
    restore_rng(state["rng"])
    optimizer_update(model, [stream.next()], opt, scale, torch.device("cpu"), "fp32", 1)
    schedule.step()
    loaded.train()
    restore_rng(state["rng"])
    optimizer_update(loaded, [stream2.next()], opt2, scale2, torch.device("cpu"), "fp32", 1)
    schedule2.step()
    for first, second in zip(model.parameters(), loaded.parameters()):
        assert torch.allclose(first, second, atol=1e-6)
    assert schedule.get_last_lr() == schedule2.get_last_lr()


def test_incomplete_checkpoint_is_rejected(tmp_path):
    (tmp_path / "step-000001").mkdir()
    (tmp_path / "latest.json").write_text(json.dumps({"path": "step-000001"}))
    with pytest.raises(RuntimeError, match="incomplete"):
        latest_checkpoint(tmp_path)


@pytest.mark.parametrize("early_stop", [False, True])
def test_training_runner_and_interrupted_resume(tmp_path, monkeypatch, early_stop):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast
    from safetensors.torch import load_file
    import src.training.trainer as trainer_module
    from src.data.batching import file_sha256

    # Local randomly initialized model and tokenizer: no Hub downloads.
    _, base = tiny_model()
    base_dir = tmp_path / "tiny-base"
    base.save_pretrained(base_dir)
    tokenizer_backend = Tokenizer(WordLevel(
        {"<eos>": 0, "<unk>": 1, "hello": 2, "world": 3}, unk_token="<unk>"
    ))
    tokenizer_backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_backend,
        eos_token="<eos>", unk_token="<unk>", pad_token="<eos>",
    )
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    tokenizer.save_pretrained(data_dir / "tokenizer")
    for split in ("train", "validation"):
        with (data_dir / f"{split}.jsonl").open("w") as file:
            for i, example in enumerate(toy_data()):
                file.write(json.dumps({
                    **example, "id": f"{split}-{i}", "source": "synthetic"
                }) + "\n")
    (data_dir / "tokenization_manifest.json").write_text(json.dumps({
        "model_name": str(base_dir), "model_revision": "local-test",
        "pad_token_id": 0,
        "output_file_sha256": {
            f"{split}.jsonl": file_sha256(data_dir / f"{split}.jsonl")
            for split in ("train", "validation")
        },
    }))
    config = {
        "smoke": True,
        "data": {"datasets": [{"name": "synthetic"}]},
        "model": {
            "name": str(base_dir), "revision": "local-test",
            "lora": {"enabled": True, "rank": 2, "alpha": 4, "dropout": 0.1,
                     "target_modules": ["query_key_value"], "bias": "none",
                     "task_type": "CAUSAL_LM"},
        },
        "training": {
            "seed": 42, "max_steps": 2, "batch_size": 2,
            "gradient_accumulation_steps": 2, "learning_rate": 0.001,
            "weight_decay": 0.01, "optimizer": "adamw", "scheduler": "cosine",
            "warmup_ratio": 0.0, "precision": "fp32", "max_grad_norm": 1.0,
            "max_sequence_length": 8, "gradient_checkpointing": True,
            "log_every_steps": 1, "evaluate_every_steps": 2, "save_every_steps": 1,
            "ema_decay": 0.9, "checkpoint_dir": "unused",
            "wandb": {"enabled": True, "project": "slm-local-tests",
                      "entity": None, "mode": "offline"},
        },
        "evaluation": {
            "split": "validation", "batch_size": 2, "max_sequence_length": 8,
            "length_bucket_upper_bounds": [4, 8],
            "checkpoint_selection_metric": "token_loss", "lower_is_better": True,
            "generation": {"prompts": ["hello world"], "max_new_tokens": 2,
                           "do_sample": False},
        },
    }
    if early_stop:
        config["training"].update({
            "max_steps": 5, "evaluate_every_steps": 1,
            "early_stopping": {"enabled": True, "patience": 1, "min_delta": 0.001},
        })
        actual_evaluate = trainer_module.evaluate_model
        def plateau(*args, **kwargs):
            metrics = actual_evaluate(*args, **kwargs)
            metrics["token_loss"] = 3.0
            metrics["ppl"] = 20.085536923187668
            return metrics
        monkeypatch.setattr(trainer_module, "evaluate_model", plateau)
    fresh = tmp_path / "fresh"
    interrupted = tmp_path / "interrupted"
    trainer_module.train(config, data_dir, fresh, device_name="cpu")
    original_save = trainer_module.save_checkpoint

    def interrupted_save(*args, **kwargs):
        result = original_save(*args, **kwargs)
        raise RuntimeError("simulated disconnect")

    monkeypatch.setattr(trainer_module, "save_checkpoint", interrupted_save)
    with pytest.raises(RuntimeError, match="simulated disconnect"):
        trainer_module.train(config, data_dir, interrupted, device_name="cpu")
    monkeypatch.setattr(trainer_module, "save_checkpoint", original_save)
    trainer_module.train(config, data_dir, interrupted, resume=True, device_name="cpu")

    first = load_file(str(fresh / "checkpoints/step-000002/adapter/adapter_model.safetensors"))
    second = load_file(str(interrupted / "checkpoints/step-000002/adapter/adapter_model.safetensors"))
    assert first.keys() == second.keys()
    for key in first:
        assert torch.allclose(first[key], second[key], atol=1e-6)
    summary = json.loads((interrupted / "summary.json").read_text())
    assert summary["completed_steps"] == 2
    assert summary["smoke"] is True

    if early_stop:
        assert summary["stop_reason"] == "early_stopping"
        assert summary["early_stopping"]["bad_evaluations"] == 1
        assert (interrupted / "best_checkpoint.json").exists()
        with pytest.raises(ValueError, match="already finished by early stopping"):
            trainer_module.train(config, data_dir, interrupted, resume=True, device_name="cpu")
