"""Offline tiny-model tests; no Hub model or test corpus is downloaded."""
import json
import math
from types import SimpleNamespace
import pytest
import torch
from transformers import GPTNeoXConfig, GPTNeoXForCausalLM
from src.data.batching import CausalTextDataset
from src.evaluation.final_evaluate import evaluate
from src.inference.benchmark import select_prompts, measured_generation, summarize_measurements
from src.inference.runtime import resolve_adapter, write_json, file_sha256


def tiny():
    torch.set_num_threads(1)
    torch.manual_seed(42)
    model = GPTNeoXForCausalLM(GPTNeoXConfig(vocab_size=17, hidden_size=16,
        intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
        max_position_embeddings=64, bos_token_id=0, eos_token_id=0))
    return model.eval()


def test_evaluation_counts_and_dataset_aggregation(tmp_path):
    path = tmp_path / "test.jsonl"
    docs = [{"id":"a", "source":"one", "input_ids":[2,3,4,5,6,0]},
            {"id":"b", "source":"two", "input_ids":[7,0]}]
    path.write_text("\n".join(json.dumps(d) for d in docs))
    data = CausalTextDataset(path, 4)
    result = evaluate(tiny(), data, 0, 2, [2,4], torch.device("cpu"), "fp32")
    assert result["metrics"]["n_tokens"] == 6
    assert result["metrics"]["n_sequences"] == 3
    assert result["by_dataset"]["one"]["n_tokens"] == 5
    assert result["by_dataset"]["two"]["n_tokens"] == 1
    weighted = sum(m["token_loss"]*m["n_tokens"] for m in result["by_dataset"].values())/6
    assert result["metrics"]["token_loss"] == pytest.approx(weighted)
    second = evaluate(tiny(), data, 0, 1, [2,4], torch.device("cpu"), "fp32")
    assert second["metrics"]["token_loss"] == pytest.approx(weighted, abs=1e-6)


def test_fixed_generation_counts_even_when_model_predicts_eos():
    model = tiny()
    for p in model.parameters():
        p.data.zero_()  # Tied logits -> argmax token 0 (EOS).
    tok = SimpleNamespace(eos_token_id=0, pad_token_id=0, bos_token_id=0)
    ids = torch.tensor([[2,3,4],[5,6,7]])
    measured = measured_generation(model, tok, ids, "fp32", 4)
    assert measured["generated_tokens"] == 8
    assert measured["seconds"] > 0


def test_speed_uses_total_tokens_over_total_time():
    samples = [{"seconds":s,"generated_tokens":8,"peak_allocated_mib":None,
                "peak_reserved_mib":None} for s in (1.0,3.0)]
    result = summarize_measurements(samples, 2)
    assert result["generated_tokens_per_sec"] == 4
    assert result["sequences_per_sec"] == 1
    assert result["per_sequence_tokens_per_sec"] == 2
    assert result["latency_mean_ms"] == 2000


def test_prompt_sampling_reproducible_no_eos(tmp_path):
    path = tmp_path / "val.jsonl"
    docs = [{"id":str(i), "source":"one", "input_ids":[2,3,4,5,0]} for i in range(20)]
    docs.append({"id":"short", "source":"one", "input_ids":[0]})
    path.write_text("\n".join(json.dumps(d) for d in docs))
    a = select_prompts(path, 4, 3, 42, 0)
    assert a == select_prompts(path, 4, 3, 42, 0)
    assert len({d["id"] for d in a}) == 3
    assert all(len(d["input_ids"]) == 4 and 0 not in d["input_ids"] for d in a)
    with pytest.raises(ValueError):
        select_prompts(path, 5, 3, 42, 0)


def test_adapter_selection_checks_completed_run_and_manifest(tmp_path):
    run = tmp_path / "runs/example"
    adapter = run / "checkpoints/step-000100/adapter"
    adapter.mkdir(parents=True)
    (adapter/"adapter_model.safetensors").write_bytes(b"fixture-only")
    write_json(adapter/"adapter_config.json", {"base_model_name_or_path":"model"})
    best = {"checkpoint":"step-000100","step":100,"token_loss":3.0,"ppl":math.exp(3)}
    write_json(run/"best_checkpoint.json", best)
    write_json(run/"summary.json", {"smoke":False,"best_trained_checkpoint":best})
    write_json(adapter.parent/"COMPLETE.json", {"step":100})
    manifest = {"model_name":"model","model_revision":"abc",
                "output_file_sha256":{"train.jsonl":"t","validation.jsonl":"v"}}
    write_json(run/"resolved_config.json", {"model":{"name":"model","revision":"abc","lora":{"rank":16}},
        "training":{"learning_rate":0.0005}, "data_fingerprint":manifest["output_file_sha256"]})
    actual, info = resolve_adapter(tmp_path,"example",manifest)
    assert actual == adapter and info["lora_rank"] == 16
    manifest["output_file_sha256"]["train.jsonl"] = "changed"
    with pytest.raises(ValueError, match="different data"):
        resolve_adapter(tmp_path,"example",manifest)


def test_real_adapter_test_and_benchmark_commands_resume(tmp_path, monkeypatch):
    """Run both CLIs with local model/tokenizer and real PEFT, replacing only W&B transport."""
    import sys
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast, AutoModelForCausalLM
    from peft import get_peft_model, LoraConfig
    import yaml
    import src.inference.runtime as runtime
    import src.inference.benchmark as benchmark
    import src.evaluation.final_evaluate as final

    model = tiny()
    base = tmp_path / "tiny-base"
    model.save_pretrained(base)
    data = tmp_path / "data/tokenized"
    data.mkdir(parents=True)
    backend = Tokenizer(WordLevel({"<eos>":0,"<unk>":1,"hello":2,"world":3}, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, eos_token="<eos>",
                                       pad_token="<eos>", unk_token="<unk>")
    tokenizer.save_pretrained(data/"tokenizer")
    documents = [{"id":"a", "source":"one", "input_ids":[2,3,4,5,6,0]},
                 {"id":"b", "source":"two", "input_ids":[7,8,9,10,11,0]}]
    for split in ("train","validation","test"):
        (data/f"{split}.jsonl").write_text("\n".join(json.dumps(d) for d in documents))
    manifest = {"model_name":str(base),"model_revision":"local", "pad_token_id":0,
        "output_file_sha256":{f"{s}.jsonl":file_sha256(data/f"{s}.jsonl") for s in ("train","validation","test")},
        "splits":{"test":{"tokens_including_eos":12,"documents":2}}}
    write_json(data/"tokenization_manifest.json",manifest)
    loaded = AutoModelForCausalLM.from_pretrained(base)
    peft_model = get_peft_model(loaded, LoraConfig(r=2,lora_alpha=4,target_modules=["query_key_value"],task_type="CAUSAL_LM"))
    run = tmp_path/"runs/selected"
    checkpoint = run/"checkpoints/step-000001"
    peft_model.save_pretrained(checkpoint/"adapter")
    best = {"checkpoint":"step-000001","step":1,"token_loss":3.0,"ppl":math.exp(3)}
    write_json(run/"best_checkpoint.json",best)
    write_json(run/"summary.json",{"smoke":False,"best_trained_checkpoint":best})
    write_json(checkpoint/"COMPLETE.json",{"step":1})
    write_json(run/"resolved_config.json",{"model":{"name":str(base),"revision":"local","lora":{"rank":2}},
       "training":{"learning_rate":0.0005},"data_fingerprint":manifest["output_file_sha256"]})
    monkeypatch.setattr(runtime,"load_configs",lambda: {
        "model":{"name":str(base),"revision":"local"},"data":{"datasets":[{"name":"synthetic"}]},
        "training":{"wandb":{"enabled":True,"project":"offline-test","entity":None,"mode":"offline"}}})
    class Run:
        id = "local-test"
        url = "offline-test"
        def log(self,*a,**k): pass
        def finish(self,*a,**k): pass
    monkeypatch.setattr(runtime.wandb,"init",lambda **kwargs: Run())
    cfg = tmp_path/"eval.yaml"
    cfg.write_text(yaml.safe_dump({"selected_run":"selected","expected_checkpoint":"step-000001",
        "split":"test","precision":"fp32","batch_size":2,"max_sequence_length":4,
        "length_bucket_upper_bounds":[2,4]}))
    args = ["final_evaluate","--project-dir",str(tmp_path),"--config",str(cfg),"--device","cpu"]
    monkeypatch.setattr(sys,"argv",args)
    final.main()
    summary = json.loads((tmp_path/"results/final_test/summary.json").read_text())
    assert summary["results"]["selected"]["metrics"]["n_tokens"] == 10
    monkeypatch.setattr(final,"evaluate",lambda *a,**k: pytest.fail("Completed test stage was reevaluated"))
    monkeypatch.setattr(sys,"argv",args+["--resume"])
    final.main()
    cfg = tmp_path/"bench.yaml"
    cfg.write_text(yaml.safe_dump({"variants":["baseline","selected"],"precisions":["fp32"],
        "batch_sizes":[1,2],"prompt_lengths":[3],"max_new_tokens":2,"warmup_runs":1,"measured_runs":2,"seed":42}))
    args = ["benchmark","--project-dir",str(tmp_path),"--config",str(cfg),"--device","cpu"]
    monkeypatch.setattr(sys,"argv",args)
    benchmark.main()
    summary = json.loads((tmp_path/"results/inference_benchmarks/summary.json").read_text())
    assert len(summary["rows"]) == 4
    assert all(r["status"] == "ok" for r in summary["rows"])
    monkeypatch.setattr(benchmark,"measured_generation",lambda *a,**k: pytest.fail("Completed case reran"))
    monkeypatch.setattr(sys,"argv",args+["--resume"])
    benchmark.main()
