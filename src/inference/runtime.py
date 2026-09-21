"""Shared, pinned model loading and auditable evaluation/benchmark sessions."""
from contextlib import nullcontext
import gc
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import subprocess
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from peft import PeftModel
import wandb

from src.data.batching import file_sha256
from src.utils.helpers import load_configs


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def checked_manifest(project, splits):
    data = Path(project) / "data/tokenized"
    manifest = read_json(data / "tokenization_manifest.json")
    config = load_configs()
    if (manifest["model_name"], manifest["model_revision"]) != (
        config["model"]["name"], config["model"]["revision"]
    ):
        raise ValueError("Shared model configuration differs from tokenization manifest")
    for split in splits:
        key = split + ".jsonl"
        if file_sha256(data / key) != manifest["output_file_sha256"][key]:
            raise ValueError(f"Tokenized checksum mismatch: {key}")
    return data, manifest


def resolve_adapter(project, run_name, manifest):
    """Select using recorded validation best, never using test performance."""
    run = Path(project) / "runs" / run_name
    summary = read_json(run / "summary.json")
    best = read_json(run / "best_checkpoint.json")
    config = read_json(run / "resolved_config.json")
    if summary["smoke"] or best != summary["best_trained_checkpoint"]:
        raise ValueError(f"Incomplete, smoke or inconsistent run: {run_name}")
    if (config["model"]["name"], config["model"]["revision"]) != (
        manifest["model_name"], manifest["model_revision"]
    ):
        raise ValueError("Adapter base model differs from tokenization")
    for split in ("train.jsonl", "validation.jsonl"):
        if config["data_fingerprint"][split] != manifest["output_file_sha256"][split]:
            raise ValueError("Adapter used a different data mixture")
    name = f"step-{best['step']:06d}"
    if best["checkpoint"] != name:
        raise ValueError("Unexpected checkpoint name")
    checkpoint = run / "checkpoints" / name
    if read_json(checkpoint / "COMPLETE.json")["step"] != best["step"]:
        raise ValueError("Incomplete checkpoint")
    adapter = checkpoint / "adapter"
    peft_config = read_json(adapter / "adapter_config.json")
    if peft_config["base_model_name_or_path"] != manifest["model_name"]:
        raise ValueError("PEFT base model identity mismatch")
    info = {
        "run": run_name, "checkpoint": name, "validation_selection": best,
        "lora_rank": config["model"]["lora"]["rank"],
        "learning_rate": config["training"]["learning_rate"],
        "adapter_sha256": file_sha256(adapter / "adapter_model.safetensors"),
        "adapter_config_sha256": file_sha256(adapter / "adapter_config.json"),
        "resolved_config_sha256": file_sha256(run / "resolved_config.json"),
    }
    return adapter, info


def choose_device(name):
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Enable a Colab GPU runtime")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return torch.device(name)


def autocast(device, precision):
    if precision == "fp32":
        return nullcontext()
    if precision != "fp16" or device.type != "cuda":
        raise ValueError("FP16 autocast requires CUDA; CPU checks use fp32")
    return torch.autocast("cuda", dtype=torch.float16)


def load_model(data, manifest, device, adapter=None):
    tokenizer = AutoTokenizer.from_pretrained(data / "tokenizer", local_files_only=True)
    if tokenizer.pad_token_id != manifest["pad_token_id"]:
        raise ValueError("Tokenizer pad token mismatch")
    model = AutoModelForCausalLM.from_pretrained(
        manifest["model_name"], revision=manifest["model_revision"],
        dtype=torch.float32, trust_remote_code=False,
        attn_implementation="sdpa",
    )
    if adapter is not None:
        model = PeftModel.from_pretrained(model, adapter, is_trainable=False)
    model.to(device).eval().requires_grad_(False)
    return model, tokenizer


def release():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def generation_config(tokenizer, count, fixed_length=False):
    # Fixed length prevents EOS from making speed comparisons use different workloads.
    return GenerationConfig(
        max_new_tokens=count, min_new_tokens=count if fixed_length else 0,
        do_sample=False, num_beams=1, use_cache=True,
        eos_token_id=None if fixed_length else tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id, bos_token_id=tokenizer.bos_token_id,
    )


def environment(device):
    result = {"python": platform.python_version(), "torch": str(torch.__version__),
              "transformers": version("transformers"), "peft": version("peft"),
              "wandb": version("wandb"), "cuda": torch.version.cuda,
              "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"}
    try:
        result["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        result["git_dirty"] = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        result["git_commit"] = None
    return result


class Session:
    """Atomic result files permit restart; every execution has its own W&B segment."""
    def __init__(self, output, specification, device, resume=False):
        self.output = Path(output)
        env = environment(device)
        signature = identity(specification)
        if resume:
            existing = read_json(self.output / "session.json")
            if existing["signature"] != signature:
                raise ValueError("Configuration, selected weights or data changed on resume")
            initial = existing.get("environment", {})
            if initial and any(initial[k] != env[k] for k in ("torch", "transformers", "peft", "cuda", "device")):
                raise ValueError("GPU or core software changed; use a new output directory for comparable measurements")
        else:
            self.output.mkdir(parents=True, exist_ok=False)
            write_json(self.output / "session.json", {"signature": signature, "specification": specification, "environment": env})
        settings = load_configs()["training"]["wandb"]
        if not settings["enabled"]:
            raise ValueError("W&B tracking must be enabled")
        self.run = wandb.init(project=settings["project"], entity=settings["entity"],
                              mode=settings["mode"], dir=str(self.output),
                              name=self.output.name + ("-resume" if resume else ""),
                              group=self.output.name, job_type=specification["stage"],
                              config={**specification, "environment": env},
                              tags=[specification["stage"], "pythia-410m", "mixture-11",
                                    *[d["name"] for d in load_configs()["data"]["datasets"]]])
        write_json(self.output / f"environment-{self.run.id}.json", env)

    def table(self, key, rows):
        if rows:
            columns = list(rows[0])
            self.run.log({key: wandb.Table(columns=columns, data=[[r.get(k) for k in columns] for r in rows])})

    def finish(self, failed):
        self.run.finish(exit_code=1 if failed else 0)
