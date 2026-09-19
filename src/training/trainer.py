"""Single-device LoRA training with exact token-normalized accumulation."""
from contextlib import nullcontext
import copy
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import time
import traceback

import torch
from torch.utils.data import DataLoader, Subset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from importlib.metadata import version

from src.data.batching import CausalTextDataset, CausalCollator, file_sha256
from src.evaluation.metrics import causal_nll, MetricAccumulator, summarize
from src.training.checkpoints import (
    write_json, save_checkpoint, latest_checkpoint, read_state,
)
from src.utils.seed import seed_everything, capture_rng, restore_rng
from src.utils.logger import RunLogger


class BatchStream:
    """Deterministic epoch permutations, with a serializable cursor."""
    def __init__(self, dataset, collator, batch_size, seed):
        self.dataset, self.collator = dataset, collator
        self.batch_size, self.seed = batch_size, seed
        self.epoch, self.position = 0, 0
        self._permute()

    def _permute(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        self.order = torch.randperm(len(self.dataset), generator=generator).tolist()

    def next(self):
        if self.position == len(self.order):
            self.epoch += 1
            self.position = 0
            self._permute()
        indices = self.order[self.position:self.position + self.batch_size]
        self.position += len(indices)
        return self.collator([self.dataset[index] for index in indices])

    def state_dict(self):
        return {"epoch": self.epoch, "position": self.position}

    def load_state_dict(self, state):
        self.epoch, self.position = state["epoch"], state["position"]
        if not 0 <= self.position <= len(self.dataset):
            raise ValueError("Invalid batch cursor.")
        self._permute()

    @property
    def fractional_epoch(self):
        return self.epoch + self.position / len(self.dataset)


def amp_context(device, precision):
    if precision == "fp32":
        return nullcontext()
    if device.type != "cuda":
        raise ValueError("FP16/BF16 require CUDA in this trainer.")
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.autocast("cuda", dtype=dtype)


def attach_lora(base, settings):
    return get_peft_model(base, LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=settings["rank"], lora_alpha=settings["alpha"],
        lora_dropout=settings["dropout"],
        target_modules=settings["target_modules"], bias=settings["bias"],
    ))


def optimizer_update(model, batches, optimizer, scaler, device, precision, max_norm):
    """Optimize the sum of token losses divided by ALL window target tokens."""
    total_tokens = sum(int((b["labels"][:, 1:] != -100).sum()) for b in batches)
    total_sequences = sum(b["input_ids"].size(0) for b in batches)
    if total_tokens == 0:
        raise ValueError("Accumulation window has no prediction targets.")
    parameters = [p for p in model.parameters() if p.requires_grad]
    initial_rng = capture_rng()
    for attempt in range(6):
        if attempt:
            restore_rng(initial_rng)
        optimizer.zero_grad(set_to_none=True)
        total_nll = 0.0
        for cpu_batch in batches:
            batch = {key: value.to(device) for key, value in cpu_batch.items()}
            with amp_context(device, precision):
                output = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"], use_cache=False,
                )
            nll, counts = causal_nll(output.logits, batch["labels"])
            loss = nll.sum() / total_tokens
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite loss; check precision/data.")
            scaler.scale(loss).backward()
            total_nll += float(nll.detach().double().sum())
            del output, nll, counts, loss, batch
        scaler.unscale_(optimizer)
        # Returned norm is measured before clipping.
        norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm)
        if torch.isfinite(norm):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            result = summarize(total_nll, total_tokens, total_sequences)
            result.update({"grad_norm": float(norm), "amp_retries": attempt})
            return result
        optimizer.zero_grad(set_to_none=True)
        if not scaler.is_enabled():
            raise FloatingPointError("Non-finite FP32 gradients.")
        scaler.update()  # Reduce scale and retry the SAME window and dropout state.
    raise FloatingPointError("FP16 gradients remained non-finite after six attempts.")


def evaluate_model(model, dataset, collator, config, device, precision):
    rng, was_training = capture_rng(), model.training
    accumulator = MetricAccumulator(config["length_bucket_upper_bounds"])
    loader = DataLoader(
        dataset, batch_size=config["batch_size"], shuffle=False,
        collate_fn=collator, num_workers=0,
        generator=torch.Generator().manual_seed(0),
    )
    try:
        model.eval()
        with torch.inference_mode():
            for index, batch in enumerate(loader, 1):
                batch = {k: v.to(device) for k, v in batch.items()}
                with amp_context(device, precision):
                    output = model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"], use_cache=False,
                    )
                nll, counts = causal_nll(output.logits, batch["labels"])
                accumulator.update(nll, counts)
                if index % 200 == 0:
                    print(f"Validation {index}/{len(loader)} batches", flush=True)
        return accumulator.compute()
    finally:
        model.train(was_training)
        restore_rng(rng)


def generate_examples(model, tokenizer, config, device, precision):
    rng, was_training = capture_rng(), model.training
    samples = []
    try:
        model.eval()
        with torch.inference_mode():
            for prompt in config["prompts"]:
                inputs = tokenizer(prompt, return_tensors="pt", return_token_type_ids=False).to(device)
                with amp_context(device, precision):
                    output = model.generate(
                        **inputs, max_new_tokens=config["max_new_tokens"],
                        do_sample=config["do_sample"], use_cache=True,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )
                text = tokenizer.decode(
                    output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True
                )
                samples.append({"prompt": prompt, "continuation": text})
        return samples
    finally:
        model.train(was_training)
        restore_rng(rng)


def validate_configuration(config, device):
    t, lora, e = config["training"], config["model"]["lora"], config["evaluation"]
    if t["optimizer"] != "adamw" or t["scheduler"] != "cosine":
        raise ValueError("This version supports adamw with cosine scheduling.")
    for key in ("max_steps", "batch_size", "gradient_accumulation_steps",
                "log_every_steps", "evaluate_every_steps", "save_every_steps"):
        if type(t[key]) is not int or t[key] <= 0:
            raise ValueError(f"training.{key} must be a positive integer.")
    if t["precision"] not in ("fp32", "fp16", "bf16"):
        raise ValueError("Unsupported precision.")
    if t["precision"] != "fp32" and device.type != "cuda":
        raise ValueError("CPU training requires precision=fp32.")
    if t["precision"] == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("This GPU does not support BF16. T4 experiments use FP16.")
    if not 0 <= t["ema_decay"] < 1 or not 0 <= t["warmup_ratio"] < 1:
        raise ValueError("EMA decay and warmup ratio must be in [0, 1).")
    if t["learning_rate"] <= 0 or t["weight_decay"] < 0 or t["max_grad_norm"] <= 0:
        raise ValueError("Invalid optimizer hyperparameters.")
    if not lora["enabled"] or lora["rank"] <= 0 or lora["bias"] != "none":
        raise ValueError("Enable positive-rank LoRA with bias=none.")
    if lora["task_type"] != "CAUSAL_LM":
        raise ValueError("LoRA task_type must be CAUSAL_LM.")
    if e["split"] != "validation":
        raise ValueError("Model selection uses validation; test stays untouched.")
    if e["checkpoint_selection_metric"] != "token_loss" or not e["lower_is_better"]:
        raise ValueError("Checkpoint selection must minimize validation token_loss.")


def train(config, data_dir, output_dir, baseline_path=None, resume=False, device_name=None):
    config = copy.deepcopy(config)
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    t, e, m = config["training"], config["evaluation"], config["model"]
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    validate_configuration(config, device)
    seed_everything(t["seed"])
    manifest = json.loads((data_dir / "tokenization_manifest.json").read_text())
    if m["revision"] != manifest["model_revision"] or m["name"] != manifest["model_name"]:
        raise ValueError("Model name/revision differ from tokenized data.")
    for split in ("train", "validation"):
        if file_sha256(data_dir / f"{split}.jsonl") != manifest["output_file_sha256"][f"{split}.jsonl"]:
            raise ValueError(f"Data checksum mismatch: {split}")

    collator = CausalCollator(manifest["pad_token_id"])
    training_data = CausalTextDataset(data_dir / "train.jsonl", t["max_sequence_length"])
    validation_data = CausalTextDataset(data_dir / "validation.jsonl", e["max_sequence_length"])
    if config["smoke"]:
        # Two sequences per source, never test data. Scores are smoke-only.
        indices, per_source = [], {}
        for index, (doc_index, _, _) in enumerate(validation_data.spans):
            source = validation_data.documents[doc_index]["source"]
            if per_source.get(source, 0) < 2:
                indices.append(index)
                per_source[source] = per_source.get(source, 0) + 1
        validation_data = Subset(validation_data, indices)

    config["data_fingerprint"] = {
        key: manifest["output_file_sha256"][key]
        for key in ("train.jsonl", "validation.jsonl")
    }
    config["training"]["checkpoint_dir"] = str(output_dir / "checkpoints")
    signature_config = copy.deepcopy(config)
    # Output location may change after copying a run, without changing its experiment.
    signature_config["training"].pop("checkpoint_dir", None)
    signature = hashlib.sha256(
        json.dumps(signature_config, sort_keys=True).encode()
    ).hexdigest()
    checkpoint_root = output_dir / "checkpoints"
    resume_checkpoint, state = None, None
    if resume:
        resume_checkpoint = latest_checkpoint(checkpoint_root)
        state = read_state(resume_checkpoint)
        if state["signature"] != signature:
            raise ValueError("Resume config or data differs from checkpoint.")
        if state["step"] >= t["max_steps"]:
            raise ValueError("This run has already completed its step budget.")
    elif output_dir.exists():
        raise FileExistsError("Output exists. Use --resume or a new output directory.")

    baseline = None
    if not config["smoke"]:
        if baseline_path is None:
            raise ValueError("Full training requires --baseline pointing to metrics.json.")
        baseline = json.loads(Path(baseline_path).read_text())
        bc = baseline["configuration"]
        if baseline["stage"] != "baseline":
            raise ValueError("Expected base-model baseline results.")
        if (
            bc["resolved_model_revision"] != m["revision"]
            or bc["model"]["name"] != m["name"]
            or bc["validation_sha256"] != config["data_fingerprint"]["validation.jsonl"]
            or bc["evaluation"]["max_sequence_length"] != e["max_sequence_length"]
            or bc["effective_precision"] != t["precision"] + "_autocast"
        ):
            raise ValueError("Baseline model, validation data, length, or precision mismatch.")
        if baseline["metrics"]["n_tokens"] != validation_data.n_targets:
            raise ValueError("Baseline target count differs from this validation set.")

    tokenizer = AutoTokenizer.from_pretrained(data_dir / "tokenizer", local_files_only=True)
    base = AutoModelForCausalLM.from_pretrained(
        m["name"], revision=m["revision"], dtype=torch.float32,
        trust_remote_code=False,
    ).to(device)
    if max(t["max_sequence_length"], e["max_sequence_length"]) > base.config.max_position_embeddings:
        raise ValueError("Sequence length exceeds model context limit.")
    model = (
        PeftModel.from_pretrained(base, resume_checkpoint / "adapter", is_trainable=True)
        if resume else attach_lora(base, m["lora"])
    )
    model.config.use_cache = False
    if t["gradient_checkpointing"]:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable or any(
        p.requires_grad and "lora_" not in name for name, p in model.named_parameters()
    ):
        raise RuntimeError("Expected only LoRA parameters to be trainable.")
    model.print_trainable_parameters()
    optimizer = torch.optim.AdamW(
        trainable, lr=t["learning_rate"], weight_decay=t["weight_decay"]
    )
    scheduler = get_scheduler(
        "cosine", optimizer, num_warmup_steps=math.ceil(t["warmup_ratio"] * t["max_steps"]),
        num_training_steps=t["max_steps"],
    )
    scaler = torch.amp.GradScaler("cuda", enabled=t["precision"] == "fp16", init_scale=1024)
    stream = BatchStream(training_data, collator, t["batch_size"], t["seed"])
    step, ema, tokens_seen, rankings = 0, None, 0, []
    baseline_metrics = baseline["metrics"] if baseline is not None else None
    if state:
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        stream.load_state_dict(state["stream"])
        step, ema, tokens_seen, rankings = (
            state["step"], state["ema"], state["tokens_seen"], state["rankings"]
        )
        baseline_metrics = state["baseline_metrics"]

    output_dir.mkdir(parents=True, exist_ok=True)
    environment = {
        "python": platform.python_version(), "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "transformers": version("transformers"), "peft": version("peft"),
        "wandb": version("wandb"),
    }
    try:
        environment["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        environment["git_dirty"] = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], text=True
        ).strip())
    except (OSError, subprocess.CalledProcessError):
        environment["git_commit"] = None
    write_json(output_dir / "resolved_config.json", config)
    write_json(output_dir / "environment.json", environment)
    logger = RunLogger(
        output_dir, config, output_dir.name,
        str(resume_checkpoint) if resume else None,
    )
    failed = True
    try:
        if baseline_metrics is None:
            with model.disable_adapter():
                baseline_metrics = evaluate_model(
                    model, validation_data, collator, e, device, t["precision"]
                )
            write_json(output_dir / "smoke_baseline.json", baseline_metrics)
        if state:
            restore_rng(state["rng"])
        model.train()
        last_checkpoint = resume_checkpoint
        while step < t["max_steps"]:
            if device.type == "cuda":
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            batches = [stream.next() for _ in range(t["gradient_accumulation_steps"])]
            learning_rate = optimizer.param_groups[0]["lr"]
            stats = optimizer_update(
                model, batches, optimizer, scaler, device, t["precision"], t["max_grad_norm"]
            )
            scheduler.step()
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            step += 1
            tokens_seen += stats["n_tokens"]
            ema = stats["token_loss"] if ema is None else (
                t["ema_decay"] * ema + (1 - t["ema_decay"]) * stats["token_loss"]
            )
            values = {
                "step": step, "epoch": stream.fractional_epoch,
                **{f"train/{key}": stats[key] for key in
                   ("token_loss", "sequence_loss", "ppl", "bpt", "grad_norm")},
                "train/lr": learning_rate, "train/weight_decay": t["weight_decay"],
                "train/throughput_tokens_per_sec": stats["n_tokens"] / elapsed,
                "train/samples_per_sec": stats["n_sequences"] / elapsed,
                "train/step_time": elapsed, "train/ema_token_loss": ema,
                "train/optimizer": "adamw", "train/precision": t["precision"],
                "train/lora_rank": m["lora"]["rank"],
                "train/grad_accum_steps": t["gradient_accumulation_steps"],
                "train/seed": t["seed"], "train/tokens_seen": tokens_seen,
                "train/amp_retries": stats["amp_retries"],
            }
            if device.type == "cuda":
                values["train/peak_memory_allocated_mib"] = (
                    torch.cuda.max_memory_allocated() / 1024**2
                )
            if step == 1 or step % t["log_every_steps"] == 0 or step == t["max_steps"]:
                logger.log(values)
                print(
                    f"Step {step}/{t['max_steps']} | loss {stats['token_loss']:.4f} | "
                    f"{stats['n_tokens']/elapsed:.0f} target tokens/s", flush=True
                )
            evaluate_now = step % t["evaluate_every_steps"] == 0 or step == t["max_steps"]
            if evaluate_now:
                metrics = evaluate_model(
                    model, validation_data, collator, e, device, t["precision"]
                )
                examples = generate_examples(
                    model, tokenizer, e["generation"], device, t["precision"]
                )
                name = f"step-{step:06d}"
                rankings.append({
                    "checkpoint": name, "step": step,
                    "token_loss": metrics["token_loss"], "ppl": metrics["ppl"],
                })
                write_json(output_dir / f"validation-{step:06d}.json", {
                    "step": step, "metrics": metrics, "examples": examples,
                    "delta_ppl": metrics["ppl"] - baseline_metrics["ppl"],
                    "smoke": config["smoke"],
                })
                logger.validation(step, metrics, baseline_metrics["ppl"], examples, [
                    {"checkpoint": "base", "step": 0,
                     "token_loss": baseline_metrics["token_loss"], "ppl": baseline_metrics["ppl"]},
                    *rankings,
                ])
                print(
                    f"Validation ppl {metrics['ppl']:.4f} | "
                    f"delta {metrics['ppl']-baseline_metrics['ppl']:+.4f}", flush=True
                )
            if evaluate_now or step % t["save_every_steps"] == 0:
                last_checkpoint = save_checkpoint(
                    checkpoint_root, model, tokenizer, optimizer, scheduler, scaler,
                    {
                        "signature": signature, "step": step, "ema": ema,
                        "tokens_seen": tokens_seen, "stream": stream.state_dict(),
                        "rankings": rankings, "baseline_metrics": baseline_metrics,
                    },
                )
                logger.log({
                    "step": step, "train/checkpoint": str(last_checkpoint),
                    "train/checkpoint_epoch": stream.fractional_epoch,
                    "train/checkpoint_step": step,
                })
                if rankings:
                    write_json(output_dir / "best_checkpoint.json",
                               min(rankings, key=lambda row: row["token_loss"]))
        write_json(output_dir / "summary.json", {
            "completed_steps": step, "tokens_seen": tokens_seen,
            "fractional_epoch": stream.fractional_epoch,
            "baseline_ppl": baseline_metrics["ppl"], "rankings": rankings,
            "best_trained_checkpoint": min(rankings, key=lambda row: row["token_loss"]),
            "smoke": config["smoke"], "wandb_url": logger.run.url,
            "latest_checkpoint": str(last_checkpoint),
        })
        print(f"PASS: run complete. Results and checkpoints: {output_dir}", flush=True)
        failed = False
    except BaseException:
        write_json(output_dir / f"failure-{int(time.time())}.json", {
            "last_completed_step": step, "traceback": traceback.format_exc(),
            "recovery": "Use --resume with identical arguments to recover latest completed checkpoint.",
        })
        raise
    finally:
        logger.finish(failed)
