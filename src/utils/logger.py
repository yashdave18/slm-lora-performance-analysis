"""Local JSONL plus W&B logging for training and validation."""
import json
import hashlib
from pathlib import Path
import wandb


class RunLogger:
    def __init__(self, directory, config, name, resumed_from=None):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        tracking = config["training"]["wandb"]
        if not tracking.get("enabled", True):
            raise ValueError("W&B logging is required; use mode=offline if needed.")
        tags = [
            "smoke" if config["smoke"] else "training",
            config["model"]["name"],
            f"rank-{config['model']['lora']['rank']}",
            f"seq-{config['training']['max_sequence_length']}",
            f"batch-{config['training']['batch_size']}",
            f"lr-{config['training']['learning_rate']}",
            config["training"]["precision"],
        ] + [item["name"] for item in config["data"]["datasets"]]
        tags = [tag if len(tag) <= 64 else tag[:54] + "-" + hashlib.sha256(tag.encode()).hexdigest()[:8] for tag in tags]
        # Resumes are separate W&B segments in the same group; no history is overwritten.
        self.run = wandb.init(
            project=tracking["project"], entity=tracking["entity"],
            mode=tracking.get("mode", "online"), group=name,
            name=name + ("-resume" if resumed_from else ""),
            job_type="smoke" if config["smoke"] else "train",
            config={**config, "resumed_from": resumed_from},
            tags=tags, dir=str(self.directory),
        )
        self.run.define_metric("step")
        self.run.define_metric("train/*", step_metric="step")
        self.run.define_metric("val/*", step_metric="step")

    def log(self, values):
        with (self.directory / "history.jsonl").open("a", encoding="utf-8") as file:
            file.write(json.dumps(
                {"segment_id": self.run.id, **values}, allow_nan=False
            ) + "\n")
        self.run.log(values)

    def validation(self, step, metrics, baseline_ppl, examples, rankings):
        numeric = {
            "step": step,
            **{f"val/{key}": value for key, value in metrics.items()
               if key != "ppl_by_length_bucket"},
            "val/delta_ppl": metrics["ppl"] - baseline_ppl,
        }
        self.log(numeric)
        buckets = wandb.Table(columns=["length_bucket", "ppl", "n_tokens", "n_sequences"])
        for label, result in metrics["ppl_by_length_bucket"].items():
            buckets.add_data(label, result["ppl"], result["n_tokens"], result["n_sequences"])
        sample_table = wandb.Table(
            columns=["prompt", "continuation"],
            data=[[item["prompt"], item["continuation"]] for item in examples],
        )
        rank_table = wandb.Table(
            columns=["checkpoint", "step", "token_loss", "ppl"],
            data=[[item["checkpoint"], item["step"], item["token_loss"], item["ppl"]]
                  for item in sorted(rankings, key=lambda row: row["token_loss"])],
        )
        self.run.log({
            "step": step,
            "val/ppl_by_length_bucket": buckets,
            "val/examples": sample_table,
            "val/checkpoint_rankings": rank_table,
        })

    def finish(self, failed=False):
        self.run.finish(exit_code=int(failed))
