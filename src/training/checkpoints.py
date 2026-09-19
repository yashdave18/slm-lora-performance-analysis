"""Save complete optimizer-boundary checkpoints with completion markers."""
import json
import time
from pathlib import Path
import torch
from src.utils.seed import capture_rng


def write_json(path, content):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(content, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def save_checkpoint(root, model, tokenizer, optimizer, scheduler, scaler, state):
    root = Path(root)
    destination = root / f"step-{state['step']:06d}"
    if destination.exists():
        # Preserve stale/partial output from a disconnected attempt before retrying.
        destination.rename(root / f"recovered-{destination.name}-{time.time_ns()}")
    # Incomplete directories are never published by latest.json.
    destination.mkdir(parents=True)
    model.save_pretrained(destination / "adapter", safe_serialization=True)
    tokenizer.save_pretrained(destination / "tokenizer")
    payload = {
        **state,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "rng": capture_rng(),
    }
    torch.save(payload, destination / "training_state.pt")
    write_json(destination / "COMPLETE.json", {"step": state["step"]})
    write_json(root / "latest.json", {"path": destination.name, "step": state["step"]})
    return destination


def latest_checkpoint(root):
    root = Path(root)
    latest = json.loads((root / "latest.json").read_text(encoding="utf-8"))
    checkpoint = root / latest["path"]
    if not (checkpoint / "COMPLETE.json").is_file():
        raise RuntimeError(f"Checkpoint is incomplete: {checkpoint}")
    return checkpoint


def read_state(checkpoint):
    # Only load training_state.pt produced by this project from your own Drive.
    # It contains Python/NumPy RNG state and therefore requires pickle loading.
    return torch.load(
        Path(checkpoint) / "training_state.pt", map_location="cpu", weights_only=False
    )
