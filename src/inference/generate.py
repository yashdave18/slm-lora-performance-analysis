"""Generate a continuation from the base model or a validation-selected adapter."""
import argparse
from pathlib import Path
import torch
from src.inference.runtime import (checked_manifest, resolve_adapter, choose_device,
    load_model, autocast, generation_config)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project-dir", type=Path, required=True)
    p.add_argument("--run", default="lora-r16-lr5e4", help="Training run name, or baseline")
    p.add_argument("--prompt", required=True)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--precision", choices=["fp16", "fp32"], default="fp16")
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = p.parse_args()
    device = choose_device(args.device)
    data, manifest = checked_manifest(args.project_dir, [])
    adapter = None if args.run == "baseline" else resolve_adapter(args.project_dir, args.run, manifest)[0]
    model, tokenizer = load_model(data, manifest, device, adapter)
    ids = tokenizer(args.prompt, return_tensors="pt", return_token_type_ids=False).to(device)
    if ids["input_ids"].numel() == 0 or args.max_new_tokens < 1:
        raise ValueError("Provide a nonempty prompt and positive generation length")
    if ids["input_ids"].shape[1] + args.max_new_tokens > model.config.max_position_embeddings:
        raise ValueError("Prompt plus generation exceeds model context")
    with torch.inference_mode(), autocast(device, args.precision):
        output = model.generate(**ids, generation_config=generation_config(tokenizer, args.max_new_tokens),
                                use_model_defaults=False)
    print(tokenizer.decode(output[0, ids["input_ids"].shape[1]:], skip_special_tokens=True))


if __name__ == "__main__":
    main()
