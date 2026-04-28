#!/usr/bin/env python3
"""
Label → Image: build prompts from semantic labels and run Stable Diffusion.

Example:
  python3 src/run_label_to_image.py --labels "upbeat,electronic,futuristic"
  python3 src/run_label_to_image.py --labels-json '["mellow","piano","sad"]'
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.generation.generate_image import DEFAULT_MODEL_ID, generate_image
from src.generation.prompt_builder import build_prompts, parse_labels_arg


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EchoVision: labels → Stable Diffusion image")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument(
        "--labels",
        type=str,
        help='Comma-separated labels, e.g. "pop,mellow piano,sad"',
    )
    g.add_argument(
        "--labels-json",
        type=str,
        help='JSON array of strings, e.g. \'["upbeat","electronic"]\'',
    )
    p.add_argument("--output", type=Path, default=Path("artifacts/generated/label_image.png"))
    p.add_argument("--meta-output", type=Path, default=None, help="Save prompt + params JSON next to image")
    p.add_argument("--max-labels", type=int, default=8)
    p.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID)
    p.add_argument("--steps", type=int, default=45)
    p.add_argument("--guidance", type=float, default=8.5)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument(
        "--use-hf-token",
        action="store_true",
        help="Use Hugging Face credentials (needed for gated checkpoints)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    label_list = parse_labels_arg(args.labels, args.labels_json)
    prompt, negative = build_prompts(label_list, max_labels=args.max_labels)

    out = generate_image(
        prompt,
        negative,
        output_path=args.output,
        model_id=args.model_id,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        height=args.height,
        width=args.width,
        seed=args.seed,
        device=args.device,
        use_hf_token=args.use_hf_token,
    )

    meta_path = args.meta_output
    if meta_path is None:
        meta_path = out.with_suffix(".json")

    meta = {
        "labels": label_list,
        "prompt": prompt,
        "negative_prompt": negative,
        "model_id": args.model_id,
        "num_inference_steps": args.steps,
        "guidance_scale": args.guidance,
        "height": args.height,
        "width": args.width,
        "seed": args.seed,
        "use_hf_token": args.use_hf_token,
        "output_path": str(out),
    }
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Labels: {label_list}")
    print(f"Saved image: {out}")
    print(f"Saved meta: {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
