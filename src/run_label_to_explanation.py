#!/usr/bin/env python3
"""
Label → Text: use Google Gemini (default) or an OpenAI-compatible API.

Examples (Gemini — get a key from Google AI Studio):
  export GEMINI_API_KEY=...   # or: export GOOGLE_API_KEY=...
  python3 src/run_label_to_explanation.py --labels "upbeat,electronic,futuristic"

  python3 src/run_label_to_explanation.py --from-meta artifacts/generated/label_image.json

OpenAI-compatible (optional):
  export OPENAI_API_KEY=sk-...
  python3 src/run_label_to_explanation.py --provider openai --labels-json '["mellow","piano","sad"]'
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.explanation.llm_explain import (
    DEFAULT_GEMINI_MODEL,
    DEFAULT_OPENAI_MODEL,
    explain_from_labels,
    load_labels_from_meta_json,
)
from src.generation.prompt_builder import parse_labels_arg


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EchoVision: labels → LLM text explanation")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--labels", type=str, help='Comma-separated labels')
    src.add_argument("--labels-json", type=str, help='JSON array of label strings')
    src.add_argument(
        "--from-meta",
        type=Path,
        help="Path to Stage-2 JSON (e.g. label_image.json) with 'labels' and optional 'prompt'",
    )
    p.add_argument(
        "--provider",
        choices=("gemini", "openai"),
        default="gemini",
        help="LLM backend (default: gemini)",
    )
    p.add_argument(
        "--image-prompt",
        type=str,
        default=None,
        help="Optional Stable Diffusion prompt text to ground the explanation",
    )
    p.add_argument("--extra-context", type=str, default=None, help="Optional extra user context string")
    p.add_argument(
        "--model",
        type=str,
        default=None,
        help=f"Model id (Gemini default: {DEFAULT_GEMINI_MODEL}; OpenAI default: {DEFAULT_OPENAI_MODEL})",
    )
    p.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Max output tokens (raise if the model stops mid-sentence; default 2048)",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=0.82,
        help="Sampling temperature (default 0.82 for richer visual variation)",
    )
    p.add_argument(
        "--gemini-max-retries",
        type=int,
        default=3,
        help="On 429/quota errors (Gemini only), retry up to this many extra attempts with backoff",
    )
    p.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Override GEMINI_API_KEY / GOOGLE_API_KEY (Gemini) or OPENAI_API_KEY (openai)",
    )
    p.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="OpenAI only: override OPENAI_BASE_URL (e.g. Ollama http://127.0.0.1:11434/v1)",
    )
    p.add_argument("--output", type=Path, default=Path("artifacts/explanations/label_explanation.txt"))
    p.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Also write a JSON file with labels, model, provider, and explanation",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    image_prompt = args.image_prompt
    if args.from_meta:
        labels, meta_prompt = load_labels_from_meta_json(args.from_meta)
        if image_prompt is None and meta_prompt:
            image_prompt = meta_prompt
    else:
        labels = parse_labels_arg(args.labels, args.labels_json)

    if args.model is None:
        resolved_model = DEFAULT_GEMINI_MODEL if args.provider == "gemini" else DEFAULT_OPENAI_MODEL
    else:
        resolved_model = args.model

    explanation = explain_from_labels(
        labels,
        provider=args.provider,
        image_prompt=image_prompt,
        extra_context=args.extra_context,
        model=resolved_model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        api_key=args.api_key,
        base_url=args.base_url,
        gemini_max_retries=args.gemini_max_retries,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(explanation + "\n", encoding="utf-8")

    meta_out = args.json_output
    if meta_out is None:
        meta_out = args.output.with_suffix(".json")

    payload = {
        "provider": args.provider,
        "gemini_max_retries": args.gemini_max_retries,
        "labels": labels,
        "model": resolved_model,
        "image_prompt": image_prompt,
        "explanation": explanation,
        "text_output_path": str(args.output),
    }
    meta_out.parent.mkdir(parents=True, exist_ok=True)
    meta_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(explanation)
    print(f"\nSaved text: {args.output}")
    print(f"Saved JSON: {meta_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
