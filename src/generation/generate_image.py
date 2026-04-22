"""Stable Diffusion image generation from text prompts (pretrained weights)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

# Lazy import diffusers inside functions so `import src.generation` works without SD deps.


def resolve_device(explicit: str | None = None) -> str:
    if explicit:
        return explicit
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def default_dtype(device: str) -> torch.dtype:
    if device == "cuda":
        return torch.float16
    return torch.float32


DEFAULT_MODEL_ID = "runwayml/stable-diffusion-v1-5"


def generate_image(
    prompt: str,
    negative_prompt: str,
    *,
    output_path: Path,
    model_id: str = DEFAULT_MODEL_ID,
    num_inference_steps: int = 35,
    guidance_scale: float = 7.5,
    height: int = 512,
    width: int = 512,
    seed: int | None = None,
    device: str | None = None,
    use_hf_token: bool = False,
) -> Path:
    """
    Run a pretrained Stable Diffusion pipeline and save a PNG.

    First call downloads weights from Hugging Face (network required).

    If ``use_hf_token`` is False (default), passes ``token=False`` so a broken
    ``HF_TOKEN`` in the environment does not break public repos. Set
    ``use_hf_token=True`` (or CLI ``--use-hf-token``) for gated models after
    ``huggingface-cli login``.
    """
    try:
        from diffusers import StableDiffusionPipeline
    except ImportError as e:
        raise ImportError(
            "Install generation deps: pip install diffusers transformers accelerate safetensors Pillow"
        ) from e

    dev = resolve_device(device)
    dtype = default_dtype(dev)

    load_kw: dict = {"torch_dtype": dtype}
    if not use_hf_token:
        load_kw["token"] = False

    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        **load_kw,
    )
    pipe = pipe.to(dev)

    if dev == "mps":
        pipe.enable_attention_slicing()

    gen = torch.Generator(device=dev)
    if seed is not None:
        gen.manual_seed(seed)

    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        height=height,
        width=width,
        generator=gen,
    )
    image = result.images[0]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return output_path


def main_argv(argv: list[str] | None = None) -> int:
    """CLI wrapper for testing generation in isolation."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate image from prompt text.")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--negative-prompt", type=str, default="")
    parser.add_argument("--output", type=Path, default=Path("artifacts/generated/image.png"))
    parser.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument("--steps", type=int, default=35)
    parser.add_argument("--guidance", type=float, default=7.5)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--use-hf-token",
        action="store_true",
        help="Use Hugging Face login / HF_TOKEN (needed for gated models)",
    )
    args = parser.parse_args(argv)

    neg = args.negative_prompt or (
        "blurry, low quality, watermark, text, deformed"
    )
    path = generate_image(
        args.prompt,
        neg,
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
    print(f"Saved: {path}")
    return 0


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    raise SystemExit(main_argv())
