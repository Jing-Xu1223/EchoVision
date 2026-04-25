#!/usr/bin/env python3
"""
EchoVision end-to-end: one audio file → predicted labels → image + LLM explanation.

Stages:
1) Music → labels (your trained CNN + Mel spectrogram, same preprocessing as MusicCaps prep)
2) Labels → image (Stable Diffusion + prompt templates)
3) Labels (+ diffusion prompt) → text (Gemini by default)

Example (use a real file path — not a placeholder):
  export GEMINI_API_KEY=...
  python3 src/run_echovision.py --audio ./my_clip.wav \\
    --checkpoint artifacts/music_label_cnn/best_model.pt \\
    --data-dir data/processed/musiccaps
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.audio_mel import audio_to_logmel, fit_mel_to_frames, load_audio_segment, normalize_mel
from src.label_text import sanitize_vocab_label
from src.explanation.llm_explain import DEFAULT_GEMINI_MODEL, explain_from_labels
from src.generation.generate_image import generate_image
from src.generation.prompt_builder import build_prompts
from src.models.music_label_cnn import MusicLabelCNN


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EchoVision: audio → labels → image + explanation")
    p.add_argument(
        "--audio",
        type=Path,
        required=True,
        help="Path to your audio file (wav/mp3/flac, etc.); must exist on disk",
    )
    p.add_argument("--checkpoint", type=Path, default=Path("artifacts/music_label_cnn/best_model.pt"))
    p.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/processed/musiccaps"),
        help="Directory containing preprocess_config.json (Mel parameters)",
    )
    p.add_argument("--output-dir", type=Path, default=None, help="Run output folder (default: artifacts/echovision_runs/<name>_<ts>)")
    p.add_argument("--start-s", type=float, default=0.0)
    p.add_argument("--duration-s", type=float, default=10.0, help="Seconds of audio to use (matches ~MusicCaps clip length)")
    p.add_argument("--target-frames", type=int, default=None, help="Override CNN time dimension (else from checkpoint)")
    p.add_argument("--threshold", type=float, default=None, help="Label probability threshold (else best_val_threshold from checkpoint)")
    p.add_argument("--max-labels", type=int, default=12, help="Max labels passed to SD / LLM after thresholding")
    p.add_argument("--min-top-k", type=int, default=5, help="If nothing passes threshold, take this many top labels by score")

    p.add_argument("--sd-model-id", type=str, default=None, help="Override Stable Diffusion model id")
    p.add_argument("--sd-steps", type=int, default=35)
    p.add_argument("--sd-guidance", type=float, default=7.5)
    p.add_argument("--sd-seed", type=int, default=None)
    p.add_argument("--sd-device", type=str, default=None)
    p.add_argument("--use-hf-token", action="store_true")

    p.add_argument("--skip-image", action="store_true")
    p.add_argument("--skip-explanation", action="store_true")
    p.add_argument("--explanation-provider", choices=("gemini", "openai"), default="gemini")
    p.add_argument("--explanation-model", type=str, default=None)
    p.add_argument(
        "--explanation-max-tokens",
        type=int,
        default=2048,
        help="LLM output token cap (increase if explanation truncates mid-sentence)",
    )
    p.add_argument("--explanation-temperature", type=float, default=0.82)
    p.add_argument("--gemini-max-retries", type=int, default=3)
    return p.parse_args()


def load_preprocess_config(data_dir: Path) -> dict:
    path = data_dir / "preprocess_config.json"
    defaults = {
        "sample_rate": 16000,
        "n_mels": 128,
        "n_fft": 1024,
        "hop_length": 512,
    }
    if not path.exists():
        return defaults
    cfg = json.loads(path.read_text(encoding="utf-8"))
    return {**defaults, **{k: cfg[k] for k in defaults if k in cfg}}


def load_checkpoint(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def select_labels(
    probs: np.ndarray,
    vocab: list[str],
    threshold: float,
    *,
    max_labels: int,
    min_top_k: int,
) -> tuple[list[str], list[dict[str, float | str]]]:
    indexed = sorted(
        [(float(probs[i]), i) for i in range(len(vocab))],
        key=lambda x: x[0],
        reverse=True,
    )
    chosen_idx = [i for p, i in indexed if p >= threshold]
    if not chosen_idx:
        chosen_idx = [i for _, i in indexed[:min_top_k]]
    chosen_idx = chosen_idx[:max_labels]
    cleaned = [sanitize_vocab_label(vocab[i]) for i in chosen_idx]
    labels = list(dict.fromkeys([t for t in cleaned if t]))
    detail = [
        {
            "label": sanitize_vocab_label(vocab[i]),
            "label_raw": vocab[i],
            "probability": float(probs[i]),
        }
        for i in chosen_idx
    ]
    return labels, detail


def main() -> int:
    args = parse_args()
    if not args.audio.exists():
        ap = str(args.audio).lower()
        if "path/to" in ap or "your_clip" in ap or "sometid" in ap:
            hint = (
                " This looks like a placeholder filename — use a real path "
                "(e.g. ./recordings/song.wav or an absolute path)."
            )
        else:
            hint = " Check spelling and working directory; use an absolute path if unsure."

        audio_dir = args.data_dir / "audio"
        if audio_dir.is_dir():
            sample = sorted(audio_dir.glob("*.wav"))[:1]
            if sample:
                hint += f" Example clip from --data-dir: {sample[0]}"

        raise FileNotFoundError(f"Audio not found: {args.audio}{hint}")

    ckpt = load_checkpoint(args.checkpoint)
    label_vocab: list[str] = ckpt["label_vocab"]
    threshold = float(args.threshold if args.threshold is not None else ckpt.get("best_val_threshold", 0.5))
    train_args = ckpt.get("args") or {}
    target_frames = int(args.target_frames or train_args.get("target_frames", 320))

    preprocess = load_preprocess_config(args.data_dir)
    sr = int(preprocess["sample_rate"])
    n_mels = int(preprocess["n_mels"])
    n_fft = int(preprocess["n_fft"])
    hop_length = int(preprocess["hop_length"])

    audio = load_audio_segment(
        args.audio,
        sample_rate=sr,
        start_s=args.start_s,
        duration_s=args.duration_s,
    )
    mel = audio_to_logmel(audio, sr, n_mels, n_fft, hop_length)
    mel = fit_mel_to_frames(mel, target_frames)
    mel = normalize_mel(mel)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MusicLabelCNN(num_labels=len(label_vocab)).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    with torch.no_grad():
        x = torch.from_numpy(mel).unsqueeze(0).unsqueeze(0).float().to(device)
        logits = model(x)
        probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()

    labels, label_details = select_labels(
        probs,
        label_vocab,
        threshold,
        max_labels=args.max_labels,
        min_top_k=args.min_top_k,
    )

    ts = time.strftime("%Y%m%d_%H%M%S")
    stem = args.audio.stem.replace(" ", "_")[:40]
    out_dir = args.output_dir or (Path("artifacts/echovision_runs") / f"{stem}_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)

    image_path = out_dir / "generated_image.png"
    image_meta_path = out_dir / "image_generation.json"
    explanation_path = out_dir / "explanation.txt"
    explanation_json_path = out_dir / "explanation.json"
    manifest_path = out_dir / "run_manifest.json"

    if args.sd_model_id:
        sd_model = args.sd_model_id
    else:
        from src.generation.generate_image import DEFAULT_MODEL_ID

        sd_model = DEFAULT_MODEL_ID

    prompt, negative = build_prompts(labels)

    if not args.skip_image:
        generate_image(
            prompt,
            negative,
            output_path=image_path,
            model_id=sd_model,
            num_inference_steps=args.sd_steps,
            guidance_scale=args.sd_guidance,
            height=512,
            width=512,
            seed=args.sd_seed,
            device=args.sd_device,
            use_hf_token=args.use_hf_token,
        )
        image_meta = {
            "labels": labels,
            "prompt": prompt,
            "negative_prompt": negative,
            "model_id": sd_model,
            "num_inference_steps": args.sd_steps,
            "guidance_scale": args.sd_guidance,
            "seed": args.sd_seed,
            "output_path": str(image_path),
        }
        image_meta_path.write_text(json.dumps(image_meta, indent=2), encoding="utf-8")
    else:
        image_path = Path("")
        image_meta_path.write_text("{}", encoding="utf-8")

    explanation_text = ""
    if not args.skip_explanation:
        exp_model = args.explanation_model or (
            DEFAULT_GEMINI_MODEL if args.explanation_provider == "gemini" else "gpt-4o-mini"
        )
        explanation_text = explain_from_labels(
            labels,
            provider=args.explanation_provider,
            image_prompt=prompt,
            model=exp_model,
            max_tokens=args.explanation_max_tokens,
            temperature=args.explanation_temperature,
            gemini_max_retries=args.gemini_max_retries,
        )
        explanation_path.write_text(explanation_text + "\n", encoding="utf-8")
        explanation_json_path.write_text(
            json.dumps(
                {
                    "labels": labels,
                    "image_prompt": prompt,
                    "provider": args.explanation_provider,
                    "model": exp_model,
                    "explanation": explanation_text,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    else:
        explanation_path.write_text("", encoding="utf-8")
        explanation_json_path.write_text("{}", encoding="utf-8")

    manifest = {
        "audio_path": str(args.audio.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "preprocess": preprocess,
        "target_frames": target_frames,
        "threshold": threshold,
        "labels": labels,
        "label_scores": label_details,
        "image_prompt": prompt,
        "negative_prompt": negative,
        "image_path": str(image_path) if not args.skip_image else None,
        "image_meta_path": str(image_meta_path) if not args.skip_image else None,
        "explanation_path": str(explanation_path) if not args.skip_explanation else None,
        "explanation_json_path": str(explanation_json_path) if not args.skip_explanation else None,
        "output_dir": str(out_dir.resolve()),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(json.dumps({"output_dir": str(out_dir), "labels": labels}, indent=2))
    if not args.skip_image:
        print(f"Image: {image_path}")
    if not args.skip_explanation:
        print(f"Explanation: {explanation_path}")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
