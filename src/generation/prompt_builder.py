"""Turn semantic music labels into stronger Stable Diffusion prompts."""

from __future__ import annotations

import json
import re
from typing import Sequence


DEFAULT_NEGATIVE = (
    "blurry, low quality, worst quality, jpeg artifacts, watermark, signature, "
    "text, logo, deformed, ugly, duplicate, cropped, out of frame, "
    "flat lighting, overexposed, underexposed, plain background, simple gradient, "
    "low contrast, muddy colors"
)

NOISY_LABEL_PATTERNS = (
    "low quality",
    "inferior audio quality",
    "amateur recording",
    "mediocre sound recording",
    "youtube",
    "two tracks",
    "unrelated",
    "quality of",
    "seconds",
)

STYLE_HINTS = {
    "electronic": "futuristic neon cityscape at night with reflective wet streets",
    "synth": "futuristic neon cityscape at night with reflective wet streets",
    "disco": "retro-futuristic dance floor environment with glowing lights",
    "sci-fi": "cinematic sci-fi environment with dramatic depth and atmosphere",
    "orchestra": "epic cinematic environment with monumental architecture",
    "classical": "grand timeless architecture under dramatic cinematic light",
    "salsa": "vibrant latin street festival scene with dynamic movement",
    "metal": "dark industrial urban environment with aggressive lighting",
    "meditation": "serene sacred interior with incense haze and soft glow",
    "chant": "serene sacred interior with incense haze and soft glow",
}

MOOD_HINTS = {
    "mellow": "calm ambience, soft diffusion, gentle shadows",
    "sad": "melancholic atmosphere, desaturated tones, lingering mist",
    "dark": "low-key lighting, deep shadows, ominous mood",
    "energetic": "high energy, dynamic motion trails, vivid contrast",
    "upbeat": "bright rhythm, lively movement, saturated highlights",
    "suspense": "tense atmosphere, dramatic rim light, cinematic fog",
    "spooky": "eerie ambience, volumetric haze, unsettling shadows",
}


def parse_labels_arg(labels: str | None, labels_json: str | None) -> list[str]:
    if labels_json:
        data = json.loads(labels_json)
        if isinstance(data, list):
            raw = [str(x).strip() for x in data if str(x).strip()]
        else:
            raise ValueError("--labels-json must be a JSON array of strings")
    elif labels:
        raw = re.split(r"[,;|]", labels)
        raw = [re.sub(r"\s+", " ", x.strip()) for x in raw if x.strip()]
    else:
        raise ValueError("Provide --labels or --labels-json")

    seen: set[str] = set()
    out: list[str] = []
    for item in raw:
        key = item.lower()
        if key and key not in seen:
            seen.add(key)
            out.append(item)
    return out


def build_prompts(
    labels: Sequence[str],
    *,
    max_labels: int = 8,
    style_anchor: str = "cinematic digital painting, highly detailed, volumetric lighting, rich textures",
    scene_anchor: str = "music-inspired cinematic scene",
) -> tuple[str, str]:
    """
    Build (positive_prompt, negative_prompt) from label strings.

    Labels are treated as mood/genre/atmosphere cues and woven into a
    single coherent image description for diffusion.
    """
    cleaned = [str(l).strip().lower() for l in labels if str(l).strip()]
    filtered: list[str] = []
    for tag in cleaned:
        if any(noisy in tag for noisy in NOISY_LABEL_PATTERNS):
            continue
        filtered.append(tag)
    # Keep order while de-duplicating and trimming.
    trimmed = list(dict.fromkeys(filtered))[:max_labels]
    if not trimmed:
        trimmed = list(dict.fromkeys(cleaned))[:max_labels]
    if not trimmed:
        raise ValueError("At least one non-empty label is required")

    scene_hint = "cinematic urban environment with depth and atmosphere"
    for key, value in STYLE_HINTS.items():
        if any(key in t for t in trimmed):
            scene_hint = value
            break

    mood_phrase = "cinematic mood with balanced color contrast"
    for key, value in MOOD_HINTS.items():
        if any(key in t for t in trimmed):
            mood_phrase = value
            break

    # Keep this short enough to stay coherent for SD while still expressive.
    label_phrase = ", ".join(trimmed[:6])
    positive = (
        f"{scene_anchor}, {scene_hint}. "
        f"Musical cues: {label_phrase}. "
        f"{mood_phrase}. "
        f"{style_anchor}, dramatic composition, foreground-midground-background depth, "
        "sharp focus, high detail, 4k"
    )
    return positive, DEFAULT_NEGATIVE
