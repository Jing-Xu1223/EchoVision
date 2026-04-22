"""Turn semantic music labels into Stable Diffusion prompts."""

from __future__ import annotations

import json
import re
from typing import Sequence


DEFAULT_NEGATIVE = (
    "blurry, low quality, worst quality, jpeg artifacts, watermark, signature, "
    "text, logo, deformed, ugly, duplicate, cropped, out of frame"
)


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
    style_anchor: str = "cinematic digital painting, highly detailed, vibrant colors",
    scene_anchor: str = "abstract atmospheric scene inspired by music",
) -> tuple[str, str]:
    """
    Build (positive_prompt, negative_prompt) from label strings.

    Labels are treated as mood/genre/atmosphere cues and woven into a
    single coherent image description for diffusion.
    """
    trimmed = [str(l).strip() for l in labels if str(l).strip()][:max_labels]
    if not trimmed:
        raise ValueError("At least one non-empty label is required")

    label_phrase = ", ".join(trimmed)
    positive = (
        f"{scene_anchor}. Musical character: {label_phrase}. "
        f"{style_anchor}, professional composition, 4k, sharp focus"
    )
    return positive, DEFAULT_NEGATIVE
