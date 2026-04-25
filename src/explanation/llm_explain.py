"""
Use a pretrained LLM to explain how semantic music labels relate to mood and
to an imagined/generated image.

Default provider: **Google Gemini** via the current **`google-genai`** SDK
(``GEMINI_API_KEY`` or ``GOOGLE_API_KEY``). Optional provider: OpenAI-compatible
Chat Completions.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Literal, Sequence

Provider = Literal["gemini", "openai"]

DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"

ECHO_SYSTEM_PROMPT = """You are the EchoVision caption writer: you turn shared semantic music labels into a single, highly visual piece of prose that could inspire or match an image.

Your output must feel like a vivid scene description or art-direction brief—not a generic summary.

Requirements:
- Write **4 to 7 sentences** in flowing prose (no bullet lists, no label-by-label checklist).
- **Prioritize concrete visuals**: palette (specific colors), light source and quality (neon rim-light, haze, chrome reflections), architecture or landscape, materials (glass, wet asphalt, brushed metal), weather or atmosphere, depth (foreground vs skyline), implied motion (traffic streaks, drifting fog, crowd energy).
- **Anchor mood from the labels** (tempo, energy, genre cues) in **one or two short phrases**, then spend most of the text on **imagery** that a painter or cinematographer could use.
- Be **specific and imaginative** while staying **loyal to the labels**—extend them with plausible visual detail, but do not invent wholly unrelated genres or instruments the labels do not suggest.
- Do **not** say you listened to audio or saw an image file; phrase as "this could read as…" or "the scene might open on…" if needed.
- Avoid cliché padding ("vibrant and dynamic"); prefer **one sharp image** over vague adjectives.
- End on a strong visual beat (a single memorable image or contrast), not mid-thought."""


def build_user_message(
    labels: Sequence[str],
    *,
    image_prompt: str | None = None,
    extra_context: str | None = None,
) -> str:
    lines = [
        "Here are the semantic labels extracted from the music (shared label space between audio and image):",
        ", ".join(labels) + ".",
    ]
    if image_prompt:
        lines.append("")
        lines.append("For context, the following text was used to guide image generation (Stable Diffusion prompt):")
        lines.append(image_prompt.strip())
    if extra_context:
        lines.append("")
        lines.append("Additional context from the pipeline:")
        lines.append(extra_context.strip())
    lines.append("")
    lines.append(
        "Task: produce one continuous, highly visual description (as instructed in your system prompt). "
        "Paint a specific scene a reader could almost see—light, color, space, texture, and motion—not a dry commentary."
    )
    return "\n".join(lines)


def resolve_gemini_api_key(api_key: str | None) -> str:
    if api_key:
        return api_key
    for env_name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        value = os.environ.get(env_name)
        if value:
            return value
    raise ValueError(
        "Missing Gemini API key. Set GEMINI_API_KEY or GOOGLE_API_KEY "
        "(from Google AI Studio), or pass api_key=..."
    )


def _is_retryable_quota_error(exc: BaseException) -> bool:
    name = type(exc).__name__
    msg = str(exc).lower()
    if "resourceexhausted" in name.lower() or "resource_exhausted" in msg:
        return True
    if "429" in msg or "quota" in msg or "rate limit" in msg:
        return True
    if "too many requests" in msg:
        return True
    return False


def _explain_gemini_once(
    user_content: str,
    *,
    model: str,
    max_tokens: int,
    temperature: float,
    api_key: str | None,
) -> str:
    try:
        from google import genai
        from google.genai import types
    except ImportError as e:
        raise ImportError(
            "Install Google's Gemini SDK: pip install google-genai "
            "(the legacy `google-generativeai` package is deprecated)."
        ) from e

    key = resolve_gemini_api_key(api_key)
    client = genai.Client(api_key=key)

    try:
        config = types.GenerateContentConfig(
            system_instruction=ECHO_SYSTEM_PROMPT,
            max_output_tokens=max_tokens,
            temperature=temperature,
        )
        contents: str = user_content
    except TypeError:
        config = types.GenerateContentConfig(
            max_output_tokens=max_tokens,
            temperature=temperature,
        )
        contents = f"{ECHO_SYSTEM_PROMPT}\n\n{user_content}"

    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=config,
    )

    text = (getattr(response, "text", None) or "").strip()
    if text:
        return text

    parts_out: list[str] = []
    for part in getattr(response, "parts", []) or []:
        t = getattr(part, "text", None)
        if t:
            parts_out.append(t)
    merged = "\n".join(parts_out).strip()
    if merged:
        return merged

    raise RuntimeError("Gemini returned no text in the response.")


def _explain_gemini(
    user_content: str,
    *,
    model: str,
    max_tokens: int,
    temperature: float,
    api_key: str | None,
    max_retries: int = 3,
) -> str:
    for attempt in range(max_retries + 1):
        try:
            return _explain_gemini_once(
                user_content,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                api_key=api_key,
            )
        except BaseException as exc:
            if attempt < max_retries and _is_retryable_quota_error(exc):
                wait_s = min(90.0, 8.0 + 24.0 * attempt)
                time.sleep(wait_s)
                continue
            raise


def _explain_openai(
    user_content: str,
    *,
    model: str,
    max_tokens: int,
    temperature: float,
    api_key: str | None,
    base_url: str | None,
) -> str:
    try:
        from openai import OpenAI
    except ImportError as e:
        raise ImportError("Install the OpenAI SDK: pip install openai") from e

    key = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY")
    url = base_url if base_url is not None else os.environ.get("OPENAI_BASE_URL")

    if not key:
        raise ValueError(
            "Missing API key. Set OPENAI_API_KEY, or pass api_key=... "
            "(for Ollama: OPENAI_BASE_URL=http://127.0.0.1:11434/v1 and OPENAI_API_KEY=ollama)."
        )

    client_kwargs: dict[str, Any] = {"api_key": key}
    if url:
        client_kwargs["base_url"] = url

    client = OpenAI(**client_kwargs)
    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": ECHO_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
    )

    choice = response.choices[0]
    text = (choice.message.content or "").strip()
    if not text:
        raise RuntimeError("LLM returned an empty response.")
    return text


def explain_from_labels(
    labels: Sequence[str],
    *,
    provider: Provider = "gemini",
    image_prompt: str | None = None,
    extra_context: str | None = None,
    model: str | None = None,
    max_tokens: int = 2048,
    temperature: float = 0.82,
    api_key: str | None = None,
    base_url: str | None = None,
    gemini_max_retries: int = 3,
) -> str:
    """
    Call a pretrained LLM and return the explanation text.

    **Gemini (default):** set ``GEMINI_API_KEY`` or ``GOOGLE_API_KEY``.
    Default model: ``gemini-2.5-flash`` (override with ``model=`` if needed).

    If the explanation **cuts off mid-sentence**, raise ``max_tokens`` (output
    token budget); vivid multi-sentence prose needs headroom beyond ~500 tokens.

    **OpenAI-compatible:** ``provider="openai"``; set ``OPENAI_API_KEY`` and
    optionally ``OPENAI_BASE_URL``.
    """
    user_content = build_user_message(
        labels,
        image_prompt=image_prompt,
        extra_context=extra_context,
    )

    if provider == "gemini":
        resolved_model = model or DEFAULT_GEMINI_MODEL
        return _explain_gemini(
            user_content,
            model=resolved_model,
            max_tokens=max_tokens,
            temperature=temperature,
            api_key=api_key,
            max_retries=gemini_max_retries,
        )

    if provider == "openai":
        resolved_model = model or DEFAULT_OPENAI_MODEL
        return _explain_openai(
            user_content,
            model=resolved_model,
            max_tokens=max_tokens,
            temperature=temperature,
            api_key=api_key,
            base_url=base_url,
        )

    raise ValueError(f"Unknown provider: {provider!r}. Use 'gemini' or 'openai'.")


def load_labels_from_meta_json(path: Path) -> tuple[list[str], str | None]:
    """Load labels (and optional image prompt) from Stage-2 metadata JSON."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Meta JSON must be an object.")
    raw = data.get("labels")
    if raw is None:
        raise ValueError("Meta JSON must contain a 'labels' field (array of strings).")
    if isinstance(raw, str):
        labels = [x.strip() for x in raw.split(",") if x.strip()]
    elif isinstance(raw, list):
        labels = [str(x).strip() for x in raw if str(x).strip()]
    else:
        raise ValueError("'labels' must be a list or comma-separated string.")
    prompt = data.get("prompt")
    image_prompt = str(prompt).strip() if prompt else None
    return labels, image_prompt
