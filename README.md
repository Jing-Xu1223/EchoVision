# EchoVision: Music-to-Image Generation via Shared Semantic Label Space

EchoVision is a multimodal generation project that translates music into images through an interpretable semantic layer. Instead of using a fully end-to-end black-box model, EchoVision predicts human-readable labels from audio (for example: `melancholic`, `energetic`, `orchestral`, `cinematic`) and uses those labels to guide image synthesis and explanation generation.

## Overview

Music and visual art often communicate similar emotions, moods, and atmospheres. EchoVision explores this connection computationally by building a system that:

1. Listens to a piece of music.
2. Extracts semantic descriptors of its character.
3. Generates a matching image from those descriptors.
4. Produces a natural-language explanation connecting the music and image.

## Why It Matters

EchoVision demonstrates how a **shared semantic label space** can act as a universal translator between different modalities.

- **Interpretable by design**: intermediate labels make model behavior inspectable.
- **Controllable generation**: prompt construction from labels allows style/content steering.
- **Extensible architecture**: each module can be improved independently.
- **Practical potential**: useful for creative tools, music visualization, and accessibility.

## System Pipeline

The project is organized into three core stages:

### 1) Music Understanding Module

- Input: raw audio clip uploaded by a user.
- Preprocessing: audio is converted into a **Mel spectrogram** representation.
- Modeling: a CNN processes the spectrogram and predicts semantic labels that describe the music's mood, genre cues, and aesthetic character.

These labels form the shared semantic bridge between audio and vision.

### 2) Label-to-Image Generation Module

- The predicted labels are converted into a structured natural-language prompt.
- The prompt is passed to a **Stable Diffusion** pipeline.
- The diffusion model generates an image aligned with the inferred musical semantics.

Example mapping:

- Labels: `{upbeat, electronic, futuristic}`
- Possible output style: a vivid neon-lit cityscape.

This prompt-engineering layer provides meaningful control over visual style and content.

### 3) LLM Explanation Module

- The same predicted labels are sent to an LLM (**Gemini by default** in this repo’s implementation).
- The LLM returns a **vivid, readable visual description** grounded in the labels:
  - brief musical mood/energy from the tags, and
  - concrete imagery (light, color, setting, texture) aligned with an imagined or generated image.

This module improves transparency and helps users understand model outputs.

## User Interface

All components are integrated in a simple **Gradio** application.

Users can:

- upload an audio clip,
- generate a corresponding image, and
- read a text explanation describing the music-image relationship.

## Project Goal

EchoVision aims to show that semantic labels can serve as a robust, interpretable intermediary for multimodal translation from sound to vision while preserving creative flexibility.

## Stage 1 Implementation: Music -> Label (CNN + Mel Spectrogram)

This repository now includes a baseline pipeline to build the first module (audio to semantic labels) with your own model.

### 1) Install dependencies

```bash
pip install -r requirements.txt
```

### 2) Prepare MusicCaps data

This step downloads/loads MusicCaps metadata, fetches audio from YouTube, trims clips, computes log-Mel spectrograms, and builds multi-label targets from `aspect_list`.

```bash
python3 src/data/prepare_musiccaps.py \
  --output-dir data/processed/musiccaps \
  --cache-dir data/raw/musiccaps
```

By default, preprocessing downloads the first `100` examples to keep iteration fast.

To run on the full dataset:

```bash
python3 src/data/prepare_musiccaps.py --download-all
```

Optional custom subset:

```bash
python3 src/data/prepare_musiccaps.py --max-items 100
```

### 3) Train the CNN label model

```bash
python3 src/train_music_label_cnn.py \
  --data-dir data/processed/musiccaps \
  --output-dir artifacts/music_label_cnn \
  --epochs 20 \
  --batch-size 32
```

### Outputs

After preprocessing:

- `data/processed/musiccaps/metadata.csv`
- `data/processed/musiccaps/label_vocab.json`
- `data/processed/musiccaps/labels_multihot.npy`
- `data/processed/musiccaps/mel/*.npy`

After training:

- `artifacts/music_label_cnn/best_model.pt`
- `artifacts/music_label_cnn/history.json`
- `artifacts/music_label_cnn/train_summary.json`

## Stage 2: Label → Image (Stable Diffusion + prompt engineering)

Uses a **pretrained** checkpoint (default: `runwayml/stable-diffusion-v1-5`, public on the Hub). Labels are turned into a positive and negative prompt, then an image is saved under `artifacts/generated/`.

**If you see `401` / `Repository Not Found`:** an invalid `HF_TOKEN` in your environment often causes this. By default this project downloads **without** using that token (`token=False`). For gated Stability models, run `huggingface-cli login` and pass `--use-hf-token`, or fix/remove the bad token.

Install generation dependencies (if not already):

```bash
pip install diffusers transformers accelerate safetensors Pillow
```

Run from predicted labels or any label list:

```bash
python3 src/run_label_to_image.py --labels "upbeat,electronic,futuristic"
```

Or JSON labels (e.g. from your own pipeline):

```bash
python3 src/run_label_to_image.py --labels-json '["mellow","piano melody","sad"]'
```

Optional flags: `--output`, `--model-id`, `--steps`, `--guidance`, `--seed`, `--device cuda|mps|cpu`, `--use-hf-token` (for gated / private repos).

Outputs:

- PNG image (default `artifacts/generated/label_image.png`)
- Sidecar JSON with prompt and generation settings (same basename, `.json`)

Programmatic use: `build_prompts(labels)` in `src/generation/prompt_builder.py`, then `generate_image(...)` in `src/generation/generate_image.py`.

## Stage 3: Label → Text explanation (pretrained LLM)

**Default: Google Gemini** (API key from [Google AI Studio](https://aistudio.google.com/apikey)). No fine-tuning—you call an existing model.

Optionally use **`--provider openai`** for an OpenAI-compatible Chat Completions API (OpenAI, Azure, Ollama `v1`, etc.).

### Install

```bash
pip install google-genai
# Optional, for --provider openai only:
pip install openai
```

Use the **`google-genai`** package (current SDK). The older `google-generativeai` package is deprecated and triggers warnings.

### Authentication (Gemini)

Set **one** of:

- `GEMINI_API_KEY` — explicit name for this project  
- `GOOGLE_API_KEY` — same key as in Google AI Studio docs  

Override per run with `--api-key`.

**Default model:** `gemini-2.5-flash`. If your API key has no quota for it, pass e.g. `--model gemini-1.5-flash` or check [rate limits](https://ai.google.dev/gemini-api/docs/rate-limits) / billing in AI Studio.

**429 / quota:** the CLI retries with backoff (`--gemini-max-retries`). If errors persist, wait for the reset window, adjust billing, or switch `--model`.

**Output style:** explanations are tuned for **rich, specific visual prose** (light, color, materials, space, motion)—not a short abstract blurb. Default `--max-tokens` is **2048** so replies rarely cut off mid-sentence; if they still do, raise it (e.g. `4096`) or lower temperature slightly.

### Run from labels (Gemini)

```bash
export GEMINI_API_KEY=your_key_here
python3 src/run_label_to_explanation.py --labels "upbeat,electronic,futuristic"
```

```bash
python3 src/run_label_to_explanation.py --labels-json '["mellow","piano melody","sad"]'
```

### Run from Stage-2 metadata (labels + image prompt)

After `run_label_to_image.py`, reuse the sidecar JSON so the explanation aligns with the diffusion prompt:

```bash
python3 src/run_label_to_explanation.py --from-meta artifacts/generated/label_image.json
```

### OpenAI-compatible provider (optional)

```bash
export OPENAI_API_KEY=sk-...
python3 src/run_label_to_explanation.py --provider openai --labels "upbeat,electronic"
```

Ollama example:

```bash
export OPENAI_BASE_URL=http://127.0.0.1:11434/v1
export OPENAI_API_KEY=ollama
python3 src/run_label_to_explanation.py --provider openai --model llama3.2 --labels-json '["mellow","piano"]'
```

### Useful flags

- `--provider` — `gemini` (default) or `openai`.
- `--model` — override model id (defaults: `gemini-2.5-flash` / `gpt-4o-mini`).
- `--image-prompt` — optional grounding (ignored if `--from-meta` already includes `prompt`).
- `--output` — text file (default `artifacts/explanations/label_explanation.txt`).
- `--json-output` — structured output path (default: same basename as `--output` with `.json`).
- `--api-key` — override env key for the active provider.
- `--base-url` — **OpenAI provider only:** e.g. Ollama endpoint.

### Programmatic use

```python
from src.explanation.llm_explain import explain_from_labels

text = explain_from_labels(
    ["upbeat", "electronic"],
    provider="gemini",
    image_prompt="optional SD prompt...",
)
```

Outputs: a **vivid, scene-forward paragraph** (several sentences) grounded in the labels, emphasizing concrete visual imagery usable for art direction or diffusion-style prompts.

## End-to-end pipeline (one audio → labels → image + explanation)

Script: `src/run_echovision.py` — runs all three stages in order on a **single** audio file (first `--duration-s` seconds, default 10, aligned with MusicCaps-style clips).

**Prerequisites:** trained checkpoint (`artifacts/music_label_cnn/best_model.pt`), processed data directory with `preprocess_config.json` (same Mel settings as training, usually `data/processed/musiccaps`), and API access for Gemini (or use `--skip-explanation`) and Hugging Face for Stable Diffusion weights.

```bash
export GEMINI_API_KEY=...   # or GOOGLE_API_KEY
# Use any real audio path. If you preprocessed MusicCaps, clips live under data-dir/audio/, e.g.:
#   data/processed/musiccaps/audio/00000_-0Gj8-vB1q4.wav
python3 src/run_echovision.py \
  --audio data/processed/musiccaps/audio/00000_-0Gj8-vB1q4.wav \
  --checkpoint artifacts/music_label_cnn/best_model.pt \
  --data-dir data/processed/musiccaps
```

Default output folder: `artifacts/echovision_runs/<audio_stem>_<timestamp>/` containing:

- `generated_image.png` — Stable Diffusion output  
- `image_generation.json` — prompts + SD settings  
- `explanation.txt` / `explanation.json` — LLM text (Gemini by default)  
- `run_manifest.json` — full trace (audio path, label scores, thresholds, paths)

Useful flags:

- `--output-dir` — fixed output directory instead of auto-named run folder  
- `--duration-s` / `--start-s` — which segment of the file to analyze  
- `--threshold` / `--target-frames` — override values stored in the checkpoint  
- `--max-labels` / `--min-top-k` — control how many labels feed SD + LLM  
- `--sd-model-id`, `--sd-steps`, `--sd-seed`, `--use-hf-token` — image stage  
- `--explanation-provider`, `--explanation-model`, `--explanation-max-tokens` (default **2048**; raise to `4096` if text still truncates) — LLM stage  
- `--skip-image` / `--skip-explanation` — run only part of the pipeline (e.g. labels-only debugging)

Mel preprocessing matches `prepare_musiccaps.py` (see `preprocess_config.json` in `--data-dir`).

If MusicCaps `aspect_list` values looked like Python lists, older runs may have odd tokens in `label_vocab.json` (extra quotes/brackets). Re-run `prepare_musiccaps.py` to rebuild vocab, then retrain; `run_echovision.py` also **sanitizes** label strings at inference so prompts stay readable with existing checkpoints.
