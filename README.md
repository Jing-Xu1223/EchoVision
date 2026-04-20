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

- The same predicted labels are sent to an LLM.
- The LLM returns a short, readable explanation of:
  - the detected musical mood/character, and
  - how those attributes relate to the generated image.

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
