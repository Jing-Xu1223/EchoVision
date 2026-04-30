# EchoVision

EchoVision is a multimodal pipeline that maps one audio clip to:

1. semantic music labels (via a CNN on log-Mel spectrograms),
2. a generated image (via Stable Diffusion prompt construction), and
3. a natural-language visual explanation (via an LLM).

The key design choice is an interpretable shared label space between sound and image.

## What Is Implemented

- **Stage 1 (Music -> Labels):** trained CNN classifier from MusicCaps-derived labels.
- **Stage 2 (Labels -> Image):** prompt builder + Stable Diffusion generation.
- **Stage 3 (Labels -> Explanation):** Gemini (default) or OpenAI-compatible text generation.
- **End-to-end CLI:** single command from audio file to all outputs.

## Repository Layout

```text
EchoVision/
├── src/
│   ├── data/
│   │   └── prepare_musiccaps.py
│   ├── models/
│   │   └── music_label_cnn.py
│   ├── generation/
│   │   ├── prompt_builder.py
│   │   └── generate_image.py
│   ├── explanation/
│   │   └── llm_explain.py
│   ├── train_music_label_cnn.py
│   ├── run_label_to_image.py
│   ├── run_label_to_explanation.py
│   └── run_echovision.py
├── data/
│   └── processed/
├── artifacts/
│   ├── music_label_cnn/
│   ├── generated/
│   ├── explanations/
│   └── echovision_runs/
└── requirements.txt
```

## Environment Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick Start (Run Existing Model End-to-End)

Use this if you already have:

- a checkpoint at `artifacts/music_label_cnn/best_model.pt`,
- processed data directory with `preprocess_config.json`,
- one audio file to test.

```bash
export GEMINI_API_KEY=your_key_here

python3 src/run_echovision.py \
  --audio data/processed/musiccaps_full/audio/00000_-0Gj8-vB1q4.wav \
  --checkpoint artifacts/music_label_cnn/best_model.pt \
  --data-dir data/processed/musiccaps_full
```

Output folder (auto-created):

`artifacts/echovision_runs/<audio_stem>_<timestamp>/`

with:

- `generated_image.png`
- `image_generation.json`
- `explanation.txt`
- `explanation.json`
- `run_manifest.json`

## Full Reproducibility Pipeline

### 1) Prepare MusicCaps Data

```bash
python3 src/data/prepare_musiccaps.py \
  --output-dir data/processed/musiccaps \
  --cache-dir data/raw/musiccaps
```

Useful options:

- `--download-all`
- `--max-items 100`
- `--start-index 640`
- `--cookies-file /path/to/cookies.txt`

### 2) Train Label CNN

```bash
python3 src/train_music_label_cnn.py \
  --data-dir data/processed/musiccaps \
  --output-dir artifacts/music_label_cnn \
  --epochs 20 \
  --batch-size 32
```

### 3) Labels -> Image (Standalone)

```bash
python3 src/run_label_to_image.py --labels "upbeat,electronic,futuristic"
```

or

```bash
python3 src/run_label_to_image.py --labels-json '["mellow","piano melody","sad"]'
```

### 4) Labels -> Explanation (Standalone)

```bash
python3 src/run_label_to_explanation.py --labels "upbeat,electronic,futuristic"
```

or from Stage-2 metadata:

```bash
python3 src/run_label_to_explanation.py --from-meta artifacts/generated/label_image.json
```

## API Keys and External Services

- **Gemini (default explanation backend):**
  - set `GEMINI_API_KEY` (or `GOOGLE_API_KEY`)
- **OpenAI-compatible backend (optional):**
  - set `OPENAI_API_KEY`
  - optional `OPENAI_BASE_URL` for local providers (e.g., Ollama)
- **Stable Diffusion weights:**
  - default model is public (`runwayml/stable-diffusion-v1-5`)
  - for gated/private repos, use `huggingface-cli login` and `--use-hf-token`

## Common Flags (End-to-End Script)

`src/run_echovision.py`:

- `--output-dir` fixed output path
- `--duration-s`, `--start-s` choose clip segment
- `--threshold`, `--max-labels`, `--min-top-k` label selection
- `--sd-model-id`, `--sd-steps`, `--sd-guidance`, `--sd-seed` image generation control
- `--skip-image`, `--skip-explanation` partial pipeline runs
- `--explanation-provider`, `--explanation-model`, `--explanation-max-tokens`

## Notes on Current Workspace Data Paths

Your current checked-in sample data is under:

- `data/processed/musiccaps_full/`

while script defaults may still point to:

- `data/processed/musiccaps/`

If a command fails due to missing files, pass `--data-dir data/processed/musiccaps_full` (or regenerate into `musiccaps`).

## License

This project is released under the MIT License. See `LICENSE`.
