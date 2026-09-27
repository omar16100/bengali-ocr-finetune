# C4 model: bengali-ocr-finetune

Purpose: architecture source of truth for this repository. Status: current. Last updated 27 Sep 2026.

## Context

A single researcher runs Python scripts on an Apple Silicon machine to fine-tune vision-language models for Bengali OCR and to score them. External systems:

- Hugging Face Hub: the `rifathridoy/bengali-ocr-synthetic` dataset and the `mlx-community/gemma-4-e4b-it-4bit` and `mlx-community/PaddleOCR-VL-1.5-4bit` models.
- mlx-vlm (with local patches for Gemma 4 gradients, not in this repo) on top of MLX.
- Tesseract, EasyOCR and PaddleOCR for baselines.
- GitHub Actions for the metrics unit tests.

There is no service, server or deployed component.

## Containers

| Container | Technology | Responsibility |
| --- | --- | --- |
| Training scripts (`train/`) | Python, MLX, mlx-vlm | Load a model, apply LoRA, train, save adapters and loss curves, run an evaluation pass |
| Evaluation (`eval/`) | Python, mlx-vlm | `metrics.py` scoring library; `evaluate_finetuned.py` scores a saved Gemma adapter |
| Baselines (`baselines/`) | Python, pytesseract, EasyOCR, PaddleOCR | Score traditional OCR engines on the test split |
| Local data (`data/`, gitignored) | Parquet via `datasets` | Downloaded dataset |
| Results (`results/`) | JSON files; adapters gitignored | Metrics, sample predictions, training curves |
| CI (`.github/workflows/ci.yml`) | GitHub Actions, uv, pytest | Runs `tests/` on every push to main and every pull request |

## Components

| Component | File | Notes |
| --- | --- | --- |
| Metrics | `eval/metrics.py` | NFC normalization, output cleanup, Levenshtein CER and WER, Bengali grapheme splitter and GER, `evaluate_batch` (mean and corpus scores). Standard library only. |
| Gemma 4 E4B LoRA trainer | `train/finetune_bengali_ocr.py` | Model from `GEMMA_MODEL_PATH` (default `mlx-community/gemma-4-e4b-it-4bit`) |
| Gemma adapter evaluator | `eval/evaluate_finetuned.py` | Uses `mlx_vlm.generate` with `apply_chat_template` so image tokens are inserted |
| Gradient smoke test | `train/smoke_test_gradient.py` | One LoRA step, checks gradients are finite |
| PaddleOCR-VL-1.5 LoRA trainers | `train/finetune_paddleocr_vl.py` (v1), `train/finetune_paddleocr_v2.py` (v2) | Model from `PADDLEOCR_MODEL_PATH` (default `mlx-community/PaddleOCR-VL-1.5-4bit`) |
| Baseline runner | `baselines/run_baselines.py` | Tesseract `ben`, EasyOCR `bn`, PaddleOCR `bn` |
| Unit tests | `tests/test_metrics.py` | 21 tests for `eval/metrics.py` |

## Data flows

1. Dataset: Hugging Face Hub, then `hf download` into `data/bengali-ocr-synthetic`, then `datasets.load_dataset` in each script.
2. Training: model (Hub or local directory) plus train subset (`shuffle(seed=42)`), then LoRA steps, then adapters in `results/adapters/` (gitignored) and loss curves in `results/training_curve*.json`.
3. Model evaluation: 200 test rows after `shuffle(seed=42)`, then generation, then `eval/metrics.py`, then `results/*eval*.json` and `results/*samples.json`.
4. Baselines: first 200 non-empty test rows in dataset order, then each engine, then `eval/metrics.py`, then `results/baselines.json`. This is a different subset from flow 3.
5. CI: checkout, uv, `pytest tests/`. The metrics tests need no model or dataset.

## Change log

| Date | Change |
| --- | --- |
| 27 Sep 2026 | Created. Model paths moved from hardcoded local directories to `GEMMA_MODEL_PATH` / `PADDLEOCR_MODEL_PATH` with Hub ids as defaults. Added CI for the metrics tests. |
