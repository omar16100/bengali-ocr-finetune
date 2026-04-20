# Bengali OCR Fine-tuning — TODO

## Phase 1: Data + Pipeline (in progress)
- [~] Download rifathridoy/bengali-ocr-synthetic (30K) — downloading (~7.8K files)
- [ ] Download BN-HTRd handwritten test set
- [x] Build CER/WER/GER evaluation harness — eval/metrics.py, 16 unit tests passing
- [x] Build baseline runner — baselines/run_baselines.py (Tesseract + EasyOCR)
- [ ] Add PaddleOCR (traditional, lang=bn) baseline — CPU, fast
- [ ] Add Qwen2.5-VL-7B as VLM baseline (via mlx-vlm, strong multilingual OCR)
- [ ] Skip SeaLLion VLM (no Bengali support — SEA languages only)
- [ ] Skip PaddleOCR VLM / PP-DocBee (PaddlePaddle-only, no MPS, Chinese/English)
- [~] Install mlx-vlm from patched clone — in progress
- [~] Codex review of eval harness — submitted
- [ ] Run all baselines on test split
- [ ] Smoke test mlx-vlm gradient fix with SuperGemma4-26B

## Phase 2: Gemma 4 Fine-tuning
- [ ] Fine-tune Gemma 4 E4B on Bengali OCR synthetic (LoRA via mlx-vlm)
- [ ] Evaluate CER/WER/GER on BN-HTRd + BCD3 test sets
- [ ] Compare against all baselines
- [ ] Scale to 31B if E4B promising

## Phase 3: autoresearch Loop
- [ ] Adapt autoresearch-qwen for Gemma 4 + Bengali OCR data
- [ ] Define metric: 1 - CER
- [ ] Run autonomous experiment loop
- [ ] Document all experiments

## Decisions log
- See /Users/macmini/projects/llm-benchmark/docs/19042026_bengali_ocr_finetuning.md
