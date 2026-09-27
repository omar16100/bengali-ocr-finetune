# bengali-ocr-finetune

Experiments in LoRA fine-tuning of vision-language models for Bengali OCR with [MLX](https://github.com/ml-explore/mlx) and [mlx-vlm](https://github.com/Blaizzy/mlx-vlm), plus a Bengali CER / WER / grapheme error rate harness and Tesseract and EasyOCR baselines.

Status: research log. No trained adapters are committed. In the committed results, every fine-tuned run has a higher character error rate than the EasyOCR baseline, although the baseline used a different test subset (see [Dataset](#dataset) and [Results](#results)). The reasoning behind each step is in [docs/decision_log.md](docs/decision_log.md).

## Models

| Model | Hugging Face id | Scripts |
| --- | --- | --- |
| Gemma 4 E4B, 4-bit MLX | [`mlx-community/gemma-4-e4b-it-4bit`](https://huggingface.co/mlx-community/gemma-4-e4b-it-4bit) | `train/finetune_bengali_ocr.py`, `eval/evaluate_finetuned.py`, `train/smoke_test_gradient.py` |
| PaddleOCR-VL-1.5, 4-bit MLX | [`mlx-community/PaddleOCR-VL-1.5-4bit`](https://huggingface.co/mlx-community/PaddleOCR-VL-1.5-4bit) | `train/finetune_paddleocr_vl.py` (v1), `train/finetune_paddleocr_v2.py` (v2) |

Both ids are passed to `mlx_vlm.load`, which accepts a Hugging Face repo id or a local directory. Override them with the `GEMMA_MODEL_PATH` and `PADDLEOCR_MODEL_PATH` environment variables.

## Dataset

[`rifathridoy/bengali-ocr-synthetic`](https://huggingface.co/datasets/rifathridoy/bengali-ocr-synthetic): `image` and `text` columns, 27,007 train and 2,993 test rows, CC BY 4.0 (from its dataset card, as of 27 Sep 2026). The scripts load it with `datasets.load_dataset("data/bengali-ocr-synthetic")`; `data/` is gitignored.

Every evaluation uses 200 test rows, but not the same 200:

- `baselines/run_baselines.py` takes the first 200 test rows with non-empty text, in dataset order.
- The model evaluations take `test.shuffle(seed=42).select(range(200))`. The 20 reference strings saved in each `results/*samples.json` file match this order.

The two subsets share only 19 distinct reference texts (checked 27 Sep 2026), so baseline and model numbers are not a like-for-like comparison.

## Training setup

Values are read from the scripts. All runs use Adam, one sample per step, and a training subset drawn with `shuffle(seed=42)`. Steps walk that subset in order, so a run of N steps sees its first N samples.

| Run | Script | LoRA targets | Rank | `alpha` passed to `get_peft_model` | Learning rate | Steps | Train subset | Loss | Prompt |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Gemma 4 E4B | `train/finetune_bengali_ocr.py` | all linear layers (`find_all_linear_names`) | 16 | 2.0 (32 / 16) | 1e-4 | 500 | 5,000 | all tokens | `এই ছবি থেকে বাংলা টেক্সট পড়ুন।` |
| PaddleOCR-VL v1 | `train/finetune_paddleocr_vl.py` | all linear layers (`find_all_linear_names`) | 8 | 1.0 | 5e-5 | 1,000 | 10,000 | all tokens | `OCR:` |
| PaddleOCR-VL v2 | `train/finetune_paddleocr_v2.py` | `q_proj`, `k_proj`, `v_proj`, `o_proj` | 4 | 8.0 | 2e-5 | 1,000 | 10,000 | target tokens only | `OCR:` |

Training loss, computed from the committed curves (one value per step):

| Source file | Steps | First loss | Final loss | Min loss | Mean of last 10 |
| --- | --- | --- | --- | --- | --- |
| `results/training_curve.json` (Gemma 4 E4B) | 500 | 25.625 | 0.046 | 0.024 | 0.083 |
| `results/training_curve_paddleocr.json` (PaddleOCR-VL v1) | 1000 | 14.188 | 7.031 | 6.781 | 7.062 |
| `results/training_curve_paddleocr_v2.json` (PaddleOCR-VL v2) | 1000 | 6.000 | 0.075 | 0.000 | 1.044 |

## Results

Metrics come from `eval/metrics.py`. Text is NFC-normalized first. CER and WER are edit distance divided by reference length (characters or words); GER does the same over Bengali grapheme clusters. "corpus" divides total edits by total reference length; "mean" averages per-sample scores. Values above 1.0 occur when predictions are much longer than the references, for example repetition loops.

Every number below is the named JSON key rounded to 3 decimals. Lower is better.

| Run | Source file | Written by | n | CER corpus | CER mean | WER corpus | WER mean | GER mean | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EasyOCR (`bn`) | `results/baselines.json` (`easyocr`) | `baselines/run_baselines.py` | 200 | 0.153 | 0.182 | 0.446 | 0.474 | 0.258 | baseline subset |
| Tesseract (`ben`) | `results/baselines.json` (`tesseract`) | `baselines/run_baselines.py` | 200 | 0.722 | 0.769 | 0.907 | 0.949 | 0.864 | baseline subset |
| PaddleOCR (traditional, `bn`) | `results/baselines.json` (`paddleocr`) | `baselines/run_baselines.py` | | | | | | | failed: "No models are available for the language 'bn' and OCR version None." |
| PaddleOCR-VL-1.5, zero-shot | `results/paddleocr_zeroshot_eval.json` | not in this repo | 200 | 0.668 | 0.734 | 0.756 | 0.813 | 1.209 | the script that wrote this file is not committed |
| PaddleOCR-VL-1.5, LoRA v1 | `results/paddleocr_finetuned_eval.json` | `train/finetune_paddleocr_vl.py` | 200 | 9.548 | 13.040 | 2.608 | 2.907 | 11.067 | output collapsed into repeated text (Decision 16) |
| PaddleOCR-VL-1.5, LoRA v2 | `results/paddleocr_finetuned_v2_eval.json` | `train/finetune_paddleocr_v2.py` | 200 | 3.220 | 4.939 | 2.159 | 2.535 | 4.959 | some samples near-exact, others loop (Decision 17) |
| Gemma 4 E4B, LoRA, in-script decode | `results/finetuned_eval.json` | `train/finetune_bengali_ocr.py` | 200 | 0.917 | 0.925 | 1.000 | 1.000 | 0.991 | invalid: decoded token by token without a KV cache (Decision 12); Decision 13 marks the Gemma results invalid |
| Gemma 4 E4B, LoRA | `results/finetuned_eval_v2.json` | `eval/evaluate_finetuned.py` | 200 | 13.697 | 18.625 | 1.000 | 1.000 | 32.991 | invalid: committed in `68e02dd`, before the image-token fix in `b9f0d8b` (Decisions 13, 14) |

Decision 14 in the decision log also mentions a 10-sample Gemma check after the image-token fix. Its output was not saved under `results/`, so it is not in this table.

The first 20 reference and prediction pairs are saved for four of the model evaluations: `results/paddleocr_zeroshot_samples.json` (zero-shot), `results/paddleocr_eval_samples.json` (v1), `results/paddleocr_v2_eval_samples.json` (v2) and `results/eval_samples.json` (Gemma, `eval/evaluate_finetuned.py`).

## How to run

Run everything from the repository root: the scripts use relative paths (`data/`, `results/`) and write `.log` files into their own directories.

There is no requirements or lock file. The scripts import:

- training and evaluation: `mlx`, `mlx-vlm`, `datasets`, `safetensors`, `Pillow`
- baselines: `pytesseract` (plus Tesseract with Bengali `ben` data), `easyocr`, `numpy`, `paddleocr`

The Gemma 4 runs depended on three local mlx-vlm patches for NaN gradients, described in [docs/decision_log.md](docs/decision_log.md) (Decision 3) and [docs/mlx_vlm_gemma4_fix_journey.md](docs/mlx_vlm_gemma4_fix_journey.md). The patched mlx-vlm is not part of this repository.

```bash
# 1. Metrics unit tests (standard library only, no model or dataset needed)
uv run --no-project --with pytest pytest tests/

# 2. Dataset into data/bengali-ocr-synthetic (the two parquet files total 274,424,389 bytes
#    per the Hugging Face file listing, as of 27 Sep 2026)
hf download rifathridoy/bengali-ocr-synthetic README.md \
  data/train-00000-of-00001.parquet data/test-00000-of-00001.parquet \
  --repo-type dataset --local-dir data/bengali-ocr-synthetic

# 3. Baselines -> results/baselines.json
python baselines/run_baselines.py

# 4. PaddleOCR-VL v2 LoRA -> results/paddleocr_finetuned_v2_eval.json, adapters in results/adapters/
python train/finetune_paddleocr_v2.py

# 5. Gemma 4 E4B LoRA, then evaluate the saved adapter -> results/finetuned_eval_v2.json
python train/finetune_bengali_ocr.py
python eval/evaluate_finetuned.py
```

As of 27 Sep 2026, steps 1 and 2 were checked (21 tests pass; the dataset loads with the expected split sizes). The training, evaluation and baseline scripts were not re-run for this README.

## Repository layout

| Path | Contents |
| --- | --- |
| `eval/metrics.py` | CER, WER, grapheme error rate, Bengali normalization |
| `eval/evaluate_finetuned.py` | evaluates a saved Gemma 4 E4B LoRA adapter |
| `baselines/run_baselines.py` | Tesseract, EasyOCR and PaddleOCR baselines |
| `train/` | LoRA fine-tuning scripts and a gradient smoke test |
| `tests/test_metrics.py` | unit tests for `eval/metrics.py` |
| `results/` | metrics, sample predictions and training curves (JSON) |
| `docs/` | decision log, mlx-vlm fix notes, [docs index](docs/index.md), [architecture](docs/c4model.md) |

## License

MIT, see [LICENSE](LICENSE). The dataset and the models are under their own licenses.
