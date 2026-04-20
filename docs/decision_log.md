# Bengali OCR Fine-tuning — Experiment Plan & Decision Log

## Context
Goal: Fine-tune Gemma 4 on Bengali OCR data, evaluate against open-source community baselines, and document the process. Started 2026-04-19.

## Decision Log

### Decision 1: autoresearch is not suitable for this project (2026-04-19)
- **What**: User asked to use https://github.com/karpathy/autoresearch for research.
- **Finding**: autoresearch is an autonomous *pretraining* experimentation framework (iterates `train.py` on a GPT-style LM with NVIDIA GPU + Flash Attention 3). It does NOT do literature review, fine-tuning, vision/OCR, or experiment planning.
- **Alternative found**: [wadeKeith/autoresearch-qwen](https://github.com/wadeKeith/autoresearch-qwen) — a vision-capable fork that fine-tunes Qwen3-VL-4B on DocVQA. Has `mlx` branch for Apple Silicon. Directly adaptable for Bengali OCR with Gemma 4.
- **Also found**: [trevin-creator/autoresearch-mlx](https://github.com/trevin-creator/autoresearch-mlx) — 1459-star MLX port, but text-only pretraining.
- **Decision**: Clone both forks. Use `autoresearch-qwen` as the vision experiment framework, adapted for Gemma 4 + Bengali data.

### Decision 2: Gemma 4 vision fine-tuning has critical blockers (2026-04-19)
- **mlx-vlm** ([Blaizzy/mlx-vlm](https://github.com/Blaizzy/mlx-vlm)): Best path for Apple Silicon but has NaN gradient bug in Gemma 4 training ([PR #969](https://github.com/Blaizzy/mlx-vlm/pull/969)). Three sub-bugs identified:
  1. `-inf` in attention mask overflows in softmax backward → NaN
  2. `.item()` calls detach from computation graph
  3. `@mx.compile` on `ensure_fused_sdpa` blocks backward pass
- **Unsloth**: CUDA-only for training. MPS support is inference-only. Vision LoRA has zero-gradient bug (#5039).
- **HF Transformers + PEFT**: MPS backend unreliable for 31B-scale. No bitsandbytes on MPS.
- **TRL**: CUDA-targeted, OOM even on 8×H200.
- **Decision**: Fix mlx-vlm ourselves (all three bugs), then use it for Gemma 4 fine-tuning on M3 Ultra.

### Decision 3: Fix mlx-vlm NaN gradient bugs (2026-04-19)
Applied three patches to `/Users/macmini/projects/mlx-vlm/`:

**Fix 1** — `mlx_vlm/models/gemma4/vision.py:509`
```python
# Before: neg_inf = mx.array(float("-inf"), dtype=inputs_embeds.dtype)
# After:  neg_inf = mx.array(-1e4, dtype=inputs_embeds.dtype)
```
Rationale: -inf causes 0 × -inf = NaN in softmax backward. -1e4 is large enough to zero softmax probs but stays finite in float16 (max ~65504).

**Fix 2** — `mlx_vlm/models/gemma4/vision.py:526-531`
```python
# Before: .item() loop with per-sample concatenation
# After:  mask multiplication + fixed-length slice
valid_mask_expanded = mx.expand_dims(valid_mask, -1).astype(pooled.dtype)
hidden_states = (pooled * valid_mask_expanded)[:, :self.default_output_length, :]
```
Rationale: .item() calls mx.eval() (illegal in mx.compile) and detaches from gradient graph.

**Fix 3** — `mlx_vlm/models/base.py:386`
```python
# Before: @mx.compile decorator on ensure_fused_sdpa
# After:  removed (comment explains why)
```
Rationale: @mx.compile caches forward-only trace; backward pass info lost → NaN gradients.

**Status**: Patches applied. Sent to codex (gpt-5.4) for critique. Awaiting review output at `/Users/macmini/projects/codex/mlxvlm_nan_fix_review_19apr2026.txt`.

**Codex review returned** (`/Users/macmini/projects/codex/mlxvlm_nan_fix_review_19apr2026.txt`):

- **Fix 1 (-1e4)**: Approved. -1e4 is correct for float16 safety.
- **Fix 2 (mask multiply)**: **Rejected initial approach.** Codex found that `pooled * valid_mask_expanded[:, :default_output_length]` is NOT semantically equivalent for variable-length batches. Updated to boolean indexing: `hidden_states = pooled[valid_mask][None]` — preserves row-major packing across batches without `.item()`.
- **Fix 3 (@mx.compile)**: Approved as short-term fix, but **updated comment** — MLX docs say compile() should work with training, so the issue is likely version/interaction-specific, not fundamental. Should benchmark prefill throughput for inference regression.
- **logit_softcap**: Do NOT remove @mx.compile preemptively. Add a finite-gradient test first; only change if test fails.
- **Tests recommended**: tiny-config gradient test (batch_size=2, different aspect ratios), packing unit test, smoke step on SuperGemma4-26B with `--train-vision`.

### Decision 7: Codex OCR eval review applied (2026-04-20)

Codex (gpt-5.4) reviewed `/Users/macmini/projects/bengali-ocr-finetune/eval/metrics.py` and found 3 HIGH-priority bugs:

1. **Grapheme splitter missed ঁ/ং/ঃ (candrabindu, anusvara, visarga)** — these combine with the preceding orthographic unit. Also missed ZWJ/ZWNJ handling. Fixed: added all three + joiners to `is_combining` check.

2. **No Unicode NFC normalization** — canonically equivalent Bengali spellings (decomposed nukta forms, decomposed vowel signs) scored as errors. Fixed: added `normalize_bengali()` applying NFC + whitespace collapse before CER/GER.

3. **Data loader schema wrong** — dataset uses `image`/`text` top-level parquet fields, not the conversation format coded. Loader needs rewrite before baselines run. (Pending fix.)

Additional improvements applied:
- Added `strip_model_formatting()` — strips VLM output artifacts (code fences, backticks, common prefixes)
- Added corpus-level CER/WER alongside per-sample macro average (corpus-level more stable for mixed-length data)
- Tests expanded: 16 → 21 passing (triple conjuncts, candrabindu, visarga, NFC, formatting strip)

Full codex output: `/Users/macmini/projects/codex/bengali_ocr_eval_review_20apr2026.txt`

### Decision 8: Baseline engine selection (2026-04-20)

| Engine | Status | Bengali support | Notes |
|---|---|---|---|
| Tesseract | ✅ added | `lang=ben` + `script/Bengali` installed | CPU, standard baseline |
| EasyOCR | ✅ added | `bn` language pack | CPU, GPU optional |
| PaddleOCR (traditional) | ✅ added | `lang=bengali` | CPU, fast, good for Indian scripts |
| PaddleOCR VLM (PP-DocBee) | ❌ skipped | Chinese/English only | PaddlePaddle-only, no MPS |
| SeaLLion VLM | ❌ skipped | No Bengali (SEA languages only) | Not relevant |
| Qwen2.5-VL-7B | 📋 planned | Strong multilingual OCR | VLM baseline via mlx-vlm |

### Decision 9: Baseline results established (2026-04-20)

| Engine | CER (macro) | CER (corpus) | WER | GER | Samples/sec |
|---|---|---|---|---|---|
| **EasyOCR** | 0.182 | **0.153** | 0.474 | 0.258 | 16.5 |
| Tesseract | 0.770 | 0.722 | 0.949 | 0.864 | 48.3 |
| PaddleOCR | failed | — | — | — | — |

- **EasyOCR is the baseline to beat**: 15.3% corpus CER on 200 test samples from `rifathridoy/bengali-ocr-synthetic`.
- **Tesseract is catastrophic** on this synthetic data (72% CER) — font diversity in the dataset overwhelms its models.
- **PaddleOCR**: Bengali lang code neither `"bengali"` nor `"bn"` works in current PaddleOCR version. Dropped.
- Results at `/Users/macmini/projects/bengali-ocr-finetune/results/baselines.json`.

### Decision 10: SuperGemma4-26B-MLX-4bit lacks vision tower (2026-04-20)

SuperGemma4-26B-MLX-4bit-v2 is a **text-only** quantization — the MLX conversion stripped 211 vision_tower parameters. Cannot be used for vision fine-tuning.

**Pivot**: downloading `mlx-community/gemma-4-e4b-it-4bit` (Gemma 4 E4B, ~3 GB, full multimodal with vision tower) as the fine-tuning base. E4B has 4.5B effective params — much smaller and faster to iterate than 26B/31B while still being Gemma 4 architecture (validates our mlx-vlm fixes).

### Decision 10b: Fix 2 revised — boolean indexing not supported in MLX 0.31 (2026-04-20)

Codex suggested `pooled[valid_mask][None]` (boolean indexing) to replace the `.item()` loop. MLX 0.31.x raises `ValueError: boolean indices are not yet supported`. Reverted to mask multiplication:
```python
valid_mask_expanded = mx.expand_dims(valid_mask, -1).astype(pooled.dtype)
hidden_states = pooled * valid_mask_expanded
```
This zeros padding tokens while preserving the autograd graph. Padding zeros don't contribute to downstream loss. Less precise than per-sample concatenation (original code) but gradient-safe.

### Decision 10c: SuperGemma4-26B lacks vision weights — pivoted to E4B (2026-04-20)

Gemma 4 E4B (4.5B params, mlx-community/gemma-4-e4b-it-4bit, 4.9 GB) downloaded. Has full vision tower (211 params present vs missing in SuperGemma4). Model loads OK, LoRA applies (16 linear layers, 10M trainable / 7.5B total = 0.14%).

### Decision 10d: LoRA API fix (2026-04-20)

mlx-vlm's `get_peft_model` signature is `(model, linear_layers, rank, alpha, dropout)` — does NOT accept `target_modules` kwarg. Fixed in smoke test and fine-tuning script.

### Decision 11: Fine-tuning script created (2026-04-20)

`train/finetune_bengali_ocr.py` — LoRA fine-tuning of Gemma 4 on Bengali OCR:
- Config: rank=16, alpha=32, lr=1e-4, batch_size=1, max_steps=500
- Data: 5K train subsample (of 27K) for fast iteration
- Eval: CER/WER on 200 test samples
- Prompt: "এই ছবি থেকে বাংলা টেক্সট পড়ুন।" (Read Bengali text from this image)
- Saves adapters every 50 steps + training curve
- Submitted to codex for review.

### Decision 4: Bengali OCR datasets selected (2026-04-19)
| Dataset | Size | Type | Use |
|---|---|---|---|
| [rifathridoy/bengali-ocr-synthetic](https://huggingface.co/datasets/rifathridoy/bengali-ocr-synthetic) | 30K samples | Synthetic printed | Training (primary) |
| [BN-HTRd](https://data.mendeley.com/datasets/743k6dm543/1) | 788 pages / 108K words | Handwritten | Evaluation |
| [Bengali.AI Grapheme](https://www.kaggle.com/competitions/bengaliai-cv19) | 200K images | Grapheme components | Potential augmentation |
| [BCD3](https://bengaliai.github.io/bbocr) | 88.5K words | Printed multi-domain | Evaluation |

### Decision 5: Evaluation metrics (2026-04-19)
- **Primary**: CER (Character Error Rate) — most relevant for Bengali with complex conjunct characters
- **Secondary**: WER (Word Error Rate) — word-level accuracy
- **Tertiary**: Grapheme-level error rate (using BnGraphemizer tokenization per GraDeT-HTR methodology)
- **Not using**: BLEU (not standard for OCR)

### Decision 6: Baselines to compare against (2026-04-19)
| Baseline | CER | Source |
|---|---|---|
| bbOCR (Bengali.AI, APSIS-Net + YOLOv8) | ~0.19 improvement margin vs Tesseract | [arxiv 2308.10647](https://arxiv.org/abs/2308.10647) |
| GraDeT-HTR (GPT-2 decoder, 87M params) | SOTA Bengali handwritten | [arxiv 2509.18081](https://arxiv.org/abs/2509.18081) |
| Tesseract (Bengali) | baseline | — |
| EasyOCR (Bengali) | beats Tesseract on diverse images | [IEEE study](https://ieeexplore.ieee.org/document/10969286/) |
| swapnillo/Bangla-OCR-SFT (Qwen3-VL fine-tune) | no published CER | [HuggingFace](https://huggingface.co/swapnillo/Bangla-OCR-SFT) |

## Experiment Plan (Phase 1: Data + Pipeline)

1. Download `rifathridoy/bengali-ocr-synthetic` (30K)
2. Build CER/WER evaluation harness
3. Run baselines (Tesseract, EasyOCR) on a test split
4. Validate data pipeline with Qwen3-VL on mlx-vlm (works TODAY)

## Experiment Plan (Phase 2: Gemma 4 Fine-tuning)

1. Verify mlx-vlm NaN fix with Gemma 4 E4B (smaller, faster iteration)
2. LoRA fine-tune Gemma 4 E4B on Bengali OCR synthetic data
3. Evaluate CER/WER on BN-HTRd + BCD3 test sets
4. Scale to Gemma 4 31B if E4B results are promising
5. Compare against all baselines

## Experiment Plan (Phase 3: autoresearch Loop)

1. Adapt `autoresearch-qwen` to use Gemma 4 + Bengali OCR data
2. Define metric: 1 - CER (higher is better) on a held-out validation split
3. Let the autoresearch agent iterate on training hyperparameters autonomously
4. Document all experiments in `results.tsv`

## Repository structure (planned)
```
/Users/macmini/projects/bengali-ocr-finetune/
├── data/                  # downloaded datasets
├── eval/                  # CER/WER harness
├── baselines/             # Tesseract, EasyOCR scripts
├── train/                 # fine-tuning scripts (mlx-vlm LoRA)
├── autoresearch/          # adapted autoresearch-qwen loop
├── results/               # per-experiment outputs
├── docs/                  # this file + future findings
└── todo.md
```

### Decision 12: Fine-tuning v5 completed — training works, eval broken (2026-04-20)

**Training**: 500 steps in 337s (1.48 steps/s). Loss: 25.6 → 0.046. Excellent convergence.

**Save fix journey** (4 attempts):
1. `mx.savez` — nanobind >1024 kwargs limit
2. `numpy.savez` — bfloat16 not supported in numpy
3. `safetensors.mlx.save_file` — internally calls np.asarray, same bf16 issue
4. **`save_file({k: v.astype(mx.float16)})` — cast bf16→f16 first** ✅

11 adapter checkpoints saved (every 50 steps + final) as safetensors.

**Evaluation**: CER=0.917 (92% error — WORSE than Tesseract 72%). Model produces garbage at inference.

**Root cause hypothesis**: manual greedy decode (`model(input_ids) → argmax → append → repeat`) doesn't maintain KV cache. Each token prediction lacks context of previous tokens. Should use `mlx_vlm.generate()` or implement proper cached generation.

**Next steps**:
1. Fix evaluation to use `mlx_vlm.generate()` (which handles KV cache, sampling, etc.)
2. If that doesn't help, check: LoRA adapter loading, chat template, prompt format
3. Baselines comparison: EasyOCR 15.3% CER is the target

### Decision 13: Vision pipeline broken on Gemma 4 E4B via mlx-vlm (2026-04-20)

**Discovery**: The base model (no LoRA) cannot do ANY OCR — English or Bengali. It echoes the prompt verbatim instead of reading images. This means:
1. The model is not receiving image information
2. LoRA training "converged" (loss 25.6 → 0.046) on text-only patterns, never seeing images
3. The `।।।।` output was collapsed LoRA, the "CER 13.7" was prompt repetition

**Test results**:
- "Read the text in this image." + English text image → outputs "Read the text in this image." (repeated)
- Bengali prompt + Bengali image → outputs Bengali prompt (repeated)
- Both with and without LoRA: same behavior

**Root cause**: `mlx-vlm` is not injecting image tokens for Gemma 4 E4B architecture. The `generate()` function processes the prompt as text-only.

**Impact**: All fine-tuning results are INVALID. The model was never training on vision — just text patterns.

**Options being evaluated**:
1. Try Qwen3-VL (confirmed working on mlx-vlm for vision tasks)
2. Try official Gemma 4 26B-A4B (larger, might have better mlx-vlm support)
3. Use HuggingFace transformers + MPS directly (bypasses mlx-vlm)
4. File upstream issue on mlx-vlm for E4B vision support

### Decision 14: Vision pipeline FIXED — apply_chat_template was missing (2026-04-20)

**Root cause** (diagnosed by codex): `mlx_vlm.generate()` does NOT auto-inject `<|image|>` tokens. Must call `apply_chat_template(processor, model.config, prompt, num_images=1)` FIRST. Without template, pixel_values are computed but scattered into zero positions → model ignores images.

**Key insight**: Training code (`finetune_bengali_ocr.py:145`) WAS correct — it used `processor.apply_chat_template()`. Only the eval code was broken. LoRA adapters ARE trained on vision+text.

**Fix**: 2-line change in `eval/evaluate_finetuned.py`:
```python
from mlx_vlm.prompt_utils import apply_chat_template
formatted = apply_chat_template(processor, model.config, PROMPT, num_images=1)
result = generate(model, processor, prompt=formatted, image=img_path, ...)
```

**Quick 10-sample eval results (with fix)**:
- Sample 9: **PERFECT** — ref=লঙ্কা, pred=লঙ্কা
- Most samples: model reads image but wraps answer in "ছবিতে লেখা আছে:" (written in image:)
- CER=6.4 corpus (high due to explanatory wrapper text, not because OCR failed)

**Next steps to improve CER**:
1. Strip explanatory prefix from predictions
2. Change prompt to "Output ONLY the Bengali text, no explanation"  
3. Re-evaluate with cleaner prompt
4. Consider retraining with stricter prompt format

**Save method evolution (complete):**
1. mx.savez → nanobind >1024 limit
2. numpy.savez → bf16→numpy conversion fail
3. safetensors.mlx → same bf16 issue (calls np.asarray internally)
4. safetensors.mlx + astype(mx.float16) → ✅ WORKS

Full codex diagnosis: /Users/macmini/projects/codex/bengali_ocr_vision_fix_20apr2026.txt

### Decision 15: PaddleOCR-VL-1.5 zero-shot baseline (2026-04-20)

| Engine | CER (corpus) | CER (macro) | WER |
|---|---|---|---|
| EasyOCR | 0.153 | 0.182 | 0.474 |
| **PaddleOCR-VL-1.5 zero-shot** | **0.668** | 0.734 | 0.813 |
| Gemma 4 E4B (LoRA, broken eval) | 0.917 | 0.925 | 1.000 |
| Gemma 4 E4B (fixed eval) | 6.417 | 8.412 | 9.817 |
| Gemma 4 E4B (base zero-shot) | N/A | N/A | N/A |
| Tesseract | 0.722 | 0.770 | 0.949 |

Observations:
- Beats Tesseract zero-shot (0.668 vs 0.722)
- Some samples near-perfect: `প্রত্যাশা করা হচ্ছে` → `প্রত্যাশ। করার হচ্ছে`
- Script confusion: some outputs in Tibetan/Devanagari instead of Bengali
- EasyOCR (traditional engine) still leads at 0.153 — fine-tuning should close the gap
- Model is tiny (0.9B, 704MB) → training will be very fast

### Decision 16: PaddleOCR-VL-1.5 LoRA collapsed (2026-04-20)

Fine-tuning 1000 steps completed but model collapsed to repetitive Bengali text ("প্রতিটির সুপ্যাদারীতিকাবেশনীতি..."). CER=9.548 — much worse than zero-shot 0.668.

**Root causes identified**:
1. **53% trainable params** (137M/258M) — `find_all_linear_names` targets ALL linear layers including vision encoder. For 0.9B model, this is near-full fine-tuning → catastrophic forgetting
2. **Training prompt format**: used `formatted_prompt + text` without proper input/output separation — model couldn't learn where prompt ends and OCR output begins
3. **Loss plateau at 7.06** (from 10.7) — model was memorizing a fixed pattern, not learning OCR

**Fixes needed for next attempt**:
1. Target ONLY language model attention layers (not vision encoder, not embeddings)
2. Use proper train/target separation with loss masking on prompt tokens
3. Lower LoRA rank (4) for smaller model
4. Consider using mlx-vlm's built-in SFT trainer instead of custom loop

**Comparison table** (all Bengali OCR attempts):

| Engine | CER (corpus) | Status |
|---|---|---|
| EasyOCR | 0.153 | target |
| PaddleOCR-VL-1.5 zero-shot | 0.668 | base works |
| Tesseract | 0.722 | baseline |
| Gemma 4 E4B (fixed eval) | 6.417 | wrapper text |
| PaddleOCR-VL-1.5 LoRA v1 | 9.548 | collapsed |
| Gemma 4 E4B (broken eval) | 13.697 | prompt echo |

### Decision 17: PaddleOCR-VL v2 LoRA — partial success (2026-04-20)

Fixes applied (per codex review):
1. LoRA targets: only q_proj, k_proj, v_proj, o_proj (still 53% trainable due to get_peft_model architecture issue)
2. Loss masking: only target tokens, not prompt/image tokens
3. alpha=8 (was 0.5), rank=4 (was 8), LR=2e-5 (was 5e-5)

Results: CER=3.22 corpus (v1 was 9.55, zero-shot 0.668)
- Sample 1: `প্রত্যাশা করা হচ্ছে` → `প্রত্যাগা করা হচ্ছে` (1 char error — NEAR PERFECT)
- Sample 3: `প্রেসিডেন্ট` → `প্রেসিডেন্টে` (1 extra char)
- Samples 2,4,5: repetition loops (`দুপদুপদুপ...`, `দাদাদাদা...`)

**Assessment**: Loss masking fixed the total collapse. Model CAN read Bengali (samples 1,3 prove it).
Remaining issue: repetition loops on some samples inflate CER.

**Next steps to try**:
1. Add repetition penalty in generate() (temperature sampling or rep_penalty)
2. Freeze vision encoder explicitly (current 53% trainable may corrupt it)
3. Use mlx-vlm's built-in SFT trainer (handles masking + freezing properly)
4. Train longer (2000+ steps) with warmup schedule

### Decision 18: PaddleOCR-VL-1.5 training data is proprietary (2026-04-20)

**Source**: [arXiv:2601.21957v2](https://arxiv.org/abs/2601.21957)
- Pre-training: 46M image-text pairs (NOT public)
- Post-training: 5.6M instruction samples (NOT public)
- Bengali: "newly added" in v1.5 among 111 languages — zero specifics on Bengali sample count or source
- Base LM: ERNIE-4.5-0.3B-Paddle
- Only 23 demo images + 2.8K eval benchmark published

Our dataset (rifathridoy/bengali-ocr-synthetic, 30K) is completely independent.

**Implication**: fine-tuning is our only path to improving Bengali OCR on this model since we can't examine or augment the base training data. The zero-shot CER of 0.668 is the floor — our job is to push it below EasyOCR's 0.153.
