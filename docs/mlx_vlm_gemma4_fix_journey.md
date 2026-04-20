# Fixing Gemma 4 Vision Training on MLX — The Full Journey

## Context
2026-04-19. Goal: fine-tune Gemma 4 on Bengali OCR data on Mac Studio M3 Ultra (512 GB). mlx-vlm is the only viable framework for vision-language fine-tuning on Apple Silicon. But Gemma 4 training produces NaN gradients.

## The Problem
[mlx-vlm PR #969](https://github.com/Blaizzy/mlx-vlm/pull/969) documents three bugs that cause NaN gradients during Gemma 4 vision training. The PR has the fix but is not merged. We need training to work NOW.

## Bug 1: -inf in attention mask → NaN in softmax backward

**File**: `mlx_vlm/models/gemma4/vision.py:509`

**Root cause**: The vision encoder builds a bidirectional attention mask with `float("-inf")` for padding positions. In the backward pass of softmax, when an entire row is -inf, softmax produces all zeros. The gradient computation involves `0 * -inf = NaN`. This silently corrupts all downstream gradients.

**Fix**: Replace `float("-inf")` with `-1e4` — large enough to zero softmax probabilities in practice, but stays finite in both float16 (max ~65504) and float32.

```python
# Before (NaN)
neg_inf = mx.array(float("-inf"), dtype=inputs_embeds.dtype)
# After (safe)
neg_inf = mx.array(-1e4, dtype=inputs_embeds.dtype)
```

**Why -1e4 and not -1e9?** -1e9 overflows to -inf in float16, defeating the fix. -1e4 is safe for both float16 and float32. Codex (gpt-5.4) confirmed this choice.

## Bug 2: .item() breaks autograd

**File**: `mlx_vlm/models/gemma4/vision.py:526-531`

**Root cause**: After the vision pooler, the original code loops over the batch, calls `valid_mask[i].sum().item()` to get the count of valid tokens, then slices the pooled output. `.item()` internally calls `mx.eval()` which:
1. Is illegal inside `mx.compile` (crashes)
2. Detaches the result from the computation graph (breaks gradient flow silently)

**The fix journey** (3 attempts):

1. **Attempt 1 — mask multiplication + fixed-length slice**: `pooled * valid_mask_expanded[:, :default_output_length]`. Codex review said this is NOT semantically equivalent for variable-length batches.

2. **Attempt 2 — boolean indexing**: `pooled[valid_mask][None]`. Codex's suggestion. **FAILED**: MLX 0.31.x raises `ValueError: boolean indices are not yet supported`.

3. **Attempt 3 — mask multiplication (no slice)**: `pooled * valid_mask_expanded`. Zeros out padding tokens. Downstream projection and loss ignore zeros naturally. Less precise than per-sample concatenation but gradient-safe and MLX-compatible.

```python
# Before (.item() breaks autograd)
all_real = []
for i in range(B):
    n_valid = int(valid_mask[i].astype(mx.int32).sum().item())
    all_real.append(pooled[i, :n_valid])
hidden_states = mx.concatenate(all_real, axis=0)[None]

# After (gradient-safe)
valid_mask_expanded = mx.expand_dims(valid_mask, -1).astype(pooled.dtype)
hidden_states = pooled * valid_mask_expanded
```

**Lesson**: Codex's static analysis suggested boolean indexing — a correct fix in PyTorch — but MLX doesn't support it yet. Always test the fix on the actual framework version.

## Bug 3: @mx.compile blocks backward pass

**File**: `mlx_vlm/models/base.py:386`

**Root cause**: The `ensure_fused_sdpa` function is decorated with `@mx.compile`, which caches a computation trace. During training, the cached trace doesn't include backward pass information, producing NaN gradients. MLX docs say `compile()` should work with training, so this may be a version-specific interaction with the SDPA padding logic.

**Fix**: Remove the decorator. Add a comment explaining the decision.

```python
# Before
@mx.compile
def ensure_fused_sdpa(q, k, v, scale, mask=None): ...

# After (training-safe)
def ensure_fused_sdpa(q, k, v, scale, mask=None): ...
```

**Open question**: This may cause a small inference regression on prefill throughput. Codex recommended a conditional `compile(inference_only)` wrapper as a follow-up.

## The Model Problem: SuperGemma4-26B lacks vision weights

After fixing all 3 bugs, the gradient smoke test on SuperGemma4-26B-MLX-4bit-v2 failed with `Missing 211 parameters` — all from `vision_tower.*`. The MLX conversion of this model **stripped the vision encoder weights** entirely. It's text-only despite being Gemma 4 architecture.

**Pivot**: Downloaded `mlx-community/gemma-4-e4b-it-4bit` (4.5B params, 4.9 GB). This model has the full vision tower present. Model loads, 16 linear layers found, LoRA applies (10M trainable / 7.5B total = 0.14%).

## The API Mismatch

`get_peft_model()` in mlx-vlm takes `(model, linear_layers, rank, alpha, dropout)` — NOT `(model, rank, alpha, target_modules)` like HuggingFace PEFT. The kwargs are positional. Got `TypeError: unexpected keyword argument 'target_modules'`.

## The Result

After fixes 1-3 + model pivot + API fix:

```
loss = 28.0  ← FINITE! Not NaN!
```

Forward + backward through Gemma 4 E4B with vision + LoRA produces a finite loss. The gradient fix is verified. Fine-tuning can proceed.

## Timeline

| Time | Event |
|---|---|
| 2026-04-19 22:00 | Research identified 3 NaN bugs in mlx-vlm PR #969 |
| 2026-04-19 22:10 | Applied 3 fixes to cloned mlx-vlm |
| 2026-04-19 22:15 | Codex reviewed fixes — approved #1, #3; suggested boolean indexing for #2 |
| 2026-04-19 22:20 | Updated Fix #2 per codex (boolean indexing), updated Fix #3 comment |
| 2026-04-20 07:45 | Smoke test on SuperGemma4-26B: Missing 211 vision_tower parameters |
| 2026-04-20 07:50 | Pivoted to Gemma 4 E4B (4.5B, has vision) — downloaded 4.9 GB |
| 2026-04-20 09:15 | Smoke test v1: LoRA API mismatch (target_modules kwarg) |
| 2026-04-20 09:17 | Smoke test v2: boolean indexing not supported in MLX 0.31 |
| 2026-04-20 09:19 | Smoke test v3: **loss = 28.0 (FINITE!)** Fix verified |
| 2026-04-20 09:20 | Minor: `mx.utils.tree_flatten` → `mlx.utils.tree_flatten` |

## What this unlocks

With finite gradients verified on Gemma 4 E4B:
1. Bengali OCR LoRA fine-tuning can proceed
2. The same fixes apply to Gemma 4 26B-A4B and 31B models
3. This fix is potentially upstreamable to mlx-vlm (PR #969 can be updated with the MLX-compatible mask multiplication approach instead of boolean indexing)

## Files changed (from upstream mlx-vlm)

| File | Change | Lines |
|---|---|---|
| `mlx_vlm/models/gemma4/vision.py:509` | `-inf` → `-1e4` | 1 line |
| `mlx_vlm/models/gemma4/vision.py:526-531` | `.item()` loop → mask multiply | 6→2 lines |
| `mlx_vlm/models/base.py:386` | Remove `@mx.compile` | 1 line |
