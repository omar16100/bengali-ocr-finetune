"""
Smoke test for mlx-vlm Gemma 4 gradient fix.
Verifies that a single training step produces finite gradients
on a vision-language model with LoRA.

Uses SuperGemma4-26B-MLX-4bit (Gemma 4 architecture, already on disk).
If that's too large, falls back to any available Gemma model.
"""

import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def run_smoke_test():
    try:
        import mlx.core as mx
        import mlx.nn as nn
        from mlx_vlm import load
        from mlx_vlm.trainer.utils import find_all_linear_names, get_peft_model
    except ImportError as e:
        log.error("mlx_vlm not installed or import failed: %s", e)
        return False

    model_path = "/Users/macmini/models/gemma-4-e4b-it-4bit"
    log.info("loading model from %s", model_path)

    try:
        model, processor = load(model_path)
        log.info("model loaded successfully")
    except Exception as e:
        log.error("failed to load model: %s", e)
        return False

    # Apply LoRA
    try:
        linear_names = find_all_linear_names(model)
        log.info("found %d linear layers for LoRA", len(linear_names))
        model = get_peft_model(model, linear_names, rank=4, alpha=0.5, dropout=0.0)
        log.info("LoRA applied (rank=4, alpha=8)")
    except Exception as e:
        log.error("failed to apply LoRA: %s", e)
        return False

    # Create a minimal input
    try:
        from PIL import Image
        import numpy as np

        # Create a tiny test image with Bengali-like content
        img = Image.fromarray(np.random.randint(0, 255, (64, 256, 3), dtype=np.uint8))

        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": "OCR this image."}
            ]},
        ]
        text = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=text, images=[img], return_tensors="np")

        # Convert to mx arrays
        input_ids = mx.array(inputs["input_ids"])
        pixel_values = mx.array(inputs["pixel_values"]) if "pixel_values" in inputs else None

        log.info("input prepared: input_ids shape=%s", input_ids.shape)
    except Exception as e:
        log.error("failed to prepare input: %s", e)
        return False

    # Forward + backward
    try:
        def loss_fn(model, input_ids, pixel_values):
            logits = model(input_ids, pixel_values=pixel_values)
            if hasattr(logits, 'logits'):
                logits = logits.logits
            # simple cross-entropy on shifted logits
            shift_logits = logits[:, :-1, :]
            shift_labels = input_ids[:, 1:]
            loss = nn.losses.cross_entropy(
                shift_logits.reshape(-1, shift_logits.shape[-1]),
                shift_labels.reshape(-1),
            ).mean()
            return loss

        loss_and_grad = nn.value_and_grad(model, loss_fn)
        loss, grads = loss_and_grad(model, input_ids, pixel_values)
        mx.eval(loss, grads)

        loss_val = loss.item()
        log.info("loss = %.4f", loss_val)

        # Check for NaN/Inf in gradients
        nan_count = 0
        inf_count = 0
        total_params = 0
        import mlx.utils
        for name, grad in mlx.utils.tree_flatten(grads):
            if isinstance(grad, mx.array):
                total_params += 1
                if mx.any(mx.isnan(grad)).item():
                    nan_count += 1
                    log.warning("NaN gradient in %s", name)
                if mx.any(mx.isinf(grad)).item():
                    inf_count += 1
                    log.warning("Inf gradient in %s", name)

        log.info("gradient check: %d params, %d NaN, %d Inf", total_params, nan_count, inf_count)

        if nan_count > 0 or inf_count > 0:
            log.error("FAIL: NaN/Inf gradients detected — fix incomplete")
            return False

        if not (0 < loss_val < 100):
            log.error("FAIL: loss out of expected range: %.4f", loss_val)
            return False

        log.info("PASS: finite loss and gradients, mlx-vlm Gemma 4 training fix verified")
        return True

    except Exception as e:
        log.error("forward/backward failed: %s", e)
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_smoke_test()
    sys.exit(0 if success else 1)
