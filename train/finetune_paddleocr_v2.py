"""
PaddleOCR-VL-1.5 Bengali OCR fine-tuning — v2 (fixed).

Fixes from v1 collapse:
1. Target ONLY language model attention (q_proj, k_proj, v_proj, o_proj)
   — NOT vision encoder (qkv, fc1, fc2, out_proj) or MLP (gate/up/down)
2. Loss masking: only compute loss on TARGET tokens, not prompt/image tokens
3. Lower LoRA rank (4) for 0.9B model
4. Lower learning rate (2e-5)
"""

import json
import logging
import sys
import time
from pathlib import Path

import mlx
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler("train/finetune_paddleocr_v2.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

MODEL_PATH = "/Users/macmini/models/PaddleOCR-VL-1.5-4bit"
DATA_DIR = Path("data/bengali-ocr-synthetic")
RESULTS_DIR = Path("results")
ADAPTER_DIR = Path("results/adapters/paddleocr_vl_bengali_lora_v2")

# Only target language model attention — freeze vision encoder + MLP
LORA_TARGET_LAYERS = ["q_proj", "k_proj", "v_proj", "o_proj"]
LORA_RANK = 4
LORA_ALPHA = 8.0  # codex: alpha=8 at rank=4, not 0.5
LEARNING_RATE = 2e-5
MAX_STEPS = 1000
EVAL_EVERY = 200
MAX_SAMPLES_TRAIN = 10000
MAX_SAMPLES_EVAL = 200
PROMPT = "OCR:"


def prepare_data():
    import datasets
    train_ds = datasets.load_dataset(str(DATA_DIR), split="train")
    test_ds = datasets.load_dataset(str(DATA_DIR), split="test")
    if len(train_ds) > MAX_SAMPLES_TRAIN:
        train_ds = train_ds.shuffle(seed=42).select(range(MAX_SAMPLES_TRAIN))
    if len(test_ds) > MAX_SAMPLES_EVAL:
        test_ds = test_ds.shuffle(seed=42).select(range(MAX_SAMPLES_EVAL))
    log.info("train: %d, test: %d", len(train_ds), len(test_ds))
    return train_ds, test_ds


def run_finetuning(train_ds, test_ds):
    from mlx_vlm import load
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.utils import prepare_inputs
    from mlx_vlm.trainer.utils import get_peft_model
    from safetensors.mlx import save_file as save_mlx

    log.info("loading model")
    model, processor = load(MODEL_PATH)

    # Apply LoRA ONLY to language model attention layers
    log.info("applying LoRA to %s (rank=%d)", LORA_TARGET_LAYERS, LORA_RANK)
    model = get_peft_model(
        model, LORA_TARGET_LAYERS,
        rank=LORA_RANK, alpha=LORA_ALPHA, dropout=0.0
    )

    trainable = sum(p.size for _, p in mlx.utils.tree_flatten(model.trainable_parameters()))
    total_p = sum(p.size for _, p in mlx.utils.tree_flatten(model.parameters()))
    log.info("LoRA: %d trainable / %d total (%.2f%%)", trainable, total_p, 100*trainable/total_p)

    if trainable / total_p > 0.10:
        log.warning("trainable ratio >10%% — risk of catastrophic forgetting")

    optimizer = optim.Adam(learning_rate=LEARNING_RATE)
    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    losses = []
    t0 = time.time()

    # Get formatted prompt (with image tokens) to measure prompt length
    formatted_prompt = apply_chat_template(processor, model.config, PROMPT, num_images=1)
    log.info("prompt template: %s", formatted_prompt[:80])

    def loss_fn(model, input_ids, pixel_values, image_grid_thw, prompt_len):
        logits = model(input_ids, pixel_values=pixel_values, image_grid_thw=image_grid_thw)
        if hasattr(logits, 'logits'):
            logits = logits.logits

        # Shift for next-token prediction
        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]

        # Loss masking: only compute loss on TARGET tokens (after prompt)
        # prompt_len includes image tokens, so tokens before prompt_len are input
        seq_len = shift_labels.shape[1]
        # Create mask: 0 for prompt tokens, 1 for target tokens
        mask = mx.arange(seq_len) >= (prompt_len - 1)
        mask = mask.astype(mx.float32)

        per_token_loss = nn.losses.cross_entropy(
            shift_logits.reshape(-1, shift_logits.shape[-1]),
            shift_labels.reshape(-1),
        )
        per_token_loss = per_token_loss.reshape(seq_len)

        # Masked mean: only target tokens contribute to loss
        masked_loss = (per_token_loss * mask).sum() / mx.maximum(mask.sum(), mx.array(1.0))
        return masked_loss

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    for step in range(MAX_STEPS):
        sample = train_ds[step % len(train_ds)]
        img = sample["image"]
        text = sample["text"]

        try:
            tmp_img = "/tmp/paddleocr_train_tmp.png"
            img.save(tmp_img)

            # Prepare prompt-only inputs to measure prompt length
            prompt_inputs = prepare_inputs(processor, images=[tmp_img], prompts=formatted_prompt)
            prompt_len = prompt_inputs["input_ids"].shape[1]

            # Prepare full inputs (prompt + target)
            full_prompt = formatted_prompt + text
            inputs = prepare_inputs(processor, images=[tmp_img], prompts=full_prompt)

            input_ids = inputs["input_ids"]
            pixel_values = inputs.get("pixel_values")
            image_grid_thw = inputs.get("image_grid_thw")

            loss, grads = loss_and_grad(model, input_ids, pixel_values, image_grid_thw, prompt_len)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state, loss)

            loss_val = loss.item()
            losses.append(loss_val)

            if (step + 1) % 10 == 0:
                avg = sum(losses[-10:]) / min(10, len(losses))
                elapsed = time.time() - t0
                log.info("step %d/%d  loss=%.4f  avg=%.4f  elapsed=%.0fs  (%.2f steps/s)",
                         step+1, MAX_STEPS, loss_val, avg, elapsed, (step+1)/elapsed)

            if (step + 1) % EVAL_EVERY == 0:
                adapter_path = ADAPTER_DIR / f"step_{step+1}"
                adapter_path.mkdir(exist_ok=True)
                params = {k: v.astype(mx.float16) for k, v in mlx.utils.tree_flatten(model.trainable_parameters())}
                save_mlx(params, str(adapter_path / "adapters.safetensors"))
                log.info("saved adapter at step %d", step+1)

                with open(RESULTS_DIR / "training_curve_paddleocr_v2.json", "w") as f:
                    json.dump({"losses": losses}, f)

        except Exception as e:
            log.error("step %d failed: %s", step+1, e)
            import traceback
            traceback.print_exc()
            if step < 5:
                log.error("failing early — aborting")
                return None, None
            continue

    # Save final
    final_path = ADAPTER_DIR / "final"
    final_path.mkdir(exist_ok=True)
    params = {k: v.astype(mx.float16) for k, v in mlx.utils.tree_flatten(model.trainable_parameters())}
    save_mlx(params, str(final_path / "adapters.safetensors"))

    elapsed = time.time() - t0
    log.info("training complete: %d steps in %.0fs, final_loss=%.4f",
             MAX_STEPS, elapsed, losses[-1] if losses else 0)

    with open(RESULTS_DIR / "training_curve_paddleocr_v2.json", "w") as f:
        json.dump({"losses": losses}, f)

    return model, processor


def evaluate(model, processor, test_ds):
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from eval.metrics import evaluate_batch, normalize_bengali, strip_model_formatting
    from mlx_vlm import generate
    from mlx_vlm.prompt_utils import apply_chat_template

    log.info("evaluating on %d samples", len(test_ds))
    predictions, references = [], []
    t0 = time.time()

    for i, sample in enumerate(test_ds):
        img = sample["image"]
        ref = normalize_bengali(sample["text"])
        tmp = "/tmp/paddleocr_eval_tmp.png"
        img.save(tmp)
        try:
            formatted = apply_chat_template(processor, model.config, PROMPT, num_images=1)
            result = generate(model, processor, prompt=formatted, image=tmp, max_tokens=64, temp=0.0)
            pred = result.text if hasattr(result, 'text') else str(result)
            pred = strip_model_formatting(normalize_bengali(pred))
        except Exception as e:
            pred = ""
        predictions.append(pred)
        references.append(ref)
        if (i+1) % 50 == 0:
            log.info("eval: %d/%d (%.1fs)", i+1, len(test_ds), time.time()-t0)

    metrics = evaluate_batch(predictions, references)
    for k in ["cer_per_sample", "wer_per_sample", "ger_per_sample"]:
        metrics.pop(k, None)
    metrics["model"] = "paddleocr-vl-1.5-lora-v2-bengali"
    metrics["elapsed_s"] = round(time.time()-t0, 2)

    log.info("CER=%.4f WER=%.4f corpus_CER=%.4f", metrics["cer_mean"], metrics["wer_mean"], metrics["cer_corpus"])

    with open(RESULTS_DIR / "paddleocr_finetuned_v2_eval.json", "w") as f:
        json.dump(metrics, f, indent=2)

    samples = [{"ref": references[i], "pred": predictions[i]} for i in range(min(20, len(predictions)))]
    with open(RESULTS_DIR / "paddleocr_v2_eval_samples.json", "w") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)

    return metrics


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    train_ds, test_ds = prepare_data()
    result = run_finetuning(train_ds, test_ds)
    if result[0] is None:
        return
    model, processor = result
    metrics = evaluate(model, processor, test_ds)
    log.info("EasyOCR baseline: CER=0.153")
    log.info("PaddleOCR-VL v2 fine-tuned: CER=%.3f", metrics["cer_corpus"])
    log.info("PaddleOCR-VL zero-shot: CER=0.668")
    if metrics["cer_corpus"] < 0.153:
        log.info("BEATS EasyOCR baseline!")
    elif metrics["cer_corpus"] < 0.668:
        log.info("BEATS zero-shot baseline!")


if __name__ == "__main__":
    main()
