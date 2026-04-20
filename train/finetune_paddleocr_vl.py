"""
Fine-tune PaddleOCR-VL-1.5 on Bengali OCR data using mlx-vlm LoRA.

Model: mlx-community/PaddleOCR-VL-1.5-4bit (0.9B params, 704 MB)
Dataset: rifathridoy/bengali-ocr-synthetic (27K train, 3K test)
Method: LoRA (rank=8, alpha=1.0) — smaller model needs smaller rank
Prompt: "OCR:" (PaddleOCR-VL's standard OCR prompt)
"""

import json
import logging
import os
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
        logging.FileHandler("train/finetune_paddleocr.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

MODEL_PATH = "/Users/macmini/models/PaddleOCR-VL-1.5-4bit"
DATA_DIR = Path("data/bengali-ocr-synthetic")
RESULTS_DIR = Path("results")
ADAPTER_DIR = Path("results/adapters/paddleocr_vl_bengali_lora")

LORA_RANK = 8
LORA_ALPHA = 1.0
LEARNING_RATE = 5e-5  # conservative for tiny model
BATCH_SIZE = 1
MAX_STEPS = 1000  # more steps for smaller model
EVAL_EVERY = 100
MAX_SAMPLES_TRAIN = 10000
MAX_SAMPLES_EVAL = 200
PROMPT = "OCR:"


def prepare_data():
    import datasets
    log.info("loading dataset from %s", DATA_DIR)
    train_ds = datasets.load_dataset(str(DATA_DIR), split="train")
    test_ds = datasets.load_dataset(str(DATA_DIR), split="test")
    log.info("train: %d, test: %d", len(train_ds), len(test_ds))

    if MAX_SAMPLES_TRAIN and len(train_ds) > MAX_SAMPLES_TRAIN:
        train_ds = train_ds.shuffle(seed=42).select(range(MAX_SAMPLES_TRAIN))
    if MAX_SAMPLES_EVAL and len(test_ds) > MAX_SAMPLES_EVAL:
        test_ds = test_ds.shuffle(seed=42).select(range(MAX_SAMPLES_EVAL))

    log.info("using %d train, %d test samples", len(train_ds), len(test_ds))
    return train_ds, test_ds


def run_finetuning(train_ds, test_ds):
    from mlx_vlm import load
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.utils import prepare_inputs
    from mlx_vlm.trainer.utils import find_all_linear_names, get_peft_model
    from safetensors.mlx import save_file as save_mlx

    log.info("loading model from %s", MODEL_PATH)
    model, processor = load(MODEL_PATH)
    log.info("model loaded")

    linear_names = find_all_linear_names(model)
    log.info("found %d linear layers for LoRA", len(linear_names))
    model = get_peft_model(model, linear_names, rank=LORA_RANK, alpha=LORA_ALPHA, dropout=0.0)

    trainable = sum(p.size for _, p in mlx.utils.tree_flatten(model.trainable_parameters()))
    total_p = sum(p.size for _, p in mlx.utils.tree_flatten(model.parameters()))
    log.info("LoRA: %d trainable / %d total (%.2f%%)", trainable, total_p, 100*trainable/total_p)

    optimizer = optim.Adam(learning_rate=LEARNING_RATE)
    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    losses = []
    t0 = time.time()

    # Format prompt with chat template for image tokens
    try:
        formatted_prompt = apply_chat_template(processor, model.config, PROMPT, num_images=1)
    except Exception:
        formatted_prompt = PROMPT
    log.info("formatted prompt: %s", formatted_prompt[:100])

    def loss_fn(model, input_ids, pixel_values, image_grid_thw, labels):
        logits = model(input_ids, pixel_values=pixel_values, image_grid_thw=image_grid_thw)
        if hasattr(logits, 'logits'):
            logits = logits.logits
        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]
        loss = nn.losses.cross_entropy(
            shift_logits.reshape(-1, shift_logits.shape[-1]),
            shift_labels.reshape(-1),
        ).mean()
        return loss

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    for step in range(MAX_STEPS):
        sample = train_ds[step % len(train_ds)]
        img = sample["image"]
        text = sample["text"]

        try:
            # Save image to temp file
            tmp_img = "/tmp/paddleocr_train_tmp.png"
            img.save(tmp_img)

            # Use prepare_inputs (same as generate()) — handles image tokens,
            # grid_thw, and pixel_values correctly for PaddleOCR-VL.
            formatted = apply_chat_template(processor, model.config, PROMPT, num_images=1)
            # Append the target text for training
            train_prompt = formatted + text
            inputs = prepare_inputs(processor, images=[tmp_img], prompts=train_prompt)

            input_ids = inputs["input_ids"]
            pixel_values = inputs.get("pixel_values")
            labels = input_ids

            image_grid_thw = inputs.get("image_grid_thw")
            loss, grads = loss_and_grad(model, input_ids, pixel_values, image_grid_thw, labels)
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

                with open(RESULTS_DIR / "training_curve_paddleocr.json", "w") as f:
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
    log.info("training complete: %d steps in %.0fs (%.2f steps/s), final_loss=%.4f",
             MAX_STEPS, elapsed, MAX_STEPS/elapsed, losses[-1] if losses else 0)

    with open(RESULTS_DIR / "training_curve_paddleocr.json", "w") as f:
        json.dump({"losses": losses}, f)

    return model, processor


def evaluate(model, processor, test_ds):
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from eval.metrics import evaluate_batch, normalize_bengali, strip_model_formatting
    from mlx_vlm import generate
    from mlx_vlm.prompt_utils import apply_chat_template

    log.info("evaluating on %d test samples", len(test_ds))
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
            log.warning("sample %d failed: %s", i, e)
            pred = ""

        predictions.append(pred)
        references.append(ref)

        if (i+1) % 50 == 0:
            log.info("eval: %d/%d (%.1fs)", i+1, len(test_ds), time.time()-t0)

    metrics = evaluate_batch(predictions, references)
    for k in ["cer_per_sample", "wer_per_sample", "ger_per_sample"]:
        metrics.pop(k, None)
    metrics["model"] = "paddleocr-vl-1.5-lora-bengali"
    metrics["elapsed_s"] = round(time.time()-t0, 2)

    log.info("CER=%.4f WER=%.4f corpus_CER=%.4f", metrics["cer_mean"], metrics["wer_mean"], metrics["cer_corpus"])

    with open(RESULTS_DIR / "paddleocr_finetuned_eval.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Save samples
    samples = [{"ref": references[i], "pred": predictions[i]} for i in range(min(20, len(predictions)))]
    with open(RESULTS_DIR / "paddleocr_eval_samples.json", "w") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)

    return metrics


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    log.info("=== Phase 1: Data ===")
    train_ds, test_ds = prepare_data()
    log.info("=== Phase 2: Fine-tune ===")
    result = run_finetuning(train_ds, test_ds)
    if result[0] is None:
        log.error("fine-tuning failed")
        return
    model, processor = result
    log.info("=== Phase 3: Evaluate ===")
    metrics = evaluate(model, processor, test_ds)
    log.info("=== Done ===")
    log.info("EasyOCR baseline: CER=0.153")
    log.info("PaddleOCR-VL fine-tuned: CER=%.3f", metrics["cer_corpus"])
    if metrics["cer_corpus"] < 0.153:
        log.info("BEATS EasyOCR baseline!")


if __name__ == "__main__":
    main()
