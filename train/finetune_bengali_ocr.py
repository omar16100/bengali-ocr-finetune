"""
Fine-tune Gemma 4 on Bengali OCR data using mlx-vlm LoRA.

Uses the patched mlx-vlm from /Users/macmini/projects/mlx-vlm/ which
fixes NaN gradients in Gemma 4 vision training (3 bugs: -inf mask,
.item() autograd break, @mx.compile gradient block).

Dataset: rifathridoy/bengali-ocr-synthetic (27K train, 3K test)
Model: SuperGemma4-26B-MLX-4bit (Gemma 4 architecture, 14 GB)
Method: LoRA (rank=16, alpha=32) on language model linear layers

Evaluated via CER/WER on held-out test split.
"""

import json
import logging
import os
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler("train/finetune.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# Paths
MODEL_PATH = "/Users/macmini/models/gemma-4-e4b-it-4bit"
DATA_DIR = Path("data/bengali-ocr-synthetic")
RESULTS_DIR = Path("results")
ADAPTER_DIR = Path("results/adapters/bengali_ocr_lora")

# Training config
LORA_RANK = 16
LORA_ALPHA = 32
LEARNING_RATE = 1e-4
BATCH_SIZE = 1
MAX_STEPS = 500
EVAL_EVERY = 50
MAX_SAMPLES_TRAIN = 5000  # start small, scale up if quality is good
MAX_SAMPLES_EVAL = 200


def prepare_data():
    """Load and format the Bengali OCR dataset for mlx-vlm SFT."""
    import datasets

    log.info("loading dataset from %s", DATA_DIR)
    train_ds = datasets.load_dataset(str(DATA_DIR), split="train")
    test_ds = datasets.load_dataset(str(DATA_DIR), split="test")

    log.info("train: %d, test: %d", len(train_ds), len(test_ds))

    # Subsample for initial experiment
    if MAX_SAMPLES_TRAIN and len(train_ds) > MAX_SAMPLES_TRAIN:
        train_ds = train_ds.shuffle(seed=42).select(range(MAX_SAMPLES_TRAIN))
        log.info("subsampled train to %d", len(train_ds))

    if MAX_SAMPLES_EVAL and len(test_ds) > MAX_SAMPLES_EVAL:
        test_ds = test_ds.shuffle(seed=42).select(range(MAX_SAMPLES_EVAL))
        log.info("subsampled test to %d", len(test_ds))

    # Format as conversation for mlx-vlm SFT trainer
    def format_sample(sample):
        return {
            "image": sample["image"],
            "messages": [
                {"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": "এই ছবি থেকে বাংলা টেক্সট পড়ুন।"}  # Read Bengali text from this image
                ]},
                {"role": "assistant", "content": sample["text"]},
            ],
        }

    train_formatted = [format_sample(s) for s in train_ds]
    test_formatted = [format_sample(s) for s in test_ds]

    log.info("formatted %d train, %d test samples", len(train_formatted), len(test_formatted))
    return train_formatted, test_formatted, test_ds


def run_finetuning(train_data, test_data):
    """Run LoRA fine-tuning using mlx-vlm's SFT trainer."""
    try:
        import mlx
        import mlx.core as mx
        import mlx.nn as nn
        import mlx.optimizers as optim
        from mlx_vlm import load
        from mlx_vlm.trainer.utils import find_all_linear_names, get_peft_model
    except ImportError as e:
        log.error("mlx_vlm not available: %s", e)
        return None

    log.info("loading model from %s", MODEL_PATH)
    model, processor = load(MODEL_PATH)
    log.info("model loaded")

    # Apply LoRA
    linear_names = find_all_linear_names(model)
    model = get_peft_model(
        model, linear_names, rank=LORA_RANK, alpha=LORA_ALPHA / LORA_RANK,
        dropout=0.0
    )
    trainable = sum(p.size for _, p in mlx.utils.tree_flatten(model.trainable_parameters()))
    total = sum(p.size for _, p in mlx.utils.tree_flatten(model.parameters()))
    log.info("LoRA applied: %d trainable / %d total params (%.2f%%)",
             trainable, total, 100 * trainable / total)

    # Optimizer
    optimizer = optim.Adam(learning_rate=LEARNING_RATE)

    # Training loop
    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    losses = []
    t0 = time.time()

    def loss_fn(model, input_ids, pixel_values, labels):
        logits = model(input_ids, pixel_values=pixel_values)
        if hasattr(logits, 'logits'):
            logits = logits.logits
        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]
        mask = shift_labels != -100
        loss = nn.losses.cross_entropy(
            shift_logits.reshape(-1, shift_logits.shape[-1]),
            shift_labels.reshape(-1),
        )
        loss = (loss * mask.reshape(-1)).sum() / mask.sum()
        return loss

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    for step in range(MAX_STEPS):
        sample = train_data[step % len(train_data)]

        try:
            # Prepare inputs
            messages = sample["messages"]
            text = processor.apply_chat_template(messages, add_generation_prompt=False)
            inputs = processor(
                text=text, images=[sample["image"]], return_tensors="np"
            )

            input_ids = mx.array(inputs["input_ids"])
            pixel_values = mx.array(inputs["pixel_values"]) if "pixel_values" in inputs else None
            labels = input_ids  # MLX arrays are immutable; no copy needed

            # Forward + backward
            loss, grads = loss_and_grad(model, input_ids, pixel_values, labels)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state, loss)

            loss_val = loss.item()
            losses.append(loss_val)

            if (step + 1) % 10 == 0:
                avg_loss = sum(losses[-10:]) / min(10, len(losses))
                elapsed = time.time() - t0
                log.info("step %d/%d  loss=%.4f  avg_loss=%.4f  elapsed=%.0fs",
                         step + 1, MAX_STEPS, loss_val, avg_loss, elapsed)

            # Periodic eval
            if (step + 1) % EVAL_EVERY == 0:
                log.info("saving adapter at step %d", step + 1)
                adapter_path = ADAPTER_DIR / f"step_{step + 1}"
                adapter_path.mkdir(exist_ok=True)
                # Save LoRA adapters via safetensors (mx.savez has nanobind
                # >1024 kwarg limit; np.array has bfloat16 conversion issues).
                from safetensors.mlx import save_file as save_mlx
                params = dict(mlx.utils.tree_flatten(model.trainable_parameters()))
                save_mlx(params, str(adapter_path / "adapters.safetensors"))

                # Save training curve
                with open(RESULTS_DIR / "training_curve.json", "w") as f:
                    json.dump({"losses": losses, "steps": list(range(1, len(losses) + 1))}, f)

        except Exception as e:
            log.error("step %d failed: %s", step + 1, e)
            import traceback
            traceback.print_exc()
            if step < 3:
                log.error("failing on early steps — likely model/data incompatibility. Aborting.")
                return None
            continue

    # Save final adapter
    final_path = ADAPTER_DIR / "final"
    final_path.mkdir(exist_ok=True)
    from safetensors.mlx import save_file as save_mlx
    params = dict(mlx.utils.tree_flatten(model.trainable_parameters()))
    save_mlx(params, str(final_path / "adapters.safetensors"))

    elapsed = time.time() - t0
    log.info("training complete: %d steps in %.0fs (%.2f steps/s), final_loss=%.4f",
             MAX_STEPS, elapsed, MAX_STEPS / elapsed, losses[-1] if losses else 0)

    return model, processor


def evaluate_finetuned(model, processor, test_ds):
    """Evaluate the fine-tuned model on CER/WER."""
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from eval.metrics import evaluate_batch, normalize_bengali, strip_model_formatting
    import mlx.core as mx

    log.info("evaluating on %d test samples", len(test_ds))
    predictions = []
    references = []

    for i, sample in enumerate(test_ds):
        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": "এই ছবি থেকে বাংলা টেক্সট পড়ুন।"}
            ]},
        ]
        text = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=text, images=[sample["image"]], return_tensors="np")
        input_ids = mx.array(inputs["input_ids"])
        pixel_values = mx.array(inputs["pixel_values"]) if "pixel_values" in inputs else None

        # Generate
        logits = model(input_ids, pixel_values=pixel_values)
        if hasattr(logits, 'logits'):
            logits = logits.logits
        next_tokens = mx.argmax(logits[:, -1, :], axis=-1)
        # Simple greedy decode for up to 128 tokens
        generated = [next_tokens.item()]
        for _ in range(127):
            new_input = mx.array([[generated[-1]]])
            logits = model(new_input)
            if hasattr(logits, 'logits'):
                logits = logits.logits
            tok = mx.argmax(logits[:, -1, :], axis=-1).item()
            if tok == processor.tokenizer.eos_token_id:
                break
            generated.append(tok)

        pred = processor.tokenizer.decode(generated, skip_special_tokens=True)
        pred = strip_model_formatting(normalize_bengali(pred))
        ref = normalize_bengali(sample["text"])

        predictions.append(pred)
        references.append(ref)

        if (i + 1) % 50 == 0:
            log.info("eval: %d/%d", i + 1, len(test_ds))

    metrics = evaluate_batch(predictions, references)
    for key in ["cer_per_sample", "wer_per_sample", "ger_per_sample"]:
        metrics.pop(key, None)

    log.info("CER=%.4f WER=%.4f GER=%.4f (corpus CER=%.4f)",
             metrics["cer_mean"], metrics["wer_mean"], metrics["ger_mean"],
             metrics["cer_corpus"])

    with open(RESULTS_DIR / "finetuned_eval.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    log.info("=== Phase 1: Prepare data ===")
    train_data, test_data, test_ds = prepare_data()

    log.info("=== Phase 2: Fine-tune ===")
    result = run_finetuning(train_data, test_data)
    if result is None:
        log.error("fine-tuning failed, aborting")
        return

    model, processor = result

    log.info("=== Phase 3: Evaluate ===")
    metrics = evaluate_finetuned(model, processor, test_ds)

    log.info("=== Done ===")
    log.info("Final metrics: %s", json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
