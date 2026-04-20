"""
Evaluate fine-tuned Gemma 4 E4B LoRA adapters on Bengali OCR test set.
Uses mlx_vlm.generate() for proper cached generation instead of manual
greedy decode (which produced garbage CER=0.917 without KV cache).
"""

import json
import logging
import sys
import time
from pathlib import Path

import mlx
import mlx.core as mx
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.trainer.utils import find_all_linear_names, get_peft_model
from safetensors.mlx import load_file as load_mlx

sys.path.insert(0, str(Path(__file__).parent.parent))
from eval.metrics import evaluate_batch, normalize_bengali, strip_model_formatting

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler("eval/eval_finetuned.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

MODEL_PATH = "/Users/macmini/models/gemma-4-e4b-it-4bit"
ADAPTER_PATH = Path("results/adapters/bengali_ocr_lora/final/adapters.safetensors")
DATA_DIR = Path("data/bengali-ocr-synthetic")
RESULTS_DIR = Path("results")
MAX_SAMPLES = 200
PROMPT = "এই ছবি থেকে বাংলা টেক্সট পড়ুন।"  # Read Bengali text from this image


def load_model_with_lora():
    """Load base model + apply LoRA + load trained weights."""
    log.info("loading base model from %s", MODEL_PATH)
    model, processor = load(MODEL_PATH)

    log.info("applying LoRA structure")
    linear_names = find_all_linear_names(model)
    model = get_peft_model(model, linear_names, rank=16, alpha=2.0, dropout=0.0)

    log.info("loading trained adapters from %s", ADAPTER_PATH)
    adapter_weights = load_mlx(str(ADAPTER_PATH))

    # Load weights into model
    loaded = 0
    for name, param in mlx.utils.tree_flatten(model.trainable_parameters()):
        if name in adapter_weights:
            # The adapter was saved as float16; load it
            param_shape = param.shape
            loaded_param = adapter_weights[name]
            if loaded_param.shape == param_shape:
                # Set the parameter via tree_unflatten approach
                loaded += 1

    # Use model.update() to load the weights
    model.update(mlx.utils.tree_unflatten(list(adapter_weights.items())))
    mx.eval(model.parameters())
    log.info("loaded %d adapter parameters", len(adapter_weights))

    return model, processor


def evaluate(model, processor):
    """Run evaluation using mlx_vlm.generate() for proper generation."""
    import datasets

    log.info("loading test data")
    ds = datasets.load_dataset(str(DATA_DIR), split="test")
    if len(ds) > MAX_SAMPLES:
        ds = ds.shuffle(seed=42).select(range(MAX_SAMPLES))
    log.info("evaluating on %d samples", len(ds))

    predictions = []
    references = []
    t0 = time.time()

    for i, sample in enumerate(ds):
        img = sample["image"]
        ref = normalize_bengali(sample["text"])

        # Save image to temp file for mlx_vlm.generate
        tmp_img = Path("/tmp/ocr_eval_tmp.png")
        img.save(str(tmp_img))

        try:
            # CRITICAL: must apply_chat_template to inject <|image|> tokens.
            # Without this, pixel_values are computed but scattered into zero
            # positions and the model ignores images entirely (codex fix 2026-04-20).
            formatted_prompt = apply_chat_template(
                processor, model.config, PROMPT, num_images=1
            )
            result = generate(
                model, processor,
                prompt=formatted_prompt,
                image=str(tmp_img),
                max_tokens=128,
                temp=0.0,
            )
            pred = result.text if hasattr(result, 'text') else str(result)
            pred = strip_model_formatting(normalize_bengali(pred))
        except Exception as e:
            log.warning("sample %d failed: %s", i, e)
            pred = ""

        predictions.append(pred)
        references.append(ref)

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            log.info("eval: %d/%d (%.1fs)", i + 1, len(ds), elapsed)

    elapsed = time.time() - t0
    metrics = evaluate_batch(predictions, references)

    # Strip per-sample arrays for compact output
    for key in ["cer_per_sample", "wer_per_sample", "ger_per_sample"]:
        metrics.pop(key, None)
    metrics["elapsed_s"] = round(elapsed, 2)
    metrics["model"] = "gemma4-e4b-lora-bengali-ocr"
    metrics["adapter"] = str(ADAPTER_PATH)

    log.info("CER=%.4f WER=%.4f GER=%.4f corpus_CER=%.4f (%.1fs)",
             metrics["cer_mean"], metrics["wer_mean"], metrics["ger_mean"],
             metrics["cer_corpus"], elapsed)

    # Save
    out_path = RESULTS_DIR / "finetuned_eval_v2.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    log.info("results saved to %s", out_path)

    # Also save some sample predictions for inspection
    samples_path = RESULTS_DIR / "eval_samples.json"
    sample_data = [
        {"reference": references[i], "prediction": predictions[i]}
        for i in range(min(20, len(predictions)))
    ]
    with open(samples_path, "w") as f:
        json.dump(sample_data, f, indent=2, ensure_ascii=False)
    log.info("sample predictions saved to %s", samples_path)

    return metrics


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    log.info("=== Loading model with LoRA adapters ===")
    model, processor = load_model_with_lora()

    log.info("=== Evaluating ===")
    metrics = evaluate(model, processor)

    log.info("=== Comparison ===")
    log.info("EasyOCR baseline: CER=0.153 (corpus)")
    log.info("Tesseract baseline: CER=0.722 (corpus)")
    log.info("Fine-tuned Gemma 4: CER=%.3f (corpus)", metrics["cer_corpus"])

    if metrics["cer_corpus"] < 0.153:
        log.info("BEATS EasyOCR baseline!")
    elif metrics["cer_corpus"] < 0.722:
        log.info("Beats Tesseract but not EasyOCR")
    else:
        log.info("Worse than Tesseract — needs investigation")


if __name__ == "__main__":
    main()
