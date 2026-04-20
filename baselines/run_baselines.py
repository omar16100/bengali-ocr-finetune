"""
Run Tesseract and EasyOCR baselines on Bengali OCR test data.
Outputs CER/WER/GER metrics to results/baselines.json.
"""

import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.metrics import evaluate_batch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler("baselines/baseline.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

DATA_DIR = Path("data/bengali-ocr-synthetic")
RESULTS_DIR = Path("results")


def load_test_data(max_samples: int = 500) -> list[dict]:
    """Load test split from the bengali-ocr-synthetic dataset.
    Format: HF dataset with 'image' (PIL) and 'text' (string) columns."""
    try:
        import datasets
        ds = datasets.load_dataset(str(DATA_DIR), split="test")
        log.info("loaded %d test samples", len(ds))
    except Exception as e:
        log.error("failed to load dataset: %s", e)
        return []

    samples = []
    for item in ds:
        if len(samples) >= max_samples:
            break
        text = (item.get("text") or "").strip()
        image = item.get("image")
        if text and image is not None:
            samples.append({"image": image, "reference": text})
    log.info("prepared %d samples with reference text", len(samples))
    return samples


def run_tesseract(samples: list[dict]) -> dict:
    """Run Tesseract OCR on samples. Requires pytesseract + tesseract-ocr with ben lang pack."""
    try:
        import pytesseract
        from PIL import Image
    except ImportError:
        log.error("pytesseract not installed: pip install pytesseract Pillow")
        return {"error": "pytesseract not installed"}

    predictions = []
    references = []
    t0 = time.time()

    for i, sample in enumerate(samples):
        img = sample["image"]
        if isinstance(img, (str, Path)):
            img = Image.open(img)
        try:
            text = pytesseract.image_to_string(img, lang="ben")
            predictions.append(text.strip())
        except Exception as e:
            log.warning("tesseract failed on sample %d: %s", i, e)
            predictions.append("")
        references.append(sample["reference"])

        if (i + 1) % 50 == 0:
            log.info("tesseract: %d/%d", i + 1, len(samples))

    elapsed = time.time() - t0
    metrics = evaluate_batch(predictions, references)
    metrics["engine"] = "tesseract"
    metrics["elapsed_s"] = round(elapsed, 2)
    metrics["samples_per_sec"] = round(len(samples) / elapsed, 2) if elapsed > 0 else 0
    log.info(
        "tesseract: CER=%.4f WER=%.4f GER=%.4f (%d samples in %.1fs)",
        metrics["cer_mean"], metrics["wer_mean"], metrics["ger_mean"],
        len(samples), elapsed,
    )
    return metrics


def run_easyocr(samples: list[dict]) -> dict:
    """Run EasyOCR on samples."""
    try:
        import easyocr
        from PIL import Image
        import numpy as np
    except ImportError:
        log.error("easyocr not installed: pip install easyocr")
        return {"error": "easyocr not installed"}

    reader = easyocr.Reader(["bn"], gpu=False)
    predictions = []
    references = []
    t0 = time.time()

    for i, sample in enumerate(samples):
        img = sample["image"]
        if isinstance(img, (str, Path)):
            img = Image.open(img)
        try:
            img_np = np.array(img)
            result = reader.readtext(img_np, detail=0)
            text = " ".join(result)
            predictions.append(text.strip())
        except Exception as e:
            log.warning("easyocr failed on sample %d: %s", i, e)
            predictions.append("")
        references.append(sample["reference"])

        if (i + 1) % 50 == 0:
            log.info("easyocr: %d/%d", i + 1, len(samples))

    elapsed = time.time() - t0
    metrics = evaluate_batch(predictions, references)
    metrics["engine"] = "easyocr"
    metrics["elapsed_s"] = round(elapsed, 2)
    metrics["samples_per_sec"] = round(len(samples) / elapsed, 2) if elapsed > 0 else 0
    log.info(
        "easyocr: CER=%.4f WER=%.4f GER=%.4f (%d samples in %.1fs)",
        metrics["cer_mean"], metrics["wer_mean"], metrics["ger_mean"],
        len(samples), elapsed,
    )
    return metrics


def run_paddleocr(samples: list[dict]) -> dict:
    """Run PaddleOCR (traditional pipeline, not VLM) on samples.
    Uses the Bengali language model for text recognition."""
    try:
        from paddleocr import PaddleOCR
        from PIL import Image
        import numpy as np
    except ImportError:
        log.error("paddleocr not installed: pip install paddleocr paddlepaddle")
        return {"error": "paddleocr not installed"}

    # PaddleOCR uses "bn" not "bengali" for Bengali
    ocr = PaddleOCR(use_angle_cls=True, lang="bn", show_log=False)
    predictions = []
    references = []
    t0 = time.time()

    for i, sample in enumerate(samples):
        img = sample["image"]
        if isinstance(img, (str, Path)):
            img = Image.open(img)
        try:
            img_np = np.array(img)
            result = ocr.ocr(img_np, cls=True)
            lines = []
            if result and result[0]:
                for line in result[0]:
                    if line and len(line) >= 2:
                        lines.append(line[1][0])
            text = " ".join(lines)
            predictions.append(text.strip())
        except Exception as e:
            log.warning("paddleocr failed on sample %d: %s", i, e)
            predictions.append("")
        references.append(sample["reference"])

        if (i + 1) % 50 == 0:
            log.info("paddleocr: %d/%d", i + 1, len(samples))

    elapsed = time.time() - t0
    metrics = evaluate_batch(predictions, references)
    metrics["engine"] = "paddleocr"
    metrics["elapsed_s"] = round(elapsed, 2)
    metrics["samples_per_sec"] = round(len(samples) / elapsed, 2) if elapsed > 0 else 0
    log.info(
        "paddleocr: CER=%.4f WER=%.4f GER=%.4f (%d samples in %.1fs)",
        metrics["cer_mean"], metrics["wer_mean"], metrics["ger_mean"],
        len(samples), elapsed,
    )
    return metrics


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    samples = load_test_data(max_samples=200)
    if not samples:
        log.error("no test data loaded, aborting")
        return

    results = {}
    out_path = RESULTS_DIR / "baselines.json"

    engines = [
        ("tesseract", run_tesseract),
        ("easyocr", run_easyocr),
        ("paddleocr", run_paddleocr),
    ]
    for name, fn in engines:
        log.info("=== Running %s baseline ===", name)
        try:
            result = fn(samples)
            for key in ["cer_per_sample", "wer_per_sample", "ger_per_sample"]:
                result.pop(key, None)
            results[name] = result
        except Exception as e:
            log.error("%s failed: %s", name, e)
            results[name] = {"error": str(e)}
        # save after each engine so partial results survive crashes
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)

    log.info("results written to %s", out_path)


if __name__ == "__main__":
    main()
