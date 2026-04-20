"""
Bengali OCR evaluation metrics: CER, WER, and grapheme-level error rate.
Uses Levenshtein distance for character and word error rate computation.

Codex-reviewed 2026-04-20: fixed grapheme splitting for ঁ/ং/ঃ,
added NFC normalization, added corpus-level CER/WER.
"""

import logging
import re
import unicodedata
from typing import Optional

log = logging.getLogger(__name__)


def normalize_bengali(text: str) -> str:
    """Normalize Bengali text before CER/GER comparison.
    - NFC normalization (collapses equivalent Unicode representations:
      decomposed nukta forms ড়/ঢ়/য়, decomposed vowel signs ো/ৌ)
    - Collapse multiple whitespace to single space
    - Strip leading/trailing whitespace
    Does NOT normalize ঁ/ং/ঃ into each other (real distinctions)."""
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def levenshtein_distance(s1: str, s2: str) -> int:
    """Compute the Levenshtein (edit) distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    prev_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row
    return prev_row[-1]


def cer(prediction: str, reference: str) -> float:
    """Character Error Rate: edit_distance(pred, ref) / len(ref).
    Returns 0.0 for perfect match. Can exceed 1.0 if prediction is much
    longer than reference."""
    if not reference:
        return 0.0 if not prediction else 1.0
    dist = levenshtein_distance(prediction, reference)
    return dist / len(reference)


def wer(prediction: str, reference: str) -> float:
    """Word Error Rate: edit_distance on word tokens."""
    pred_words = prediction.split()
    ref_words = reference.split()
    if not ref_words:
        return 0.0 if not pred_words else 1.0
    dist = levenshtein_distance_list(pred_words, ref_words)
    return dist / len(ref_words)


def levenshtein_distance_list(s1: list, s2: list) -> int:
    """Levenshtein distance on lists of tokens (for WER)."""
    if len(s1) < len(s2):
        return levenshtein_distance_list(s2, s1)
    if len(s2) == 0:
        return len(s1)

    prev_row = list(range(len(s2) + 1))
    for i, t1 in enumerate(s1):
        curr_row = [i + 1]
        for j, t2 in enumerate(s2):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (t1 != t2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row
    return prev_row[-1]


def split_bengali_graphemes(text: str) -> list[str]:
    """Split Bengali text into grapheme clusters.
    A grapheme cluster is a base consonant/vowel + any combining marks
    (matras, hasanta, nukta). This gives linguistically meaningful units
    for Bengali OCR evaluation.

    Uses Unicode category-based splitting:
    - Base characters: Bengali letters (U+0985-U+09B9, U+09CE, U+09DC-U+09DF)
    - Combining marks: U+09BE-U+09CC, U+09CD (hasanta), U+09D7, U+09BC (nukta)
    """
    graphemes = []
    current = ""
    prev_was_hasanta = False
    for ch in text:
        cp = ord(ch)
        is_combining = (
            (0x09BE <= cp <= 0x09CC)  # dependent vowel signs (matras)
            or cp == 0x09CD            # hasanta (virama)
            or cp == 0x09D7            # au length mark
            or cp == 0x09BC            # nukta
            or cp == 0x0981            # candrabindu (ঁ) — nasalization mark
            or cp == 0x0982            # anusvara (ং) — nasal mark
            or cp == 0x0983            # visarga (ঃ) — aspiration mark
            or cp == 0x200D            # ZWJ — zero-width joiner (part of conjunct)
            or cp == 0x200C            # ZWNJ — zero-width non-joiner
        )
        # A consonant after hasanta is part of the same conjunct cluster
        is_post_hasanta_consonant = prev_was_hasanta and not is_combining

        if (is_combining or is_post_hasanta_consonant) and current:
            current += ch
        else:
            if current:
                graphemes.append(current)
            current = ch
        prev_was_hasanta = cp == 0x09CD
    if current:
        graphemes.append(current)
    return graphemes


def grapheme_error_rate(prediction: str, reference: str) -> float:
    """Grapheme-level error rate for Bengali text.
    Splits into grapheme clusters before computing Levenshtein distance.
    More linguistically meaningful than raw CER for Bengali conjuncts."""
    pred_graphemes = split_bengali_graphemes(prediction)
    ref_graphemes = split_bengali_graphemes(reference)
    if not ref_graphemes:
        return 0.0 if not pred_graphemes else 1.0
    dist = levenshtein_distance_list(pred_graphemes, ref_graphemes)
    return dist / len(ref_graphemes)


def strip_model_formatting(text: str) -> str:
    """Strip VLM output formatting artifacts before scoring.
    Removes code fences, backticks, and common assistant prefixes
    while preserving Bengali content."""
    text = re.sub(r"```[a-z]*\n?", "", text)
    text = text.replace("`", "")
    # strip common prefixes from VLM outputs
    for prefix in ["The text in the image reads:", "The Bengali text is:",
                    "Here is the extracted text:", "OCR result:"]:
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix):]
    return text.strip()


def evaluate_batch(predictions: list[str], references: list[str],
                   normalize: bool = True) -> dict:
    """Evaluate a batch of predictions against references.
    Returns both per-sample (macro) and corpus-level metrics.
    Applies NFC normalization + formatting strip by default."""
    assert len(predictions) == len(references), "predictions and references must be same length"

    cers, wers, gers = [], [], []
    total_char_edits, total_char_ref = 0, 0
    total_word_edits, total_word_ref = 0, 0

    for pred, ref in zip(predictions, references):
        if normalize:
            pred = strip_model_formatting(normalize_bengali(pred))
            ref = normalize_bengali(ref)
        else:
            pred = pred.strip()
            ref = ref.strip()

        # per-sample metrics
        cers.append(cer(pred, ref))
        wers.append(wer(pred, ref))
        gers.append(grapheme_error_rate(pred, ref))

        # corpus-level accumulators (edit distance / total ref length)
        total_char_edits += levenshtein_distance(pred, ref)
        total_char_ref += len(ref) if ref else 1
        pred_words, ref_words = pred.split(), ref.split()
        total_word_edits += levenshtein_distance_list(pred_words, ref_words)
        total_word_ref += len(ref_words) if ref_words else 1

    n = len(predictions)
    return {
        "n": n,
        # per-sample macro average (overweights short samples)
        "cer_mean": sum(cers) / n if n else 0,
        "wer_mean": sum(wers) / n if n else 0,
        "ger_mean": sum(gers) / n if n else 0,
        # corpus-level (stabler for mixed-length data)
        "cer_corpus": total_char_edits / total_char_ref if total_char_ref else 0,
        "wer_corpus": total_word_edits / total_word_ref if total_word_ref else 0,
        "cer_per_sample": cers,
        "wer_per_sample": wers,
        "ger_per_sample": gers,
    }
