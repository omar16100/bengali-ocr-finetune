"""Unit tests for OCR evaluation metrics."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.metrics import (
    levenshtein_distance,
    cer,
    wer,
    split_bengali_graphemes,
    grapheme_error_rate,
    evaluate_batch,
    normalize_bengali,
    strip_model_formatting,
)


def test_levenshtein_identical():
    assert levenshtein_distance("hello", "hello") == 0


def test_levenshtein_empty():
    assert levenshtein_distance("", "abc") == 3
    assert levenshtein_distance("abc", "") == 3


def test_levenshtein_insertion():
    assert levenshtein_distance("abc", "abcd") == 1


def test_levenshtein_substitution():
    assert levenshtein_distance("abc", "axc") == 1


def test_levenshtein_deletion():
    assert levenshtein_distance("abcd", "acd") == 1


def test_cer_perfect():
    assert cer("আমি", "আমি") == 0.0


def test_cer_empty_reference():
    assert cer("abc", "") == 1.0
    assert cer("", "") == 0.0


def test_cer_one_char_diff():
    # "আমি" (3 chars) vs "আমা" (3 chars) — 1 substitution
    result = cer("আমা", "আমি")
    assert abs(result - 1 / 3) < 0.01


def test_wer_perfect():
    assert wer("আমি বাংলায় গান গাই", "আমি বাংলায় গান গাই") == 0.0


def test_wer_one_word_diff():
    # 4 words, 1 wrong → 0.25
    assert wer("আমি বাংলায় গান করি", "আমি বাংলায় গান গাই") == 0.25


def test_bengali_grapheme_split_simple():
    # "বাং" = ব + া + ং → ["বাং"] — ং (anusvara) combines with preceding
    graphemes = split_bengali_graphemes("বাং")
    assert len(graphemes) == 1
    assert graphemes[0] == "বাং"


def test_bengali_grapheme_split_conjunct():
    # "ক্ষ" = ক + ্ + ষ → conjunct, should be ["ক্ষ"]
    graphemes = split_bengali_graphemes("ক্ষ")
    assert len(graphemes) == 1


def test_bengali_grapheme_split_triple_conjunct():
    # "ক্ষ্ণ" = ক + ্ + ষ + ্ + ণ → one cluster
    graphemes = split_bengali_graphemes("ক্ষ্ণ")
    assert len(graphemes) == 1


def test_bengali_grapheme_candrabindu():
    # "চাঁদ" = চা + ঁ + দ → ঁ combines with preceding: ["চাঁ", "দ"]
    graphemes = split_bengali_graphemes("চাঁদ")
    assert len(graphemes) == 2
    assert "ঁ" in graphemes[0]  # candrabindu attached to চা


def test_bengali_grapheme_visarga():
    # "দুঃখ" = দু + ঃ + খ → ঃ combines with preceding: ["দুঃ", "খ"]
    graphemes = split_bengali_graphemes("দুঃখ")
    assert len(graphemes) == 2
    assert "ঃ" in graphemes[0]


def test_normalize_bengali_nfc():
    # NFC should collapse decomposed forms
    import unicodedata
    text = unicodedata.normalize("NFD", "কো")
    normalized = normalize_bengali(text)
    assert normalized == unicodedata.normalize("NFC", "কো")


def test_strip_model_formatting():
    assert strip_model_formatting("```\nআমি\n```") == "আমি"
    assert strip_model_formatting("The text in the image reads: আমি") == "আমি"


def test_grapheme_error_rate_perfect():
    assert grapheme_error_rate("বাংলা", "বাংলা") == 0.0


def test_grapheme_error_rate_empty():
    assert grapheme_error_rate("abc", "") == 1.0
    assert grapheme_error_rate("", "") == 0.0


def test_evaluate_batch():
    preds = ["আমি", "বাংলা"]
    refs = ["আমি", "বাংলা"]
    result = evaluate_batch(preds, refs)
    assert result["n"] == 2
    assert result["cer_mean"] == 0.0
    assert result["wer_mean"] == 0.0
    assert result["ger_mean"] == 0.0


def test_evaluate_batch_with_errors():
    preds = ["আমি", "বাংলো"]  # second has 1 char diff
    refs = ["আমি", "বাংলা"]
    result = evaluate_batch(preds, refs)
    assert result["n"] == 2
    assert result["cer_mean"] > 0.0
    assert result["cer_per_sample"][0] == 0.0
    assert result["cer_per_sample"][1] > 0.0


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
