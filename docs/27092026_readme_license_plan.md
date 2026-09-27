# Plan: README, license, docs and CI (27 Sep 2026)

Purpose: record of the 27 Sep 2026 maintenance pass. Status: done. Last updated 27 Sep 2026.

## Goal

The repository had no README and no LICENSE. Add both, with every number in the README traced to a committed file, and add the docs index, C4 model and a minimal CI workflow.

## Scope

- `README.md`: models, dataset, training setup and results, built only from the scripts, `results/*.json`, `docs/decision_log.md` and live Hugging Face metadata (dated).
- `LICENSE`: MIT, copyright 2026 (first commit 20 Apr 2026), holder omar shabab.
- `docs/index.md`, `docs/c4model.md`, this plan.
- `.github/workflows/ci.yml`: uv plus pytest for `tests/` (the only tests).
- Model paths: replace hardcoded local directories with `GEMMA_MODEL_PATH` / `PADDLEOCR_MODEL_PATH`, defaulting to the Hugging Face ids named in the docs.
- Remove absolute local paths from `docs/decision_log.md` and `todo.md`.

Out of scope: re-running training or evaluation, changing `results/`, rewriting history.

## Decisions

- Results table values are the JSON keys rounded to 3 decimals, each row naming its source file and the script that writes it.
- The Gemma results are included but labelled invalid, as `docs/decision_log.md` (Decisions 12 to 14) records. The 10-sample Gemma check after the image-token fix is not in the table because its output was not saved.
- `results/paddleocr_zeroshot_eval.json` has no producing script in the repo; the README says so.
- The baselines and the model evaluations use different 200-row test subsets (first 200 in order vs `shuffle(seed=42)`). Verified by matching the saved sample references against the dataset; the subsets share 19 distinct texts. The README states this instead of presenting a like-for-like comparison.
- CI runs `uv run --no-project --with pytest pytest tests/` because the repo has no `pyproject.toml`, and the tested module uses only the standard library.

## Status

- Done: all files above; 21 metrics tests pass locally; dataset download command and split sizes (27,007 / 2,993) verified.
- Not verified: training, evaluation and baseline scripts were not re-run (they need MLX models and, for Gemma, patched mlx-vlm).
