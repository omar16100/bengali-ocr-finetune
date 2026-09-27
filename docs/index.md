# Documentation index: bengali-ocr-finetune

LoRA fine-tuning experiments for Bengali OCR with MLX and mlx-vlm. Start with the [README](../README.md) for models, dataset, results and how to run; this folder holds the reasoning and build records.

Last updated 27 Sep 2026.

## Conventions

- Dated docs (a plan, a decision, a point-in-time record): `DDMMYYYY_topic.md`, for example `27092026_readme_license_plan.md`. They are not rewritten after the fact; if the facts change, add a new dated doc and link it.
- Evergreen docs (kept current with the code): `topic.md`, for example `c4model.md`.
- Every new doc opens with its purpose, status and last-updated date.
- `c4model.md` is the architecture source of truth. Read it before an architecture change and update it for every change to containers, components, dependencies or data flows.
- Register every new doc in the table below.

## Categories

| Category | Required sections |
| --- | --- |
| Architecture | Context, Containers, Components, Data flows, Change log |
| Plan | Goal, Scope, Decisions, Status (plus Deviations when the work departed from the plan) |
| Research log | Context, dated decisions with evidence |

## Documents

| Path | Category | Description | Date |
| --- | --- | --- | --- |
| [c4model.md](c4model.md) | Architecture | Context, containers, components and data flows: dataset, MLX training scripts, evaluation harness, baselines, results files, tests and CI. Evergreen. | Created 27 Sep 2026 |
| [decision_log.md](decision_log.md) | Research log | Experiment plan and numbered decisions: framework choice, mlx-vlm NaN gradient patches, datasets, metrics, baselines, Gemma 4 E4B and PaddleOCR-VL-1.5 runs and why the Gemma results are invalid. | 19 to 20 Apr 2026 |
| [mlx_vlm_gemma4_fix_journey.md](mlx_vlm_gemma4_fix_journey.md) | Research log | The three mlx-vlm bugs behind NaN gradients in Gemma 4 vision training and how each was patched. | 19 to 20 Apr 2026 |
| [27092026_readme_license_plan.md](27092026_readme_license_plan.md) | Plan | Adding the README (numbers traced to `results/*.json`), MIT license, this index, the C4 model and a CI workflow for the metrics tests; parameterizing model paths. | 27 Sep 2026 |
