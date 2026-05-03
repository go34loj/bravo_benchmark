# 1.1 Scene Perception: Inputs / Outputs by Notebook

This document describes, for each notebook in `eval/1_1_scene_perception`:

1. required inputs;
2. where inputs come from;
3. generated outputs (`csv` / `xlsx`);
4. files needed to run the pipeline.

## Shared prerequisites for inference notebooks (`Claude` / `Gemini` / `GPT`)

- Working directory: `eval/1_1_scene_perception`.
- Input dataset: `dataset/1_1_scene_perception.csv`.
- Environment file setup in `eval/`:
  - `eval/.env.template` (public template to copy/paste for users)
  - `eval/.env` (local private file, ignored by git)
- API key source:
  - `OPENROUTER_API_KEY` is loaded from `eval/.env` via `python-dotenv`.

Required package for env loading:

- `python-dotenv`

Recommended core input columns in `dataset/1_1_scene_perception.csv`:

- `generated_question_id`
- `question_text`
- `ground_truth_answer`

`1_1_Eval_Analysis.ipynb` does not call OpenRouter, so it does not require API key loading.

---

## Why notebooks are separated by model

We keep separate inference notebooks for `Claude`, `Gemini`, and `GPT` intentionally:

- model/provider-specific request parameters differ (for example reasoning/thinking controls and payload options);
- easier reruns and resume checkpoints per model without affecting other runs;
- cleaner audit trail: each model has its own `1_1_with_answers_*.csv` and raw `jsonl` log.

`1_1_Eval_Analysis.ipynb` is separate because it aggregates already-generated CSV outputs and computes final metrics.

---

## Notebook: `1_1_Eval_Claude_Opus_4_6.ipynb`

### Purpose

Runs inference for section 1.1 using Claude Opus 4.6 via OpenRouter.

### Inputs

- `dataset/1_1_scene_perception.csv`
- `../.env` (contains `OPENROUTER_API_KEY`)

### Outputs

- Main output CSV:
  - `1_1_with_answers_claude-opus_4_6.csv`
- Raw API log:
  - `1_1_with_answers_claude-opus_4_6_raw.jsonl`
- Preview sample outputs (6 random questions: 3 from `1_1_1` + 3 from `1_1_2`):
  - `1_1_preview_claude-opus_4_6.csv`
  - `1_1_preview_claude-opus_4_6_raw.jsonl`

### Output columns added/updated

- `model_answer`
- `openrouter_request_id`
- `error_reason` (when request/parsing fails)
- `bool_model_answer` (after post-processing)

---

## Notebook: `1_1_Eval_Gemini.ipynb`

### Purpose

Runs inference for section 1.1 using Gemini via OpenRouter.

### Inputs

- `dataset/1_1_scene_perception.csv`
- `../.env` (contains `OPENROUTER_API_KEY`)

### Outputs

- Main output CSV:
  - `1_1_with_answers_gemini.csv`
- Raw API log:
  - `1_1_with_answers_gemini_raw.jsonl`
- Preview sample outputs (6 random questions: 3 from `1_1_1` + 3 from `1_1_2`):
  - `1_1_preview_gemini.csv`
  - `1_1_preview_gemini_raw.jsonl`

### Note

For section 1.1, the expected output names are:

- `1_1_with_answers_gemini.csv`
- `1_1_with_answers_gemini_raw.jsonl`

---

## Notebook: `1_1_Eval_GPT_5_2.ipynb`

### Purpose

Runs inference for section 1.1 using GPT-5.2 via OpenRouter.

### Inputs

- `dataset/1_1_scene_perception.csv`
- `../.env` (contains `OPENROUTER_API_KEY`)

### Outputs

- Main output CSV:
  - `1_1_with_answers_gpt-5_2.csv`
- Raw API log:
  - `1_1_with_answers_gpt-5_2_raw.jsonl`
- Preview sample outputs (6 random questions: 3 from `1_1_1` + 3 from `1_1_2`):
  - `1_1_preview_gpt-5_2.csv`
  - `1_1_preview_gpt-5_2_raw.jsonl`

### Note

Legacy `1_2*` filename references were removed.  
For section 1.1, the expected output names are:

- `1_1_with_answers_gpt-5_2.csv`
- `1_1_with_answers_gpt-5_2_raw.jsonl`

---

## Notebook: `1_1_Eval_Analysis.ipynb`

### Purpose

Computes evaluation metrics and summary tables across boolean model outputs for section 1.1.

### Inputs

- `1_1_with_answers_claude-opus_4_6.csv`
- `1_1_with_answers_gemini.csv`
- `1_1_with_answers_gpt-5_2.csv`

No `.env` / API key is required for this notebook.

For semantic similarity metrics:

- `sentence-transformers`
- model: `paraphrase-multilingual-MiniLM-L12-v2`

### Outputs

- Summary CSV:
  - `1_1_scene_perception_summary.csv`
- Summary Excel:
  - `1_1_scene_perception_summary.xlsx`

### Minimum required columns in model-output CSVs

- `ground_truth_answer`
- `model_answer`

---

## Minimal file set to run full 1.1 pipeline

1. `eval/.env.template` (reference template)
2. `eval/.env` (local private key file with `OPENROUTER_API_KEY`)
3. `dataset/1_1_scene_perception.csv`
4. `1_1_Eval_Claude_Opus_4_6.ipynb` -> `1_1_with_answers_claude-opus_4_6.csv`
5. `1_1_Eval_Gemini.ipynb` -> `1_1_with_answers_gemini.csv`
6. `1_1_Eval_GPT_5_2.ipynb` -> `1_1_with_answers_gpt-5_2.csv`
7. `1_1_Eval_Analysis.ipynb` -> `1_1_scene_perception_summary.csv`, `1_1_scene_perception_summary.xlsx`

Final artifacts for section 1.1:

- `1_1_with_answers_*.csv` (one per model)
- `1_1_preview_*.csv` (preview sample outputs)
- `1_1_preview_*_raw.jsonl` (raw preview logs)
- `1_1_scene_perception_summary.csv`
- `1_1_scene_perception_summary.xlsx`
