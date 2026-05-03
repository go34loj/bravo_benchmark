# 3.1 Scene Sufficiency: Inputs / Outputs by Notebook

This document describes inputs and outputs for notebooks in:
`eval/3_compliance_reasoning/3_1_scene_sufficiency`.

## Notebooks in This Section

1. `3_1_Eval_Claude_Opus_4_6.ipynb`
2. `3_1_Eval_Gemini.ipynb`
3. `3_1_Eval_GPT_5_2.ipynb`
4. `3_1_Eval_yes_no_Analysis.ipynb`

---

## Overall Run Order

1. Run the three inference notebooks first (Claude, Gemini, GPT-5.2).
2. Each inference notebook reads `dataset/3_compliance_reasoning.csv` and keeps only `template_id == "39"`.
3. Each inference notebook writes model answers (`with_answers`) and preview files.
4. Run `3_1_Eval_yes_no_Analysis.ipynb` to build final summary tables (`csv` / `xlsx`).

---

## Shared Prerequisites for Inference Notebooks

- Working directory: `eval/3_compliance_reasoning/3_1_scene_sufficiency`.
- Source dataset: `dataset/3_compliance_reasoning.csv`.
- Template filter: `template_id == "39"`.
- OpenRouter key:
  - loaded from `eval.env` / `.env` via `python-dotenv`,
  - read from env var `OPENROUTER_API_KEY`.

Minimum expected input columns:

- `generated_question_id`
- `template_id`
- `question_text`
- `file_path`
- `ground_truth_answer`

---

## Why Inference Notebooks Are Separate

Inference notebooks are split by model because:

- request payload details differ by provider/model;
- retries and resume are easier per model;
- each model has isolated `with_answers` and raw `jsonl` audit files.

`3_1_Eval_yes_no_Analysis.ipynb` is separate because it only aggregates model outputs and computes metrics.

---

## Notebook: `3_1_Eval_Claude_Opus_4_6.ipynb`

### Purpose

Runs section 3.1 inference with Claude Opus 4.6 through OpenRouter.

### Inputs

- `dataset/3_compliance_reasoning.csv` (filtered to `template_id == "39"`)
- `OPENROUTER_API_KEY` from `eval.env/.env`

### Outputs

- Main output:
  - `3_1_with_answers_claude_opus_4_6.csv`
- Raw log:
  - `3_1_with_answers_claude_opus_4_6_raw.jsonl`
- Preview:
  - `3_1_preview_claude_opus_4_6.csv`
  - `3_1_preview_claude_opus_4_6_raw.jsonl`

### Preview Logic

- `PREVIEW_N = 6` random rows from filtered input;
- rows already present in preview are skipped (`generated_question_id` / `row_index`);
- new preview rows are appended to existing preview files.

---

## Notebook: `3_1_Eval_Gemini.ipynb`

### Purpose

Runs section 3.1 inference with Gemini through OpenRouter.

### Inputs

- `dataset/3_compliance_reasoning.csv` (filtered to `template_id == "39"`)
- `OPENROUTER_API_KEY` from `eval.env/.env`

### Outputs

- Main output:
  - `3_1_with_answers_gemini.csv`
- Raw log:
  - `3_1_with_answers_gemini_raw.jsonl`
- Preview:
  - `3_1_preview_gemini.csv`
  - `3_1_preview_gemini_raw.jsonl`
- Extra checkpoint file used by one resume block:
  - `3_1_with_answers_gemini_V1.csv`

### Preview Logic

Same pattern as Claude:

- `PREVIEW_N = 6`;
- sample from filtered (`template_id == "39"`) rows;
- append-only behavior for preview files.

---

## Notebook: `3_1_Eval_GPT_5_2.ipynb`

### Purpose

Runs section 3.1 inference with GPT-5.2 through OpenRouter.

### Inputs

- `dataset/3_compliance_reasoning.csv` (filtered to `template_id == "39"`)
- `OPENROUTER_API_KEY` from `eval.env/.env`

### Outputs

- Main output:
  - `3_1_with_answers_gpt_5_2.csv`
- Raw log:
  - `3_1_with_answers_gpt_5_2_raw.jsonl`
- Preview:
  - `3_1_preview_gpt_5_2.csv`
  - `3_1_preview_gpt_5_2_raw.jsonl`

### Preview Logic

Same pattern as Claude/Gemini:

- `PREVIEW_N = 6`;
- sample from filtered (`template_id == "39"`) rows;
- append-only behavior for preview files.

---

## Notebook: `3_1_Eval_yes_no_Analysis.ipynb`

### Purpose

Computes section 3.1 yes/no metrics and exports final summary tables.

### Inputs

- `3_1_with_answers_claude-opus_4_6.csv`
- `3_1_with_answers_gemini.csv`
- `3_1_with_answers_gpt-5_2.csv`

The notebook does not call OpenRouter for metric computation.

### Outputs

- `3_1_yes_no_summary.csv`
- `3_1_yes_no_summary_ambiguity_overall.csv`
- `3_1_yes_no_summary.xlsx`

### Metrics Scope in Current Notebook

- ambiguity split (`Ambiguity=yes` / `Ambiguity=no`);
- overall metrics;
- parent rule vs atomic rule splits.

---

## Current-State Notes

1. Inference and analysis naming are not fully uniform:
   inference notebooks currently write `3_1_with_answers_claude_opus_4_6.csv` and `3_1_with_answers_gpt_5_2.csv`,
   while analysis reads legacy names `3_1_with_answers_claude-opus_4_6.csv` and `3_1_with_answers_gpt-5_2.csv`.
2. Practical step before analysis:
   rename/copy outputs to the filenames expected by `3_1_Eval_yes_no_Analysis.ipynb`.
3. Some notebook output-history cells still show old paths/filenames from earlier runs.

---

## Minimal File Set for Full 3.1 Evaluation

1. `dataset/3_compliance_reasoning.csv`
2. `3_1_Eval_Claude_Opus_4_6.ipynb` -> `3_1_with_answers_claude_opus_4_6.csv`
3. `3_1_Eval_Gemini.ipynb` -> `3_1_with_answers_gemini.csv`
4. `3_1_Eval_GPT_5_2.ipynb` -> `3_1_with_answers_gpt_5_2.csv`
5. Rename/copy `3_1_with_answers_claude_opus_4_6.csv` -> `3_1_with_answers_claude-opus_4_6.csv`.
6. Rename/copy `3_1_with_answers_gpt_5_2.csv` -> `3_1_with_answers_gpt-5_2.csv`.
7. `3_1_Eval_yes_no_Analysis.ipynb` -> `3_1_yes_no_summary.csv`, `3_1_yes_no_summary_ambiguity_overall.csv`, `3_1_yes_no_summary.xlsx`
