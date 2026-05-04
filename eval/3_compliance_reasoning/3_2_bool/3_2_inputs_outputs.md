# 3.2 Compliance Bool: Inputs / Outputs by Notebook

This document describes inputs and outputs for:
`eval/3_compliance_reasoning/3_2_bool`.

## Notebooks in This Section

1. `3_2_Eval_Claude_Opus.ipynb`
2. `3_2_Eval_gemini_3_pro_prev.ipynb`
3. `3_2_Eval_GPT_5_2.ipynb`
4. `3_2_Eval_yes_no_Analysis.ipynb`

---

## Overall Run Order

1. Run inference notebooks first (Claude, Gemini, GPT-5.2).
2. Each inference notebook reads `dataset/3_compliance_reasoning.csv` and filters to `template_id != "39"`.
3. Each inference notebook exports `with_answers` and preview files.
4. Run `3_2_Eval_yes_no_Analysis.ipynb` to produce final summary outputs.

---

## Shared Prerequisites for Inference Notebooks

- Working directory: `eval/3_compliance_reasoning/3_2_bool`.
- Input dataset: `dataset/3_compliance_reasoning.csv`.
- Slice for section 3.2: `template_id != "39"`.
- API key source:
  - loaded from `eval.env` / `.env` via `python-dotenv`,
  - fallback: environment variable `OPENROUTER_API_KEY`.

Minimum expected input columns:

- `generated_question_id`
- `template_id`
- `question_text`
- `file_path`
- `figure_path`
- `ground_truth_answer`

---

## Notebook: `3_2_Eval_Claude_Opus.ipynb`

### Purpose

Runs section 3.2 inference with Claude Opus via OpenRouter.

### Inputs

- `dataset/3_compliance_reasoning.csv` (`template_id != "39"`)
- `OPENROUTER_API_KEY` from `eval.env/.env`

### Outputs

- main:
  - `3_2_with_answers_claude_opus_4_6.csv`
  - `3_2_with_answers_claude_opus_4_6_raw.jsonl`
- preview:
  - `3_2_preview_claude_opus_4_6.csv`
- clean/resume file:
  - `3_2_clean_claude_opus_4_6.csv`

### Preview behavior

- `PREVIEW_N = 5`, `PREVIEW_SEED = 60`
- sampled from filtered data (`template_id != "39"`)
- can skip rows already present in clean/preview files

---

## Notebook: `3_2_Eval_gemini_3_pro_prev.ipynb`

### Purpose

Runs section 3.2 inference with Gemini 3 Pro Preview via OpenRouter.

### Inputs

- `dataset/3_compliance_reasoning.csv` (`template_id != "39"`)
- `OPENROUTER_API_KEY` from `eval.env`

### Outputs

- main:
  - `3_2_with_answers_gemini_3_pro_prev.csv`
  - `3_2_with_answers_gemini_3_pro_prev_raw.jsonl`
- preview:
  - `3_2_preview_gemini_3_pro_prev.csv`
  - `3_2_preview_gemini_3_pro_prev_raw.jsonl`
- clean:
  - `3_2_clean_gemini_3_pro_prev.csv`

### Preview behavior

- `PREVIEW_N = 5`, `PREVIEW_SEED = 60`
- sampled from filtered data (`template_id != "39"`)
- can skip rows already present in clean/preview files

---

## Notebook: `3_2_Eval_GPT_5_2.ipynb`

### Purpose

Runs section 3.2 inference with GPT-5.2 via OpenRouter.

### Inputs

- `dataset/3_compliance_reasoning.csv` (`template_id != "39"`)
- `OPENROUTER_API_KEY` from `eval.env`

### Outputs

- main:
  - `3_2_with_answers_gpt_5_2.csv`
  - `3_2_with_answers_gpt_5_2_raw.jsonl`
- preview:
  - `3_2_preview_gpt_5_2.csv`
  - `3_2_preview_gpt_5_2_raw.jsonl`
- clean:
  - `3_2_clean_gpt_5_2.csv`

### Preview behavior

- `PREVIEW_N = 5`, `PREVIEW_SEED = 60`
- sampled from filtered data (`template_id != "39"`)
- can skip rows already present in clean/preview files

---

## Notebook: `3_2_Eval_yes_no_Analysis.ipynb`

### Purpose

Computes yes/no metrics across Claude/Gemini/GPT outputs and exports final summary tables.

### Inputs

- `3_2_clean_claude_opus_4_6.csv`
- `3_2_clean_gemini_3_pro_prev.csv`
- `3_2_clean_gpt_5_2.csv`

No OpenRouter API calls are required in this notebook.

### Final outputs

- `3_2_yesno_summary.csv`
- `3_2_yesno_summary.xlsx`

---

## Current-State Notes

1. Inference notebooks are aligned to:
   - `dataset/3_compliance_reasoning.csv`,
   - `template_id != "39"`,
   - `3_2_*` naming.
2. `3_2_Eval_yes_no_Analysis.ipynb` also uses `3_2_*` outputs for final summary.
3. The first “Update ground truth” block in analysis still contains old absolute paths from earlier local runs (`WebstormProjects/...`) in its setup/history cell; this does not affect the final summary-export section.

---

## Minimal File Set for Running 3.2

1. `dataset/3_compliance_reasoning.csv`
2. `3_2_Eval_Claude_Opus.ipynb`
3. `3_2_Eval_gemini_3_pro_prev.ipynb`
4. `3_2_Eval_GPT_5_2.ipynb`
5. `3_2_Eval_yes_no_Analysis.ipynb` -> `3_2_yesno_summary.csv`, `3_2_yesno_summary.xlsx`
