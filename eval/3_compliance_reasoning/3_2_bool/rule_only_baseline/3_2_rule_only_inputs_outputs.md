# 3.2 Rule-Only Baseline: Inputs / Outputs

This document reflects the current state of files in:
`eval/3_compliance_reasoning/3_2_bool/rule_only_baseline`.

## Files in This Folder

1. `3_2_to_3_5_Rule-only_Claude_Opus.ipynb`
2. `3_2_to_3_5_rule_only_gemini_3_pro_prev.ipynb`
3. `3_2_to_3_5_rule-only_GPT_5_2.ipynb`
4. `3_2_to_3_5_rule_only_analysis.py`
5. `3_2_to_3_5_(2)_rule_onl_subset.csv`

---

## Overall Run Order

1. Run model notebooks (Claude / Gemini / GPT) to generate predictions.
2. Run `3_2_to_3_5_rule_only_analysis.py` to merge outputs and calculate metrics.

---

## Shared Inference Setup

- Input dataset used by all model notebooks:
  - `C:\Users\ritaMZ\PycharmProjects\bravo_benchmark\eval\3_compliance_reasoning\3_2_bool\rule_only_baseline\3_2_to_3_5_(2)_rule_onl_subset.csv`
- API key source:
  - `eval.env` (`OPENROUTER_API_KEY`, loaded via `python-dotenv`)
- Core columns:
  - `generated_question_id`, `question_text`, `figure_path`, `correct_answer`

---

## Notebook: `3_2_to_3_5_Rule-only_Claude_Opus.ipynb`

### Main outputs

- `3_2_rule_only_with_answers_claude_opus_4_6.csv`
- `3_2_rule_only_with_answers_claude_opus_4_6_raw.jsonl`

### Preview / merged outputs

- `3_2_rule_only_preview_claude_opus_4_6.csv`
- `3_2_rule_only_with_answers_claude_opus_4_6_merged80.csv`

---

## Notebook: `3_2_to_3_5_rule_only_gemini_3_pro_prev.ipynb`

### Main outputs

- `3_2_rule_only_gemini_3_pro_prev.csv`
- `3_2_rule_only_gemini_3_pro_prev_raw.jsonl`

### Preview / clean outputs

- `3_2_rule_only_preview_gemini_3_pro_prev.csv`
- `3_2_rule_only_preview_gemini_3_pro_prev_raw.jsonl`
- `3_2_rule_only_clean_gemini_3_pro_prev.csv`

---

## Notebook: `3_2_to_3_5_rule-only_GPT_5_2.ipynb`

### Main outputs

- `3_2_rule_only_with_answers_gpt_5_2.csv`
- `3_2_rule_only_with_answers_gpt_5_2_raw.jsonl`

### Preview / clean outputs

- `3_2_rule_only_preview_gpt_5_2.csv`
- `3_2_rule_only_preview_gpt_5_2_raw.jsonl`
- `3_2_rule_only_clean_gpt_5_2.csv`

---

## Script: `3_2_to_3_5_rule_only_analysis.py`

### Short description (3 sentences)

The script builds a consolidated rule-only evaluation table from the base dataset and model output tables.  
It normalizes predicted yes/no labels, resolves duplicate answers via ranking logic, and computes binary metrics (accuracy/F1/baselines) for key slices.  
It then exports merged intermediate tables and a final metrics table for reporting.

### Default inputs (current script defaults)

- `--dataset`: `3_2_to_3_5_(2)_rule_onl_subset.csv`
- `--claude-merged-v1`: `3_2_to_3_5_rule-only_claude-opus-4.6_merged80_V1.csv`
- `--claude-main`: `3_2_to_3_5_rule-only_claude-opus-4.6.csv`
- `--claude-preview`: `3_2_to_3_5_preview_rule_onl_claude-opus-4.6.csv`
- `--gpt-main`: `3_2_to_3_5_rule-only_gpt-5_2.csv`
- `--gemini-main`: `3_2_to_3_5_clean_rule-only_gemini.csv`
- `--gemini-preview`: `3_2_to_3_5_preview_rule-only_gemini.csv`

### Default outputs (current script defaults)

- `--claude-merged-out`: `3_2_to_3_5_rule-only_claude-opus-4.6_merged80.csv`
- `--gemini-merged-out`: `3_2_to_3_5_rule-only_gemini_3_pro_prev_merged80.csv`
- `--metrics-out`: `3_2_to_3_5_rule_only_baseline_metrics.csv`

### Important note

The analysis script currently uses legacy `3_2_to_3_5_*` filename defaults and requires adaptation/unification to the newer `3_2_rule_only*` naming used by the notebooks.
