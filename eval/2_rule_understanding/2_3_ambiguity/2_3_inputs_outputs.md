# 2_3 Ambiguity: Inputs / Outputs

This document reflects the current state of files in:
`eval/2_rule_understanding/2_3_ambiguity`.

## Notebooks in This Section

1. `2_3_Eval_Claude_Opus.ipynb`
2. `2_3_Eval_gemini_3_pro_prev.ipynb`
3. `2_3_Eval_GPT_5_2.ipynb`

---

## Important Note About Analysis

For section `2_3`, there is **no separate shared analysis notebook**.
Analysis is done inside each model notebook and/or via the CSV exports produced by that notebook (including `summary_metrics` files).

---

## Common Inputs (Current State)

- Working directory: `eval/2_rule_understanding/2_3_ambiguity`
- Main dataset used by all three notebooks:
  - `dataset/2_rules_understanding.csv`
- Row filter:
  - `layer_id == "2_3"`

---

## 1) `2_3_Eval_Claude_Opus.ipynb`

### Inputs

- `dataset/2_rules_understanding.csv` (filtered to `layer_id == "2_3"`)

### Main Outputs

- `2_3_with_answers_claude-opus-4_6.csv`
- `2_3_with_answers_claude-opus-4_6_raw.jsonl`
- `2_3_clean_claude-opus-4_6.csv`

### Preview Outputs

- No dedicated preview table (`*_preview*.csv`) is currently saved.
- The notebook has a single-row preview check (`PREVIEW_ROW = 10`) without separate CSV export.

### Summary Metrics Output

- `results/summary_metrics_Claude_4_6.csv`

---

## 2) `2_3_Eval_gemini_3_pro_prev.ipynb`

### Inputs

- `dataset/2_rules_understanding.csv` (filtered to `layer_id == "2_3"`)

### Main Outputs

- `2_3_with_answers_gemini-3_1-pro-preview.csv`
- `2_3_gemini-3_1-pro-preview_raw.jsonl`
- `2_3_clean_gemini-3_1-pro-preview.csv`

### Preview Outputs

- The notebook has a random preview block (`PREVIEW_N = 12`), but no explicit `preview_*.csv` save in code.

### Summary Metrics Output

- `results/2_3_summary_metrics_gemini-3_1-pro-preview.csv`

---

## 3) `2_3_Eval_GPT_5_2.ipynb`

### Inputs

- `dataset/2_rules_understanding.csv` (filtered to `layer_id == "2_3"`)

### Main Outputs

- `2_3_with_answers_gpt_5_2.csv`
- `2_3_gpt_5_2_raw.jsonl`
- `2_3_clean_gpt-5_2.csv`

### Preview Outputs

- The notebook has a random preview block (`PREVIEW_N = 12`), but no explicit `preview_*.csv` save in code.

### Summary Metrics Output

- `results/2_3_summary_metrics_gpt-5_2.csv`

---

## Notebook Differences

The three notebooks are separated intentionally:

- They target different models (`Claude Opus 4.6`, `Gemini 3.1 Pro Preview`, `GPT-5.2`).
- Output file names are model-specific.
- Each notebook contains its own in-notebook evaluation and exports its own `summary_metrics` CSV.

Core flow is the same in all three:

1. Load `dataset/2_rules_understanding.csv`.
2. Filter to `layer_id == "2_3"`.
3. Run inference through OpenRouter.
4. Save answers (`with_answers` + raw `jsonl` + clean CSV).
5. Build and export model-specific summary metrics.

---

## Minimal File Set for Running 2_3

1. `dataset/2_rules_understanding.csv`
2. One or more model notebooks listed above
3. For metrics: corresponding `results/*summary_metrics*.csv` exported by each notebook
