# 2_1-2 Rule Understanding: Inputs / Outputs

This document describes inputs and outputs for notebooks in:  
`eval/2_rule_understanding/2_1-2_rule_classific_decompos`.

## Notebooks in This Section

1. `2_1-2_Eval_Claude_Opus_4_6.ipynb`
2. `2_1-2_Eval_Gemini.ipynb`
3. `2_1-2_Eval_GPT_5_2.ipynb`
4. `2_1-2_Analysis.ipynb`

---

## Overall Run Order

1. Run the three inference notebooks first (Claude, Gemini, GPT-5.2).
2. Each notebook reads the shared dataset `dataset/2_rules_understanding.csv`, filters rows by `layer_id in {"2_1","2_2"}`, and writes its own answer files.
3. Then run `2_1-2_Analysis.ipynb`, which reads the three answer files and computes metrics/summaries.

---

## Common Prerequisites for Inference Notebooks

- Working directory: `eval/2_rule_understanding/2_1-2_rule_classific_decompos`.
- OpenRouter API key:
  - preferred: `eval/.env`
  - fallback: `eval.env`
  - fallback: environment variable `OPENROUTER_API_KEY`
- Input dataset: `dataset/2_rules_understanding.csv`.

Minimum important columns in input CSV:

- `generated_question_id`
- `question_text`
- `ground_truth_answer`
- `layer_id`

---

## 1) `2_1-2_Eval_Claude_Opus_4_6.ipynb`

### Purpose

Generates Claude Opus 4.6 answers for the 2_1-2 benchmark section.

### Inputs

- `dataset/2_rules_understanding.csv` (filtered to `layer_id` = `2_1` / `2_2`)
- OpenRouter API key (`eval/.env` or `eval.env`)

### Outputs

- `2_1-2_with_answers_claude-opus_4_6.csv`
- `2_1-2_with_answers_claude-opus_4_6_raw.jsonl`
- `2_1-2_preview_claude-opus_4_6.csv`
- `2_1-2_preview_claude-opus_4_6_raw.jsonl`

### How Preview Is Built

- Only questions from `layer_id` = `2_1` and `2_2` are eligible.
- Stratified sampling by layer:
  - `PREVIEW_LAYER_IDS = ["2_1", "2_2"]`
  - `PREVIEW_PER_LAYER = 2`
- Target preview size is up to 4 rows total (2 from `2_1`, 2 from `2_2`).
- If fewer than 2 rows are available in one layer, it takes the available minimum.
- On repeated runs, existing preview rows are skipped (by `generated_question_id` / `row_index`), and only new rows are appended.

---

## 2) `2_1-2_Eval_Gemini.ipynb`

### Purpose

Generates Gemini answers for the 2_1-2 benchmark section.

### Inputs

- `dataset/2_rules_understanding.csv` (filtered to `layer_id` = `2_1` / `2_2`)
- OpenRouter API key (`eval/.env` or `eval.env`)

### Outputs

- `2_1-2_with_answers_gemini.csv`
- `2_1-2_with_answers_gemini_raw.jsonl`
- `2_1-2_preview_gemini.csv`
- `2_1-2_preview_gemini_raw.jsonl`

### How Preview Is Built

Same logic as Claude:

- only `layer_id` = `2_1` / `2_2`
- 2 questions per layer
- up to 4 total questions
- no duplicate append on repeated runs

---

## 3) `2_1-2_Eval_GPT_5_2.ipynb`

### Purpose

Generates GPT-5.2 answers for the 2_1-2 benchmark section.

### Inputs

- `dataset/2_rules_understanding.csv` (filtered to `layer_id` = `2_1` / `2_2`)
- OpenRouter API key (`eval/.env` or `eval.env`)

### Outputs

- `2_1-2_with_answers_gpt-5_2.csv`
- `2_1-2_with_answers_gpt-5_2_raw.jsonl`
- `2_1-2_preview_gpt-5_2.csv`
- `2_1-2_preview_gpt-5_2_raw.jsonl`

### How Preview Is Built

Same logic as Claude/Gemini:

- only `layer_id` = `2_1` / `2_2`
- 2 questions per layer
- up to 4 total questions
- no duplicate append on repeated runs

---

## How Inference Notebooks Differ

`2_1-2_Eval_Claude_Opus_4_6.ipynb`, `2_1-2_Eval_Gemini.ipynb`, and `2_1-2_Eval_GPT_5_2.ipynb` differ mainly by:

- target OpenRouter model:
  - Claude: `anthropic/claude-opus-4.6`
  - Gemini: `google/gemini-3.1-pro-preview`
  - GPT: `openai/gpt-5.2`
- model-specific retry/parameter behavior
- output file suffixes (`claude-opus_4_6`, `gemini`, `gpt-5_2`)

The base pipeline logic is the same across all three.

---

## 4) `2_1-2_Analysis.ipynb`

### Purpose

Computes quality metrics for Claude, Gemini, and GPT-5.2; builds intermediate matching tables and final summary outputs.

### Inputs

- `2_1-2_with_answers_claude-opus_4_6.csv`
- `2_1-2_with_answers_gemini.csv`
- `2_1-2_with_answers_gpt-5_2.csv`

`OpenRouter API key` is not used in the analysis notebook.

### How Layer `2_2` Is Evaluated in This Notebook

- `2_2` rows are selected via `layer_id` routing (`df_rule_understanding` slice).
- Rule-classification matching tables (`all_pairwise`, `candidate_matches`, `gt_coverage_summary`) are for `2_1`, not `2_2`.
- `2_2` metrics are computed in the model-bundle stage (`_build_model_rule_understanding_bundle`) under the *rule picture reading* block.
- Metrics computed for `2_2`: `ROUGE-L (avg per rule)`, `Sentence-BERT similarity`, `BLEU`, and `n_questions`.
- Aggregation principle for `2_2`:
  - metrics are first computed per sample in `layer_id == "2_2"` (GT vs model answer),
  - then final `2_2` values in summary are averages across all `2_2` samples.
- `2_2` results are written into:
  - `2_1-2_rule_understanding_summary_table.csv`
  - `2_1-2_rule_understanding_summary_table.xlsx`

### Intermediate Analysis Outputs (2_1)

Detailed methodology and field-level explanation:  
[2_1-2_Rule_Classification_Eval.md](/C:/Users/ritaMZ/PycharmProjects/bravo_benchmark/eval/2_rule_understanding/2_1-2_rule_classific_decompos/2_1-2_Rule_Classification_Eval.md)

- `2_1-2_all_pairwise_similarity.csv`  
  Full pairwise `GT x Pred` matrix per question.  
  Used as a raw similarity audit layer and for traceability of candidate selection.

- `2_1-2_rule_classification_candidate_matches.csv`  
  Predicted-centered compressed candidate layer per `pred_rule_id` (selected/best match + adjudication fields).  
  Used as a supplementary diagnostic/control artifact, not as the mandatory final decision source.

- `2_1-2_gt_coverage_summary.csv`  
  GT-centered coverage table: for each GT rule, whether and how it is covered by predictions.  
  Used for coverage-based metrics and missing-rule analysis.

- `2_1-2_sample_level_metrics.csv`  
  Per-sample metrics table (`gt_total`, `gt_covered_count`, recall/rates, etc.).  
  Used for sample-level diagnostics and aggregation.

### Final Analysis Outputs

- `2_1-2_rule_understanding_summary_table.csv`  
  Final cross-model metric summary (overall, rule classification, rule picture reading).

- `2_1-2_rule_understanding_summary_table.xlsx`  
  Excel version of the same final summary.

---

## Minimal File Set for a Full 2_1-2 Run

1. `dataset/2_rules_understanding.csv`
2. `2_1-2_Eval_Claude_Opus_4_6.ipynb` -> `2_1-2_with_answers_claude-opus_4_6.csv`
3. `2_1-2_Eval_Gemini.ipynb` -> `2_1-2_with_answers_gemini.csv`
4. `2_1-2_Eval_GPT_5_2.ipynb` -> `2_1-2_with_answers_gpt-5_2.csv`
5. `2_1-2_Analysis.ipynb` -> intermediate outputs + final summary CSV/XLSX
