# 3.3 Reasoning Analysis: Inputs / Outputs

Scope: executable files in `eval/3_compliance_reasoning/3_3_reasoning_analysis`.

This document is compact by design.  
Detailed methodology and field-level specs are documented here:

- [3_3_Stepwise_Eval.md](/C:/Users/ritaMZ/PycharmProjects/bravo_benchmark/eval/3_compliance_reasoning/3_3_reasoning_analysis/3_3_Stepwise_Eval.md) (human-centered stepwise evaluation)
- [3_3_Stepwise_LLM-as-a-judge.md](/C:/Users/ritaMZ/PycharmProjects/bravo_benchmark/eval/3_compliance_reasoning/3_3_reasoning_analysis/3_3_Stepwise_LLM-as-a-judge.md) (automatic + LLM-as-a-judge flow)
- [LLM-as-a-judge_V1.md](/C:/Users/ritaMZ/PycharmProjects/bravo_benchmark/eval/3_compliance_reasoning/3_3_reasoning_analysis/3_3_archive_LLM_judge/LLM-as-a-judge_V1.md) (archived v1 approach)

Note: many exported files keep the historical `3_2_to_3_5_*` prefix.

---

## Files in Scope

1. `3_3_Stepwise_LLM_judge_Eval.ipynb`
2. `3_3_Stepwise_Analysis.ipynb`
3. `human_alignment.py`
4. `3_3_archive_LLM_judge/LLM-as-a-judge.ipynb` (archive)

---

## Recommended Run Order

1. Run `3_3_Stepwise_LLM_judge_Eval.ipynb` for the main LLM-as-a-judge pipeline.
2. Run `3_3_Stepwise_Analysis.ipynb` for the parallel non-judge (human-centered) stepwise pipeline.
3. Run `human_alignment.py` only when you need explicit judge-vs-final alignment metrics.
4. Use the archive notebook only for historical reference.

---

## 1) `3_3_Stepwise_LLM_judge_Eval.ipynb`

### Purpose
Main automatic stepwise evaluation notebook with optional OpenRouter adjudication.

### Key inputs
- Model CSVs:
  - `3_2_to_3_5_clean_claude-opus-4.6_V2.csv`
  - `3_2_to_3_5_clean_gemini_3_pro_prev_V2.csv`
  - `3_2_to_3_5_clean_gpt-5_2_V2.csv`
- GT/rubric sources:
  - `03_04_reasonLLMjudgeCalibr.csv`
  - `final_VQA_metric_rubrics.CSV`
- OpenRouter (if judge is enabled): `OPENROUTER_API_KEY` environment variable.

### Main outputs
- `3_2_to_3_5_step_candidate_matches_LLM.csv`
- `3_2_to_3_5_all_pairwise_similarity_LLM.csv`
- `all_pairwise_similarity_LLM.csv`
- `3_2_to_3_5_gt_coverage_summary_LLM.csv`
- `gt_step_candidates_auto_LLM.csv`
- `3_2_to_3_5_sample_level_metrics_LLM.csv`
- `3_2_to_3_5_gt_lookup_audit_LLM.csv`
- `3_2_to_3_5_stepwise_summary_table_LLM.csv`
- `3_2_to_3_5_stepwise_summary_table_LLM.xlsx`

### Judge-specific outputs
- `3_2_to_3_5_judge_row_results_LLM.jsonl`
- `3_2_to_3_5_gt_coverage_judge_review_LLM.csv`

---

## 2) `3_3_Stepwise_Analysis.ipynb`

### Purpose
Stepwise analysis notebook without LLM judge calls (human-centered adjudication path).

### Key inputs
- Model CSVs:
  - `3_2_to_3_5_clean_claude-opus-4.6_V2.csv`
  - `3_2_to_3_5_clean_gemini_3_pro_prev_V2.csv`
  - `3_2_to_3_5_clean_gpt-5_2_V2.csv`
- GT source:
  - `03_04_reasonLLMjudgeCalibr.csv`

### Outputs
- `3_2_to_3_5_step_candidate_matches.csv`
- `3_2_to_3_5_all_pairwise_similarity.csv`
- `3_2_to_3_5_gt_coverage_summary.csv`
- `3_2_to_3_5_sample_level_metrics.csv`
- `3_2_to_3_5_gt_lookup_audit.csv`
- `3_2_to_3_5_stepwise_summary_table.csv`
- `3_2_to_3_5_stepwise_summary_table.xlsx`

---

## 3) `human_alignment.py`

### Purpose
Computes judge-vs-final alignment metrics from GT-coverage exports.

### Short technical note
The script merges automatic GT coverage with judge review data on strict row keys (`model_name`, `generated_question_id`, `row_index`, `gt_step_id`, with fallback by `gt_step_position`), then computes:
- `Matched ID Overlap` (Jaccard overlap of matched predicted ID sets),
- `Unchanged Percentage` (agreement of structure/quality labels),
- reviewed-step and reviewed-question counts per step and overall.

### Default inputs
- `3_2_to_3_5_gt_coverage_summary_automatic.csv`
- `3_2_to_3_5_gt_coverage_judge_review_LLM.csv`

### Default output
- `3_2_to_3_5_human_alignment_metrics_LLM.csv`

### CLI arguments
- `--automatic-csv`
- `--review-csv`
- `--output-csv`

---

## 4) `3_3_archive_LLM_judge/LLM-as-a-judge.ipynb` (Archive)

### Purpose
Initial LLM-as-a-judge design retained for historical reproducibility and comparison.

### Typical outputs (`../results`)
- `llm_judge_stage1_metric_level.csv`
- `llm_judge_stage1_raw.jsonl`
- `llm_judge_stage1_merged.csv`
- `llm_judge_stage1b_composites.csv`
- `llm_judge_stage2_overall.csv`
- `llm_judge_final_merged.csv`

---

## Quick Navigation

- Main automatic+judge flow: [3_3_Stepwise_LLM-as-a-judge.md](/C:/Users/ritaMZ/PycharmProjects/bravo_benchmark/eval/3_compliance_reasoning/3_3_reasoning_analysis/3_3_Stepwise_LLM-as-a-judge.md)
- Main human-centered flow: [3_3_Stepwise_Eval.md](/C:/Users/ritaMZ/PycharmProjects/bravo_benchmark/eval/3_compliance_reasoning/3_3_reasoning_analysis/3_3_Stepwise_Eval.md)
- Archived v1 flow: [LLM-as-a-judge_V1.md](/C:/Users/ritaMZ/PycharmProjects/bravo_benchmark/eval/3_compliance_reasoning/3_3_reasoning_analysis/3_3_archive_LLM_judge/LLM-as-a-judge_V1.md)
