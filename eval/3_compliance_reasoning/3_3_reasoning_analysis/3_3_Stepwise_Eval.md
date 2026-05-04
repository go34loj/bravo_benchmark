# Stepwise Reasoning Evaluation (Human-Centered): Methodology and Technical Specification

This document describes the evaluation logic implemented in
[3_3_Stepwise_Analysis.ipynb](/C:/Users/ritaMZ/PycharmProjects/bravo_benchmark/eval/3_compliance_reasoning/3_3_reasoning_analysis/3_3_Stepwise_Analysis.ipynb).

Primary scope:
- stepwise reasoning quality for compliance Q&A,
- GT-centric human review and adjudication,
- export artifacts used for audit and reporting.

The workflow is configured for **human evaluation** first. Automatic matching is used to pre-fill review tables; final metrics are computed from reviewed/final labels.

---

## Contents

1. [Evaluation Scope](#1-evaluation-scope)  
2. [Core Methodology](#2-core-methodology)  
3. [Metric Definitions](#3-metric-definitions)  
4. [Stage-to-Artifact Mapping](#4-stage-to-artifact-mapping)  
5. [Input Artifacts (Current Notebook Configuration)](#5-input-artifacts-current-notebook-configuration)  
6. [Output Tables and Field Reference](#6-output-tables-and-field-reference)  
7. [Human-Alignment Metrics in Final Summary](#7-human-alignment-metrics-in-final-summary)  
8. [Human-Evaluation Operating Notes](#8-human-evaluation-operating-notes)  
9. [Quick Run Procedure](#9-quick-run-procedure)

---

## 1. Evaluation Scope

## 1.1 Target subset

- Input rows are routed by `template_id`.
- Stepwise reasoning subset in this notebook: `template_id in {41, 42}`.

## 1.2 Input text fields used for scoring

- Ground truth reasoning text: `ground_truth_answer`
- Predicted reasoning text: `model_answer`

## 1.3 GT retrieval prerequisite

Before matching, each row is linked to canonical GT reasoning via composite key:
- `scene_id`
- `ParentRuleID` (normalized)
- `rule_figure_id` (normalized)

Rows are labeled with GT retrieval status (`gt_match_status`) and only unique-match rows proceed to auto metrics. Non-eligible rows are retained for manual review queue/audit.

---

## 2. Core Methodology

## 2.1 Step decomposition

Both GT and prediction are split into reasoning steps. The parser supports:
- explicit `Step N` format,
- fallback numbered lists (`1.`, `2.`, ...).

Step metadata includes IDs, positions, labels, and normalized text.

## 2.2 Pairwise semantic matching

For each sample, all `Pred x GT` step pairs are scored with Sentence-BERT cosine similarity.

Thresholds used in current notebook:
- `RULE_WEAK_MATCH_THRESHOLD = 0.58`
- `RULE_MATCH_THRESHOLD = 0.71`
- `RULE_STRONG_MATCH_THRESHOLD = 0.82`
- `RULE_AUTO_ACCEPT_THRESHOLD = 0.82`
- `RULE_AUTO_REJECT_THRESHOLD = 0.58`

## 2.3 GT-centric coverage assignment

For each GT step:
- collect matched predicted steps (`similarity >= RULE_MATCH_THRESHOLD`),
- assign structure label:
  - `missing` (0 matches),
  - `exact` (1 match),
  - `split` (>1 matches),
- assign quality label:
  - `strong` / `weak` for matched rows,
  - blank for `missing`.

## 2.4 Final-step guardrails

To prevent unstable matching of conclusion steps:
- final-step candidates are detected in GT and prediction,
- compatibility guardrails are applied so final vs non-final steps do not match incorrectly,
- fallback label-level matching is used for final step when text-level match is absent,
- such fallback is recorded with `coverage_quality = wrong` when applicable.

## 2.5 Human adjudication logic (authoritative layer)

Human review columns:
- `human_coverage_structure`
- `human_coverage_quality`
- `human_best_pred_ids`
- `human_notes`

Final labels are computed as:
- use human value if provided,
- otherwise keep automatic value.

Final columns:
- `final_coverage_structure`
- `final_coverage_quality`
- `final_best_pred_ids`

All sample-level metrics are computed from these `final_*` fields.

---

## 3. Metric Definitions

## 3.1 Sample-level metrics

From final GT-centric labels:

- `gt_total_steps`: GT step count.
- `gt_covered_count`: number of GT steps with final structure in `{exact, split}`.
- `gt_missing_count`: number of GT steps with final structure `missing`.
- `gt_split_count`: number of GT steps with final structure `split`.
- `gt_weak_count`: number of non-missing GT steps with final quality `weak`.
- `pred_total_steps`: predicted step count.
- `pred_extra_count`: predicted steps not used in `final_best_pred_ids`.

Rates:
- `coverage_recall = gt_covered_count / gt_total_steps`
- `split_step_rate = gt_split_count / gt_total_steps`
- `missing_step_rate = gt_missing_count / gt_total_steps`
- `weak_match_rate = gt_weak_count / gt_total_steps`
- `extra_step_rate = pred_extra_count / pred_total_steps`

## 3.2 Order metrics

Order is evaluated after semantic alignment:
- anchor predicted position per GT step (policy: `min_pred_position`),
- compare relative order across comparable matched GT step pairs.

![evaluation_of_logical_flow copy.png](evaluation_of_logical_flow%20copy.png)

Outputs:
- `order_preservation_score`
- `order_error_count`
- `strict_position_match_rate`
- `comparable_matched_pairs`

## 3.3 Text similarity diagnostics

- `overall Sentence-BERT similarity`: mean `best_similarity` over GT rows.
- `ROUGE-L (avg per step)`: mean ROUGE-L F1 between `gt_step_text` and matched predicted text.
- Optional diagnostics may appear in summary formatting (`METEOR` marker), but GT-centric counts and SBERT/ROUGE-L are the principal generated signals in this notebook.

---

## 4. Stage-to-Artifact Mapping

| Pipeline stage | Internal product | Output artifact |
| --- | --- | --- |
| Pairwise scoring | all `Pred x GT` rows with similarity and triage | `3_2_to_3_5_all_pairwise_similarity.csv` |
| Predicted-centered helper | one row per predicted step with selected GT mapping | `3_2_to_3_5_step_candidate_matches.csv` |
| GT-centric coverage + review | one row per GT step with auto + human + final labels | `3_2_to_3_5_gt_coverage_summary.csv` |
| Sample aggregation | one row per sample with counts/rates/order metrics | `3_2_to_3_5_sample_level_metrics.csv` |
| GT retrieval audit | per-row GT match status and key fields | `3_2_to_3_5_gt_lookup_audit.csv` |
| Final summary table | cross-model, stepwise + overall view | `3_2_to_3_5_stepwise_summary_table.csv`, `3_2_to_3_5_stepwise_summary_table.xlsx` |

Note: file names currently keep the historical `3_2_to_3_5_*` prefix even though this notebook is located in section `3_3`.

---

## 5. Input Artifacts (Current Notebook Configuration)

## 5.1 Model input CSVs

- `3_2_to_3_5_clean_claude-opus-4.6_V2.csv`
- `3_2_to_3_5_clean_gemini_3_pro_prev_V2.csv`
- `3_2_to_3_5_clean_gpt-5_2_V2.csv`

## 5.2 GT lookup source

- `03_04_reasonLLMjudgeCalibr.csv` (referenced via absolute path in notebook config).

## 5.3 Required operational columns (minimum)

For routing and matching:
- `generated_question_id`, `row_index`, `template_id`, `question_text`,
- `scene_id`, `ParentRuleID`, `rule_figure_id`,
- `ground_truth_answer`, `model_answer`.

---

## 6. Output Tables and Field Reference

## 6.1 `3_2_to_3_5_step_candidate_matches.csv`

Purpose: predicted-centered diagnostic/review helper.

Columns:
- `generated_question_id`, `row_index`, `model_name`, `question_text`
- `pred_step_id`, `pred_step_position`, `pred_step_label`, `pred_text`
- `matched_gt_id`, `matched_gt_position`, `matched_gt_text`
- `full_ground_truth_answer`, `full_model_answer`
- `similarity_score`
- `triage_zone` (`auto_accept` / `uncertain` / `auto_reject`)
- `Human Review Required` (exported review flag)
- `candidate_flag`, `auto_match`
- adjudication helper fields (produced in notebook): `auto_label`, `human_label`, `final_label`, `human_gt_assignment`, `final_gt_assignment`, `human_pred_assignment`, `final_pred_assignment` (when present in working frame).

## 6.2 `3_2_to_3_5_all_pairwise_similarity.csv`

Purpose: full pairwise audit matrix (`GT x Pred`) for traceability.

Columns:
- `generated_question_id`, `row_index`, `model_name`, `question_text`
- `full_ground_truth_answer`, `full_model_answer`
- `gt_step_id`, `gt_step_position`, `gt_is_final_step`, `pred_final_step_ids`, `gt_step_text`
- `pred_step_id`, `pred_step_position`, `pred_step_label`, `pred_text`
- `similarity_score`
- `triage_zone`
- `Human Review Required`
- `candidate_flag`, `auto_match`

## 6.3 `3_2_to_3_5_gt_coverage_summary.csv`

Purpose: authoritative GT-centric review/adjudication table.

Columns:
- `generated_question_id`, `row_index`, `model_name`
- `question_text`, `full_ground_truth_answer`, `full_model_answer`
- `gt_step_id`, `gt_step_position`, `gt_is_final_step`, `pred_final_step_ids`
- `matched_pred_step_ids`, `matched_pred_positions`, `matched_pred_texts`, `matched_pred_labels`
- `gt_step_text`
- `best_similarity`, `triage_zone`, `matched_pred_count`
- `coverage_structure`, `coverage_quality` (auto)
- `human_coverage_structure`, `human_coverage_quality`, `human_best_pred_ids`, `human_notes` (review)
- `final_coverage_structure`, `final_coverage_quality`, `final_best_pred_ids` (adjudicated)
- `Human Review Required` (exported review flag)

## 6.4 `3_2_to_3_5_sample_level_metrics.csv`

Purpose: one row per sample for scoring and comparison.

Columns:
- identity/context:
  - `generated_question_id`, `row_index`, `model_name`, `question_text`
  - `full_ground_truth_answer`, `full_model_answer`
- counts:
  - `gt_total_steps`, `pred_total_steps`, `unique_matched_pred_count`
  - `gt_covered_count`, `gt_missing_count`, `gt_split_count`, `gt_weak_count`
  - `pred_extra_count`
- rates:
  - `coverage_recall`, `split_step_rate`, `missing_step_rate`, `weak_match_rate`, `extra_step_rate`
- order:
  - `order_preservation_score`, `order_error_count`, `strict_position_match_rate`, `comparable_matched_pairs`
- triage aggregates:
  - `pair_auto_accept_count`, `pair_uncertain_count`, `pair_auto_reject_count`
- optional merged metadata:
  - `sample_key`, `template_id`, `scene_id`, `parent_rule_id_norm`, `figure_id_norm`, `gt_match_status`.

## 6.5 `3_2_to_3_5_gt_lookup_audit.csv`

Purpose: GT retrieval traceability before step matching.

Columns (as exported by notebook):
- `generated_question_id`
- `row_index`
- `template_id`
- `scene_id`
- `ParentRuleID`
- `rule_figure_id`
- `gt_match_status`
- `gt_source_row_id`
- `gt_total_steps`

## 6.6 `3_2_to_3_5_stepwise_summary_table.csv/.xlsx`

Purpose: reviewer-facing final summary by reasoning step and by model, including human-alignment diagnostics.

Columns:
- `Reasoning step`
- `Primary automatic metrics`
- `Claude Opus 4.6`
- `Gemini 3 Pro Preview`
- `GPT-5.2`
- `Dataset characteristics`
- `Human alignment metrics`

---

## 7. Human-Alignment Metrics in Final Summary

The notebook computes human-alignment from reviewed GT rows:

- `matched_id_overlap`:
  mean Jaccard overlap between auto matched IDs (`matched_pred_step_ids`) and final reviewed IDs (`final_best_pred_ids`).

- `unchanged_percentage`:
  proportion of reviewed GT rows where auto and final labels are unchanged
  (`coverage_structure` vs `final_coverage_structure`, and `coverage_quality` vs `final_coverage_quality`).

- `n_reviewed_steps`:
  number of reviewed GT-step rows.

- `n_reviewed_questions`:
  number of unique questions represented by reviewed rows.

If no human review exists for a model/scope, these values are `NaN` or zero by design.

---

## 8. Human-Evaluation Operating Notes

1. The GT-centric table is the single authoritative adjudication surface.
2. Candidate and pairwise tables are diagnostic aids and triage support, not final truth tables.
3. Final metrics and final summary must be regenerated after review updates in `3_2_to_3_5_gt_coverage_summary.csv`.
4. Non-eligible GT retrieval rows remain part of manual-review audit and should not be silently discarded.

---

## 9. Quick Run Procedure

This notebook is designed for a two-pass human-evaluation loop.

1. Run `3_3_Stepwise_Analysis.ipynb` end-to-end once.
   - This produces initial automatic artifacts, including:
     - `3_2_to_3_5_gt_coverage_summary.csv`
     - `3_2_to_3_5_sample_level_metrics.csv`
     - `3_2_to_3_5_stepwise_summary_table.csv/.xlsx`

2. Open `3_2_to_3_5_gt_coverage_summary.csv` and perform human review.
   - Fill reviewer columns where needed:
     - `human_coverage_structure`
     - `human_coverage_quality`
     - `human_best_pred_ids`
     - `human_notes`
   - Save the file in place with the same filename.

3. Run the notebook again.
   - The notebook merges human review into GT-centric rows.
   - It recomputes final labels and sample-level metrics from `final_*` columns.
   - It rewrites final exports (`sample_level_metrics` and `stepwise_summary_table`) using adjudicated values.

This is the expected operating mode for final human-validated results.
