# Stepwise Reasoning Evaluation (LLM-as-a-Judge, Automatic): Methodology and 
Technical Specification

This document describes the evaluation logic implemented in
[3_3_Stepwise_LLM_judge_Eval.ipynb](/C:/Users/ritaMZ/PycharmProjects/bravo_benchmark/eval/3_compliance_reasoning/3_3_reasoning_analysis/3_3_Stepwise_LLM_judge_Eval.ipynb).

Primary scope:
- automatic stepwise reasoning evaluation for compliance Q&A;
- GT-centric candidate matching with LLM adjudication;
- reproducible export artifacts for audit and reporting.

The workflow is configured for **LLM-as-a-judge + automatic aggregation**.  
It is not a full-answer black-box score.

---

## Contents

1. [Evaluation Scope](#1-evaluation-scope)  
2. [Core Methodology](#2-core-methodology)  
3. [Metric Definitions](#3-metric-definitions)  
4. [Stage-to-Artifact Mapping](#4-stage-to-artifact-mapping)  
5. [Input Artifacts (Current Notebook Configuration)](#5-input-artifacts-current-notebook-configuration)  
6. [Output Tables and Field Reference](#6-output-tables-and-field-reference)  
7. [Judge Alignment Metrics in Final Summary](#7-judge-alignment-metrics-in-final-summary)  
8. [Operating Notes](#8-operating-notes)

---

## 1. Evaluation Scope

### 1.1 Target subset

- Input rows are routed by `template_id`.
- Stepwise subset in this notebook: `template_id in {41, 42}`.

### 1.2 Input text fields used for scoring

- Ground truth reasoning: `ground_truth_answer`
- Predicted reasoning: `model_answer`

### 1.3 GT retrieval prerequisite

Before step matching, each row is linked to canonical GT reasoning via:
- `scene_id`
- `ParentRuleID` (normalized)
- `rule_figure_id` (normalized)

Rows with `gt_match_status != unique_match` are excluded from judge adjudication and written to audit outputs.

---

## 2. Core Methodology

### 2.1 Step decomposition

Both GT and prediction are split into ordered reasoning steps (`Step N` parser + numbered-list fallback).

### 2.2 Pairwise semantic matching

For each sample, all `GT x Pred` step pairs are scored with Sentence-BERT cosine similarity.

Thresholds in current notebook:
- `RULE_WEAK_MATCH_THRESHOLD = 0.58`
- `RULE_MATCH_THRESHOLD = 0.71`
- `RULE_STRONG_MATCH_THRESHOLD = 0.82`
- `RULE_AUTO_ACCEPT_THRESHOLD = 0.82`
- `RULE_AUTO_REJECT_THRESHOLD = 0.58`

### 2.3 GT-centric automatic coverage

For each GT step:
- matched predicted IDs are collected from thresholded candidates;
- structure is assigned as:
  - `missing` (0 matches),
  - `exact` (1 match),
  - `split` (>1 matches);
- automatic quality is assigned (`strong` / `weak`) for matched rows.

### 2.4 LLM judge adjudication

The judge runs per GT step (single-call adjudication) on prepared payload:
- GT step metadata and text;
- candidate pool;
- fallback pool (bounded rescue count);
- rubric definitions (`strong`, `weak`, `no_match`).

Judge output fields:
- `judge_coverage_structure`
- `judge_coverage_quality`
- `judge_best_pred_ids`
- `judge_final_step_flag`
- `judge_notes`
- `judge_request_id`
- `judge_error`

### 2.5 Final labels and aggregation

Final columns (`final_*`) are computed from automatic baseline plus judge overrides:
- `final_coverage_structure`
- `final_coverage_quality`
- `final_best_pred_ids`
- `final_step_special_flag`

Sample-level and summary metrics are calculated from these final columns.

---

## 3. Metric Definitions

### 3.1 Sample-level counts

- `gt_total_steps`
- `pred_total_steps`
- `unique_matched_pred_count`
- `gt_covered_count`
- `gt_missing_count`
- `gt_split_count`
- `gt_weak_count`
- `pred_extra_count`

### 3.2 Sample-level rates

- `coverage_recall = gt_covered_count / gt_total_steps`
- `split_step_rate = gt_split_count / gt_total_steps`
- `missing_step_rate = gt_missing_count / gt_total_steps`
- `weak_match_rate = gt_weak_count / gt_total_steps`
- `extra_step_rate = pred_extra_count / pred_total_steps`

### 3.3 Order metrics

- `order_preservation_score`: share of correctly ordered comparable matched GT-step pairs.
- `order_error_count`: count of comparable pairs with order inversion.
- `strict_position_match_rate`: share of matched GT steps where GT position equals anchor predicted position.
- `comparable_matched_pairs`: number of pair comparisons used for order score.
![evaluation_of_logical_flow copy.png](evaluation_of_logical_flow%20copy.png)

### 3.4 Summary-level text/similarity diagnostics

- overall Sentence-BERT similarity (`best_similarity` aggregate);
- ROUGE-L average per step (`gt_step_text` vs matched predicted text);
- optional markers may appear in formatted summary text, but principal generated signals are SBERT/ROUGE-L + coverage/order metrics.

---

## 4. Stage-to-Artifact Mapping

| Pipeline stage | Internal product | Output artifact |
| --- | --- | --- |
| Pairwise scoring | all `GT x Pred` pairs with similarity and triage | `3_2_to_3_5_all_pairwise_similarity_LLM.csv` |
| Candidate shortlist | predicted-centered candidate rows | `3_2_to_3_5_step_candidate_matches_LLM.csv` |
| Judge run | row-level judge decisions | `3_2_to_3_5_judge_row_results_LLM.jsonl`, `3_2_to_3_5_gt_coverage_judge_review_LLM.csv` |
| GT-centric final coverage | one row per GT step with auto, judge, and final labels | `3_2_to_3_5_gt_coverage_summary_LLM.csv` |
| Sample aggregation | one row per sample with counts/rates/order | `3_2_to_3_5_sample_level_metrics_LLM.csv` |
| GT retrieval audit | lookup status per dataset row | `3_2_to_3_5_gt_lookup_audit_LLM.csv` |
| Final summary | cross-model automatic vs judge table | `3_2_to_3_5_stepwise_summary_table_LLM.csv`, `3_2_to_3_5_stepwise_summary_table_LLM.xlsx` |

Notebook also exports methodology aliases:
- `all_pairwise_similarity_LLM.csv`
- `gt_step_candidates_auto_LLM.csv`

---

## 5. Input Artifacts (Current Notebook Configuration)

### 5.1 Model input CSVs

- `3_2_to_3_5_clean_claude-opus-4.6_V2.csv`
- `3_2_to_3_5_clean_gemini_3_pro_prev_V2.csv`
- `3_2_to_3_5_clean_gpt-5_2_V2.csv`

### 5.2 GT lookup source

- `03_04_reasonLLMjudgeCalibr.csv` (absolute path in notebook config).

### 5.3 Required operational columns (minimum)

- `generated_question_id`, `row_index`, `template_id`, `question_text`
- `scene_id`, `ParentRuleID`, `rule_figure_id`
- `ground_truth_answer`, `model_answer`

### 5.4 OpenRouter configuration

Judge calls use `OPENROUTER_API_KEY` from environment and are controlled by `JUDGE_RUN` / `JUDGE_TARGET_MODE`.

---

## 6. Output Tables and Field Reference

### 6.1 `3_2_to_3_5_step_candidate_matches_LLM.csv`

Purpose: predicted-centered diagnostic shortlist for judge routing.

Columns:
- `generated_question_id`, `row_index`, `model_name`, `question_text`
- `pred_step_id`, `pred_step_position`, `pred_step_label`, `pred_text`
- `matched_gt_id`, `matched_gt_position`, `matched_gt_text`
- `full_ground_truth_answer`, `full_model_answer`
- `similarity_score`
- `triage_zone` (`auto_accept` / `uncertain` / `auto_reject`)
- `Judge Review Suggested`
- `candidate_flag`, `auto_match`

### 6.2 `3_2_to_3_5_all_pairwise_similarity_LLM.csv`

Purpose: full `GT x Pred` audit matrix.

Columns:
- `generated_question_id`, `row_index`, `model_name`, `question_text`
- `full_ground_truth_answer`, `full_model_answer`
- `gt_step_id`, `gt_step_position`, `gt_is_final_step`, `pred_final_step_ids`, `gt_step_text`
- `pred_step_id`, `pred_step_position`, `pred_step_label`, `pred_text`
- `similarity_score`
- `triage_zone`
- `Judge Review Suggested`
- `candidate_flag`, `auto_match`

### 6.3 `3_2_to_3_5_gt_coverage_summary_LLM.csv`

Purpose: authoritative GT-centric adjudication table used for final metric computation.

Columns:
- context:  
  `generated_question_id`, `row_index`, `model_name`, `question_text`,  
  `full_ground_truth_answer`, `full_model_answer`
- GT step identity:  
  `gt_step_id`, `gt_step_position`, `gt_is_final_step`, `pred_final_step_ids`, `gt_step_text`
- automatic candidate/match fields:  
  `candidate_pool`, `fallback_pool`, `accepted_pred_ids_auto`, `matched_pred_step_ids`,  
  `matched_pred_positions`, `matched_pred_texts`, `matched_pred_labels`,  
  `best_similarity`, `triage_zone`, `matched_pred_count`,  
  `coverage_structure`, `coverage_quality_auto`, `coverage_quality`, `final_step_special_flag_auto`
- judge fields:  
  `judge_coverage_structure`, `judge_coverage_quality`, `judge_best_pred_ids`,  
  `judge_final_step_flag`, `judge_notes`, `judge_request_id`, `judge_error`
- final adjudicated fields:  
  `final_coverage_structure`, `final_coverage_quality`, `final_best_pred_ids`, `final_step_special_flag`
- review flag:  
  `Judge Review Suggested`

### 6.4 `3_2_to_3_5_sample_level_metrics_LLM.csv`

Purpose: one row per sample for automatic scoring and comparison.

Columns:
- context:
  `generated_question_id`, `row_index`, `model_name`, `question_text`, `full_ground_truth_answer`, `full_model_answer`
- counts:
  `gt_total_steps`, `pred_total_steps`, `unique_matched_pred_count`, `gt_covered_count`, `gt_missing_count`, `gt_split_count`, `gt_weak_count`, `pred_extra_count`
- rates:
  `coverage_recall`, `split_step_rate`, `missing_step_rate`, `weak_match_rate`, `extra_step_rate`
- order:
  `order_preservation_score`, `order_error_count`, `strict_position_match_rate`, `comparable_matched_pairs`
- triage aggregates:
  `pair_auto_accept_count`, `pair_uncertain_count`, `pair_auto_reject_count`
- optional merged metadata:
  `sample_key`, `template_id`, `scene_id`, `parent_rule_id_norm`, `figure_id_norm`, `gt_match_status`

### 6.5 `3_2_to_3_5_gt_lookup_audit_LLM.csv`

Purpose: traceability of GT retrieval before step matching.

Columns:
- `generated_question_id`
- `row_index`
- `template_id`
- `scene_id`
- `ParentRuleID`
- `rule_figure_id`
- `gt_match_status`
- `gt_source_row_id`
- `gt_total_steps`

### 6.6 `3_2_to_3_5_stepwise_summary_table_LLM.csv/.xlsx`

Purpose: final reviewer-facing cross-model table.

Columns:
- `Reasoning step`
- `Primary automatic metrics`
- `Claude Opus 4.6 (Automatic)`
- `Claude Opus 4.6 (LLM-as-a-judge)`
- `Gemini 3 Pro Preview (Automatic)`
- `Gemini 3 Pro Preview (LLM-as-a-judge)`
- `GPT-5.2 (Automatic)`
- `GPT-5.2 (LLM-as-a-judge)`
- `Dataset characteristics`
- `Judge alignment metrics`

---

## 7. Judge Alignment Metrics in Final Summary

The notebook reports alignment between automatic labels and LLM-adjudicated final labels.

- `matched_id_overlap`:
  mean Jaccard overlap between `matched_pred_step_ids` (automatic) and `final_best_pred_ids` (judge-finalized).
- `unchanged_percentage`:
  proportion of reviewed GT rows where `coverage_structure` and `coverage_quality` remain unchanged after judge adjudication.
- `n_reviewed_steps`:
  reviewed GT-step row count.
- `n_reviewed_questions`:
  number of unique questions represented by reviewed rows.

If no judge-adjudicated rows exist for a scope/model, alignment values are not reported for that scope.

---

## 8. Operating Notes

1. This notebook is designed for automatic evaluation with optional judge augmentation.
2. `JUDGE_RUN=False` keeps the pure automatic path and still exports all audit and metrics tables.
3. `JUDGE_RUN=True` produces judge artifacts and final labels with LLM overrides.
4. Candidate and pairwise tables are diagnostic layers; final scoring is based on GT-centric final fields.
