# LLM-as-a-Judge V1 (Archive): Technical Documentation

This document describes the archived notebook implementation:
`eval/3_compliance_reasoning/3_3_reasoning_analysis/3_3_archive_LLM_judge/LLM-as-a-judge.ipynb`.

Scope of this document:
- notebook architecture and execution stages;
- input artifacts and required schemas;
- output artifacts and field-level definitions.

---

## 1. Purpose and Status

`LLM-as-a-judge.ipynb` is the initial (V1) implementation of a stepwise LLM-judge pipeline for compliance reasoning evaluation.

The notebook runs in two stages:
1. **Stage 1 (metric-level scoring):** one evaluation record per `(generated_question_id, metric_name)`.
2. **Stage 2 (overall synthesis):** one overall decision record per `(generated_question_id, candidate_model)`.

The notebook supports both:
- `DRY_RUN=True` (no API calls),
- real OpenRouter calls (`DRY_RUN=False`) with `OPENROUTER_API_KEY` from `eval.env`.

---

## 2. Pipeline Structure

### Stage A. Data Ingestion and Normalization

The notebook loads four normative/calibration tables and multiple model-output tables, then canonicalizes column names via alias maps.

### Stage B. Retrieval Grounding

For each evaluation item and each metric:
- rule-specific context is assembled from rule tables;
- calibration examples are selected via cascading retrieval;
- retrieval diagnostics are stored per request.

### Stage C. Stage-1 Metric Judging

One record is generated per metric call. Deterministic fallback rows are produced in `DRY_RUN`.

### Stage D. Stage-1b Deterministic Composites

Composite metrics are computed from Stage-1 outputs without additional judge calls:
- `CRCS_deterministic_score_1_5`,
- `RSF_deterministic_score_1_5`.

### Stage E. Stage-2 Overall Assessment

A compact overall assessment is produced from fixed metric values.

### Stage F. Export

All stage artifacts are written to `../results`.

---

## 3. Input Artifacts

## 3.1 Core input files (configured in notebook)

- `20_03_VQA_Rules_Scenes_Templates(rules_sort).csv`
- `09_01_VQA_Rules_Scenes_Templates(rule_figure).csv`
- `03_04_reasonLLMjudgeCalibr.csv`
- `03_04_VQA_metric_rubrics.csv`
- model outputs:
  - `3_2_to_3_5_clean_claude-opus-4.6_V2.csv`
  - `3_2_to_3_5_clean_gemini_3_pro_prev_V2.csv`
  - `3_2_to_3_5_clean_gpt-5_2_V2.csv`

## 3.2 Required schema by logical table

## `rules_table` (required)
- `parent_rule_id`
- `parent_rule_text`
- `rule_id`
- `rule_text_atomic`
- `rule_figure_caption`
- `ambiguity_remark`

## `calibration_examples` (required)
- `reason_gt_id`
- `parent_rule_id`
- `question_id`
- `scene_id`
- `task_family`
- `ground_truth_answer`
- `answer_to_evaluate`
- `survey_question_id`
- `human_verdict_distribution`
- `human_verdict_consensus`
- `human_verdict_agreement`
- `human_completeness_mean`
- `human_completeness_std`
- `human_completeness_agreement`
- `human_logical_mean`
- `human_logical_std`
- `human_redundancy_mean`
- `human_redundancy_std`
- `counter_note`
- `rule_interpretation_note`

## `metric_rubrics` (required)
- `metric_name`
- `definition`
- `score_scale`
- `scoring_guidelines`
- `anti_bias_note`

## `evaluation_items` (required)
- `generated_question_id`
- `template_id`
- `question_text`
- `scene_id`
- `file_path`
- `figure_path`
- `RuleID`
- `ParentRuleID`
- `Classification`
- `Ambiguity`
- `ClassificationParent`
- `rule_figure_id`
- `correct_answer`
- `CorrectAnswer`
- `model_answer`
- `openrouter_request_id`
- `error_reason`
- `row_index`
- `bool_model_answer`

---

## 4. Stage-to-Artifact Mapping

| Notebook stage | Internal product | Exported artifact |
| --- | --- | --- |
| Stage 1 metric loop | one row per metric call with score/rationale/retrieval diagnostics | `llm_judge_stage1_metric_level.csv` |
| Stage 1 raw request log | full prompt/response JSON payloads | `llm_judge_stage1_raw.jsonl` |
| Stage 1 pivot + deterministic metrics | one row per item-model with metric columns + deterministic CRCS/RSF | `llm_judge_stage1_merged.csv` |
| Stage 1b compact composite view | selected CRCS/RSF fields for reporting | `llm_judge_stage1b_composites.csv` |
| Stage 2 overall loop | one row per item-model overall label/summary | `llm_judge_stage2_overall.csv` |
| Final merge | Stage1 merged + Stage2 overall | `llm_judge_final_merged.csv` |

Preview/test run exports (strict split-aware mini run):
- `3_2_to_3_5_stage1_metric_scores_preview.csv`
- `3_2_to_3_5_test_stage1_vs_reference.csv`
- `3_2_to_3_5_test_stage2_overall.csv`
- `3_2_to_3_5_test_raw_openrouter.jsonl`

---

## 5. Output Schemas and Field Definitions

## 5.1 `llm_judge_stage1_metric_level.csv`

Granularity: one row per `(generated_question_id, candidate_model, metric_name)`.

Core fields:
- `generated_question_id`: item identifier.
- `template_id`: template identifier from source data.
- `scene_id`: scene identifier.
- `RuleID`, `ParentRuleID`: rule linkage for the evaluated item.
- `candidate_model`: model that produced `model_answer`.
- `source_eval_file`: source model-output file.
- `judge_model`: LLM judge model name.
- `metric_name`: judged metric.
- `score`: metric score (1-5 after normalization; may be empty on error).
- `max_score`: upper bound (default `5`).
- `short_rationale`: short textual justification.
- `ambiguity_flag`: boolean ambiguity signal.
- `ambiguity_level`: textual ambiguity level.
- `retrieval_level1_n`..`retrieval_level4_n`: counts of examples selected at each retrieval level.
- `local_pool_sufficient`: whether strict local pool was sufficient.
- `used_fallback`: whether fallback retrieval was used.
- `retrieved_examples_n`: total calibration examples in prompt context.
- `openrouter_request_id`: request identifier.
- `error_reason`: request/parsing error text (empty if successful).

## 5.2 `llm_judge_stage1_raw.jsonl`

Granularity: one JSON object per Stage-1 request.

Key fields:
- `request_id`
- `metric_name`
- `generated_question_id`
- `candidate_model`
- `source_eval_file`
- `retrieval_diag`
- `prompt_messages`
- `response`
- `parsed`
- `parsed_normalized`
- `error`

## 5.3 `llm_judge_stage1_merged.csv`

Granularity: one row per `(generated_question_id, candidate_model, judge_model)`.

Content:
- pivoted Stage-1 metric scores (columns named by metric);
- source fields (`model_answer`, `CorrectAnswer`, `correct_answer`);
- deterministic components:
  - `CRCS - Final Verdict Accuracy`
  - `CRCS_deterministic_unit`
  - `CRCS_deterministic_score_1_5`
  - `RSF_deterministic_unit`
  - `RSF_deterministic_score_1_5`
- `ambiguity_flag` (aggregated from Stage-1 rows).

## 5.4 `llm_judge_stage1b_composites.csv`

Compact projection for composite review:
- `generated_question_id`
- `candidate_model`
- `judge_model`
- `CRCS - Final Verdict Accuracy`
- `CRCS - Reference Mention Accuracy`
- `CRCS - Entity Grounding`
- `CRCS - Visual Alignment`
- `CRCS_deterministic_score_1_5`
- `RSF_deterministic_score_1_5`
- `ambiguity_flag`

## 5.5 `llm_judge_stage2_overall.csv`

Granularity: one row per `(generated_question_id, candidate_model, judge_model)`.

Fields:
- `generated_question_id`
- `candidate_model`
- `judge_model`
- `overall_label`
- `overall_summary`
- `openrouter_request_id`
- `error_reason`

## 5.6 `llm_judge_final_merged.csv`

Final merged table:
- left side: all Stage-1 merged fields;
- right side: Stage-2 overall fields (`overall_label`, `overall_summary`, request/error fields).

Join keys:
- `generated_question_id`
- `candidate_model`
- `judge_model`

## 5.7 Preview/test output fields

## `3_2_to_3_5_stage1_metric_scores_preview.csv`
- same schema family as Stage-1 metric output, but only preview subset rows.

## `3_2_to_3_5_test_stage1_vs_reference.csv`
- `generated_question_id`, `candidate_model`, `metric_name`
- `score`, `expected_score`, `expected_source`
- `abs_error`, `exact_match`
- audit helpers (for example `short_rationale`, `error_reason`, rule/scene keys if present).

## `3_2_to_3_5_test_stage2_overall.csv`
- same schema as Stage-2 overall output, for preview subset only.

## `3_2_to_3_5_test_raw_openrouter.jsonl`
- raw OpenRouter request/response logs for preview subset.

---

## 6. OpenRouter Configuration

The notebook expects `OPENROUTER_API_KEY` from `eval.env`.
No key value is embedded in the documented workflow.

Execution modes:
- `DRY_RUN=True`: no external calls.
- `DRY_RUN=False`: OpenRouter calls enabled for Stage-1/Stage-2 judging.

---

## 7. Notes on V1 Scope

- This document describes the archived V1 implementation as-is.
- The notebook can be used for reproducibility/reference; newer evaluation flows may use updated pipelines in neighboring notebooks.
