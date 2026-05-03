# Semantic Evaluation Methodology for Rule Classification and Decomposition

## 1. Overview

This document defines the evaluation procedure for rule classification and rule decomposition tasks.  
The primary objective is semantic alignment between predicted and reference rules, not exact wording overlap.
The described implementation corresponds to the notebook [2_1-2_Analysis.ipynb](/C:/Users/ritaMZ/PycharmProjects/bravo_benchmark/eval/2_rule_understanding/2_1-2_rule_classific_decompos/2_1-2_Analysis.ipynb).

## 2. Task Definition

Each sample includes:

- a ground-truth answer containing one or more atomic rules
- a model answer containing one or more predicted rules

The evaluation checks whether the model:

- identifies relevant rule components
- reconstructs them with correct semantic meaning
- avoids unsupported or irrelevant rules

## 3. Routing Strategy

Routing is based on `layer_id`:

- `layer_id == "2_1"`: rule classification / rule decomposition (covered here)
- `layer_id == "2_2"`: rule understanding (image description), evaluated separately

This markdown document covers only layer `2_1`.

## 4. Evaluation Pipeline

### Step 1. Decompose Answers into Atomic Rules

Both ground truth and model answers are split into lists of atomic rule statements:

- ground truth: `GT = {GT_1, GT_2, ..., GT_n}`
- prediction: `P = {P_1, P_2, ..., P_m}`

Each GT item represents one logical requirement or constraint.  
For consistent extraction, split both texts by numbered fragments.

### Step 2. Compute Pairwise Semantic Similarity Matrix

For each sample, compute semantic similarity for all `P_i x GT_j` pairs using Sentence-BERT cosine similarity:

`sim(P_i, GT_j) = cosine_similarity(embedding(P_i), embedding(GT_j))`

The pairwise matrix is a raw audit layer and must be preserved.

### Step 3. Build a GT-Centric Coverage Table

The GT-centric table is the primary review artifact and source of truth.  
Each row corresponds to one GT rule.

For each `GT_j`, record:

- matched predicted IDs
- matched predicted positions
- matched predicted texts
- best similarity
- matched predicted fragment count
- coverage structure and quality

Coverage columns:

- `coverage_structure`: `exact`, `split`, `missing`
- `coverage_quality`: `strong`, `weak` (blank for `missing`)

Interpretation:

- `exact + strong`: full high-quality coverage by one predicted fragment
- `split + strong`: good coverage distributed across fragments
- `split + weak`: distributed but weak/incomplete coverage
- `exact + weak`: one fragment exists but semantic quality is weak
- `missing + blank`: no coverage found

`coverage_quality` is not defined only by `best_similarity`; it reflects the quality of the selected coverage set.

#### Operational Rule for Automatic GT Coverage Assignment

Use these rules:

- matched predicted fragments = all fragments with `similarity >= RULE_MATCH_THRESHOLD`
- if matched count = `0` -> `coverage_structure = missing`
- if matched count = `1` -> `coverage_structure = exact`
- if matched count > `1` -> `coverage_structure = split`
- if structure is `missing` -> `coverage_quality` is blank
- if structure is `exact` and similarity >= `RULE_STRONG_MATCH_THRESHOLD` -> `strong`, else `weak`
- if structure is `split` and all matched similarities >= `RULE_STRONG_MATCH_THRESHOLD` -> `strong`, else `weak`

### Step 4. Human Review and Adjudication

`2_1-2_gt_coverage_summary.csv` is the authoritative review file.

Reviewer columns used in the current pipeline:

- `human_coverage_structure`
- `human_coverage_quality`
- `human_best_pred_ids`
- `human_notes`

Adjudication logic:

- human values override automatic values when provided
- otherwise automatic values are retained
- machine and human values are stored side by side for auditability
- required final GT-centric fields:
  - `final_coverage_structure`
  - `final_coverage_quality`
  - `final_best_pred_ids`

Final counts must be computed from final labels.

### Step 5. Export Intermediate and Review CSV Files

All review-oriented CSV files should include a shared context block:

- `generated_question_id`
- `layer_id`
- `question_text`
- `full_ground_truth_answer`
- `full_model_answer`

#### Stage-to-Artifact Mapping (What Is Produced at Each Step)

This mapping clarifies when each intermediate/review artifact is formed and which fragments are written into it.

| Pipeline step | Internal fragment(s) produced | Output artifact(s) populated |
| --- | --- | --- |
| **Step 2. Pairwise similarity** | Pairwise records for every `GT x Pred` pair: `gt_rule_id`, `pred_rule_id`, `gt_text`, `pred_text`, `similarity_score` (+ diagnostic flags) | `2_1-2_all_pairwise_similarity.csv` |
| **Step 3. GT-centric coverage construction** | GT-side coverage fragments per `gt_rule_id`: `matched_pred_rule_ids`, `matched_pred_positions`, `matched_pred_texts`, `best_similarity`, `matched_pred_count`, `coverage_structure`, `coverage_quality` | Initial GT-centric rows later materialized into `2_1-2_gt_coverage_summary.csv` |
| **Step 4. Human review / adjudication** | Human overrides and final GT labels: `human_*`, `final_coverage_*`, `final_best_pred_ids` | Updated/authoritative `2_1-2_gt_coverage_summary.csv`; derived sample totals for metrics |
| **Candidate-layer materialization (between Step 2 and Step 4)** | Predicted-centered candidate rows per `pred_rule_id`: selected/best GT match and adjudication fields (`matched_gt_id`, `auto_label`, `human_label`, `final_label`, etc.) | `2_1-2_rule_classification_candidate_matches.csv` |
| **Final aggregation from adjudicated GT table** | One row per sample with counts/rates (`gt_total`, `gt_covered_count`, `pred_extra_count`, `coverage_recall`, etc.) | `2_1-2_sample_level_metrics.csv` |

Notes:
- `2_1-2_all_pairwise_similarity.csv` is the raw audit matrix (all combinations).
- `2_1-2_rule_classification_candidate_matches.csv` is a compressed predicted-centered helper layer (diagnostic/review support), not the full `GT x Pred` space.
- `2_1-2_gt_coverage_summary.csv` is the authoritative GT-centered review/adjudication table used for final counting.

#### `2_1-2_all_pairwise_similarity.csv`

Raw pairwise matrix for semantic-audit and traceability of candidate generation.

Columns produced by the current pipeline:

- `generated_question_id`
- `layer_id`
- `question_text`
- `full_ground_truth_answer`
- `full_model_answer`
- `gt_rule_id`
- `gt_rule_position`
- `gt_text`
- `pred_rule_id`
- `pred_rule_position`
- `pred_text`
- `similarity_score`
- `candidate_flag`
- `auto_match`

`candidate_flag` and `auto_match` are diagnostic fields, not final reviewed labels.

#### `2_1-2_rule_classification_candidate_matches.csv`

Predicted-centered candidate layer.  
Contains a compressed candidate result per `pred_rule_id` (selected/best match) and review/adjudication fields.

Typical fields include:

- `matched_gt_id`
- `auto_label`
- `human_label`
- `final_label`
- `human_gt_assignment`
- `human_pred_assignment`
- `final_gt_assignment`
- `final_pred_assignment`

This artifact is supplementary and diagnostic. It is useful for targeted inspection and additional control, but it is not the mandatory single source for final decisioning.

#### `2_1-2_gt_coverage_summary.csv`

Primary GT-centric review table (authoritative after review).

Columns produced by the current pipeline:

- `generated_question_id`
- `row_index`
- `layer_id`
- `question_text`
- `full_ground_truth_answer`
- `full_model_answer`
- `gt_rule_id`
- `gt_rule_position`
- `gt_text`
- `matched_pred_rule_ids`
- `matched_pred_positions`
- `matched_pred_texts`
- `best_similarity`
- `matched_pred_count`
- `coverage_structure`
- `coverage_quality`
- `human_coverage_structure`
- `human_coverage_quality`
- `human_best_pred_ids`
- `final_best_pred_ids`
- `human_notes`

#### `2_1-2_pred_support_summary.csv`

Optional derived artifact for inspection only.  
Can be regenerated from GT-centric reviewed outputs.

#### `2_1-2_sample_level_metrics.csv`

One-row-per-sample summary, computed from final GT-centric reviewed labels.

Columns produced by the current pipeline:

- `generated_question_id`
- `row_index`
- `layer_id`
- `question_text`
- `full_ground_truth_answer`
- `full_model_answer`
- `gt_total`
- `gt_covered_count`
- `gt_missing_count`
- `gt_split_count`
- `pred_total`
- `pred_extra_count`
- `coverage_recall`
- `split_rate`
- `missing_rule_rate`
- `extra_rule_rate`
- optional: `gt_weak_count`

## 6. Error Categories

- **Missing rules:** required GT rules were not covered
- **Weak coverage:** GT rules were covered weakly
- **Extra rules:** predicted rules with no GT correspondence

## 7. Final Counting Rules

After human validation, counts must be computed from final GT-centric labels.

For each sample:

- `gt_covered_count` = number of GT rules with final structure `exact` or `split`
- `gt_missing_count` = number of GT rules with final structure `missing`
- `gt_split_count` = number of GT rules with final structure `split`
- optional `gt_weak_count` = number of GT rules with final quality `weak` and non-`missing` structure

Extra-rule counting is GT-based:

1. Build the set of predicted IDs matched in reviewed GT rows (`final_best_pred_ids`).
2. Compute `pred_total` from decomposed predicted rules.
3. Compute `pred_extra_count = pred_total - unique_matched_pred_count` (bounded at zero if needed).
4. Compute `extra_rule_rate = pred_extra_count / pred_total`.

Derived metrics:

- `Coverage Recall = gt_covered_count / gt_total`
- `Split Rate = gt_split_count / gt_total`
- `Missing Rule Rate = gt_missing_count / gt_total`
- `Extra Rule Rate = pred_extra_count / pred_total`

## 8. Field Reference and Output Cheat Sheet

### 8.1 Quick Definitions (plain language)

- **sample**: one question row in the input file (`2_1-2_with_answers_*.csv`).  
  One sample = one question + one GT answer + one model answer.
- **GT fragment**: one atomic rule extracted from `full_ground_truth_answer`.
- **Pred fragment**: one atomic rule extracted from `full_model_answer`.
- `gt_rule_id` / `pred_rule_id`: internal fragment numbers starting from `0` (for code).
- `gt_rule_position` / `pred_rule_position`: fragment order starting from `1` (for people reading tables).
- `triage_zone`: quick confidence bucket from similarity score:
  - `auto_accept` = likely correct
  - `uncertain` = check manually
  - `auto_reject` = likely incorrect

### 8.2 Field Map: Source -> Meaning -> Where Used

| Field | Source | Meaning | Used in table(s) | Why this field is needed |
| --- | --- | --- | --- | --- |
| `generated_question_id` | input sample row | question ID | all outputs | links rows from different tables to the same question |
| `row_index` | input sample row index | row number in the run | all outputs | keeps rows aligned and easy to find |
| `layer_id` | input sample | task type (`2_1` / `2_2`) | all outputs | separates task subsets |
| `question_text` | input sample | question text | all outputs | human-readable context |
| `full_ground_truth_answer` | input sample | full GT answer text | all outputs | source text for GT fragments |
| `full_model_answer` | input sample | full model answer text | all outputs | source text for predicted fragments |
| `gt_rule_id` | GT decomposition | GT fragment ID (0-based) | `2_1-2_all_pairwise_similarity.csv`, `2_1-2_gt_coverage_summary.csv` | identifies a specific GT fragment in code |
| `gt_rule_position` | GT decomposition | GT fragment order (1-based) | `2_1-2_all_pairwise_similarity.csv`, `2_1-2_gt_coverage_summary.csv` | easier for manual reading |
| `gt_text` | GT decomposition | GT fragment text | `2_1-2_all_pairwise_similarity.csv`, `2_1-2_gt_coverage_summary.csv` | text being matched |
| `pred_rule_id` | prediction decomposition | predicted fragment ID (0-based) | `2_1-2_all_pairwise_similarity.csv`, `2_1-2_rule_classification_candidate_matches.csv` | identifies a predicted fragment in code |
| `pred_rule_position` | prediction decomposition | predicted fragment order (1-based) | `2_1-2_all_pairwise_similarity.csv`, `2_1-2_rule_classification_candidate_matches.csv` | easier for manual reading |
| `pred_text` | prediction decomposition | predicted fragment text | `2_1-2_all_pairwise_similarity.csv`, `2_1-2_rule_classification_candidate_matches.csv` | text being matched |
| `similarity_score` | SBERT comparison | semantic similarity number | `2_1-2_all_pairwise_similarity.csv`, `2_1-2_rule_classification_candidate_matches.csv`, `2_1-2_gt_coverage_summary.csv` | core score used to decide match quality |
| `candidate_flag` | pipeline diagnostic | technical marker for candidate rows | `2_1-2_all_pairwise_similarity.csv` | debugging and quality checks |
| `auto_match` | threshold on `similarity_score` | 0/1 automatic match | `2_1-2_all_pairwise_similarity.csv`, `2_1-2_rule_classification_candidate_matches.csv` | baseline machine decision before manual override |
| `triage_zone` | from score thresholds | `auto_accept` / `uncertain` / `auto_reject` | `2_1-2_all_pairwise_similarity.csv`, `2_1-2_rule_classification_candidate_matches.csv`, `2_1-2_gt_coverage_summary.csv` | shows which rows need manual attention first |
| `needs_human_review` | from `triage_zone` | 0/1 manual-check flag | same as `triage_zone` | marks rows for human review |
| `matched_gt_id` | candidate selection | selected GT for one predicted fragment | `2_1-2_rule_classification_candidate_matches.csv` | stores pred->GT candidate mapping |
| `matched_gt_position` | candidate selection | human-readable position of selected GT | `2_1-2_rule_classification_candidate_matches.csv` | easier manual validation |
| `human_review_required` | normalized review flag | explicit 0/1 review flag | `2_1-2_rule_classification_candidate_matches.csv` | stable control flag for review workflow |
| `auto_label` | from `auto_match` | automatic label | `2_1-2_rule_classification_candidate_matches.csv` | baseline label |
| `human_label` | reviewer input | manual label override | `2_1-2_rule_classification_candidate_matches.csv` | lets reviewer correct auto label |
| `final_label` | adjudication | final label after overrides | `2_1-2_rule_classification_candidate_matches.csv` | finalized candidate label |
| `human_gt_assignment` | reviewer input | manual GT assignment override | `2_1-2_rule_classification_candidate_matches.csv` | reviewer can remap GT target |
| `human_pred_assignment` | reviewer input | manual predicted assignment override | `2_1-2_rule_classification_candidate_matches.csv` | reviewer can remap predicted target |
| `final_gt_assignment` | adjudication | final GT assignment | `2_1-2_rule_classification_candidate_matches.csv` | final trace of GT mapping |
| `final_pred_assignment` | adjudication | final predicted assignment | `2_1-2_rule_classification_candidate_matches.csv` | final trace of prediction mapping |

### 8.3 Output Tables at a Glance

- `2_1-2_all_pairwise_similarity.csv`: full raw `GT x Pred` similarity matrix for audit and traceability.
- `2_1-2_rule_classification_candidate_matches.csv`: predicted-centered compressed candidate layer (one selected/best candidate per `pred_rule_id`) for diagnostic review support.
- `2_1-2_gt_coverage_summary.csv`: authoritative GT-centered review/adjudication table used for final counting.
- `2_1-2_sample_level_metrics.csv`: one-row-per-sample final metrics derived from adjudicated GT coverage.
- `2_1-2_rule_understanding_summary_table.csv`: compact cross-model metric summary table.
- `2_1-2_rule_understanding_summary_table.xlsx`: Excel version of the same final summary.
