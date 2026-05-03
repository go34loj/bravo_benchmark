# Backend Architecture (Dataset Generation Pipeline)

This document consolidates the backend architecture for dataset generation based on the current `backend/` folder structure in the repo (as of 2026-05-02). It serves as technical documentation for users who want to run and extend the generation pipeline from CLI.

## Table of Contents
- [0. Core QA generator](#0-core-qa-generator)
  - [`csv_unifier.py` (end-to-end database builder)](#csv_unifierpy-end-to-end-database-builder)
  - [`data_manipulation.py` (dataset linking and cleanup)](#data_manipulationpy-dataset-linking-and-cleanup)
- [1. Current Folder Structure](#1-current-folder-structure)
- [2. Data Access Layer (read-only adapters)](#2-data-access-layer-read-only-adapters)
- [3. Scene perception QA generation](#3-scene-perception-qa-generation)
- [4. Scene understanding QA generation](#4-scene-understanding-qa-generation)
- [5. Rule understanding QA generation](#5-rule-understanding-qa-generation)
- [6. Compliance/reasoning QA generation](#6-compliancereasoning-qa-generation)
- [7. unified_database.db structure](#7-unified_databasedb-structure)
- [8. Future work: Core Problems to Solve](#8-future-work-core-problems-to-solve)

---

## 0. Core QA generator

### `csv_unifier.py` (end-to-end database builder)
This is the main entry point that builds `unified_database.db` from CSV sources and then runs the current generation pipelines.
It loads templates and rules, optionally loads rule figures and cutouts, then writes everything into SQLite and generates derived question tables.
It also runs Scene Perception generation, Scene Understanding generation, and Compliance QA generation when their inputs are available.
It supports backend configuration via `config.yaml` (default path: `backend/config.yaml`).

Key behaviors:
- Auto-detects latest CSVs in `backend/` if paths are not provided.
- Config-aware path resolution with precedence: `CLI args > config.yaml > auto-detected files`.
- Uses schema-first DB setup from `utils/database_schema.py` (`setup_connection`, `ensure_schema`, `validate_integrity`).
- Creates/migrates master and generated tables before data load; ID columns are normalized to consistent types on insert.
- Writes tables: `templates`, `rules`, optional `rule_figures`, `cutout`, `viewpoint_scenes`, and generated question tables.
- Centralizes execution order and prints a DB summary at the end.
- Supports execution flags from `config.yaml`:
  - `run_rule_understanding`
  - `run_scene_perception`
  - `run_scene_understanding`
  - `run_compliance_reasoning`
  - `run_scene_perception_gt`
  - `run_compliance_gt`
- If GT flags are enabled:
  - Scene Perception GT is filled by `utils/scene_perc_gen_gt.py` logic (`update_ground_truth(...)`).
  - Compliance GT is filled by running `utils/compliance_bool_gt_gen.py`.

Schema-first integration in `csv_unifier.py`:
- Import:
  - `from utils.database_schema import setup_connection, ensure_schema, validate_integrity`
- DB flow:
  1. `conn = setup_connection(output_sqlite)`
  2. `ensure_schema(conn)`
  3. load/insert source tables and generated tables into pre-created schema (no `to_sql(..., replace)` for key tables)
  4. `validate_integrity(conn)` before final summary/close

### `data_manipulation.py` (dataset linking and cleanup)
This is a maintenance script used before generation to repair and enrich the templates and scene metadata.
It links ambiguous rules to templates, filters rule links by figure availability, and can update viewpoint scene file paths from the `scenes/` directory.

Key behaviors:
- `link_ambiguous_rules_to_templates`: adds rule IDs to `requires_rule` for templates with subtask `ambiguity` and applies figure-based filtering.
- `update_viewpoint_scene_paths`: overwrites `file_path` in the viewpoint scenes CSV based on `scene_id` and matching filenames.
- Produces `templates_updated.csv` by default.

---

## 1. Current Folder Structure

This section documents the folders and files that are part of the dataset generation pipeline.

### A) Top-level folders
- `rule_figure/` - rule figure images (illustrations from codes / standards)
- `scenes/` - scene images (BIM / drawing views)
- `utils/` - core utilities (see below)

### B) Utilities (current contents of `utils/`)
- `compliance_bool_gt_gen.py`
- `compliance_question_gen.py`
- `data_model.py`
- `data_parsers.py`
- `database_utils.py`
- `file_operations.py`
- `qualitycheck.py`
- `rule_question_processing.py`
- `scene_perc_correct_answ.py`
- `scene_perc_material_gt.py`
- `scene_perc_quest_gen.py`
- `scene_underst_QA_gen.py`
- `sc_underst_QA_func_group_gen.py`

### C) Top-level scripts (pipelines / tooling)
- `csv_unifier.py` (unification pipeline)
- `amb_correct_answ_gener.py` (correct answer generator; WIP)
- `data_manipulation.py`

### D) Data assets (overview)
- CSV inputs and templates:
  - `03_04_VQA_Rules_Scenes_Templates(RASE).csv`
  - `03_04_VQA_Rules_Scenes_Templates(templates).csv`
  - `10_04_VQA_Rules_Scenes_Templates(viewpoint_scenes).CSV`
  - `20_03_VQA_Rules_Scenes_Templates(cutouts).csv`
  - `20_03_VQA_Rules_Scenes_Templates(rules_sort).csv`
  - `09_01_VQA_Rules_Scenes_Templates(rule_figure).csv`
  - `scene_perception_questions.csv`
  - `03_04_reasonLLMjudgeCalibr.csv`
  - `final_VQA_metric_rubrics.CSV`
- Unified DB:
  - `unified_database.db`

### E) Docs
- `backend_architecture.md`
- `SETUP.md`
- `unified_backend_architecture.md`

### F) Backend config
- `config.yaml` - backend-only configuration file for dataset generation paths, output table names, and execution flags.

---

## 2. Data Access Layer (read-only adapters)
- Reads from `unified_database.db` (templates, viewpoint_scenes, and any helper tables).
- Normalizes column names (`_resolve_column`) and parses list-like cells (template_id lists, object_type lists, feature maps, material maps).
- Outputs strongly-typed records (`TemplateConfig`, `SceneRecord`, `SceneUnderstandingRecord`) so the next layers do not depend on raw SQL/CSV quirks.

Scripts in `utils/data_access_layer/` (excluding `qualitycheck.py`):
- `database_utils.py` - SQLite helpers: `setup_database`, `save_dataframe_to_db`, `print_database_summary`.
- `data_model.py` - shared data structures.
- `data_parsers.py` - parsing/normalization utilities.
- `file_operations.py` - CSV discovery and loading.

---

## 3. Scene perception QA generation
### 1) Sources
Entry point:
- `csv_unifier.py` (function `create_database(...)`) runs Scene Perception generation via `generate_scene_questions(...)` and writes to `generated_scene_questions`.

Scene Perception generation uses three scripts:
- `utils/scene_perc_quest_gen.py` - question generation
- `utils/scene_perc_router.py` - route classification for templates
- `utils/scene_perc_gen_gt.py` - ground truth generation/update

The generation is template-driven and scene-driven:
- templates come from table `templates` (`question_template`, `benchmark_layer`, `subtask`, `context`, `answer_type`);
- scene data comes from table `viewpoint_scenes`;
- generated rows are written to table `generated_scene_questions`.

For extension work, the key contract is: **every new Scene Perception template must be routable by `scene_perc_router.py` and must have required input fields available in `viewpoint_scenes`**.

---

### 2) High-Level Architecture (Scene Perception generation only)

#### A) Task Routing Layer (no more template_id branching)
- `scene_perc_router.py` reads the template text from `templates.question_template` (loaded in `scene_perc_quest_gen.py -> load_templates(...)`).
- Routing does **not** depend on `template_id`; it depends on:
  - `benchmark_layer` (must be `scene_perception`);
  - placeholders found in `question_template` (`{object_type}`, `{feature_name}`, `{location_context}`, `{option1..4}`, `{highlight}`);
  - `subtask`, `context`, and `answer_type`.
- Router output is one route key: `STATIC`, `OBJ_BOOL`, `FEATURE_BOOL`, `MATERIAL_COLOR_TEXTURE`, `OCR`, `REGION`, `UNSUPPORTED`, `MISSING`.
- Router is classification only; it does not create rows.
- `REGION` route is triggered by `{highlight}` placeholder in scene-perception templates and is intended for future highlighted-object/region-based questions (where specific objects are visually marked). In the current dataset stage this route is not implemented in generation and is treated as unsupported, but it is intentionally kept as an extension point for scaling.

#### B) Generation Layer (handlers + contracts)
`scene_perc_quest_gen.py` (question generation):
- Input tables/data:
  - `templates` (from templates CSV loaded by `csv_unifier.py`);
  - `viewpoint_scenes` (from viewpoint scenes CSV loaded by `csv_unifier.py`).
- Scene rows are converted to `SceneRecord`; templates are converted to `TemplateConfig`.
- For each scene, the script reads `viewpoint_scenes.template_id` list and selects matching templates.
- For each `(scene, template)`, it calls `route_scene_perception_template(...)` and dispatches to route-specific generators.
- Output table: `generated_scene_questions`.

`scene_perc_gen_gt.py` (ground truth generation/update):
- Input tables:
  - `generated_scene_questions` (generated instances);
  - `templates` (to re-route each template consistently);
  - `viewpoint_scenes` (source of object/feature/material/text/dimension truth).
- Updates `generated_scene_questions.ground_truth_answer` in place.
- For material MCQ routes it may also update `question_text` by appending options.

### 3) Current table/output structure for Scene Perception
Canonical output table in current implementation is:
- `generated_scene_questions` (in `backend/unified_database.db`).

Current columns in `generated_scene_questions`:
- `generated_question_id` INTEGER
- `template_id` INTEGER
- `layer_id` TEXT
- `scene_id` TEXT
- `file_path` TEXT
- `question_text` TEXT
- `object_type_filled` TEXT
- `feature_name_filled` TEXT
- `ground_truth_answer` TEXT
- `answer_type` TEXT
- `dimension_source` TEXT
- `dimension_key_filled` TEXT
- `dimension_unit` TEXT
- `dimension_source_view` TEXT

Additional note on dimension-related fields:
- Columns `dimension_source`, `dimension_key_filled`, `dimension_unit`, and `dimension_source_view` are used to increase the number of OCR-style dimension questions for a single view / single element context.
- Although the source data may come from documented multi-view dimensions, the resulting question type and phrasing are different from Scene Understanding multiview integration tasks.
- These Scene Perception OCR questions therefore do **not** evaluate cross-view reasoning or integration of information across multiple drawing projections.

Input tables used by Scene Perception scripts:
- `templates`:
  - required by question generation and routing (`template_id`, `question_template`, `benchmark_layer`, `subtask`, `context`, `answer_type`, optional `commands`);
- `viewpoint_scenes`:
  - required by question generation and ground truth (`scene_id`, `file_path`, `template_id`, `object_type`, `feature_name`, `material`, `text`, `multi-view_dimensions`).

Note:
- Ground truth is stored directly in `generated_scene_questions.ground_truth_answer`.
- Running only `csv_unifier.py` is not sufficient to populate Scene Perception ground truth: it creates `generated_scene_questions`, but `ground_truth_answer` is filled only after a separate run of `utils/scene_perc_gen_gt.py`.

---

## 4. Scene understanding QA generation
### 1) Sources
Entry point:
- `csv_unifier.py` (function `create_database(...)`) runs Scene Understanding generation after loading `viewpoint_scenes`.
- It calls:
  - `load_scene_understanding_scenes(...)`
  - `generate_connectivity_flagging_rows(...)`
  - `generate_connectivity_relation_rows(..., relation_type="not_direct_connectivity")`
  - `generate_adjacency_flagging_rows(...)`
  - `generate_functional_grouping_rows(...)`
  - `generate_multiview_dimension_rows(...)`
- The concatenated output is saved to `generated_scene_understanding_questions`.

Main scripts:
- `utils/scene_underst_QA_gen.py` - relation generation orchestrator and loaders
- `utils/sc_underst_QA_func_group_gen.py` - functional grouping generation
- `utils/scen_unders_OCR_QA_gen.py` - multiview dimensions OCR generation

### 2) High-Level Architecture (Scene Understanding generation only)

#### A) Input tables
- `templates`:
  - used by all Scene Understanding generators;
  - key fields: `template_id`, `layer_id`, `question_template`, `answer_type`, `context`, `subtask`, `metrics`.
- `viewpoint_scenes`:
  - relation inputs: `space_naming`, `spatial relations` (or normalized `spatial_relations`);
  - multiview OCR input: `multi_view_dimensions` / `multi-view_dimensions`;
  - common fields: `scene_id`, `file_path`.
- `cutout`:
  - optional input for functional-grouping caption templates (`cutout_id` -> `caption`).

#### B) Relation-based generation (`scene_underst_QA_gen.py`)
- `load_scene_understanding_scenes(...)` parses relation text into `SceneUnderstandingRecord` using:
  - `parse_scene_relations(space_naming, source_column=...)`;
  - `parse_scene_relations(spatial_relations, source_column=...)`.
- Connectivity:
  - templates filtered by `context` containing `connectivity flagging`;
  - `generate_connectivity_flagging_rows(...)` creates positive rows (`ground_truth_answer = "yes"`).
- Not-direct connectivity:
  - `generate_connectivity_relation_rows(..., relation_type="not_direct_connectivity")`;
  - generated with the same connectivity templates, GT is `no`.
- Adjacency:
  - templates filtered by `context` containing `adjacency flagging`;
  - positives from adjacency relations (`yes`);
  - negatives from ordered set difference `(connectivity pairs - adjacency pairs)` (`no`).

#### C) Functional grouping (`sc_underst_QA_func_group_gen.py`)
- Templates are filtered by `context` containing `functional grouping`.
- Three template classes are detected from placeholders:
  - MCQ templates (`{option1..4}`);
  - contains-item templates (`{object type}` / `{object/space}`);
  - caption templates (`{caption}`).
- Output behavior:
  - MCQ rows include `options` (JSON) and `correct_option_index`;
  - contains-item rows produce both positive and negative yes/no samples;
  - caption templates use `cutout` captions when available and also create negative rows with a non-matching group label.

#### D) MultiView Dimensions / OCR (`scen_unders_OCR_QA_gen.py`)
- Templates are selected strictly by:
  - `subtask == "multiview_ocr_text_detection"`;
  - `context == "multi_view_dimensions"`.
- For each scene and object in parsed `multi_view_dimensions`, generator fills:
  - `{object_type}`;
  - `{dimension array}` / `{dimension_array}` (JSON list of dimension keys).
- `ground_truth_answer` is stored as JSON for that object’s dimension map.

### 3) Current table/output structure for Scene Understanding
All Scene Understanding outputs are concatenated and written to:
- `generated_scene_understanding_questions`.

Current union schema (nullable fields depend on row type):
- Common core:
  - `generated_question_id`, `template_id`, `layer_id`, `answer_type`, `metrics`,
  - `scene_id`, `file_path`, `question_text`,
  - `source_column`, `entity_scope`, `relation_type`,
  - `main_entity`, `ground_truth_answer`.
- Relation rows:
  - `related_entity`.
- Functional grouping rows:
  - `group_label`,
  - optional `options` (JSON),
  - optional `correct_option_index`.
- MultiView rows:
  - `relation_type = "multi_view_dimensions"`,
  - JSON ground truth in `ground_truth_answer`.

---

## 5. Rule understanding QA generation
### 1) Sources
Entry point:
- `csv_unifier.py` calls `generate_rule_questions(...)` from `utils/rule_question_processing.py` after loading `templates`, `rules`, and optional `rule_figures`.
- Inside `generate_rule_questions(...)`, additional generators are called:
  - `generate_rase_atomic_questions(...)`
  - `generate_rase_requirement_flagging_questions(...)`
  - `generate_rule_understanding_questions(...)` (rule text atomisation)

Main scripts:
- `utils/rule_question_processing.py` - base rule-question generation and orchestration
- `utils/rule_under_classific_QA.py` - atomisation + RASE-based generators
- `utils/amb_correct_answ_gener.py` - GT labeling for ambiguity/classification tasks in `generated_questions`

### 2) High-Level Architecture (Rule Understanding generation only)

#### A) Input tables
- `templates`:
  - key fields: `template_id`, `question_template`, `context`, `answer_type`, `requires_rule`.
- `rules`:
  - key fields: `rule_id`, `rule_text_atomic`, `parent_rule_id`, `parent_rule_text`, `classification`, `ambiguity`, optional figure fields.
- `rule_figures` (optional):
  - key fields: `figure_id`, `file_path`, `caption`.
- RASE CSV (`*RASE*.csv`, auto-detected or passed explicitly):
  - used by `generate_rase_atomic_questions(...)` and `generate_rase_requirement_flagging_questions(...)`.

#### B) Base rule-question generation (`rule_question_processing.py`)
- For each template, `detect_rule_text_placeholder(...)` determines which rule text is required.
- `requires_rule` is parsed and matched against `rules` via normalized IDs.
- Question text is produced by replacing placeholders in `question_template`.
- Optional figure metadata is attached when `figure_required == yes` and figure exists in `rule_figures`.
- Output is merged into `generated_questions`.
- Also supports `context == "rule figures understanding"` templates and emits rows with caption-based ground truth.

#### C) Rule text atomisation and RASE (`rule_under_classific_QA.py`)
- `generate_rule_understanding_questions(db_path)`:
  - filters templates with `context == "rule text atomisation"`;
  - groups atomic rules by `parent_rule_id`;
  - builds numbered atomic ground-truth lists and writes to `generated_questions`.
- `generate_rase_atomic_questions(conn, ...)`:
  - filters templates with `context == "RASE for atomic"`;
  - uses RASE CSV applicability mapping to derive target rule expansions;
  - builds structured ground truth with Applicability/Requirement/Selection/Exception lines;
  - stores trace columns like `target_rule_id_used`, `applicability_rule_ids_used`, `requirement_text_used`.
- `generate_rase_requirement_flagging_questions(conn, ...)`:
  - filters templates with `context == "RASErequirement flagging"`;
  - builds numbered Requirement / Requirement (within Exception) ground truth.

#### D) GT labeling for ambiguity/classification (`amb_correct_answ_gener.py`)
- Updates `generated_questions.ground_truth_answer` in place for:
  - boolean ambiguity templates (`rule_text_atomic` and `parent_rule_text`);
  - multichoice category templates (`categories` and `sub-categories` patterns).
- Uses `rules.classification` and `rules.ambiguity` plus internal `SUBCATEGORY_MAPPING`.

### 3) Current table/output structure for Rule Understanding
Current output table:
- `generated_questions`.

Main fields used in current code:
- `generated_question_id`, `template_id`, `question_text`
- `rule_id`, `parent_rule_id`
- `ground_truth_answer`
- context/template metadata (`context`, `layer_id`, `benchmark_layer`, `subtask`, `answer_type`, `metrics`)
- optional figure fields (`figure_id`, `figure_path`, `figure_caption`, `figure_required`)
- RASE trace fields (`target_rule_id_used`, `applicability_rule_ids_used`, `requirement_text_used`).

Note:
- Rows are append-merged by multiple generators.
- Table schema is dynamic (columns can be added by merge/ALTER logic), not enforced by a static migration.

---

## 6. Compliance/reasoning QA generation
### 1) Sources
Entry point:
- `csv_unifier.py` calls `generate_compliance_questions(...)` from `utils/compliance_question_gen.py`.
- After question generation, GT is generated by running `utils/compliance_bool_gt_gen.py` (separate step) to populate `correct_answer`.

Main scripts:
- `utils/compliance_question_gen.py` - compliance question instance generation
- `utils/compliance_bool_gt_gen.py` - boolean GT assignment in generated compliance table

### 2) High-Level Architecture (Compliance/reasoning generation only)

#### A) Input tables
- `templates`:
  - generator targets template IDs `{38, 39, 41, 42}`.
- `viewpoint_scenes`:
  - `scene_id`, `template_id`, `rule_id`, optional `not sufficient`, `cutout_id`, `file_path`.
- `rules`:
  - atomic and parent rule texts + labels + optional figure references.
- `cutout`:
  - used for parent-rule expansion paths.
- `rule_figures` (optional):
  - used to attach figure assets.

#### B) Question generation (`compliance_question_gen.py`)
- Creates/recreates table `generated_compliance_questions`.
- Handles both atomic-rule and parent-rule placeholder templates.
- Parent-rule branch:
  - resolves parent rules through `cutout.parent_rule_id`;
  - expands to one question per parent rule.
- Template `39` has additional balancing logic:
  - combines baseline rule IDs with `not sufficient` negatives;
  - applies a cap based on target positive/negative share.
- Stores question text plus trace fields (`rule_id`, `rule_text_atomic_used`, `parent_rule_id`, labels, figure metadata).

#### C) Ground truth generation (`compliance_bool_gt_gen.py`)
- Updates `generated_compliance_questions.correct_answer` in place.
- Default logic:
  - if rule is present in scene non-compliance set -> `no`;
  - otherwise -> `yes`.
- Template `39` logic uses `viewpoint_scenes.not sufficient` instead of `not compliant`.
- Parent-rule rows are mapped to atomic rules through `rules.parent_rule_id` and then evaluated against scene sets.

### 3) Current table/output structure for Compliance/reasoning
Current output table:
- `generated_compliance_questions`.

Current columns (from DB):
- `generated_question_id` (PK in this table),
- `scene_id`, `file_path`, `template_id`, `question_text`,
- `rule_id`, `rule_text_atomic_used`,
- `parent_rule_id`, `parent_rule_text_used`,
- `classification`, `ambiguity`, `classification_parent`,
- `rule_figure_required`, `rule_figure_id`, `rule_figure_asset`,
- `answer_type`, `correct_answer`.

Note:
- `correct_answer` is not final after `csv_unifier.py` alone; run `compliance_bool_gt_gen.py` to populate it.

---

## 7. unified_database.db structure
`unified_database.db` is the central SQLite relational store used by the generation pipeline.
It aggregates source tables (templates/rules/scenes/cutouts/figures) and generated question tables for each benchmark layer.

### 1) Source tables
- `templates` - question template definitions and metadata.
- `rules` - atomic and parent rule texts + labels.
- `viewpoint_scenes` - scene-level annotations and links to templates/rules.
- `cutout` - cutout metadata and parent rule links.
- `rule_figures` - figure metadata and assets.

### 2) Generated tables
- `generated_questions` - rule-understanding outputs.
- `generated_scene_questions` - scene-perception outputs.
- `generated_scene_understanding_questions` - scene-understanding outputs.
- `generated_compliance_questions` - compliance reasoning outputs.

### 3) Keys and relations in current DB state
- Master table PKs (schema-first):
  - `templates.template_id` (TEXT PRIMARY KEY)
  - `rules.rule_id` (TEXT PRIMARY KEY)
  - `viewpoint_scenes.scene_id` (TEXT PRIMARY KEY)
  - `rule_figures.figure_id` (TEXT PRIMARY KEY)
  - `cutout.cutout_id` (TEXT PRIMARY KEY)
- Generated table PKs:
  - `generated_scene_questions.generated_question_id` (INTEGER PRIMARY KEY)
  - `generated_scene_understanding_questions.generated_question_id` (INTEGER PRIMARY KEY)
  - `generated_questions.generated_question_id` (INTEGER PRIMARY KEY)
  - `generated_compliance_questions.generated_question_id` (INTEGER PRIMARY KEY)
- Foreign keys (enforced when PRAGMA `foreign_keys=ON`):
  - `generated_scene_questions.template_id` -> `templates.template_id`
  - `generated_scene_questions.scene_id` -> `viewpoint_scenes.scene_id`
  - `generated_scene_understanding_questions.template_id` -> `templates.template_id`
  - `generated_scene_understanding_questions.scene_id` -> `viewpoint_scenes.scene_id`
  - `generated_questions.template_id` -> `templates.template_id`
  - `generated_questions.rule_id` -> `rules.rule_id`
  - `generated_questions.figure_id` -> `rule_figures.figure_id`
  - `generated_compliance_questions.template_id` -> `templates.template_id`
  - `generated_compliance_questions.scene_id` -> `viewpoint_scenes.scene_id`
  - `generated_compliance_questions.rule_id` -> `rules.rule_id`

### 4) Logical (application-level) relations used by code
- `generated_* .template_id` -> `templates.template_id`
- `generated_* .scene_id` -> `viewpoint_scenes.scene_id`
- `generated_questions.rule_id` / `generated_compliance_questions.rule_id` -> `rules.rule_id`
- `generated_questions.parent_rule_id` / `generated_compliance_questions.parent_rule_id` -> `rules.parent_rule_id`
- figure references (`figure_id`, `rule_figure_id`) -> `rule_figures.figure_id`
- compliance parent mapping via `cutout.cutout_id` -> `cutout.parent_rule_id`

Note:
- Relations are now defined in DB schema and validated at the end of `csv_unifier.py` via `validate_integrity(conn)` (`PRAGMA foreign_key_check` + PK uniqueness checks).

---

## 8. Future work: Core Problems to Solve
1. Missing schema constraints and migrations:
- Most tables have no enforced PK/FK, and schema evolves dynamically via `to_sql`/`ALTER`.
- Risk: silent duplicates, orphan links, and hard-to-debug drift across runs.

2. Context matching and routing stability:
- Several generators rely on context substring matching or naming heuristics.
- Risk: template renames can silently change routing behavior.

3. Ground-truth generation is split across extra scripts:
- Scene Perception and Compliance GT are separate post-steps.
- Risk: users run `csv_unifier.py` and assume dataset is complete.

4. Inconsistent ID semantics and typing:
- IDs appear as `REAL`, `TEXT`, and normalized string forms across tables.
- Risk: join mismatches, non-deterministic rule matching edge cases.

5. Wide union output tables without strict contracts:
- Generated tables accumulate nullable columns from multiple branches.
- Risk: downstream consumers need fragile per-row-type assumptions.

6. Limited reproducibility controls:
- Some branches include balancing/sampling behavior and optional randomness.
- Risk: run-to-run drift when seeds/config snapshots are not standardized.

7. Observability and auditability gaps:
- Skip reasons and processing traces are not uniformly stored in output tables.
- Risk: difficult root-cause analysis when expected questions are missing.

8. Scalability of orchestration:
- `csv_unifier.py` orchestrates many flows in one script-level sequence.
- Risk: difficult incremental reruns, parallelization, and modular ownership as dataset grows.
