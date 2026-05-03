# Dataset Contribution Guide

This file explains how to extend the dataset so the backend pipeline continues to work without code changes.

Scope: `backend/` Dataset Generation Pipeline (`data_manipulation.py` + `csv_unifier.py`).

## Table of Contents
- [1) Quick workflow](#1-quick-workflow)
- [2) What to update for each contribution type](#2-what-to-update-for-each-contribution-type)
  - [A) Add a new template](#a-add-a-new-template)
  - [B) Add a new scene (viewpoint)](#b-add-a-new-scene-viewpoint)
  - [C) Add a new rule (atomic / parent)](#c-add-a-new-rule-atomic--parent)
  - [D) Add a new normative figure](#d-add-a-new-normative-figure)
  - [E) Add a new cutout-parent mapping (for compliance parent-rule templates)](#e-add-a-new-cutout-parent-mapping-for-compliance-parent-rule-templates)
  - [F) Add or update RASE classification rows (for rule-understanding GT)](#f-add-or-update-rase-classification-rows-for-rule-understanding-gt)
  - [G) Add or update expert reasoning calibration rows](#g-add-or-update-expert-reasoning-calibration-rows)
- [3) Data format rules (critical)](#3-data-format-rules-critical)
- [4) Data Access Layer: what it does](#4-data-access-layer-what-it-does)
- [5) `data_parsers.py` reference (what format is expected)](#5-data_parserspy-reference-what-format-is-expected)
- [5.1) `viewpoint_scenes` column format guide (practical)](#51-viewpoint_scenes-column-format-guide-practical)
- [6) Minimum pre-merge checklist](#6-minimum-pre-merge-checklist)

---

## 1) Quick workflow

1. Update CSV source files (templates/rules/scenes/cutouts/figures).
2. Put any new scene images into `backend/scenes/`. 
3. Put any new normative figure image files into `backend/rule_figure/`.
4. Run:
   - `python data_manipulation.py`
   - `python csv_unifier.py`
5. If GT auto-flags are disabled in `config.yaml`, run:
   - `python utils/scene_perc_gen_gt.py --db unified_database.db`
   - `python utils/compliance_bool_gt_gen.py --db unified_database.db --table-name generated_compliance_questions`

---

## 2) What to update per each contribution type

## A) Add a new template
Update: `*VQA_Rules_Scenes_Templates(templates).csv`

Required fields:
- `template_id` (unique)
- `benchmark_layer`
- `subtask`
- `answer_type`
- `question_template`
- `context` (important for routing in scene/rule/compliance generators)

If rule-linked:
- `requires_rule` should contain rule IDs in JSON array format (preferred), e.g. `["101","102"]`.

Placeholders must match existing parser/generator logic (examples):
- Rule: `{rule_text_atomic}`, `{parent_rule_text}`
- Scene perception: `{object_type}`, `{feature_name}`, `{location_context}`, `{highlight}`
- Scene understanding functional: `{option1}`, `{option2}`, `{option3}`, `{option4}`, `{caption}`

---

## B) Add a new scene (viewpoint)
Update: `*VQA_Rules_Scenes_Templates(viewpoint_scenes).csv`

Required fields:
- `scene_id` (unique)
- `template_id` (list of applicable template IDs)
- `rule_id` (list of atomic rule IDs used by scene/rule/compliance flows)
- `file_path` (or let `data_manipulation.py` refresh it from `backend/scenes/`)

Optional but important fields:
- `object_type`, `feature_name`, `material`, `text`, `multi-view_dimensions`, `space_naming`, `spatial relations`, `not compliant`, `not sufficient`, `cutout_id`

Also add image file to:
- `backend/scenes/` (filename should start with `scene_id` for automatic path mapping)

---

## C) Add a new rule (atomic / parent)
Update: `*VQA_Rules_Scenes_Templates(rules_sort).csv`

Required fields:
- `rule_id` (unique atomic ID)
- `rule_text_atomic`
- `parent_rule_id`
- `parent_rule_text`

So that the pipeline correctly generates ground truth for rule understanding questions, please also add classification and ambiguity class to the relevant columns within the table:
- `Classification` / `classification`
- `Ambiguity` / `ambiguity`
- If a figure is related to certain rules, please add a link to it to the `rules_sort` table (`figure_required=yes`, `figure_id=<new id>`)

If you add a rule, make sure matching templates/scenes reference it via:
- `templates.requires_rule`
- `viewpoint_scenes.rule_id` / compliance metadata as needed.

---

## D) Add a new normative figure
Updates:
- `*VQA_Rules_Scenes_Templates(rule_figure).csv`
- If a figure is related to certain rules, please add a link to it to the `rules_sort` table (`figure_required=yes`, `figure_id=<new id>`)

Required figure fields:
- `figure_id` (unique)
- `file_path`
- `caption`

---

## E) Add a new cutout-parent mapping (for compliance parent-rule templates)
Update: `*VQA_Rules_Scenes_Templates(cutouts).csv`

Required fields:
- `cutout_id`
- `parent_rule_id` (list/string parsable as list)

And reference `cutout_id` in `viewpoint_scenes`.

---

## F) Add or update RASE classification rows (for rule-understanding GT)
Update: `03_04_VQA_Rules_Scenes_Templates(RASE).csv`

Why this is needed:
- `utils/rule_under_classific_QA.py` uses this CSV to generate:
  - `RASE for atomic` questions/ground truth
  - `RASErequirement flagging` ground truth

Minimum required columns (must stay present):
- `parent_rule_id`
- `parent_rule_text`
- `rule_id`
- `rule_text_atomic`
- `RASE`

Columns required for target expansion logic:
- `applies_to_rule` (especially for rows with `RASE = Applicability (A)`)

Optional but recommended:
- `figure_required`, `figure_id` (if a normative figure is linked)

Formatting rules:
- Keep one atomic statement per row.
- Keep `parent_rule_id` consistent across rows of the same parent rule.
- Use consistent RASE labels containing keywords: `Applicability`, `Requirement`, `Selection`, `Exception`.
- In `applies_to_rule`, use list-like IDs (JSON list preferred, comma-separated accepted).

---

## G) Add or update expert reasoning calibration rows
Update: `03_04_reasonLLMjudgeCalibr.csv`

Purpose:
- This CSV is used as GT lookup source in `notebooks/3_2_to_3_5_Stepwise_LLM_judge_Eval.ipynb`.
- It is **not required** to build `unified_database.db`, but is required to keep evaluation/calibration assets up to date.

Fields that are реально used by the notebook lookup:
- `ground_truth_answer`
- `scene_id`
- `parent_rule_id`
- `figure_id`
- `reason_gt_id` (used as source-row trace when a unique match is found)

Lookup rule used in the notebook:
- GT retrieval is matched by the key combination:
  - `(scene_id, parent_rule_id, figure_id)`
- This is intentionally more stable than `question_id`, because question IDs can be re-generated/re-numbered when the dataset is extended.

Contribution rule:
- For every new reasoning GT row, ensure `scene_id`, `parent_rule_id`, `figure_id` are correctly populated and uniquely identify the intended `ground_truth_answer`.

---

## 3) Data format rules (critical)

The pipeline is tolerant, but stable runs require consistent formats:

1. List-like fields (preferred JSON arrays):
- `requires_rule`, `template_id`, `rule_id`, `not compliant`, `not sufficient`, `parent_rule_id` lists
- `applies_to_rule` in RASE CSV
- Preferred: `["1","2","3"]`
- Accepted fallback: `1,2,3`

2. IDs:
- Keep IDs clean and stable (no trailing spaces).
- Numeric-like IDs are normalized by loaders/parsers; avoid mixed semantic IDs unless intentional.

3. Text encodings:
- CSV files are loaded with encoding detection/fallback; still prefer UTF-8.

---

## 4) Data Access Layer: what it does

Main files:
- `utils/data_access_layer/file_operations.py`
- `utils/data_access_layer/data_model.py`
- `utils/data_access_layer/data_parsers.py`

Responsibilities:
- discover/load CSV files
- normalize column names
- parse list-like and structured text fields into typed Python objects
- build consistent records (`TemplateConfig`, `SceneRecord`, `SceneUnderstandingRecord`)

If your CSV change introduces a new field format, update parser logic first.

---

## 5) `data_parsers.py` reference (what format is expected)

Core parser expectations:

- `parse_requires_rule(...)`:
  - expects JSON list string or comma-separated string.
  - used for list-like ID fields in:
    - `templates.requires_rule`
    - `viewpoint_scenes.rule_id`
    - `viewpoint_scenes.template_id`
    - `viewpoint_scenes.not compliant`
    - `viewpoint_scenes.not sufficient`
    - `cutout.parent_rule_id` (compliance parent mapping)

- `parse_template_id_list(...)`:
  - parses template ID lists to `List[int]`.
  - primary source column: `viewpoint_scenes.template_id`.

- `parse_feature_mapping(...)`:
  - expects blocks like `{toilet: tank, lid}, {sink: drain hole}`.
  - source column: `viewpoint_scenes.feature_name`.

- `parse_material_field(...)`:
  - expects sections like:
    - `texture={wall: rough}`
    - `material={floor: parquet/laminate}`
    - `color={wall: white}`
  - source column: `viewpoint_scenes.material`.

- `parse_scene_relations(...)`:
  - expects relation blocks from:
    - `viewpoint_scenes.space_naming`
    - `viewpoint_scenes.spatial relations` (or normalized alias `spatial_relations`)
    - `adjacency={A, B; C, D}`
    - `connectivity={A: B, C; D: E}`
    - `not_direct_connectivity={...}`
    - `functional grouping={Group A: room1, room2; ...}`

- `parse_viewpoint_text(...)`:
  - parses OCR/tag blocks from `viewpoint_scenes.text`.

- `parse_multi_view_dimensions(...)`:
  - parses structured multiview dimension payloads from `viewpoint_scenes.multi-view_dimensions` (and alias variants like `multi_view_dimensions` / `multi_view_dimension` in loaders).

When adding new notation, keep backward compatibility or update all dependent generators.

---

## 5.1) `viewpoint_scenes` column format guide (practical)

Based on `10_04_VQA_Rules_Scenes_Templates(viewpoint_scenes).CSV`, use the following formats so `data_parsers.py` can parse reliably:

- `object_type`:
  - comma-separated labels  
  - example: `handrails, toilet, sink, door`
  - allowed marker: `highlighted:<object>` (now cleaned by parser)

- `feature_name`:
  - repeated `{object: feature1, feature2}` blocks
  - example: `{toilet: tank, lid}, {sink: drain hole}`

- `material`:
  - sections by kind: `texture={object:texture}`, `material={object:material}`, `color={object:color}`
  - example: `color={paper towel dispenser:turquoise}`
  - multi-values inside one object can use `/`: `material={floor:parquet/laminate}`

- `text`:
  - OCR/tag/dimension blocks separated by commas
  - supported examples:
    - `text={"Cutout 1"}`
    - `label with a leader/room tag={room name:TOILET, room label:1D26}`
    - `label with a leader/layer tag={facade flashing:ventilated facade panels, load-bearing structure:[reinforced concrete structure, 250 mm]}`
      - Layer Tag is used to represent finish-layer compositions (e.g., wall, façade, floor, or ceiling build-ups). Each entry is written as `layer meaning: layer name/material`, and entries are separated by commas. If a layer name itself contains commas, the full layer name must be enclosed in square brackets `[...]` so it is treated as a single value during parsing.
    - `door={dimension:975}`

- `multi-view_dimensions`:
  - object-level JSON-like map of dimensions
  - expected shape:  
    `object = { "dim_key": { "value": <num>, "unit": "<unit>", "source_view": "<view>" }, "other_dim": null }`

- `space_naming` and `spatial relations`:
  - relation blocks in one cell, parser recognizes:
    - `functional grouping={group: item1, item2; ...}`
    - `connectivity={object/space: connected object/space 1, connected object/space 2; ...}` (or `direct_connectivity`)
    - `adjacency={adjacent object 1, adjacent object 2; ...}`
    - `not_direct_connectivity={object/space: connected object/space 1, connected object/space 2; ...}`

- `not compliant`, `not sufficient`, `rule_id`, `template_id`:
  - preferred: JSON list strings (e.g. `["12","14"]`)
  - accepted fallback: comma-separated list (e.g. `12, 14, 15`)

- `cutout_id`:
  - single stable ID referencing `cutout.cutout_id`
  - in `cutout` table, `file_path` is metadata for the cutout asset path.
  - `cutout_title` is currently metadata-only (not required by current generators); it is not used by scene/rule/compliance generation logic in the current pipeline.
  - A cutout represents the same underlying scene extracted from the model. In `viewpoint_scenes`, this cutout can appear as multiple viewpoints (different camera angles or projections) of that same scene. As the dataset expands, the `cutout` table can store metadata shared across all related viewpoints, reducing redundancy and keeping scene documentation more compact.


Formatting recommendations:
- keep delimiters consistent (`;` between relation groups, `,` inside group items),
- avoid unmatched braces/quotes,
- keep IDs trim-clean (no trailing spaces).

---

## 6) Minimum pre-merge checklist

- `template_id`, `rule_id`, `scene_id`, `figure_id`, `cutout_id` remain unique where expected.
- New template placeholders are supported by current generators/router.
- New rules referenced in templates/scenes exist in rules CSV.
- New figure IDs referenced by rules exist in figure CSV.
- If rule-understanding RASE templates are used, RASE CSV rows are updated and contain valid `parent_rule_id` / `rule_id` / `RASE` / `applies_to_rule`.
- New scene image files exist in `backend/scenes/`.
- Pipeline runs end-to-end:
  - `python data_manipulation.py`
  - `python csv_unifier.py`
- DB integrity check passes at the end of `csv_unifier.py`.
