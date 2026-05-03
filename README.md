# BRAVO Benchmark: Evaluating Multimodal LLMs for BIM scene-based Automated Compliance Checking

## Abstract

This study introduces BRAVO (Building Regulation Answering & Visual Observation) Bench, the first benchmark designed to evaluate the capability of Multimodal Large Language Models (MLLMs) to perform compliance checking based on BIM-derived scenes. BRAVO Bench integrates about 26 BIM-scenes, 59 textual provisions and 5 regulatory illustrations producing approximately 1500 question–answer pairs across four diagnostic layers: scene perception, scene understanding, rule interpretation, and compliance reasoning. 

The evaluation combines conventional accuracy-based metrics for closed-form tasks, BLEU/ROUGE for generated responses, and a ground-truth-centric step-level frame-work for compliance reasoning. This framework measures covered, missing, split, and extra reasoning steps through human or LLM-assisted review, and aggregates them using the Compliance Reasoning Score (CRS), a non-compensatory geometric mean designed to prevent strong performance in one dimension from masking failure in an-other. 

Results reveal a substantial cognitive barrier in current state-of-the-art MLLMs. While models achieve relatively high accuracy in primary object perception, perfor-mance declines sharply on ambiguous regulatory norms requiring expert judgment and collapses on tacit-knowledge tasks involving implicit commonsense and domain assumptions. Error analysis further identifies missing reasoning steps and weak multi-view synthesis as key failure patterns. BRAVO Bench provides a foundation for diag-nosing, adapting, and retraining MLLMs for compliance-oriented reasoning in the AEC.

## Dataset

The [`dataset/`](dataset/) directory contains two image types:

- Design scene images in [`dataset/BIM_design_scenes/`](dataset/BIM_design_scenes/), referenced by the `file_path` column.
- Rule/regulatory figures in [`dataset/rule_figures/`](dataset/rule_figures/), referenced by the `figure_path` column.

These image assets are reused across multiple dataset parts and are not tied to a single benchmark level.

CSV files in this folder are QA-style tables that can be directly inspected and used as question-answer datasets:
[`1_1_scene_perception.csv`](dataset/1_1_scene_perception.csv),
[`1_2_scene_understanding.csv`](dataset/1_2_scene_understanding.csv),
[`2_rules_understanding.csv`](dataset/2_rules_understanding.csv),
[`3_compliance_reasoning.csv`](dataset/3_compliance_reasoning.csv).

## Dataset Generation Pipeline

Code for dataset generation is located in [`scripts/`](scripts/).

Key documentation:

- [`scripts/backend_architecture.md`](scripts/backend_architecture.md) - detailed description of the functionality of dataset-generation scripts.
- [`scripts/SETUP.md`](scripts/SETUP.md) - quick guide for running the pipeline.
- [`scripts/dataset_contribution.md`](scripts/dataset_contribution.md) - guide for adding new data to the dataset and scaling it.

## License

This repository uses multiple licenses.

The source code, including preprocessing scripts, evaluation scripts, and utilities, is licensed under the Apache License 2.0. See [`LICENSE-CODE`](LICENSE-CODE).

The dataset, annotations, rendered BIM scene images, benchmark splits, ground-truth labels, prompts, question-answer pairs, and evaluation tables are licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0), unless otherwise stated. See [`LICENSE-DATA.md`](LICENSE-DATA.md).

Commercial use of the dataset, annotations, images, or benchmark materials is not permitted without prior written permission.

Third-party materials, including regulatory texts, standards, external figures, and other copyrighted sources, are not covered by these licenses and remain subject to their original copyright and licensing terms.
