# bravo_benchmark

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

## License

This repository uses multiple licenses.

The source code, including preprocessing scripts, evaluation scripts, and utilities, is licensed under the Apache License 2.0. See [`LICENSE-CODE`](LICENSE-CODE).

The dataset, annotations, rendered BIM scene images, benchmark splits, ground-truth labels, prompts, question-answer pairs, and evaluation tables are licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0), unless otherwise stated. See [`LICENSE-DATA.md`](LICENSE-DATA.md).

Commercial use of the dataset, annotations, images, or benchmark materials is not permitted without prior written permission.

Third-party materials, including regulatory texts, standards, external figures, and other copyrighted sources, are not covered by these licenses and remain subject to their original copyright and licensing terms.
