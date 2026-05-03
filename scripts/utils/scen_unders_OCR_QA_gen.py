from __future__ import annotations

import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))
"""
MultiView Dimensions generation for Scene Understanding QA.
"""

import json
import re
from typing import Dict, List, Optional

import pandas as pd

try:
    from utils.data_access_layer.data_model import TemplateConfig
    from utils.data_access_layer.data_parsers import _normalize_text, _resolve_column, parse_multi_view_dimensions
except ImportError:
    from data_access_layer.data_model import TemplateConfig
    from utils.data_access_layer.data_parsers import (
        _normalize_text,
        _resolve_column,
        parse_multi_view_dimensions,
    )


SUBTASK_VALUE = "multiview_ocr_text_detection"
CONTEXT_VALUE = "multi_view_dimensions"

OBJECT_TYPE_PATTERNS = [
    r"\{\s*object_type\s*\}",
    r"\{\s*object\s+type\s*\}",
]
DIMENSION_ARRAY_PATTERNS = [
    r"\{\s*dimension\s*array\s*\}",
    r"\{\s*dimension_array\s*\}",
]


def generate_multiview_dimension_rows(
    templates_df: pd.DataFrame,
    viewpoint_df: pd.DataFrame,
    start_id: int = 0,
) -> pd.DataFrame:
    """
    Generate MultiView Dimensions rows for Scene Understanding QA.

    Rows are created for templates where:
      templates.subtask == "multiview_ocr_text_detection"
      and templates.context == "multi_view_dimensions"
    """
    templates = _load_multiview_templates(templates_df)
    if templates_df.empty:
        print("MultiView Dimensions: templates_df is empty")
    print(f"MultiView Dimensions: templates matched = {len(templates)}")

    scene_id_col = _resolve_column(viewpoint_df, ["scene_id", "scene", "id"])
    file_path_col = _resolve_column(viewpoint_df, ["file_path", "filepath", "image_path", "path"])
    multi_view_col = _resolve_column(
        viewpoint_df,
        ["multi_view_dimensions", "multi_view_dimension", "multi-view_dimensions"],
    )

    missing = [name for name, col in [
        ("scene_id", scene_id_col),
        ("file_path", file_path_col),
        ("multi_view_dimensions", multi_view_col),
    ] if col is None]
    if missing:
        print(f"MultiView Dimensions: missing required scene columns: {missing}")
        return pd.DataFrame([])

    rows: List[Dict[str, object]] = []
    scenes_scanned = 0
    skipped_scenes = 0

    for _, row in viewpoint_df.iterrows():
        scenes_scanned += 1
        scene_id = _normalize_text(row[scene_id_col])
        file_path = _normalize_text(row[file_path_col])
        parsed = parse_multi_view_dimensions(row[multi_view_col], tolerant=True)

        if not parsed:
            skipped_scenes += 1
            continue

        for object_type, dim_map in parsed.items():
            if not object_type or not dim_map:
                continue
            dimension_params = list(dim_map.keys())
            dimension_array_json = json.dumps(dimension_params, ensure_ascii=False)
            ground_truth_json = json.dumps(dim_map, ensure_ascii=False, separators=(",", ":"))

            for template in templates:
                question_text = _fill_multiview_template(
                    template.question_template,
                    object_type=object_type,
                    dimension_array_json=dimension_array_json,
                )
                if question_text is None:
                    continue
                rows.append({
                    "scene_id": scene_id,
                    "file_path": file_path,
                    "source_column": "multi_view_dimensions",
                    "entity_scope": "objects",
                    "template_id": template.template_id,
                    "layer_id": template.layer_id,
                    "answer_type": template.answer_type,
                    "metrics": template.metrics,
                    "question_text": question_text,
                    "main_entity": object_type,
                    "relation_type": "multi_view_dimensions",
                    "ground_truth_answer": ground_truth_json,
                })

    if rows:
        for idx, row in enumerate(rows, start=start_id):
            row["generated_question_id"] = idx

    print(f"MultiView Dimensions: scenes scanned = {scenes_scanned}")
    print(f"MultiView Dimensions: skipped scenes = {skipped_scenes}")
    print(f"MultiView Dimensions: questions generated = {len(rows)}")

    return pd.DataFrame(rows)


def _load_multiview_templates(templates_df: pd.DataFrame) -> List[TemplateConfig]:
    template_id_col = _resolve_column(templates_df, ["template_id", "template"])
    layer_id_col = _resolve_column(templates_df, ["layer_id", "layer"])
    question_template_col = _resolve_column(
        templates_df, ["question_template", "question", "template_text", "template"]
    )
    answer_type_col = _resolve_column(templates_df, ["answer_type", "answer"])
    metrics_col = _resolve_column(templates_df, ["metrics"])
    context_col = _resolve_column(templates_df, ["context"])
    subtask_col = _resolve_column(templates_df, ["subtask"])

    missing = [name for name, col in [
        ("template_id", template_id_col),
        ("question_template", question_template_col),
        ("context", context_col),
        ("subtask", subtask_col),
    ] if col is None]
    if missing:
        print(f"MultiView Dimensions: missing required template columns: {missing}")
        return []

    templates: List[TemplateConfig] = []
    for _, row in templates_df.iterrows():
        subtask = str(row.get(subtask_col, "")).strip().lower()
        context = str(row.get(context_col, "")).strip().lower()
        if subtask != SUBTASK_VALUE or context != CONTEXT_VALUE:
            continue

        raw_id = row.get(template_id_col)
        if raw_id is None or str(raw_id).strip() == "":
            continue
        try:
            template_id = int(float(raw_id))
        except (TypeError, ValueError):
            continue

        templates.append(TemplateConfig(
            template_id=template_id,
            layer_id=str(row.get(layer_id_col, "")).strip() if layer_id_col else "",
            question_template=str(row.get(question_template_col, "")),
            answer_type=str(row.get(answer_type_col, "")).strip() if answer_type_col else "",
            commands=[],
            benchmark_layer="",
            requires_rule=None,
            context=str(row.get(context_col, "")).strip(),
            metrics=str(row.get(metrics_col, "")).strip() if metrics_col else None,
            subtask=str(row.get(subtask_col, "")).strip() if subtask_col else None,
        ))
    return templates


def _fill_multiview_template(
    template_text: str,
    *,
    object_type: str,
    dimension_array_json: str,
) -> Optional[str]:
    if not template_text:
        return None
    text = str(template_text)
    text, _ = _replace_placeholders(text, OBJECT_TYPE_PATTERNS, object_type)
    text, _ = _replace_placeholders(text, DIMENSION_ARRAY_PATTERNS, dimension_array_json)
    return text


def _replace_placeholders(text: str, patterns: List[str], value: str) -> tuple[str, int]:
    replaced = 0
    updated = text
    for pattern in patterns:
        updated, count = re.subn(pattern, value, updated, flags=re.IGNORECASE)
        replaced += count
    return updated, replaced
