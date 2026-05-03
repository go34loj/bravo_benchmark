from __future__ import annotations

"""
Utilities for Scene Understanding QA generation.

"""
import re
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

try:
    from utils.data_model import SceneUnderstandingRecord, TemplateConfig
    from utils.data_parsers import _normalize_key, _normalize_text, _resolve_column, parse_scene_relations
except ImportError:
    from backend.utils.data_access_layer.data_model import SceneUnderstandingRecord, TemplateConfig
    from backend.utils.data_access_layer.data_parsers import _normalize_key, _normalize_text, _resolve_column, parse_scene_relations


CONNECTIVITY_CONTEXT = "connectivity flagging"
ADJACENCY_CONTEXT = "adjacency flagging"
NOT_DIRECT_CONNECTIVITY_RELATION = "not_direct_connectivity"

MAIN_PLACEHOLDER_PATTERNS = [
    r"\{\s*object\s+type/space\s*\}",
    r"\{\s*object/space\s*\}",
]
CONNECTED_PLACEHOLDER_PATTERNS = [r"\{\s*connected\s+object/space\s*\}"]
ADJACENT_PLACEHOLDER_PATTERNS = [r"\{\s*adjacent\s+object/space\s*\}"]

def generate_functional_grouping_rows(
    templates_df: pd.DataFrame,
    scenes: List[SceneUnderstandingRecord],
    cutouts_df: Optional[pd.DataFrame] = None,
    start_id: int = 0,
) -> pd.DataFrame:
    """
    Delegate functional grouping generation to the dedicated module.
    """
    try:
        from utils.sc_underst_QA_func_group_gen import generate_functional_grouping_rows as _gen
    except ImportError:
        from backend.utils.sc_underst_QA_func_group_gen import generate_functional_grouping_rows as _gen
    return _gen(
        templates_df=templates_df,
        scenes=scenes,
        cutouts_df=cutouts_df,
        start_id=start_id,
    )


def generate_multiview_dimension_rows(
    templates_df: pd.DataFrame,
    viewpoint_df: pd.DataFrame,
    start_id: int = 0,
) -> pd.DataFrame:
    """
    Delegate MultiView Dimensions generation to the dedicated module.
    """
    try:
        from utils.scen_unders_OCR_QA_gen import generate_multiview_dimension_rows as _gen
    except ImportError:
        from backend.utils.scen_unders_OCR_QA_gen import generate_multiview_dimension_rows as _gen
    return _gen(
        templates_df=templates_df,
        viewpoint_df=viewpoint_df,
        start_id=start_id,
    )


def load_scene_understanding_scenes(viewpoint_df: pd.DataFrame) -> List[SceneUnderstandingRecord]:
    """
    Load viewpoint_scenes rows and parse relation fields with column context.
    """
    scene_id_col = _resolve_column(viewpoint_df, ["scene_id", "scene", "id"])
    file_path_col = _resolve_column(viewpoint_df, ["file_path", "filepath", "image_path", "path"])
    space_col = _resolve_column(viewpoint_df, ["space_naming", "space naming", "spaces"])
    spatial_col = _resolve_column(viewpoint_df, ["spatial_relations", "spatial relations", "spatial_relation"])

    missing = [name for name, col in [
        ("scene_id", scene_id_col),
        ("file_path", file_path_col),
        ("space_naming", space_col),
        ("spatial_relations", spatial_col),
    ] if col is None]
    if missing:
        raise ValueError(f"Missing required scene columns: {missing}")

    records: List[SceneUnderstandingRecord] = []
    for _, row in viewpoint_df.iterrows():
        scene_id = _normalize_text(row[scene_id_col])
        file_path = _normalize_text(row[file_path_col])
        space_relations = parse_scene_relations(row[space_col], source_column=space_col)
        object_relations = parse_scene_relations(row[spatial_col], source_column=spatial_col)

        fields: Dict[str, str] = {}
        for col in viewpoint_df.columns:
            key = _normalize_key(col)
            value = row[col]
            if pd.isna(value):
                continue
            value_str = str(value).strip()
            if value_str and key not in fields:
                fields[key] = value_str

        records.append(SceneUnderstandingRecord(
            scene_id=scene_id,
            file_path=file_path,
            space_relations=space_relations,
            object_relations=object_relations,
            space_source_column=space_col,
            spatial_source_column=spatial_col,
            fields=fields,
        ))

    return records


def load_scene_understanding_templates(templates_df: pd.DataFrame) -> List[TemplateConfig]:
    """
    Load template rows and keep only fields needed for scene understanding generation.
    """
    template_id_col = _resolve_column(templates_df, ["template_id", "template"])
    layer_id_col = _resolve_column(templates_df, ["layer_id", "layer"])
    question_template_col = _resolve_column(
        templates_df, ["question_template", "question", "template_text", "template"]
    )
    answer_type_col = _resolve_column(templates_df, ["answer_type", "answer"])
    context_col = _resolve_column(templates_df, ["context"])
    metrics_col = _resolve_column(templates_df, ["metrics"])

    missing = [name for name, col in [
        ("template_id", template_id_col),
        ("question_template", question_template_col),
        ("context", context_col),
    ] if col is None]
    if missing:
        raise ValueError(f"Missing required template columns: {missing}")

    templates: List[TemplateConfig] = []
    for _, row in templates_df.iterrows():
        raw_id = row[template_id_col]
        if pd.isna(raw_id) or str(raw_id).strip() == "":
            continue
        try:
            template_id = int(float(raw_id))
        except (TypeError, ValueError):
            continue

        question_template = str(row[question_template_col])
        context = str(row[context_col]).strip()
        layer_id = str(row[layer_id_col]).strip() if layer_id_col else ""
        answer_type = str(row[answer_type_col]).strip() if answer_type_col else ""
        metrics = str(row[metrics_col]).strip() if metrics_col else None

        templates.append(TemplateConfig(
            template_id=template_id,
            layer_id=layer_id,
            question_template=question_template,
            answer_type=answer_type,
            commands=[],
            benchmark_layer="",
            requires_rule=None,
            context=context,
            metrics=metrics,
        ))
    return templates


def generate_connectivity_flagging_rows(
    templates_df: pd.DataFrame,
    scenes: List[SceneUnderstandingRecord],
    start_id: int = 0,
) -> pd.DataFrame:
    """
    Generate positive rows for connectivity flagging templates.
    """
    return generate_connectivity_relation_rows(
        templates_df=templates_df,
        scenes=scenes,
        relation_type="connectivity",
        start_id=start_id,
    )


def generate_connectivity_relation_rows(
    templates_df: pd.DataFrame,
    scenes: List[SceneUnderstandingRecord],
    relation_type: str,
    start_id: int = 0,
) -> pd.DataFrame:
    """
    Generate rows for connectivity-like relations (connectivity or not_direct_connectivity).
    """
    templates = _filter_templates_by_context(load_scene_understanding_templates(templates_df), CONNECTIVITY_CONTEXT)
    return _generate_relation_rows(
        scenes,
        templates,
        relation_type=relation_type,
        start_id=start_id,
    )


def generate_adjacency_flagging_rows(
    templates_df: pd.DataFrame,
    scenes: List[SceneUnderstandingRecord],
    start_id: int = 0,
    include_negatives: bool = True,
) -> pd.DataFrame:
    """
    Generate positive rows for adjacency flagging templates.
    """
    templates = _filter_templates_by_context(load_scene_understanding_templates(templates_df), ADJACENCY_CONTEXT)
    positive_df = _generate_relation_rows(
        scenes,
        templates,
        relation_type="adjacency",
        start_id=start_id,
    )
    if not include_negatives:
        return positive_df

    neg_start = start_id + len(positive_df) if not positive_df.empty else start_id
    negative_df = _generate_adjacency_negative_rows(
        scenes,
        templates,
        start_id=neg_start,
    )
    return pd.concat([df for df in [positive_df, negative_df] if not df.empty], ignore_index=True)


def _generate_relation_rows(
    scenes: List[SceneUnderstandingRecord],
    templates: List[TemplateConfig],
    relation_type: str,
    start_id: int,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    if not scenes or not templates:
        return pd.DataFrame(rows)

    for scene in scenes:
        for source_column, relations in _iter_scene_relation_sources(scene):
            relation_map = _get_relation_mapping(relations, relation_type)
            if not relation_map:
                continue
            entity_scope = _get_entity_type(relations)
            source_label = _canonical_source_label(source_column)
            for main_entity, related_entity in _iter_relation_pairs(relation_map):
                for template in templates:
                    question_text = _fill_relation_template(
                        template.question_template,
                        main_entity=main_entity,
                        related_entity=related_entity,
                        relation_type=relation_type,
                    )
                    if question_text is None:
                        continue
                    rows.append({
                        "scene_id": scene.scene_id,
                        "file_path": scene.file_path,
                        "source_column": source_label,
                        "entity_scope": entity_scope,
                        "template_id": template.template_id,
                        "layer_id": template.layer_id,
                        "answer_type": template.answer_type,
                        "metrics": template.metrics,
                        "question_text": question_text,
                        "main_entity": main_entity,
                        "related_entity": related_entity,
                        "relation_type": relation_type,
                        "ground_truth_answer": _relation_ground_truth(relation_type),
                    })

    if rows:
        for idx, row in enumerate(rows, start=start_id):
            row["generated_question_id"] = idx
    return pd.DataFrame(rows)


def _generate_adjacency_negative_rows(
    scenes: List[SceneUnderstandingRecord],
    templates: List[TemplateConfig],
    start_id: int,
) -> pd.DataFrame:
    """
    Generate negative adjacency rows using (connectivity - adjacency) pairs.
    Pair comparison is order-sensitive (main_entity, related_entity).
    """
    rows: List[Dict[str, object]] = []
    if not scenes or not templates:
        return pd.DataFrame(rows)

    for scene in scenes:
        for source_column, relations in _iter_scene_relation_sources(scene):
            adjacency_map = _get_relation_mapping(relations, "adjacency")
            connectivity_map = _get_relation_mapping(relations, "connectivity")
            if not connectivity_map:
                continue

            adjacency_pairs = _collect_relation_pairs(adjacency_map)
            connectivity_pairs = _collect_relation_pairs(connectivity_map)
            negative_pairs = connectivity_pairs - adjacency_pairs
            if not negative_pairs:
                continue

            entity_scope = _get_entity_type(relations)
            source_label = _canonical_source_label(source_column)
            for main_entity, related_entity in negative_pairs:
                for template in templates:
                    question_text = _fill_relation_template(
                        template.question_template,
                        main_entity=main_entity,
                        related_entity=related_entity,
                        relation_type="adjacency",
                    )
                    if question_text is None:
                        continue
                    rows.append({
                        "scene_id": scene.scene_id,
                        "file_path": scene.file_path,
                        "source_column": source_label,
                        "entity_scope": entity_scope,
                        "template_id": template.template_id,
                        "layer_id": template.layer_id,
                        "answer_type": template.answer_type,
                        "metrics": template.metrics,
                        "question_text": question_text,
                        "main_entity": main_entity,
                        "related_entity": related_entity,
                        "relation_type": "adjacency",
                        "ground_truth_answer": "no",
                    })

    if rows:
        for idx, row in enumerate(rows, start=start_id):
            row["generated_question_id"] = idx
    return pd.DataFrame(rows)


def _filter_templates_by_context(
    templates: List[TemplateConfig],
    context_value: str,
) -> List[TemplateConfig]:
    target = context_value.strip().lower()
    return [t for t in templates if target in t.context.strip().lower()]


def _iter_scene_relation_sources(
    scene: SceneUnderstandingRecord,
) -> Iterable[Tuple[str, Dict[str, object]]]:
    return [
        (scene.space_source_column, scene.space_relations),
        (scene.spatial_source_column, scene.object_relations),
    ]


def _get_relation_mapping(relations: Dict[str, object], relation_type: str) -> Dict[str, List[str]]:
    mapping = relations.get(relation_type)
    if isinstance(mapping, dict):
        return mapping
    return {}


def _get_entity_type(relations: Dict[str, object]) -> str:
    entity_type = relations.get("entity_type")
    return str(entity_type) if entity_type else "unknown"


def _iter_relation_pairs(relation_map: Dict[str, List[str]]) -> Iterable[Tuple[str, str]]:
    for main_entity, related_list in relation_map.items():
        if not main_entity or not related_list:
            continue
        for related_entity in related_list:
            if related_entity:
                yield main_entity, related_entity


def _collect_relation_pairs(relation_map: Dict[str, List[str]]) -> set[Tuple[str, str]]:
    """
    Collect ordered relation pairs (main_entity, related_entity).
    """
    return set(_iter_relation_pairs(relation_map))


def _canonical_source_label(source_column: str) -> str:
    key = _normalize_key(source_column)
    if key in {"space_naming", "spatial_relations"}:
        return key
    return source_column


def _fill_relation_template(
    template_text: str,
    main_entity: str,
    related_entity: str,
    relation_type: str,
) -> Optional[str]:
    if not template_text:
        return None
    text = str(template_text)
    text, main_count = _replace_placeholders(text, MAIN_PLACEHOLDER_PATTERNS, main_entity)

    if relation_type in {"connectivity", NOT_DIRECT_CONNECTIVITY_RELATION}:
        text, related_count = _replace_placeholders(text, CONNECTED_PLACEHOLDER_PATTERNS, related_entity)
    elif relation_type == "adjacency":
        text, related_count = _replace_placeholders(text, ADJACENT_PLACEHOLDER_PATTERNS, related_entity)
    else:
        return None

    if main_count == 0 or related_count == 0:
        return None
    return text


def _replace_placeholders(text: str, patterns: List[str], value: str) -> Tuple[str, int]:
    replaced = 0
    updated = text
    for pattern in patterns:
        updated, count = re.subn(pattern, value, updated, flags=re.IGNORECASE)
        replaced += count
    return updated, replaced


def _relation_ground_truth(relation_type: str) -> str:
    if relation_type == NOT_DIRECT_CONNECTIVITY_RELATION:
        return "no"
    return "yes"
