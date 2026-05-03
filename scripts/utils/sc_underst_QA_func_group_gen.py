from __future__ import annotations

import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))
"""
Functional grouping generation for Scene Understanding QA.
Separated to keep the main relation pipeline compact.
"""

import json
import re
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

try:
    from utils.data_access_layer.data_model import SceneUnderstandingRecord, TemplateConfig
    from utils.data_access_layer.data_parsers import _resolve_column
    from utils.scene_underst_QA_gen import load_scene_understanding_templates
except ImportError:
    from data_access_layer.data_model import SceneUnderstandingRecord, TemplateConfig
    from utils.data_access_layer.data_parsers import _resolve_column
    from utils.scene_underst_QA_gen import load_scene_understanding_templates


FUNCTIONAL_GROUPING_CONTEXT = "functional grouping"

GROUP_LABEL_PLACEHOLDER_PATTERNS = [r"\{\s*group_label\s*\}\}?"]
CAPTION_PLACEHOLDER_PATTERNS = [r"\{\s*caption\s*\}"]
ITEM_PLACEHOLDER_PATTERNS = [
    r"\{\s*object\s+type\s*\}",
    r"\{\s*object\s+type/space\s*\}",
    r"\{\s*object/space\s*\}",
]
OPTION_PLACEHOLDER_PATTERNS = [
    r"\{\s*option1\s*\}",
    r"\{\s*option2\s*\}",
    r"\{\s*option3\s*\}",
    r"\{\s*option4\s*\}",
]

ROOM_PLACEHOLDER_ORDER = ["room_a", "room_b", "room_c", "room_d", "room_e"]
MIN_GROUP_ITEMS_FOR_MCQ = 2


def generate_functional_grouping_rows(
    templates_df: pd.DataFrame,
    scenes: List[SceneUnderstandingRecord],
    cutouts_df: Optional[pd.DataFrame] = None,
    start_id: int = 0,
) -> pd.DataFrame:
    """
    Generate rows for functional grouping templates (MCQ and yes/no variants).
    """
    templates = _filter_templates_by_context(
        load_scene_understanding_templates(templates_df),
        FUNCTIONAL_GROUPING_CONTEXT,
    )
    if not scenes or not templates:
        return pd.DataFrame([])

    caption_map = _build_cutout_caption_map(cutouts_df)
    group_label_pool = _build_group_label_pool(scenes)

    rows: List[Dict[str, object]] = []
    for scene in scenes:
        for source_column, relations in _iter_scene_relation_sources(scene):
            grouping = _get_relation_mapping(relations, "functional_grouping")
            if not grouping:
                continue

            entity_scope = _get_entity_scope(relations)
            source_label = _canonical_source_label(source_column)
            all_items = _collect_group_items(grouping)
            other_groups_items = [items for items in grouping.values() if items]

            for group_label, items in grouping.items():
                if not group_label or not items:
                    continue

                for template in templates:
                    kind = _classify_functional_template(template.question_template)
                    if kind == "mcq":
                        rows.extend(_generate_functional_mcq_rows(
                            template=template,
                            scene=scene,
                            source_label=source_label,
                            entity_scope=entity_scope,
                            group_label=group_label,
                            group_items=items,
                            other_groups=other_groups_items,
                            all_items=all_items,
                        ))
                    elif kind == "contains_item":
                        rows.extend(_generate_functional_contains_item_rows(
                            template=template,
                            scene=scene,
                            source_label=source_label,
                            entity_scope=entity_scope,
                            group_label=group_label,
                            group_items=items,
                            all_items=all_items,
                        ))
                    elif kind == "caption_contains_group":
                        caption = _resolve_scene_caption(scene, caption_map)
                        if not caption:
                            continue
                        rows.extend(_generate_functional_caption_rows(
                            template=template,
                            scene=scene,
                            source_label=source_label,
                            entity_scope=entity_scope,
                            group_label=group_label,
                            caption=caption,
                            group_label_pool=group_label_pool,
                        ))

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


def _get_entity_scope(relations: Dict[str, object]) -> str:
    entity_type = relations.get("entity_type")
    return str(entity_type) if entity_type else "unknown"


def _canonical_source_label(source_column: str) -> str:
    key = _normalize_source_key(source_column)
    if key in {"space_naming", "spatial_relations"}:
        return key
    return source_column


def _normalize_source_key(value: str) -> str:
    key = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower())
    return key.strip("_")


def _classify_functional_template(template_text: str) -> str:
    text = str(template_text).lower()
    if re.search(r"\{\s*option1\s*\}", text):
        return "mcq"
    if re.search(r"\{\s*caption\s*\}", text):
        return "caption_contains_group"
    if re.search(r"\{\s*object\s+type", text) or re.search(r"\{\s*object/space\s*\}", text):
        return "contains_item"
    return ""


def _template_needs_item(template_text: str) -> bool:
    text = str(template_text).lower()
    return bool(re.search(r"\{\s*object\s+type", text) or re.search(r"\{\s*object/space\s*\}", text))


def _collect_group_items(grouping: Dict[str, List[str]]) -> List[str]:
    items: List[str] = []
    seen = set()
    for group_items in grouping.values():
        for item in group_items:
            if item and item not in seen:
                items.append(item)
                seen.add(item)
    return items


def _build_cutout_caption_map(cutouts_df: Optional[pd.DataFrame]) -> Dict[str, str]:
    if cutouts_df is None or cutouts_df.empty:
        return {}
    cutout_id_col = _resolve_column(cutouts_df, ["cutout_id", "cutout"])
    caption_col = _resolve_column(cutouts_df, ["caption", "cutout_title", "title"])
    if not cutout_id_col or not caption_col:
        return {}
    mapping: Dict[str, str] = {}
    for _, row in cutouts_df.iterrows():
        raw_id = row.get(cutout_id_col)
        if pd.isna(raw_id):
            continue
        cutout_id = _normalize_cutout_id(raw_id)
        caption = str(row.get(caption_col, "")).strip()
        if cutout_id and caption and cutout_id not in mapping:
            mapping[cutout_id] = caption
    return mapping


def _resolve_scene_caption(scene: SceneUnderstandingRecord, caption_map: Dict[str, str]) -> str:
    cutout_id = _get_scene_cutout_id(scene)
    if cutout_id and cutout_id in caption_map:
        return caption_map[cutout_id]
    return ""


def _get_scene_cutout_id(scene: SceneUnderstandingRecord) -> str:
    for key in ("cutout_id", "cutout"):
        if key in scene.fields:
            return _normalize_cutout_id(scene.fields.get(key))
    return ""


def _normalize_cutout_id(raw_id: object) -> str:
    if raw_id is None or pd.isna(raw_id):
        return ""
    text = str(raw_id).strip()
    if text == "":
        return ""
    try:
        num = float(text)
        if num.is_integer():
            return str(int(num))
    except ValueError:
        pass
    return text


def _build_group_label_pool(scenes: List[SceneUnderstandingRecord]) -> Dict[Tuple[str, str], List[str]]:
    """
    Pool of group labels by (scene_id, source_column) to enable cross-scene negatives.
    """
    pool: Dict[Tuple[str, str], List[str]] = {}
    for scene in scenes:
        for source_column, relations in _iter_scene_relation_sources(scene):
            grouping = _get_relation_mapping(relations, "functional_grouping")
            if not grouping:
                continue
            key = (scene.scene_id, _canonical_source_label(source_column))
            labels = [g for g in grouping.keys() if g]
            if labels:
                pool[key] = labels
    return pool


def _generate_functional_mcq_rows(
    template: TemplateConfig,
    scene: SceneUnderstandingRecord,
    source_label: str,
    entity_scope: str,
    group_label: str,
    group_items: List[str],
    other_groups: List[List[str]],
    all_items: List[str],
) -> List[Dict[str, object]]:
    if len(group_items) < MIN_GROUP_ITEMS_FOR_MCQ:
        return []
    option_count = _count_option_placeholders(template.question_template) or 4
    options = _build_grouping_options(group_items, other_groups, all_items, option_count)
    if not options:
        return []

    correct_option = _join_items(group_items)
    if correct_option not in options:
        return []
    correct_index = options.index(correct_option)
    item_value = group_items[0] if group_items and _template_needs_item(template.question_template) else ""

    text = _fill_functional_template(
        template.question_template,
        group_label=group_label,
        caption="",
        item=item_value,
        options=options,
        room_items=all_items,
    )
    if text is None:
        return []

    return [{
        "scene_id": scene.scene_id,
        "file_path": scene.file_path,
        "source_column": source_label,
        "entity_scope": entity_scope,
        "template_id": template.template_id,
        "layer_id": template.layer_id,
        "answer_type": template.answer_type,
        "metrics": template.metrics,
        "question_text": text,
        "main_entity": group_label,
        "related_entity": correct_option,
        "relation_type": "functional_grouping",
        "ground_truth_answer": correct_option,
        "group_label": group_label,
        "options": json.dumps(options),
        "correct_option_index": correct_index,
    }]


def _generate_functional_contains_item_rows(
    template: TemplateConfig,
    scene: SceneUnderstandingRecord,
    source_label: str,
    entity_scope: str,
    group_label: str,
    group_items: List[str],
    all_items: List[str],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    group_set = set(group_items)
    other_items = [item for item in all_items if item not in group_set]

    for item in group_items:
        text = _fill_functional_template(
            template.question_template,
            group_label=group_label,
            caption="",
            item=item,
            options=None,
            room_items=None,
        )
        if text is None:
            continue
        rows.append(_build_functional_row(
            scene=scene,
            source_label=source_label,
            entity_scope=entity_scope,
            template=template,
            group_label=group_label,
            related_entity=item,
            question_text=text,
            ground_truth="yes",
            relation_type="functional_grouping",
        ))

    for item in other_items:
        text = _fill_functional_template(
            template.question_template,
            group_label=group_label,
            caption="",
            item=item,
            options=None,
            room_items=None,
        )
        if text is None:
            continue
        rows.append(_build_functional_row(
            scene=scene,
            source_label=source_label,
            entity_scope=entity_scope,
            template=template,
            group_label=group_label,
            related_entity=item,
            question_text=text,
            ground_truth="no",
            relation_type="functional_grouping",
        ))

    return rows


def _generate_functional_caption_rows(
    template: TemplateConfig,
    scene: SceneUnderstandingRecord,
    source_label: str,
    entity_scope: str,
    group_label: str,
    caption: str,
    group_label_pool: Dict[Tuple[str, str], List[str]],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    text = _fill_functional_template(
        template.question_template,
        group_label=group_label,
        caption=caption,
        item="",
        options=None,
        room_items=None,
    )
    if text is None:
        return rows

    rows.append(_build_functional_row(
        scene=scene,
        source_label=source_label,
        entity_scope=entity_scope,
        template=template,
        group_label=group_label,
        related_entity=caption,
        question_text=text,
        ground_truth="yes",
        relation_type="functional_grouping",
    ))

    negative_label = _select_negative_group_label(
        scene_id=scene.scene_id,
        source_label=source_label,
        group_label_pool=group_label_pool,
        current_labels=[group_label],
    )
    if not negative_label:
        return rows

    neg_text = _fill_functional_template(
        template.question_template,
        group_label=negative_label,
        caption=caption,
        item="",
        options=None,
        room_items=None,
    )
    if neg_text is None:
        return rows

    rows.append(_build_functional_row(
        scene=scene,
        source_label=source_label,
        entity_scope=entity_scope,
        template=template,
        group_label=negative_label,
        related_entity=caption,
        question_text=neg_text,
        ground_truth="no",
        relation_type="functional_grouping",
    ))
    return rows


def _build_functional_row(
    scene: SceneUnderstandingRecord,
    source_label: str,
    entity_scope: str,
    template: TemplateConfig,
    group_label: str,
    related_entity: str,
    question_text: str,
    ground_truth: str,
    relation_type: str,
) -> Dict[str, object]:
    return {
        "scene_id": scene.scene_id,
        "file_path": scene.file_path,
        "source_column": source_label,
        "entity_scope": entity_scope,
        "template_id": template.template_id,
        "layer_id": template.layer_id,
        "answer_type": template.answer_type,
        "metrics": template.metrics,
        "question_text": question_text,
        "main_entity": group_label,
        "related_entity": related_entity,
        "relation_type": relation_type,
        "ground_truth_answer": ground_truth,
        "group_label": group_label,
    }


def _fill_functional_template(
    template_text: str,
    group_label: str,
    caption: str,
    item: str,
    options: Optional[List[str]],
    room_items: Optional[List[str]],
) -> Optional[str]:
    if not template_text:
        return None
    text = str(template_text)
    text, _ = _replace_placeholders(text, GROUP_LABEL_PLACEHOLDER_PATTERNS, group_label)
    if caption:
        text, _ = _replace_placeholders(text, CAPTION_PLACEHOLDER_PATTERNS, caption)
    if item:
        text, _ = _replace_placeholders(text, ITEM_PLACEHOLDER_PATTERNS, item)
    if options:
        text = _replace_option_placeholders(text, options)
    if room_items:
        text = _replace_room_placeholders(text, room_items)
    return text


def _replace_placeholders(text: str, patterns: List[str], value: str) -> Tuple[str, int]:
    replaced = 0
    updated = text
    for pattern in patterns:
        updated, count = re.subn(pattern, value, updated, flags=re.IGNORECASE)
        replaced += count
    return updated, replaced


def _replace_option_placeholders(text: str, options: List[str]) -> str:
    updated = text
    for idx, pattern in enumerate(OPTION_PLACEHOLDER_PATTERNS):
        value = options[idx] if idx < len(options) else ""
        updated, _ = re.subn(pattern, value, updated, flags=re.IGNORECASE)
    return updated


def _replace_room_placeholders(text: str, items: List[str]) -> str:
    updated = text
    for idx, label in enumerate(ROOM_PLACEHOLDER_ORDER):
        if idx >= len(items):
            break
        pattern = r"\{\s*" + re.escape(label) + r"\s*\}"
        updated, _ = re.subn(pattern, items[idx], updated, flags=re.IGNORECASE)
    return updated


def _count_option_placeholders(text: str) -> int:
    return len(re.findall(r"\{\s*option\d+\s*\}", str(text), flags=re.IGNORECASE))


def _build_grouping_options(
    group_items: List[str],
    other_groups: List[List[str]],
    all_items: List[str],
    option_count: int,
) -> List[str]:
    correct = _join_items(group_items)
    candidates: List[str] = []

    for other in other_groups:
        if other and len(other) >= 2:
            candidates.append(_join_items(other))

    if len(group_items) >= 2:
        partial = group_items[:-1]
        if len(partial) >= 2:
            candidates.append(_join_items(partial))

    if other_groups and group_items:
        mixed = [group_items[0], other_groups[0][0]] if other_groups[0] else []
        if len(mixed) >= 2:
            candidates.append(_join_items(mixed))

    if len(candidates) < (option_count - 1):
        for i in range(len(all_items)):
            for j in range(i + 1, len(all_items)):
                candidates.append(_join_items([all_items[i], all_items[j]]))
                if len(candidates) >= (option_count - 1):
                    break
            if len(candidates) >= (option_count - 1):
                break

    unique: List[str] = []
    seen_keys = {_normalize_option_key(correct)}
    for cand in candidates:
        if not cand:
            continue
        key = _normalize_option_key(cand)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        if cand != correct and cand not in unique:
            unique.append(cand)

    if len(unique) < (option_count - 1):
        return []

    options = [correct] + unique[: option_count - 1]
    return options


def _select_negative_group_label(
    scene_id: str,
    source_label: str,
    group_label_pool: Dict[Tuple[str, str], List[str]],
    current_labels: List[str],
) -> str:
    current_set = set(current_labels)
    for (sid, src), labels in sorted(group_label_pool.items()):
        if sid == scene_id and src == source_label:
            continue
        for label in labels:
            if label not in current_set:
                return label
    return ""


def _join_items(items: List[str]) -> str:
    return " + ".join(items)


def _normalize_option_key(option_text: str) -> str:
    parts = [p.strip().lower() for p in str(option_text).split("+") if p.strip()]
    parts = sorted(set(parts))
    return " ".join(parts)



