from __future__ import annotations

import argparse
import random
import re
import sqlite3
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

try:
    from utils.data_model import TemplateConfig
except ImportError:
    from backend.utils.data_access_layer.data_model import TemplateConfig
try:
    from utils.data_parsers import (
        _clean_label,
        _clean_material_map,
        _normalize_key,
        _normalize_text,
        _parse_list_field,
        extract_viewpoint_all_text_ground_truth,
        extract_viewpoint_text_ground_truth,
        parse_feature_mapping,
        parse_material_field,
        parse_multi_view_dimensions,
        parse_viewpoint_text,
    )
except ImportError:
    from backend.utils.data_access_layer.data_parsers import (
        _clean_label,
        _clean_material_map,
        _normalize_key,
        _normalize_text,
        _parse_list_field,
        extract_viewpoint_all_text_ground_truth,
        extract_viewpoint_text_ground_truth,
        parse_feature_mapping,
        parse_material_field,
        parse_multi_view_dimensions,
        parse_viewpoint_text,
    )
try:
    from utils.scene_perc_router import (
        ROUTE_FEATURE_BOOL,
        ROUTE_MATERIAL_COLOR_TEXTURE,
        ROUTE_OBJ_BOOL,
        ROUTE_OCR,
        ROUTE_REGION,
        ROUTE_STATIC,
        ROUTE_UNSUPPORTED,
        is_bool_answer_type,
        is_dimension_context,
        is_text_recognition_context,
        route_scene_perception_template,
    )
except ImportError:
    from backend.utils.scene_perc_router import (
        ROUTE_FEATURE_BOOL,
        ROUTE_MATERIAL_COLOR_TEXTURE,
        ROUTE_OBJ_BOOL,
        ROUTE_OCR,
        ROUTE_REGION,
        ROUTE_STATIC,
        ROUTE_UNSUPPORTED,
        is_bool_answer_type,
        is_dimension_context,
        is_text_recognition_context,
        route_scene_perception_template,
    )

GLOBAL_COLORS = [
    "white",
    "black",
    "gray",
    "red",
    "blue",
    "green",
    "yellow",
    "brown",
    "beige",
    "turquoise",
    "pink",
    "orange",
    "purple",
    "silver",
    "gold",
]

GLOBAL_TEXTURES = [
    "soft",
    "rough",
    "smooth",
    "glossy",
    "matte",
    "fluffy",
    "hard",
    "shiny",
    "striped",
    "patterned",
    "wrinkled",
    "bumpy",
    "fuzzy",
]

GLOBAL_MATERIALS = [
    "wood",
    "metal",
    "glass",
    "plastic",
    "ceramic",
    "tiles",
    "parquet",
    "laminate",
    "marble",
    "concrete",
    "fabric",
    "leather",
    "paper",
    "stone",
    "carpet",
    "steel",
]


def _normalize_value(value: Optional[str]) -> str:
    if value is None:
        return ""
    return _clean_label(_normalize_text(value))


def _normalize_scene_id(value: object) -> Optional[object]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        number = float(text)
    except (TypeError, ValueError):
        return text
    if number.is_integer():
        return int(number)
    return text


def _resolve_db_column(columns: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    lower_map = {col.lower(): col for col in columns}
    for cand in candidates:
        key = cand.lower()
        if key in lower_map:
            return lower_map[key]
    return None


def _load_templates(
    cursor: sqlite3.Cursor,
) -> Dict[int, TemplateConfig]:
    cursor.execute(
        "SELECT name FROM sqlite_master "
        "WHERE type='table' AND lower(name)='templates' "
        "LIMIT 1"
    )
    row = cursor.fetchone()
    if not row:
        return {}
    table_name = row[0]

    cursor.execute(f"SELECT * FROM {table_name}")
    columns = [desc[0] for desc in cursor.description]

    template_id_col = _resolve_db_column(columns, ["template_id", "template"])
    question_template_col = _resolve_db_column(columns, ["question_template", "template_text", "template"])
    answer_type_col = _resolve_db_column(columns, ["answer_type", "answer"])
    layer_id_col = _resolve_db_column(columns, ["layer_id"])
    benchmark_layer_col = _resolve_db_column(columns, ["benchmark_layer"])
    commands_col = _resolve_db_column(columns, ["commands", "comments", "candidates"])
    context_col = _resolve_db_column(columns, ["context"])
    subtask_col = _resolve_db_column(columns, ["subtask"])
    metrics_col = _resolve_db_column(columns, ["metrics"])

    templates: Dict[int, TemplateConfig] = {}

    for row in cursor.fetchall():
        row_map = {columns[i]: row[i] for i in range(len(columns))}
        raw_id = row_map.get(template_id_col) if template_id_col else None
        if raw_id is None or str(raw_id).strip() == "":
            continue
        try:
            template_id = int(float(raw_id))
        except (TypeError, ValueError):
            continue

        question_template = str(row_map.get(question_template_col) or "")
        answer_type = str(row_map.get(answer_type_col) or "")
        layer_id = str(row_map.get(layer_id_col) or "") if layer_id_col else ""
        benchmark_layer = str(row_map.get(benchmark_layer_col) or "") if benchmark_layer_col else ""
        commands_raw = row_map.get(commands_col) if commands_col else None
        commands = _parse_list_field(commands_raw) if commands_raw is not None else []
        context = str(row_map.get(context_col)).strip() if context_col else None
        subtask = str(row_map.get(subtask_col)).strip() if subtask_col else None
        metrics = str(row_map.get(metrics_col)).strip() if metrics_col else None

        templates[template_id] = TemplateConfig(
            template_id=template_id,
            layer_id=layer_id,
            question_template=question_template,
            answer_type=answer_type,
            commands=commands,
            benchmark_layer=benchmark_layer,
            requires_rule=None,
            context=context,
            metrics=metrics,
            subtask=subtask,
        )
    return templates


def _parse_object_types(raw_value: Optional[str]) -> List[str]:
    if raw_value is None:
        return []
    text = str(raw_value).strip()
    if not text:
        return []
    return [_normalize_value(part) for part in text.split(",") if _normalize_value(part)]


def _load_scene_cache(
    cursor: sqlite3.Cursor,
) -> Dict[object, Dict[str, object]]:
    cursor.execute("SELECT * FROM viewpoint_scenes")
    columns = [desc[0] for desc in cursor.description]

    scene_id_col = _resolve_db_column(columns, ["scene_id", "scene", "id"])
    object_type_col = _resolve_db_column(columns, ["object_type", "objects"])
    feature_name_col = _resolve_db_column(columns, ["feature_name", "features"])
    material_col = _resolve_db_column(columns, ["material"])
    text_col = _resolve_db_column(columns, ["text"])

    cache: Dict[object, Dict[str, object]] = {}

    for row in cursor.fetchall():
        row_map = {columns[i]: row[i] for i in range(len(columns))}
        scene_id_raw = row_map.get(scene_id_col) if scene_id_col else None
        scene_id = _normalize_scene_id(scene_id_raw)
        if scene_id is None:
            continue

        objects_present: Set[str] = set()
        valid_pairs: Set[Tuple[str, str]] = set()

        obj_raw = row_map.get(object_type_col) if object_type_col else None
        row_objects = _parse_object_types(obj_raw)
        for obj in row_objects:
            objects_present.add(obj)

        feat_raw = row_map.get(feature_name_col) if feature_name_col else None
        mapping = parse_feature_mapping(feat_raw) if feat_raw is not None else {}
        for obj in row_objects:
            for feat in mapping.get(obj, []):
                feat_norm = _normalize_value(feat)
                if feat_norm:
                    valid_pairs.add((obj, feat_norm))

        material_raw = row_map.get(material_col) if material_col else None
        material_map = _clean_material_map(parse_material_field(material_raw)) if material_raw is not None else {
            "texture": {},
            "material": {},
            "color": {},
        }

        fields: Dict[str, str] = {}
        for col, value in row_map.items():
            if value is None:
                continue
            if isinstance(value, float) and value != value:
                continue
            value_str = str(value).strip()
            if not value_str:
                continue
            key = _normalize_key(col)
            if key not in fields:
                fields[key] = value_str

        parsed_text = {}
        if text_col:
            parsed_text = parse_viewpoint_text(row_map.get(text_col), tolerant=True)

        cache[scene_id] = {
            "objects_present": objects_present,
            "valid_pairs": valid_pairs,
            "material_map": material_map,
            "fields": fields,
            "parsed_text": parsed_text,
        }

    return cache


def _compute_obj_feature_truth(
    scene_id: object,
    object_type_filled: Optional[str],
    feature_name_filled: Optional[str],
    scene_cache: Dict[object, Dict[str, object]],
) -> str:
    obj = _normalize_value(object_type_filled)
    if not obj:
        return ""

    if scene_id not in scene_cache:
        return "Scene not found"

    objects_present = scene_cache[scene_id]["objects_present"]
    valid_pairs = scene_cache[scene_id]["valid_pairs"]

    if obj not in objects_present:
        return "no"

    if feature_name_filled is None or not str(feature_name_filled).strip():
        return "yes"

    feature = _normalize_value(feature_name_filled)
    if feature and (obj, feature) in valid_pairs:
        return "yes"

    return "no"


def _scene_has_ocr_text(scene_fields: Dict[str, str]) -> bool:
    keys = ("text", "multi_view_dimensions", "multi_view_dimension")
    for key in keys:
        value = scene_fields.get(key)
        if value is None:
            continue
        value_str = str(value).strip()
        if value_str and value_str.lower() not in {"nan", "none", "null"}:
            return True
    return False


def _detect_target_text_tag(question_text: Optional[str]) -> Optional[str]:
    text = str(question_text or "").lower()
    has_room = "room tag" in text
    has_layer = "layer tag" in text
    if has_room and not has_layer:
        return "room tag"
    if has_layer and not has_room:
        return "layer tag"
    return None


def _extract_ocr_text_ground_truth(
    parsed_text: Dict[str, object],
    question_text: Optional[str],
) -> str:
    if not isinstance(parsed_text, dict):
        return ""

    target_tag = _detect_target_text_tag(question_text)
    if target_tag:
        return extract_viewpoint_text_ground_truth(
            parsed_text,
            target_tag,
            highlighted_only_rule=True,
        )

    question_lower = str(question_text or "").lower()
    asks_for_all_text = (
        "all the text" in question_lower
        or "all text" in question_lower
        or "including digits" in question_lower
    )
    text_raw = str(parsed_text.get("text") or "").strip()
    room_gt = extract_viewpoint_text_ground_truth(parsed_text, "room tag", highlighted_only_rule=True)
    layer_gt = extract_viewpoint_text_ground_truth(parsed_text, "layer tag", highlighted_only_rule=True)

    if asks_for_all_text:
        all_text_gt = extract_viewpoint_all_text_ground_truth(parsed_text)
        if all_text_gt:
            return all_text_gt
        values: List[str] = []
        if text_raw:
            values.append(text_raw)
        if room_gt:
            values.append(room_gt)
        if layer_gt:
            values.append(layer_gt)
        return "; ".join(values)

    if text_raw:
        return text_raw
    if room_gt:
        return room_gt
    if layer_gt:
        return layer_gt
    return ""


def _extract_first_dimension(
    dimensions: Dict[str, Dict[str, object]],
) -> Tuple[Optional[str], Optional[float]]:
    for obj_type, payload in dimensions.items():
        if not isinstance(payload, dict):
            continue
        if "value" in payload:
            value = payload.get("value")
            if isinstance(value, (int, float)):
                return obj_type, float(value)
        params = payload.get("params")
        if isinstance(params, dict):
            for val in params.values():
                if isinstance(val, (int, float)):
                    return obj_type, float(val)
    return None, None


def _format_dimension_scalar(value: object) -> str:
    if isinstance(value, float):
        return str(int(value)) if value.is_integer() else str(value)
    if isinstance(value, int):
        return str(value)
    return str(value).strip()


def _extract_dimension_scalar_from_payload(payload: object) -> Optional[str]:
    if not isinstance(payload, dict):
        return None

    if "value" in payload:
        value = payload.get("value")
        if value is None:
            return None
        text = _format_dimension_scalar(value)
        return text if text else None

    params = payload.get("params")
    if isinstance(params, dict):
        for value in params.values():
            if value is None:
                continue
            text = _format_dimension_scalar(value)
            if text:
                return text
    return None


def _split_dimension_display_target(value: Optional[str]) -> Tuple[str, Optional[str]]:
    text = str(value or "").strip()
    if not text:
        return "", None
    match = re.match(r"^(.*)\(([^()]*)\)\s*$", text)
    if not match:
        return text, None
    object_type = match.group(1).strip()
    dimension_key = match.group(2).strip()
    if not object_type:
        return text, None
    return object_type, (dimension_key or None)


def _extract_multi_view_dimension_value(
    parsed_multi_dimensions: Dict[str, Dict[str, Optional[Dict[str, object]]]],
    object_type_filled: Optional[str],
    dimension_key: Optional[str] = None,
) -> str:
    if not isinstance(parsed_multi_dimensions, dict) or not parsed_multi_dimensions:
        return ""

    object_norm = _normalize_value(object_type_filled)
    key_norm = _normalize_value(dimension_key) if dimension_key else ""

    if object_norm:
        for obj_type, obj_dimensions in parsed_multi_dimensions.items():
            if _normalize_value(obj_type) != object_norm:
                continue
            if not isinstance(obj_dimensions, dict):
                return ""
            if key_norm:
                for dim_key, payload in obj_dimensions.items():
                    if _normalize_value(dim_key) != key_norm:
                        continue
                    value_text = _extract_dimension_scalar_from_payload(payload)
                    return value_text or ""
                return ""
            for payload in obj_dimensions.values():
                value_text = _extract_dimension_scalar_from_payload(payload)
                if value_text:
                    return value_text
            return ""

    for obj_dimensions in parsed_multi_dimensions.values():
        if not isinstance(obj_dimensions, dict):
            continue
        for payload in obj_dimensions.values():
            value_text = _extract_dimension_scalar_from_payload(payload)
            if value_text:
                return value_text
    return ""


def _compute_dimension_ground_truth(
    parsed_text: Dict[str, object],
    object_type_filled: Optional[str],
    scene_fields: Optional[Dict[str, str]] = None,
) -> str:
    dimensions = parsed_text.get("dimensions") if isinstance(parsed_text, dict) else None
    text_dimensions = dimensions if isinstance(dimensions, dict) else {}
    object_base, dimension_key = _split_dimension_display_target(object_type_filled)
    object_norm = _normalize_value(object_base)

    raw_multi_view = None
    if isinstance(scene_fields, dict):
        raw_multi_view = scene_fields.get("multi_view_dimensions") or scene_fields.get("multi_view_dimension")
    parsed_multi_view = parse_multi_view_dimensions(raw_multi_view, tolerant=True)

    if object_base:
        # Question explicitly targets object + dimension key from multi_view_dimensions.
        if dimension_key:
            multi_value = _extract_multi_view_dimension_value(
                parsed_multi_view,
                object_base,
                dimension_key=dimension_key,
            )
            return multi_value or ""

        for obj_type, payload in text_dimensions.items():
            if _normalize_value(obj_type) != object_norm:
                continue
            value_text = _extract_dimension_scalar_from_payload(payload)
            if value_text:
                return value_text

        multi_value = _extract_multi_view_dimension_value(parsed_multi_view, object_base)
        return multi_value or ""

    for payload in text_dimensions.values():
        value_text = _extract_dimension_scalar_from_payload(payload)
        if value_text:
            return value_text

    if text_dimensions:
        _, fallback_value = _extract_first_dimension(text_dimensions)
        if fallback_value is not None:
            return str(int(fallback_value)) if float(fallback_value).is_integer() else str(fallback_value)

    fallback_multi = _extract_multi_view_dimension_value(parsed_multi_view, None)
    if fallback_multi:
        return fallback_multi
    return ""



def _detect_category(question_text: Optional[str]) -> Optional[str]:
    if not question_text:
        return None
    text = question_text.lower()
    for category in ("texture", "color", "material"):
        if re.search(rf"\b{category}\b", text):
            return category
    return None


def _is_multichoice(answer_type: Optional[str]) -> bool:
    if not answer_type:
        return False
    return str(answer_type).strip().lower() in {
        "mcq",
        "multi",
        "multichoice",
        "multiple_choice",
        "choice",
        "choices",
    }


def _strip_existing_choices(question_text: str) -> str:
    if not question_text:
        return ""
    parts = re.split(r"(?i)\bchoose\s+one\s*:", question_text, maxsplit=1)
    return parts[0].strip()


def _build_choice_suffix(options: List[str]) -> str:
    return (
        f' Choose one: "{options[0]}", "{options[1]}", "{options[2]}", or "{options[3]}"'
    )


def _get_global_pool(category: str) -> List[str]:
    if category == "color":
        return list(GLOBAL_COLORS)
    if category == "texture":
        return list(GLOBAL_TEXTURES)
    return list(GLOBAL_MATERIALS)


def _select_correct_value(values: List[str], rng: random.Random) -> Optional[str]:
    if not values:
        return None
    return rng.choice(values)


def _generate_options(category: str, correct_value: str, rng: random.Random) -> List[str]:
    pool = [v for v in _get_global_pool(category) if v != correct_value]
    if len(pool) < 3:
        raise RuntimeError(
            f"Not enough distractors for category '{category}'. "
            f"Need at least 3, got {len(pool)}."
        )
    distractors = rng.sample(pool, 3)
    insert_index = rng.choice([1, 2])
    options = list(distractors)
    options.insert(insert_index, correct_value)
    return options


def _compute_material_ground_truth(
    scene_id: object,
    question_text: Optional[str],
    object_type_filled: Optional[str],
    answer_type: Optional[str],
    scene_cache: Dict[object, Dict[str, object]],
    rng: random.Random,
) -> Tuple[Optional[str], Optional[str]]:
    category = _detect_category(question_text)
    if not category:
        return None, None

    if scene_id not in scene_cache:
        return "Scene not found", question_text or ""

    obj = _normalize_value(object_type_filled)
    if not obj:
        return "", question_text or ""

    material_map = scene_cache[scene_id]["material_map"]
    category_map = material_map.get(category, {})
    values = category_map.get(obj, [])

    if category == "color" and not _is_multichoice(answer_type):
        correct_value = _select_correct_value(values, rng)
        return correct_value or "", question_text or ""

    correct_value = _select_correct_value(values, rng)
    if not correct_value:
        return "", question_text or ""

    options = _generate_options(category, correct_value, rng)
    base_text = _strip_existing_choices(question_text or "")
    updated_text = f"{base_text}{_build_choice_suffix(options)}".strip()
    return correct_value, updated_text


def _load_question_rows(
    cursor: sqlite3.Cursor,
) -> Iterable[Tuple[int, Optional[int], Optional[object], Optional[str], Optional[str], Optional[str], Optional[str]]]:
    query = (
        "SELECT generated_question_id, template_id, scene_id, question_text, "
        "object_type_filled, feature_name_filled, answer_type "
        "FROM generated_scene_questions"
    )
    cursor.execute(query)
    return cursor.fetchall()


def update_ground_truth(db_path: Path, seed: Optional[int] = None) -> None:
    rng = random.Random(seed)
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        templates = _load_templates(cursor)
        scene_cache = _load_scene_cache(cursor)

        updates: List[Tuple[str, str, int]] = []

        for (
            row_id,
            template_id,
            scene_id_raw,
            question_text,
            object_type_filled,
            feature_name_filled,
            row_answer_type,
        ) in _load_question_rows(cursor):
            scene_id = _normalize_scene_id(scene_id_raw)
            template = templates.get(int(template_id)) if template_id is not None else None
            if template is None:
                continue

            route, _ = route_scene_perception_template(template)
            if route == ROUTE_UNSUPPORTED or route == ROUTE_REGION:
                continue

            ground_truth: Optional[str] = None
            updated_text: Optional[str] = None

            if route in (ROUTE_OBJ_BOOL, ROUTE_FEATURE_BOOL):
                ground_truth = _compute_obj_feature_truth(
                    scene_id,
                    object_type_filled,
                    feature_name_filled,
                    scene_cache,
                )

            elif route == ROUTE_MATERIAL_COLOR_TEXTURE:
                effective_answer_type = template.answer_type or row_answer_type
                gt, updated = _compute_material_ground_truth(
                    scene_id,
                    question_text,
                    object_type_filled,
                    effective_answer_type,
                    scene_cache,
                    rng,
                )
                if gt is not None:
                    ground_truth = gt
                    updated_text = updated

            elif route == ROUTE_OCR:
                effective_answer_type = template.answer_type or row_answer_type
                if scene_id not in scene_cache:
                    ground_truth = "Scene not found"
                else:
                    scene_payload = scene_cache[scene_id]
                    parsed_text = scene_payload.get("parsed_text", {})
                    parsed_text_map = parsed_text if isinstance(parsed_text, dict) else {}

                    if is_dimension_context(template):
                        ground_truth = _compute_dimension_ground_truth(
                            parsed_text_map,
                            object_type_filled,
                            scene_payload.get("fields") if isinstance(scene_payload, dict) else None,
                        )
                    elif is_bool_answer_type(effective_answer_type):
                        if not is_text_recognition_context(template):
                            continue
                        has_text = _scene_has_ocr_text(scene_payload["fields"])
                        ground_truth = "yes" if has_text else "no"
                    else:
                        ground_truth = _extract_ocr_text_ground_truth(
                            parsed_text_map,
                            question_text,
                        )

            elif route == ROUTE_STATIC and is_dimension_context(template):
                if scene_id not in scene_cache:
                    ground_truth = "Scene not found"
                else:
                    parsed_text = scene_cache[scene_id].get("parsed_text", {})
                    ground_truth = _compute_dimension_ground_truth(
                        parsed_text if isinstance(parsed_text, dict) else {},
                        object_type_filled,
                        scene_cache[scene_id].get("fields") if isinstance(scene_cache[scene_id], dict) else None,
                    )

            if ground_truth is None and updated_text is None:
                continue

            if updated_text is None:
                updated_text = question_text or ""

            updates.append((ground_truth or "", updated_text, row_id))

        if updates:
            cursor.executemany(
                "UPDATE generated_scene_questions "
                "SET ground_truth_answer = ?, question_text = ? "
                "WHERE generated_question_id = ?",
                updates,
            )
            conn.commit()
    finally:
        conn.close()


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    backend_dir = script_dir.parent
    default_db = backend_dir / "unified_database.db"

    parser = argparse.ArgumentParser(
        description="Update ground_truth_answer for Scene Perception questions.",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=default_db,
        help="Path to unified_database.db",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducibility.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.db.exists():
        raise FileNotFoundError(f"Database not found: {args.db}")
    update_ground_truth(args.db, seed=args.seed)


if __name__ == "__main__":
    main()
