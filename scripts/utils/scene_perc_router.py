from __future__ import annotations

import random
import re
from typing import Dict, List, Optional, Sequence, Tuple

try:
    from utils.data_model import SceneRecord, TemplateConfig
except ImportError:
    from backend.utils.data_access_layer.data_model import SceneRecord, TemplateConfig
try:
    from utils.data_parsers import _extract_placeholders, _normalize_key, parse_multi_view_dimensions, parse_viewpoint_text
except ImportError:
    from backend.utils.data_access_layer.data_parsers import _extract_placeholders, _normalize_key, parse_multi_view_dimensions, parse_viewpoint_text

ROUTE_STATIC = "STATIC"
ROUTE_OBJ_BOOL = "OBJ_BOOL"
ROUTE_FEATURE_BOOL = "FEATURE_BOOL"
ROUTE_MATERIAL_COLOR_TEXTURE = "MATERIAL_COLOR_TEXTURE"
ROUTE_OCR = "OCR"
ROUTE_REGION = "REGION"
ROUTE_UNSUPPORTED = "UNSUPPORTED"
ROUTE_MISSING = "MISSING"

OBJECT_PLACEHOLDER = "{object_type}"
FEATURE_PLACEHOLDER = "{feature_name}"


def route_scene_perception_template(
    template: Optional[TemplateConfig],
) -> Tuple[str, Optional[str]]:
    if template is None:
        return ROUTE_MISSING, "template_not_found"

    if template.benchmark_layer.lower() != "scene_perception":
        return ROUTE_UNSUPPORTED, "benchmark_layer"

    question_text = template.question_template or ""
    answer_type = _normalize_answer_type(template.answer_type)
    placeholders = {_normalize_key(p) for p in _extract_placeholders(question_text)}

    if "highlight" in placeholders:
        return ROUTE_REGION, "highlight_placeholder"

    if _is_ocr_template(template):
        return ROUTE_OCR, "ocr_subtask"

    if is_extract_any_text_context(template):
        return ROUTE_OCR, "extract_any_text"

    if is_dimension_context(template):
        return ROUTE_OCR, "dimension_context"

    if _contains_material_keywords(question_text):
        return ROUTE_MATERIAL_COLOR_TEXTURE, "material_keyword"

    has_object = "object_type" in placeholders
    has_feature = "feature_name" in placeholders
    if has_object and has_feature:
        return ROUTE_FEATURE_BOOL, "non_bool_answer_type" if answer_type and not is_bool_answer_type(answer_type) else None
    if has_object:
        return ROUTE_OBJ_BOOL, "non_bool_answer_type" if answer_type and not is_bool_answer_type(answer_type) else None

    if _has_option_placeholders(placeholders):
        return ROUTE_OCR, "options_placeholder"

    if "location_context" in placeholders:
        return ROUTE_STATIC, "location_context"

    return ROUTE_STATIC, "static"


def _normalize_answer_type(value: Optional[str]) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


def is_bool_answer_type(answer_type: Optional[str]) -> bool:
    if not answer_type:
        return False
    return str(answer_type).strip().lower() in {"bool", "boolean", "yes/no", "yesno"}


def _contains_material_keywords(question_text: str) -> bool:
    text = question_text.lower()
    return "texture" in text or "color" in text or "material" in text


def _has_option_placeholders(placeholders: set[str]) -> bool:
    for placeholder in placeholders:
        if re.match(r"^option\d+$", placeholder):
            return True
    return False


def _record_skip(
    skip_reasons: Optional[Dict[int, Dict[str, int]]],
    template_id: int,
    reason: str,
) -> None:
    if skip_reasons is None:
        return
    template_map = skip_reasons.setdefault(template_id, {})
    template_map[reason] = template_map.get(reason, 0) + 1


def render_template_text(
    template_text: str,
    scene: SceneRecord,
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    placeholders = _extract_placeholders(template_text)
    if not placeholders:
        return template_text, None, None

    object_choice = scene.objects[0] if scene.objects else ""
    feature_choice = ""
    if any(_normalize_key(p) == "feature_name" for p in placeholders):
        for obj, feats in scene.features_map.items():
            if feats:
                feature_choice = feats[0]
                if any(_normalize_key(p) == "object_type" for p in placeholders):
                    object_choice = obj
                break
    placeholder_values: Dict[str, str] = {}
    object_filled: Optional[str] = None
    feature_filled: Optional[str] = None

    for placeholder in placeholders:
        key = _normalize_key(placeholder)
        if key == "object_type":
            value = object_choice
            object_filled = value
        elif key == "feature_name":
            value = feature_choice
            feature_filled = value
        elif key == "location_context":
            value = scene.fields.get(key, "")
            if value:
                value = value.split(",")[0].strip()
        else:
            value = scene.fields.get(key, "")

        if not value:
            return None, None, None

        placeholder_values[placeholder] = value

    question_text = template_text
    for placeholder, value in placeholder_values.items():
        question_text = question_text.replace(f"{{{placeholder}}}", value)

    return question_text, object_filled, feature_filled


def _has_any_text_content(parsed_text: Dict[str, object]) -> bool:
    if not isinstance(parsed_text, dict):
        return False
    text_tags = parsed_text.get("text_tags")
    if isinstance(text_tags, list) and text_tags:
        return True
    if parsed_text.get("text"):
        return True
    room_tags = parsed_text.get("room_tags")
    if isinstance(room_tags, list) and room_tags:
        return True
    layer_tag_blocks = parsed_text.get("layer_tag_blocks")
    if isinstance(layer_tag_blocks, list) and layer_tag_blocks:
        return True
    return False


def _material_kind_from_template(template_text: str) -> Optional[str]:
    text = template_text.lower()
    if "texture" in text:
        return "texture"
    if "color" in text:
        return "color"
    if "material" in text:
        return "material"
    return None


def generate_material_questions(
    template: TemplateConfig,
    scenes: Sequence[SceneRecord],
) -> List[Dict]:
    kind = _material_kind_from_template(template.question_template)
    if not kind:
        return []

    results: List[Dict] = []
    for scene in scenes:
        obj_labels = list(dict.fromkeys(scene.material_map.get(kind, {}).keys()))
        if not obj_labels:
            continue
        for obj in obj_labels:
            if OBJECT_PLACEHOLDER in template.question_template:
                question_text = template.question_template.replace(OBJECT_PLACEHOLDER, obj)
                object_filled = obj
            else:
                question_text = template.question_template
                object_filled = obj
            results.append({
                "template_id": template.template_id,
                "layer_id": template.layer_id,
                "scene_id": scene.scene_id,
                "file_path": scene.file_path,
                "question_text": question_text,
                "object_type_filled": object_filled,
                "feature_name_filled": None,
                "ground_truth_answer": "",
                "answer_type": template.answer_type or "text",
            })

    return results


def generate_location_context_questions(
    template: TemplateConfig,
    scenes: Sequence[SceneRecord],
) -> List[Dict]:
    results: List[Dict] = []
    for scene in scenes:
        question_text, object_filled, feature_filled = render_template_text(template.question_template, scene)
        if question_text is None:
            continue
        results.append({
            "template_id": template.template_id,
            "layer_id": template.layer_id,
            "scene_id": scene.scene_id,
            "file_path": scene.file_path,
            "question_text": question_text,
            "object_type_filled": object_filled,
            "feature_name_filled": feature_filled,
            "ground_truth_answer": "",
            "answer_type": template.answer_type or "text",
        })
    return results


def generate_static_questions(
    template: TemplateConfig,
    scenes: Sequence[SceneRecord],
) -> List[Dict]:
    if _extract_placeholders(template.question_template):
        return []
    results: List[Dict] = []
    for scene in scenes:
        results.append({
            "template_id": template.template_id,
            "layer_id": template.layer_id,
            "scene_id": scene.scene_id,
            "file_path": scene.file_path,
            "question_text": template.question_template,
            "object_type_filled": None,
            "feature_name_filled": None,
            "ground_truth_answer": "",
            "answer_type": template.answer_type or "text",
        })
    return results


def generate_ocr_questions(
    template: TemplateConfig,
    scenes: Sequence[SceneRecord],
    skip_reasons: Optional[Dict[int, Dict[str, int]]] = None,
) -> List[Dict]:
    if is_dimension_context(template):
        return []

    is_bool_template = is_bool_answer_type(template.answer_type)
    is_any_text = not is_bool_template

    results: List[Dict] = []
    for scene in scenes:
        raw_text = scene.fields.get("text")
        parsed = parse_viewpoint_text(raw_text, tolerant=True)
        text_type_value: Optional[str] = None
        if isinstance(parsed, dict):
            room_tags = parsed.get("room_tags")
            if isinstance(room_tags, list) and room_tags:
                text_type_value = "room tag"
            if text_type_value is None:
                layer_tag_blocks = parsed.get("layer_tag_blocks")
                if isinstance(layer_tag_blocks, list) and layer_tag_blocks:
                    text_type_value = "layer tag"

        if is_any_text:
            if not _has_any_text_content(parsed):
                _record_skip(
                    skip_reasons,
                    template.template_id,
                    "Template {} skipped: no text content detected in viewpoint_scenes.text".format(template.template_id),
                )
                continue
            question_text = template.question_template
            object_filled = None
            feature_filled = None
            if "{text_type}" in question_text:
                if not text_type_value:
                    _record_skip(
                        skip_reasons,
                        template.template_id,
                        "Template {} skipped: missing text_type in viewpoint_scenes.text".format(template.template_id),
                    )
                    continue
                question_text = question_text.replace("{text_type}", text_type_value)
            question_text, object_filled, feature_filled = render_template_text(question_text, scene)
            if question_text is None:
                _record_skip(
                    skip_reasons,
                    template.template_id,
                    "Template {} skipped: missing required placeholder values".format(template.template_id),
                )
                continue
        else:
            question_template = template.question_template
            if text_type_value:
                question_template = question_template.replace("{text_type}", text_type_value)

            question_text, object_filled, feature_filled = render_template_text(question_template, scene)
            if question_text is None:
                _record_skip(
                    skip_reasons,
                    template.template_id,
                    "Template {} skipped: missing required placeholder values".format(template.template_id),
                )
                continue
        default_answer_type = template.answer_type or ("text" if is_any_text else "bool")
        results.append({
            "template_id": template.template_id,
            "layer_id": template.layer_id,
            "scene_id": scene.scene_id,
            "file_path": scene.file_path,
            "question_text": question_text,
            "object_type_filled": object_filled,
            "feature_name_filled": feature_filled,
            "ground_truth_answer": "",
            "answer_type": default_answer_type,
        })

    return results


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


def _format_numeric_value(value: object) -> str:
    if isinstance(value, float):
        return str(int(value)) if value.is_integer() else str(value)
    if isinstance(value, int):
        return str(value)
    return str(value).strip()


def _flatten_multi_view_dimension_candidates(
    raw_value: Optional[str],
) -> List[Dict[str, object]]:
    parsed = parse_multi_view_dimensions(raw_value, tolerant=True)
    if not isinstance(parsed, dict) or not parsed:
        return []

    candidates: List[Dict[str, object]] = []
    for object_type, dimensions in parsed.items():
        if not object_type or not isinstance(dimensions, dict):
            continue
        for dimension_key, payload in dimensions.items():
            if not dimension_key or payload is None or not isinstance(payload, dict):
                continue
            value = payload.get("value")
            if value is None or not isinstance(value, (int, float)):
                continue
            candidates.append({
                "object_type": object_type,
                "dimension_key": dimension_key,
                "display_target": f"{object_type} ({dimension_key})",
                "value": value,
                "unit": str(payload.get("unit") or "").strip(),
                "source_view": str(payload.get("source_view") or "").strip(),
            })
    return candidates


def generate_dimension_questions(
    template: TemplateConfig,
    scenes: Sequence[SceneRecord],
    skip_reasons: Optional[Dict[int, Dict[str, int]]] = None,
    rng: Optional[random.Random] = None,
    max_multiview_extra_questions: Optional[int] = 12,
) -> List[Dict]:
    if not is_dimension_context(template):
        return []

    results: List[Dict] = []
    multiview_results: List[Dict] = []
    for scene in scenes:
        raw_text = scene.fields.get("text")
        parsed = parse_viewpoint_text(raw_text, tolerant=True)
        dimensions = parsed.get("dimensions") if isinstance(parsed, dict) else None
        has_text_dimensions = isinstance(dimensions, dict) and bool(dimensions)
        multi_view_raw = scene.fields.get("multi_view_dimensions") or scene.fields.get("multi_view_dimension")
        multi_view_candidates = _flatten_multi_view_dimension_candidates(multi_view_raw)
        if not has_text_dimensions and not multi_view_candidates:
            _record_skip(
                skip_reasons,
                template.template_id,
                "Template {} skipped: no dimension entries found in viewpoint_scenes.text or multi_view_dimensions".format(template.template_id),
            )
            continue

        if has_text_dimensions:
            for obj_type in dimensions.keys():
                if not obj_type:
                    continue
                question_template = template.question_template.replace(OBJECT_PLACEHOLDER, obj_type)
                question_text, _, feature_filled = render_template_text(
                    question_template,
                    scene,
                )
                if question_text is None:
                    _record_skip(
                        skip_reasons,
                        template.template_id,
                        "Template {} skipped: missing required placeholder values".format(template.template_id),
                    )
                    continue
                results.append({
                    "template_id": template.template_id,
                    "layer_id": template.layer_id,
                    "scene_id": scene.scene_id,
                    "file_path": scene.file_path,
                    "question_text": question_text,
                    "object_type_filled": obj_type,
                    "feature_name_filled": feature_filled,
                    "ground_truth_answer": "",
                    "answer_type": template.answer_type or "text",
                    "dimension_source": "text_dimensions",
                })

        for candidate in multi_view_candidates:
            display_target = str(candidate.get("display_target") or "").strip()
            if not display_target:
                continue
            question_template = template.question_template.replace(OBJECT_PLACEHOLDER, display_target)
            question_text, _, feature_filled = render_template_text(
                question_template,
                scene,
            )
            if question_text is None:
                _record_skip(
                    skip_reasons,
                    template.template_id,
                    "Template {} skipped: missing required placeholder values".format(template.template_id),
                )
                continue
            multiview_results.append({
                "template_id": template.template_id,
                "layer_id": template.layer_id,
                "scene_id": scene.scene_id,
                "file_path": scene.file_path,
                "question_text": question_text,
                "object_type_filled": display_target,
                "feature_name_filled": feature_filled,
                "ground_truth_answer": _format_numeric_value(candidate.get("value")),
                "answer_type": template.answer_type or "text",
                "dimension_source": "multi_view_dimensions",
                "dimension_key_filled": candidate.get("dimension_key"),
                "dimension_unit": candidate.get("unit"),
                "dimension_source_view": candidate.get("source_view"),
            })

    if max_multiview_extra_questions is not None and max_multiview_extra_questions >= 0:
        cap = int(max_multiview_extra_questions)
        if len(multiview_results) > cap:
            if rng is None:
                multiview_results = multiview_results[:cap]
            else:
                selected_indices = sorted(rng.sample(range(len(multiview_results)), cap))
                multiview_results = [multiview_results[idx] for idx in selected_indices]

    results.extend(multiview_results)
    return results


def is_text_recognition_context(template: TemplateConfig) -> bool:
    if not template.context:
        return False
    context_raw = str(template.context).strip().lower()
    return "text recognition" in context_raw or "text_recognition" in context_raw


def is_extract_any_text_context(template: TemplateConfig) -> bool:
    if not template.context:
        return False
    context_value = str(template.context).strip().lower()
    return "extract any text" in context_value or "any text" in context_value


def is_dimension_context(template: TemplateConfig) -> bool:
    if not template.context:
        return False
    return "dimension" in str(template.context).strip().lower()


def is_object_detection_template(template: TemplateConfig) -> bool:
    if not template.subtask or not template.context:
        return False
    subtask = template.subtask.strip().lower()
    context = template.context.strip().lower()
    return subtask == "object_detection_and_naming" and context == "object_type"


def is_feature_detection_template(template: TemplateConfig) -> bool:
    if not template.subtask or not template.context:
        return False
    subtask = template.subtask.strip().lower()
    context = template.context.strip().lower()
    return subtask == "object_detection_and_naming" and "feature_name" in context


def _is_ocr_template(template: TemplateConfig) -> bool:
    subtask_raw = str(template.subtask).strip().lower() if template.subtask else ""
    subtask_key = subtask_raw.replace(" ", "_")
    if subtask_key == "ocr_text_detection":
        return True
    if is_text_recognition_context(template):
        return True
    return False
