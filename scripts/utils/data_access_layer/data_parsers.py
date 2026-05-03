from __future__ import annotations

import json
import re
from typing import Dict, List, Optional, Sequence, Tuple
import pandas as pd


def parse_requires_rule(
    requires_rule_val: str,
    *,
    tolerant: bool = False,
    warnings: Optional[List[str]] = None,
) -> List[str]:
    # Field should be JSON array, doublecheck for to comma-separated case
    if pd.isna(requires_rule_val) or not str(requires_rule_val).strip():
        return []

    raw_value = str(requires_rule_val).strip()
    if tolerant:
        _check_unbalanced(raw_value, warnings, "parse_requires_rule")
    try:
        parsed = json.loads(raw_value)
        if isinstance(parsed, list):
            cleaned = []
            for item in parsed:
                txt = str(item).strip()
                if txt:
                    cleaned.append(txt)
            return cleaned
        else:
            # Single value, convert to list
            return [str(parsed).strip()]
    except (json.JSONDecodeError, ValueError):
        return [p.strip() for p in raw_value.split(",") if p.strip()]


def detect_rule_text_placeholder(question_template: str) -> Tuple[Optional[str], str]:
    """
    Detect placeholder type in template. Returns (placeholder, text_type).
    - placeholder_string: the actual placeholder to replace (e.g., '{parent_rule_text}', '{rule_text_atomic}')
    - text_type: 'parent_rule_text' or 'rule_text_atomic'

    Priority: parent_rule_text > rule_text_atomic
    """
    if pd.isna(question_template):
        return None, ""

    template_str = str(question_template)
    if "{parent_rule_text}" in template_str:
        return "{parent_rule_text}", "parent_rule_text"
    if "{rule_text_atomic}" in template_str:
        return "{rule_text_atomic}", "rule_text_atomic"
    if "[parent_rule_text]" in template_str:
        return "[parent_rule_text]", "parent_rule_text"

    if "[rule_text_atomic]" in template_str:
        return "[rule_text_atomic]", "rule_text_atomic"
    return None, ""


def normalize_id(id_value: str) -> Tuple[str, str, str]:
    """
    Returns: (normalized_primary, normalized_secondary, original_str)
    """
    id_clean = str(id_value).strip()
    id_normalized = id_clean

    if "." in id_clean:
        head, tail = id_clean.split(".", 1)
        if head.lstrip("-").isdigit() and tail.isdigit():
            tail = tail.rstrip("0")
            id_normalized = head if tail == "" else f"{head}.{tail}"

    return id_normalized, id_normalized, id_clean

def find_in_dict(dictionary: Dict, *keys: str) -> Optional[Dict]:
    """Try multiple keys to find a value in dictionary."""
    for key in keys:
        if key in dictionary:
            return dictionary[key]
    return None


def parse_material_field(
    material_value: Optional[str],
    *,
    tolerant: bool = False,
    warnings: Optional[List[str]] = None,
) -> Dict[str, Dict[str, List[str]]]:
    """
        texture={object: texture description, other: other description}
        material={floor: parquet/laminate}
        color={walls: white}
    Returns:
        {
            "texture": {"object": ["texture description"], "other": ["other description"]},
            "material": {"floor": ["parquet", "laminate"]},
            "color": {"walls": ["white"]},
        }
    """
    result: Dict[str, Dict[str, List[str]]] = {
        "texture": {},
        "material": {},
        "color": {},
    }

    if material_value is None or pd.isna(material_value) or not str(material_value).strip():
        return result

    text = str(material_value)
    if tolerant:
        _check_unbalanced(text, warnings, "parse_material_field")
    pattern = r"(texture|material|color)\s*=\s*\{([^}]*)\}"
    for kind, content in re.findall(pattern, text, flags=re.IGNORECASE):
        key = kind.lower()
        entries = _split_top_level(content, tolerant=tolerant, warnings=warnings, context="parse_material_field")
        for entry in entries:
            if ":" in entry:
                obj_part, desc_part = entry.split(":", 1)
            else:
                obj_part, desc_part = entry, ""

            obj_label = obj_part.strip()
            desc = desc_part.strip()
            if not obj_label:
                continue

            alternatives: List[str] = []
            if desc:
                for alt in desc.split("/"):
                    alt_clean = alt.strip()
                    if alt_clean:
                        alternatives.append(alt_clean)

            if obj_label not in result[key]:
                result[key][obj_label] = []
            for alt in alternatives:
                if alt not in result[key][obj_label]:
                    result[key][obj_label].append(alt)

    return result

def _normalize_text(value: str) -> str:
    return str(value).strip()


def _normalize_scene_id(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    s = str(value).strip()
    return s[:-2] if s.endswith(".0") else s


def _clean_label(value: str) -> str:
    """Remove highlight markers from object/feature labels."""
    text = str(value).strip()
    if not text:
        return text
    # Remove leading "highlighted:" (case-insensitive) with optional whitespace
    text = re.sub(r"^highlighted\s*:\s*", "", text, flags=re.IGNORECASE)
    # Remove any standalone "highlighted" token left behind
    text = re.sub(r"\bhighlighted\b\s*:?$", "", text, flags=re.IGNORECASE).strip()
    return text.strip()


def _normalize_scene_label(value: str) -> str:
    """
    Normalize scene entity labels for relations (spaces/objects).
    Lowercase to avoid case-variant duplicates.
    """
    text = _clean_label(value)
    return text.lower()


def _clean_material_map(material_map: Dict[str, Dict[str, List[str]]]) -> Dict[str, Dict[str, List[str]]]:
    cleaned: Dict[str, Dict[str, List[str]]] = {"texture": {}, "material": {}, "color": {}}
    for kind, items in material_map.items():
        if kind not in cleaned:
            cleaned[kind] = {}
        for obj, descs in items.items():
            obj_clean = _clean_label(obj)
            if not obj_clean:
                continue
            desc_cleaned: List[str] = []
            for desc in descs:
                desc_value = _clean_label(desc)
                if desc_value and desc_value not in desc_cleaned:
                    desc_cleaned.append(desc_value)
            cleaned[kind][obj_clean] = desc_cleaned
    return cleaned

def parse_template_id_list(
    raw_value: Optional[str],
    *,
    tolerant: bool = False,
    warnings: Optional[List[str]] = None,
) -> List[int]:
    items = parse_requires_rule(raw_value, tolerant=tolerant, warnings=warnings)
    template_ids: List[int] = []
    for part in items:
        try:
            template_ids.append(int(float(part)))
        except (ValueError, TypeError):
            continue
    seen = set()
    return [tid for tid in template_ids if not (tid in seen or seen.add(tid))]


def format_template_id_list(original_value: object, template_ids: List[int]) -> str:
    raw = "" if original_value is None else str(original_value).strip()
    if raw.startswith("[") and raw.endswith("]"):
        return json.dumps(template_ids)
    return ",".join(str(tid) for tid in template_ids)


def _parse_list_field(
    raw_value: Optional[str],
    *,
    tolerant: bool = False,
    warnings: Optional[List[str]] = None,
) -> List[str]:
    items = parse_requires_rule(raw_value, tolerant=tolerant, warnings=warnings)
    return [_clean_label(item) for item in items if str(item).strip()]


def parse_feature_mapping(
    raw_value: Optional[str],
    *,
    tolerant: bool = False,
    warnings: Optional[List[str]] = None,
) -> Dict[str, List[str]]:
    """
    Parse feature_name format:
    "{toilet: tank, set, lid, rim}, {sink: drain hole}"
    Returns: {"toilet": ["tank", "set", "lid", "rim"], "sink": ["drain hole"]}
    """
    if raw_value is None or pd.isna(raw_value) or not str(raw_value).strip():
        return {}

    text = str(raw_value).strip()
    if tolerant:
        _check_unbalanced(text, warnings, "parse_feature_mapping")
    blocks = re.findall(r"\{([^{}]+)\}", text)
    mapping: Dict[str, List[str]] = {}
    for block in blocks:
        if ":" not in block:
            continue
        obj_part, feat_part = block.split(":", 1)
        obj = _clean_label(_normalize_text(obj_part))
        features = [_clean_label(f.strip()) for f in feat_part.split(",") if f.strip()]
        if obj:
            mapping[obj] = features
    return mapping

def _normalize_key(value: str) -> str:
    key = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower())
    return key.strip("_")

def _resolve_column(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    lower_map = {col.lower(): col for col in df.columns}
    alias_map = {
        col.lower().replace("-", "_"): col
        for col in df.columns
        if "-" in col
    }
    for cand in candidates:
        key = cand.lower()
        if key in lower_map:
            return lower_map[key]
        key_norm = key.replace("-", "_")
        if key_norm in alias_map:
            return alias_map[key_norm]
    return None


def _extract_placeholders(template_text: str) -> List[str]:
    return re.findall(r"\{([^{}]+)\}", template_text)


def parse_scene_relations(
    raw_value: Optional[str],
    source_column: Optional[str] = None,
    *,
    tolerant: bool = False,
    warnings: Optional[List[str]] = None,
) -> Dict[str, object]:
    """
    Parse adjacency/connectivity/functional grouping blocks from a scene field.

    The entity type depends on the source column:
      - space_naming      -> spaces
      - spatial_relations -> objects

    Supported blocks:
      adjacency={A, B, C; D, E}
      connectivity={A: B, C; D, E}
      not_direct_connectivity={A: B, C; D, E}
      functional grouping={zone label: item1, item2; ...}

    Returns:
      {
        "entity_type": "spaces" | "objects" | "unknown",
        "adjacency": {"A": ["B", "C"]},
        "connectivity": {"A": ["B", "C"]},
        "not_direct_connectivity": {"A": ["B", "C"]},
        "functional_grouping": {"zone label": ["item1", "item2"]}
      }
    """
    result: Dict[str, object] = {
        "entity_type": _infer_scene_entity_type(source_column),
        "adjacency": {},
        "connectivity": {},
        "not_direct_connectivity": {},
        "functional_grouping": {},
    }

    if raw_value is None or pd.isna(raw_value) or not str(raw_value).strip():
        return result

    text = str(raw_value)
    if tolerant:
        _check_unbalanced(text, warnings, "parse_scene_relations")
    for key, body in _extract_relation_blocks(text, tolerant=tolerant, warnings=warnings):
        canonical = _normalize_relation_key(key)
        if canonical is None:
            continue
        mapping = _parse_relation_mapping(body, tolerant=tolerant, warnings=warnings)
        bucket = result[canonical]
        for subject, items in mapping.items():
            if subject not in bucket:
                bucket[subject] = []
            for item in items:
                if item not in bucket[subject]:
                    bucket[subject].append(item)
    return result


def parse_viewpoint_text(
    raw_value: Optional[str],
    *,
    tolerant: bool = False,
) -> Dict[str, object]:
    """
    Parse the `viewpoint_scenes.text` cell into structured components.

    Supported blocks (examples):
    - text={"shower curtain"}
    - highlighted:room tag={room name:Child bedroom, room area:14 m2}
    - room tag={room name:Guest bedroom, room area:24 m2}
    - label with a leader/room tag={room name:Guest WC, room area:2 m2}
    - label with a leader/layer tag={load-bearing structure:[reinforced concrete slab, 300 mm], sound insulation}
    - mirror={dimension:600 mm}
    - mirror={dimension ["unit": "mm"]:{"width"=600, "height"=900}}

    Returns:
      {
        "text": str | None,
        "room_tags": [
          {
            "tag_type": "room tag",
            "source": str,
            "is_highlighted": bool,
            "entries": [
              {"meaning": str|None, "text_items": [str], "raw_entry": str}
            ],
            "ocr_text_items": [str],
            "ocr_ground_truth": str,
            "name": str,
            "area_value": float|int|None,
            "area_unit": str,
            "fields": {raw_room_key: raw_room_value}
          }
        ],
        "layer_tags": [
          {
            "tag_type": "layer tag",
            "source": str,
            "is_highlighted": bool,
            "entries": [
              {"meaning": str|None, "text_items": [str], "raw_entry": str}
            ],
            "ocr_text_items": [str],
            "ocr_ground_truth": str
          }
        ],
        "layer_tag_blocks": [
          {
            "tag_type": "layer tag",
            "source": str,
            "is_highlighted": bool,
            "entries": [
              {"meaning": str|None, "text_items": [str], "raw_entry": str}
            ],
            "ocr_text_items": [str],
            "ocr_ground_truth": str
          }
        ],
        "text_tags": [
          # ordered mixed blocks in source order:
          # text, room tag, layer tag, dimension
          {
            "tag_type": str,
            "source": str,
            "is_highlighted": bool,
            "entries": [{"meaning": str|None, "text_items": [str], "raw_entry": str}],
            "ocr_text_items": [str],
            "ocr_ground_truth": str
          }
        ],
        "dimensions": {object_type: {"value": num, "unit": str} | {"unit": str, "params": {...}}}
      }
      When tolerant=True, the output may include:
      - "parse_warnings": list of warnings (unbalanced braces/brackets/quotes, fallback splits)
    """
    if raw_value is None or (isinstance(raw_value, float) and pd.isna(raw_value)):
        return {}
    text = str(raw_value).strip()
    if not text:
        return {}

    warnings: Optional[List[str]] = [] if tolerant else None

    result: Dict[str, object] = {}
    room_tags: List[Dict[str, object]] = []
    layer_tag_blocks: List[Dict[str, object]] = []
    text_tags: List[Dict[str, object]] = []
    dimensions: Dict[str, Dict[str, object]] = {}
    text_value: Optional[str] = None

    blocks = _split_top_level(text, tolerant=tolerant, warnings=warnings, context="parse_viewpoint_text")
    for block in blocks:
        block = block.strip().strip(",").strip()
        if not block:
            continue
        if "=" not in block:
            continue

        left, right = block.split("=", 1)
        left_clean = left.strip()
        left_norm = left_clean.lower()
        payload = _strip_outer_braces(right.strip())

        if left_norm == "text":
            text_value = _strip_quotes(payload)
            text_items = [text_value] if text_value else []
            entries = [{
                "meaning": None,
                "text_items": list(text_items),
                "raw_entry": payload,
            }]
            _, ocr_ground_truth = _build_tag_ocr_text(entries)
            text_tags.append({
                "tag_type": "text",
                "source": "text",
                "is_highlighted": False,
                "entries": entries,
                "ocr_text_items": text_items,
                "ocr_ground_truth": ocr_ground_truth,
            })
            continue

        tag_header = _parse_viewpoint_tag_header(left_clean)
        if tag_header and tag_header.get("tag_type") == "layer tag":
            block_data = _parse_layer_tag_block(payload, tolerant=tolerant, warnings=warnings)
            if not block_data:
                continue
            tag_block = {
                "tag_type": "layer tag",
                "source": tag_header.get("source", "layer tag"),
                "is_highlighted": bool(tag_header.get("is_highlighted")),
                "entries": block_data.get("entries", []),
                "ocr_text_items": block_data.get("ocr_text_items", []),
                "ocr_ground_truth": block_data.get("ocr_ground_truth", ""),
            }
            layer_tag_blocks.append(tag_block)
            text_tags.append(tag_block)
            continue

        if tag_header and tag_header.get("tag_type") == "room tag":
            room_info = _parse_room_tag_block(payload, tolerant=tolerant, warnings=warnings)
            if room_info:
                room_info["tag_type"] = "room tag"
                room_info["is_highlighted"] = bool(tag_header.get("is_highlighted"))
                room_info["source"] = str(tag_header.get("source") or "room tag")
                room_tags.append(room_info)
                text_tags.append(room_info)
            continue

        object_type = left_clean
        dimension_data = _parse_dimension_block(payload, tolerant=tolerant, warnings=warnings)
        if dimension_data and object_type:
            dimensions[object_type] = dimension_data
            text_items = _dimension_to_text_items(dimension_data)
            entries = [{
                "meaning": "dimension",
                "text_items": list(text_items),
                "raw_entry": payload,
            }]
            _, ocr_ground_truth = _build_tag_ocr_text(entries)
            text_tags.append({
                "tag_type": "dimension",
                "source": object_type,
                "is_highlighted": False,
                "object_type": object_type,
                "entries": entries,
                "ocr_text_items": text_items,
                "ocr_ground_truth": ocr_ground_truth,
            })

    if text_value is not None:
        result["text"] = text_value
    if room_tags:
        result["room_tags"] = room_tags
    if layer_tag_blocks:
        result["layer_tags"] = list(layer_tag_blocks)
        result["layer_tag_blocks"] = layer_tag_blocks
    if text_tags:
        result["text_tags"] = text_tags
    if dimensions:
        result["dimensions"] = dimensions
    if warnings:
        result["parse_warnings"] = warnings

    return result


def extract_viewpoint_text_ground_truth(
    parsed_viewpoint_text: Dict[str, object],
    target_tag_type: str,
    highlighted_only_rule: bool = True,
) -> str:
    """
    Build OCR ground truth for a target tag type from parse_viewpoint_text output.

    Rules:
    - Within one tag: text items are joined by ", ".
    - Across multiple tags of same type: tag strings are joined by "; ".
    - If highlighted_only_rule=True and at least one highlighted tag exists for this type,
      only highlighted tags are used.
    """
    normalized_target = _normalize_tag_type(target_tag_type)
    if not normalized_target or not isinstance(parsed_viewpoint_text, dict):
        return ""

    tags: List[Dict[str, object]] = []
    if normalized_target == "room tag":
        room_tags = parsed_viewpoint_text.get("room_tags")
        if isinstance(room_tags, list):
            tags = [tag for tag in room_tags if isinstance(tag, dict)]
    elif normalized_target == "layer tag":
        layer_blocks = parsed_viewpoint_text.get("layer_tag_blocks")
        if isinstance(layer_blocks, list):
            tags = [tag for tag in layer_blocks if isinstance(tag, dict)]

    if not tags:
        return ""

    if highlighted_only_rule:
        has_highlighted = any(bool(tag.get("is_highlighted")) for tag in tags)
        if has_highlighted:
            tags = [tag for tag in tags if bool(tag.get("is_highlighted"))]

    tag_ground_truths: List[str] = []
    for tag in tags:
        ground_truth = str(tag.get("ocr_ground_truth") or "").strip()
        if not ground_truth:
            entries = tag.get("entries")
            if isinstance(entries, list):
                _, ground_truth = _build_tag_ocr_text(entries)
        if ground_truth:
            tag_ground_truths.append(ground_truth)

    return "; ".join(tag_ground_truths)


def extract_viewpoint_all_text_ground_truth(
    parsed_viewpoint_text: Dict[str, object],
) -> str:
    """
    Build ordered OCR ground truth from all text-carrying blocks in text_tags.
    """
    if not isinstance(parsed_viewpoint_text, dict):
        return ""

    text_tags = parsed_viewpoint_text.get("text_tags")
    if not isinstance(text_tags, list):
        return ""

    blocks: List[str] = []
    for tag in text_tags:
        if not isinstance(tag, dict):
            continue
        block_gt = str(tag.get("ocr_ground_truth") or "").strip()
        if not block_gt:
            entries = tag.get("entries")
            if isinstance(entries, list):
                _, block_gt = _build_tag_ocr_text(entries)
                block_gt = block_gt.strip()
        if block_gt:
            blocks.append(block_gt)
    return "; ".join(blocks)


def parse_multi_view_dimensions(
    raw_value: Optional[str],
    *,
    tolerant: bool = False,
    warnings: Optional[List[str]] = None,
) -> Dict[str, Dict[str, Optional[Dict[str, object]]]]:
    """
    Parse viewpoint_scenes.multi_view_dimensions into:
      {
        "<object_type>": {
          "<dimension_param>": {"value": int, "unit": str, "source_view": str} | None
        }
      }

    The input may contain multiple `object_type = {...}` entries separated by
    top-level commas.
    """
    if raw_value is None or pd.isna(raw_value) or not str(raw_value).strip():
        return {}

    text = str(raw_value).strip()
    if tolerant:
        _check_unbalanced(text, warnings, "parse_multi_view_dimensions")

    try:
        entries = _split_top_level(
            text,
            tolerant=tolerant,
            warnings=warnings,
            context="parse_multi_view_dimensions",
        )
        result: Dict[str, Dict[str, Optional[Dict[str, object]]]] = {}
        for entry in entries:
            if "=" not in entry:
                continue
            obj_part, payload = entry.split("=", 1)
            object_type = obj_part.strip()
            if not object_type:
                continue
            block = _strip_outer_braces(payload.strip())
            if not block:
                result[object_type] = {}
                continue
            dimensions: Dict[str, Optional[Dict[str, object]]] = {}
            for dim_entry in _split_top_level(
                block,
                tolerant=tolerant,
                warnings=warnings,
                context="parse_multi_view_dimensions_entry",
            ):
                if ":" not in dim_entry:
                    continue
                key, value = dim_entry.split(":", 1)
                dim_key = _strip_quotes(key.strip())
                if not dim_key:
                    continue
                value_str = value.strip()
                if _is_null_like(value_str):
                    dimensions[dim_key] = None
                else:
                    parsed_obj = _parse_dimension_value_object(
                        value_str,
                        tolerant=tolerant,
                        warnings=warnings,
                    )
                    dimensions[dim_key] = parsed_obj
            result[object_type] = dimensions
        return result
    except Exception as exc:  # noqa: BLE001 - safety for parsing pipeline
        message = f"parse_multi_view_dimensions failed: {exc}"
        if warnings is not None:
            warnings.append(message)
        else:
            print(message)
        return {}


def _split_top_level(
    text: str,
    delimiter: str = ",",
    *,
    tolerant: bool = False,
    warnings: Optional[List[str]] = None,
    context: str = "split_top_level",
) -> List[str]:
    parts: List[str] = []
    current: List[str] = []
    brace_depth = 0
    bracket_depth = 0
    in_quote = False
    escape = False
    extra_brace = 0
    extra_bracket = 0

    for ch in text:
        if escape:
            current.append(ch)
            escape = False
            continue

        if ch == "\\" and in_quote:
            current.append(ch)
            escape = True
            continue

        if ch == '"':
            in_quote = not in_quote
            current.append(ch)
            continue

        if not in_quote:
            if ch == "{":
                brace_depth += 1
            elif ch == "}":
                if brace_depth == 0:
                    extra_brace += 1
                else:
                    brace_depth -= 1
            elif ch == "[":
                bracket_depth += 1
            elif ch == "]":
                if bracket_depth == 0:
                    extra_bracket += 1
                else:
                    bracket_depth -= 1

        if ch == delimiter and brace_depth == 0 and bracket_depth == 0 and not in_quote:
            part = "".join(current).strip()
            if part:
                parts.append(part)
            current = []
            continue

        current.append(ch)

    tail = "".join(current).strip()
    if tail:
        parts.append(tail)

    if tolerant and (brace_depth != 0 or bracket_depth != 0 or in_quote or extra_brace or extra_bracket):
        _warn(
            warnings,
            f"{context}: unbalanced structure "
            f"(brace_depth={brace_depth}, bracket_depth={bracket_depth}, "
            f"in_quote={in_quote}, extra_brace={extra_brace}, extra_bracket={extra_bracket})",
        )
        return _split_top_level_relaxed(text, delimiter, warnings=warnings, context=context)
    return parts


def _split_top_level_relaxed(
    text: str,
    delimiter: str,
    *,
    warnings: Optional[List[str]] = None,
    context: str = "split_top_level_relaxed",
) -> List[str]:
    parts: List[str] = []
    current: List[str] = []
    in_quote = False
    escape = False

    for ch in text:
        if escape:
            current.append(ch)
            escape = False
            continue

        if ch == "\\" and in_quote:
            current.append(ch)
            escape = True
            continue

        if ch == '"':
            in_quote = not in_quote
            current.append(ch)
            continue

        if ch == delimiter and not in_quote:
            part = "".join(current).strip()
            if part:
                parts.append(part)
            current = []
            continue

        current.append(ch)

    if in_quote:
        _warn(warnings, f"{context}: unbalanced quotes in relaxed split")

    tail = "".join(current).strip()
    if tail:
        parts.append(tail)
    return parts


def _strip_outer_braces(text: str) -> str:
    trimmed = text.strip()
    if trimmed.startswith("{") and trimmed.endswith("}"):
        return trimmed[1:-1].strip()
    return trimmed


def _strip_quotes(text: str) -> str:
    trimmed = text.strip()
    if trimmed.startswith('"') and trimmed.endswith('"') and len(trimmed) >= 2:
        return trimmed[1:-1]
    return trimmed


def _is_null_like(value: str) -> bool:
    return value.strip().lower() in {"null", "none", "nan", "nil"}


def _parse_dimension_value_object(
    raw_value: str,
    *,
    tolerant: bool = False,
    warnings: Optional[List[str]] = None,
) -> Optional[Dict[str, object]]:
    """
    Parse a { "value": int, "unit": str, "source_view": str } block.
    Returns None if parsing fails.
    """
    payload = raw_value.strip()
    payload = _strip_outer_braces(payload)
    if not payload:
        return None

    parsed: Dict[str, object] = {}
    for entry in _split_top_level(
        payload,
        tolerant=tolerant,
        warnings=warnings,
        context="parse_dimension_value_object",
    ):
        if ":" not in entry:
            continue
        key, value = entry.split(":", 1)
        key_clean = _strip_quotes(key.strip()).lower()
        value_clean = value.strip()
        if key_clean == "value":
            num = _parse_number(_strip_quotes(value_clean))
            if isinstance(num, float) and num.is_integer():
                num = int(num)
            parsed["value"] = num
        elif key_clean == "unit":
            parsed["unit"] = _strip_quotes(value_clean)
        elif key_clean == "source_view":
            parsed["source_view"] = _strip_quotes(value_clean)

    if not parsed:
        if tolerant:
            _warn(warnings, "parse_dimension_value_object: empty object")
        return None
    return parsed


def _parse_viewpoint_tag_header(left_part: str) -> Optional[Dict[str, object]]:
    raw = str(left_part).strip()
    if not raw:
        return None

    lowered = raw.lower()
    is_highlighted = False
    while lowered.startswith("highlighted:"):
        is_highlighted = True
        lowered = lowered[len("highlighted:"):].strip()

    if "room tag" in lowered:
        if "leader/room tag" in lowered:
            source = "label with a leader/room tag"
        else:
            source = "room tag"
        return {
            "tag_type": "room tag",
            "source": source,
            "is_highlighted": is_highlighted,
        }

    if "layer tag" in lowered:
        if "leader/layer tag" in lowered:
            source = "label with a leader/layer tag"
        else:
            source = "layer tag"
        return {
            "tag_type": "layer tag",
            "source": source,
            "is_highlighted": is_highlighted,
        }

    return None


def _normalize_tag_type(tag_type: Optional[str]) -> str:
    value = str(tag_type or "").strip().lower()
    if "room" in value and "tag" in value:
        return "room tag"
    if "layer" in value and "tag" in value:
        return "layer tag"
    return ""


def _split_ocr_text_items(
    value: str,
    *,
    tolerant: bool = False,
    warnings: Optional[List[str]] = None,
    context: str = "ocr_text_items",
) -> List[str]:
    raw = str(value).strip()
    if not raw:
        return []

    content = raw
    if raw.startswith("[") and raw.endswith("]"):
        content = raw[1:-1].strip()
        parts = _split_top_level(
            content,
            tolerant=tolerant,
            warnings=warnings,
            context=f"{context}_bracket",
        )
    else:
        parts = [raw]

    items: List[str] = []
    for part in parts:
        text = _strip_quotes(str(part).strip())
        if text:
            items.append(text)
    return items


def _parse_layer_tag_entry(
    entry: str,
    *,
    tolerant: bool = False,
    warnings: Optional[List[str]] = None,
) -> Optional[Dict[str, object]]:
    raw_entry = str(entry).strip()
    if not raw_entry:
        return None

    if ":" in raw_entry:
        meaning_part, text_part = raw_entry.split(":", 1)
        meaning = meaning_part.strip() or None
        text_items = _split_ocr_text_items(
            text_part,
            tolerant=tolerant,
            warnings=warnings,
            context="layer_tag_entry",
        )
    else:
        meaning = None
        text_items = _split_ocr_text_items(
            raw_entry,
            tolerant=tolerant,
            warnings=warnings,
            context="layer_tag_entry",
        )

    if not text_items:
        return None

    return {
        "meaning": meaning,
        "text_items": text_items,
        "raw_entry": raw_entry,
    }


def _parse_layer_tag_block(
    payload: str,
    *,
    tolerant: bool = False,
    warnings: Optional[List[str]] = None,
) -> Dict[str, object]:
    entries: List[Dict[str, object]] = []
    for entry in _split_top_level(payload, tolerant=tolerant, warnings=warnings, context="layer_tag"):
        parsed_entry = _parse_layer_tag_entry(entry, tolerant=tolerant, warnings=warnings)
        if parsed_entry:
            entries.append(parsed_entry)

    if not entries:
        return {}

    ocr_text_items, ocr_ground_truth = _build_tag_ocr_text(entries)
    return {
        "entries": entries,
        "ocr_text_items": ocr_text_items,
        "ocr_ground_truth": ocr_ground_truth,
    }


def _build_tag_ocr_text(entries: Sequence[Dict[str, object]]) -> Tuple[List[str], str]:
    ocr_text_items: List[str] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        text_items = entry.get("text_items")
        if not isinstance(text_items, list):
            continue
        for text_item in text_items:
            text = str(text_item).strip()
            if text:
                ocr_text_items.append(text)
    return ocr_text_items, ", ".join(ocr_text_items)


def _format_ocr_dimension_value(value: object, unit: str) -> str:
    value_text = str(value).strip()
    unit_text = str(unit).strip()
    if not value_text:
        return ""
    if unit_text:
        return f"{value_text} {unit_text}".strip()
    return value_text


def _dimension_to_text_items(dimension_data: Dict[str, object]) -> List[str]:
    if not isinstance(dimension_data, dict):
        return []

    items: List[str] = []
    unit = str(dimension_data.get("unit") or "").strip()

    if "value" in dimension_data:
        text = _format_ocr_dimension_value(dimension_data.get("value"), unit)
        if text:
            items.append(text)
        return items

    params = dimension_data.get("params")
    if isinstance(params, dict):
        for value in params.values():
            text = _format_ocr_dimension_value(value, unit)
            if text:
                items.append(text)
    return items


def _parse_room_tag_entry(
    entry: str,
    *,
    tolerant: bool = False,
    warnings: Optional[List[str]] = None,
) -> Optional[Dict[str, object]]:
    raw_entry = str(entry).strip()
    if not raw_entry:
        return None

    if ":" in raw_entry:
        meaning_part, text_part = raw_entry.split(":", 1)
        meaning = meaning_part.strip() or None
        text_items = _split_ocr_text_items(
            text_part,
            tolerant=tolerant,
            warnings=warnings,
            context="room_tag_entry",
        )
    else:
        meaning = None
        text_items = _split_ocr_text_items(
            raw_entry,
            tolerant=tolerant,
            warnings=warnings,
            context="room_tag_entry",
        )

    if not text_items:
        return None

    return {
        "meaning": meaning,
        "text_items": text_items,
        "raw_entry": raw_entry,
    }


def _parse_room_tag_block(
    payload: str,
    *,
    tolerant: bool = False,
    warnings: Optional[List[str]] = None,
) -> Dict[str, object]:
    room_name: Optional[str] = None
    area_value: Optional[float] = None
    area_unit = ""
    room_fields: Dict[str, str] = {}
    entries: List[Dict[str, object]] = []

    for entry in _split_top_level(payload, tolerant=tolerant, warnings=warnings, context="room_tag"):
        parsed_entry = _parse_room_tag_entry(entry, tolerant=tolerant, warnings=warnings)
        if not parsed_entry:
            continue
        entries.append(parsed_entry)

        meaning = str(parsed_entry.get("meaning") or "").strip()
        text_items = parsed_entry.get("text_items")
        value_text = ""
        if isinstance(text_items, list):
            value_text = ", ".join(str(v).strip() for v in text_items if str(v).strip())
        if meaning:
            room_fields[meaning] = value_text
        else:
            raw_entry = str(parsed_entry.get("raw_entry") or "").strip()
            if raw_entry:
                room_fields[raw_entry] = value_text or raw_entry

        meaning_norm = meaning.lower()
        if meaning_norm == "room name" and isinstance(text_items, list) and text_items and room_name is None:
            room_name = str(text_items[0]).strip()
        elif meaning_norm == "room area" and isinstance(text_items, list) and text_items and area_value is None:
            area_value, area_unit = _parse_number_with_unit(str(text_items[0]).strip())

    if room_name is None and area_value is None and not entries:
        return {}

    ocr_text_items, ocr_ground_truth = _build_tag_ocr_text(entries)
    return {
        "entries": entries,
        "ocr_text_items": ocr_text_items,
        "ocr_ground_truth": ocr_ground_truth,
        "name": room_name or "",
        "area_value": area_value,
        "area_unit": area_unit,
        "fields": room_fields,
    }


def _parse_dimension_block(
    payload: str,
    *,
    tolerant: bool = False,
    warnings: Optional[List[str]] = None,
) -> Optional[Dict[str, object]]:
    text = payload.strip()
    if not text.lower().startswith("dimension"):
        return None

    remainder = text[len("dimension"):].strip()
    if remainder.startswith("["):
        unit, rest = _parse_unit_bracket(remainder, tolerant=tolerant, warnings=warnings)
        params = _parse_param_block(rest, tolerant=tolerant, warnings=warnings)
        return {"unit": unit, "params": params}

    if remainder.startswith(":"):
        remainder = remainder[1:].strip()
    if not remainder:
        return None

    value, unit = _parse_number_with_unit(remainder)
    return {"value": value, "unit": unit}


def _parse_unit_bracket(
    text: str,
    *,
    tolerant: bool = False,
    warnings: Optional[List[str]] = None,
) -> Tuple[str, str]:
    unit = ""
    remainder = text.strip()
    if not remainder.startswith("["):
        return unit, remainder

    depth = 0
    idx = 0
    for idx, ch in enumerate(remainder):
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                break
    if depth != 0 and tolerant:
        _warn(warnings, "parse_unit_bracket: missing closing ']'")
        bracket_content = remainder[1:]
        remainder_after = ""
    else:
        bracket_content = remainder[1:idx]
        remainder_after = remainder[idx + 1:].lstrip()
    unit = _extract_unit_value(bracket_content)
    if remainder_after.startswith(":"):
        remainder_after = remainder_after[1:].lstrip()
    return unit, remainder_after


def _extract_unit_value(text: str) -> str:
    match = re.search(r'unit"\s*:\s*"([^"]+)"', text, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip()
    match = re.search(r"unit\s*:\s*([^,\]]+)", text, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip().strip('"')
    match = re.search(r"unit\s*=\s*([^,\]]+)", text, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip().strip('"')
    return ""


def _parse_param_block(
    text: str,
    *,
    tolerant: bool = False,
    warnings: Optional[List[str]] = None,
) -> Dict[str, object]:
    params: Dict[str, object] = {}
    trimmed = text.strip()
    if trimmed.startswith("{") and trimmed.endswith("}"):
        trimmed = trimmed[1:-1].strip()
    if not trimmed:
        return params
    for entry in _split_top_level(trimmed, tolerant=tolerant, warnings=warnings, context="param_block"):
        if "=" not in entry:
            continue
        key, value = entry.split("=", 1)
        key_clean = key.strip().strip('"')
        value_clean = value.strip().strip('"')
        params[key_clean] = _parse_number(value_clean)
    return params


def _parse_number_with_unit(text: str) -> Tuple[Optional[float], str]:
    match = re.match(r"^\s*(-?\d+(?:\.\d+)?)\s*(.*)$", text)
    if not match:
        return None, ""
    value = _parse_number(match.group(1))
    unit = match.group(2).strip()
    return value, unit


def _parse_number(value: str) -> object:
    if re.match(r"^-?\d+$", value):
        return int(value)
    if re.match(r"^-?\d+\.\d+$", value):
        return float(value)
    return value


def _extract_relation_blocks(
    text: str,
    *,
    tolerant: bool = False,
    warnings: Optional[List[str]] = None,
) -> List[Tuple[str, str]]:
    pattern = (
        r"(adjacency|connectivity|not_direct_connectivity|not\s+direct\s+connectivity|"
        r"functional\s+grouping|functional_grouping)\s*=\s*\{([^}]*)\}"
    )
    matches = list(re.finditer(pattern, text, flags=re.IGNORECASE | re.DOTALL))
    if matches:
        return [(m.group(1), m.group(2)) for m in matches]

    # Fallback: handle cases with missing closing brace by scanning from key to next key or end.
    blocks: List[Tuple[str, str]] = []
    keys = [
        "adjacency",
        "connectivity",
        "not_direct_connectivity",
        "not direct connectivity",
        "functional grouping",
        "functional_grouping",
    ]
    lowered = text.lower()
    positions: List[Tuple[int, str]] = []
    for key in keys:
        idx = lowered.find(key)
        if idx != -1:
            positions.append((idx, key))
    positions.sort()
    for i, (idx, key) in enumerate(positions):
        start = lowered.find("{", idx)
        if start == -1:
            continue
        end = lowered.find("}", start)
        if end == -1:
            next_idx = positions[i + 1][0] if i + 1 < len(positions) else len(text)
            body = text[start + 1:next_idx]
        else:
            body = text[start + 1:end]
        blocks.append((key, body))
    if tolerant and not blocks:
        _warn(warnings, "parse_scene_relations: no relation blocks detected")
    return blocks


def _normalize_relation_key(key: str) -> Optional[str]:
    key_norm = str(key).strip().lower()
    if "adjacency" in key_norm:
        return "adjacency"
    if "not_direct_connectivity" in key_norm or "not direct connectivity" in key_norm:
        return "not_direct_connectivity"
    if "connectivity" in key_norm or "connected" in key_norm:
        return "connectivity"
    if "functional" in key_norm:
        return "functional_grouping"
    return None


def _parse_relation_mapping(
    body: str,
    *,
    tolerant: bool = False,
    warnings: Optional[List[str]] = None,
) -> Dict[str, List[str]]:
    mapping: Dict[str, List[str]] = {}
    for entry in re.split(r"\s*;\s*", body.strip()):
        entry_clean = entry.strip()
        if not entry_clean:
            continue
        subject, items = _parse_relation_entry(entry_clean, tolerant=tolerant, warnings=warnings)
        if not subject:
            continue
        if subject not in mapping:
            mapping[subject] = []
        for item in items:
            if item and item not in mapping[subject]:
                mapping[subject].append(item)
    return mapping


def _parse_relation_entry(
    entry: str,
    *,
    tolerant: bool = False,
    warnings: Optional[List[str]] = None,
) -> Tuple[str, List[str]]:
    if ":" in entry:
        head, rest = entry.split(":", 1)
        subject = _normalize_scene_label(head)
        items = [
            _normalize_scene_label(p)
            for p in _split_top_level(rest, tolerant=tolerant, warnings=warnings, context="relation_entry")
            if p.strip()
        ]
        return subject, items

    parts = [
        _normalize_scene_label(p)
        for p in _split_top_level(entry, tolerant=tolerant, warnings=warnings, context="relation_entry")
        if p.strip()
    ]
    if not parts:
        return "", []
    return parts[0], parts[1:]


def _infer_scene_entity_type(source_column: Optional[str]) -> str:
    if source_column is None:
        return "unknown"
    col = str(source_column).strip().lower().replace(" ", "_")
    if "space" in col:
        return "spaces"
    if "spatial" in col or "object" in col:
        return "objects"
    return "unknown"


def _warn(warnings: Optional[List[str]], message: str) -> None:
    if warnings is not None:
        warnings.append(message)


def _check_unbalanced(text: str, warnings: Optional[List[str]], context: str) -> None:
    if warnings is None:
        return
    brace_depth = 0
    bracket_depth = 0
    in_quote = False
    escape = False
    extra_brace = 0
    extra_bracket = 0

    for ch in text:
        if escape:
            escape = False
            continue
        if ch == "\\" and in_quote:
            escape = True
            continue
        if ch == '"':
            in_quote = not in_quote
            continue
        if in_quote:
            continue
        if ch == "{":
            brace_depth += 1
        elif ch == "}":
            if brace_depth == 0:
                extra_brace += 1
            else:
                brace_depth -= 1
        elif ch == "[":
            bracket_depth += 1
        elif ch == "]":
            if bracket_depth == 0:
                extra_bracket += 1
            else:
                bracket_depth -= 1

    if brace_depth or bracket_depth or in_quote or extra_brace or extra_bracket:
        _warn(
            warnings,
            f"{context}: unbalanced structure "
            f"(brace_depth={brace_depth}, bracket_depth={bracket_depth}, "
            f"in_quote={in_quote}, extra_brace={extra_brace}, extra_bracket={extra_bracket})",
        )


def _test_parse_multi_view_dimensions_examples() -> None:
    """
    Minimal examples for parse_multi_view_dimensions (not executed by default).
    """
    sample = (
        'double door = { "leaf width": { "value": 875, "unit": "mm", "source_view": "plan" }, '
        '"leaf height": { "value": 2032, "unit": "mm", "source_view": "section 2" }, '
        '"leaf depth": null }, '
        'door to pediatric wts&msrs = { "leaf width": { "value": 900, "unit": "mm", "source_view": "plan" }, '
        '"leaf height": { "value": 2032, "unit": "mm", "source_view": "section 1" }, '
        '"leaf depth": null }'
    )
    parsed = parse_multi_view_dimensions(sample)
    assert parsed["double door"]["leaf width"]["value"] == 875
    assert parsed["double door"]["leaf depth"] is None
    assert parsed["door to pediatric wts&msrs"]["leaf height"]["source_view"] == "section 1"

    simple = 'cabinet = { "width": { "value": 12, "unit": "in", "source_view": "plan" } }'
    parsed_simple = parse_multi_view_dimensions(simple)
    assert parsed_simple["cabinet"]["width"]["unit"] == "in"

    null_case = 'chair = { "height": null }'
    parsed_null = parse_multi_view_dimensions(null_case)
    assert parsed_null["chair"]["height"] is None
