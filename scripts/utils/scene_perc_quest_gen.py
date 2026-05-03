from __future__ import annotations

import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))
"""
Generate question instances for the Scene Perception part of the VQA dataset.

This script reads:
1) Templates table (multiple template_ids)
2) Viewpoint scenes table (scene_id, file_path, object_type, feature_name mapping, template_id list)

It produces question instances with positive and negative samples for templates 1/2.
"""

import argparse
import math
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

try:
    from utils.data_access_layer.database_utils import print_database_summary, save_dataframe_to_db, setup_database
except ImportError:
    from data_access_layer.database_utils import print_database_summary, save_dataframe_to_db, setup_database
try:
    from utils.data_access_layer.data_model import SceneRecord, TemplateConfig
except ImportError:
    from data_access_layer.data_model import SceneRecord, TemplateConfig
try:
    from utils.data_access_layer.data_parsers import (
        _clean_label,
        _clean_material_map,
        _normalize_key,
        _normalize_text,
        _parse_list_field,
        _resolve_column,
        parse_feature_mapping,
        parse_material_field,
        parse_template_id_list,
    )
except ImportError:
    from data_access_layer.data_parsers import (
        _clean_label,
        _clean_material_map,
        _normalize_key,
        _normalize_text,
        _parse_list_field,
        _resolve_column,
        parse_feature_mapping,
        parse_material_field,
        parse_template_id_list,
    )
try:
    from utils.data_access_layer.file_operations import load_csv
except ImportError:
    from data_access_layer.file_operations import load_csv
try:
    from utils.scene_perc_router import (
        FEATURE_PLACEHOLDER,
        OBJECT_PLACEHOLDER,
        ROUTE_FEATURE_BOOL,
        ROUTE_MATERIAL_COLOR_TEXTURE,
        ROUTE_MISSING,
        ROUTE_OBJ_BOOL,
        ROUTE_OCR,
        ROUTE_REGION,
        ROUTE_STATIC,
        ROUTE_UNSUPPORTED,
        generate_dimension_questions,
        generate_location_context_questions,
        generate_material_questions,
        generate_ocr_questions,
        generate_static_questions,
        is_bool_answer_type,
        is_dimension_context,
        is_feature_detection_template,
        is_object_detection_template,
        render_template_text,
        route_scene_perception_template,
    )
except ImportError:
    from scene_perc_router import (
        FEATURE_PLACEHOLDER,
        OBJECT_PLACEHOLDER,
        ROUTE_FEATURE_BOOL,
        ROUTE_MATERIAL_COLOR_TEXTURE,
        ROUTE_MISSING,
        ROUTE_OBJ_BOOL,
        ROUTE_OCR,
        ROUTE_REGION,
        ROUTE_STATIC,
        ROUTE_UNSUPPORTED,
        generate_dimension_questions,
        generate_location_context_questions,
        generate_material_questions,
        generate_ocr_questions,
        generate_static_questions,
        is_bool_answer_type,
        is_dimension_context,
        is_feature_detection_template,
        is_object_detection_template,
        render_template_text,
        route_scene_perception_template,
    )


def _record_skip(
    skip_reasons: Optional[Dict[int, Dict[str, int]]],
    template_id: int,
    reason: str,
) -> None:
    if skip_reasons is None:
        return
    template_map = skip_reasons.setdefault(template_id, {})
    template_map[reason] = template_map.get(reason, 0) + 1


def _has_multi_view_dimensions(scene: SceneRecord) -> bool:
    value = (
        scene.fields.get("multi_view_dimensions")
        or scene.fields.get("multi_view_dimension")
    )
    if value is None:
        return False
    text = str(value).strip()
    if not text:
        return False
    return text.lower() not in {"nan", "none", "null"}


def _merge_unique_scenes(base_scenes: Sequence[SceneRecord], extra_scenes: Sequence[SceneRecord]) -> List[SceneRecord]:
    merged: List[SceneRecord] = list(base_scenes)
    seen = {str(scene.scene_id) for scene in base_scenes}
    for scene in extra_scenes:
        sid = str(scene.scene_id)
        if sid in seen:
            continue
        merged.append(scene)
        seen.add(sid)
    return merged


def load_templates(templates_df: pd.DataFrame, template_ids: Iterable[int]) -> Dict[int, TemplateConfig]:
    template_id_col = _resolve_column(templates_df, ["template_id"])
    layer_id_col = _resolve_column(templates_df, ["layer_id"])
    question_template_col = _resolve_column(templates_df, ["question_template", "template_text", "template"])
    answer_type_col = _resolve_column(templates_df, ["answer_type", "answer"])
    commands_col = _resolve_column(templates_df, ["commands", "comments", "candidates"])
    benchmark_layer_col = _resolve_column(templates_df, ["benchmark_layer"])
    context_col = _resolve_column(templates_df, ["context"])
    subtask_col = _resolve_column(templates_df, ["subtask"])

    missing = [name for name, col in [
        ("template_id", template_id_col),
        ("question_template", question_template_col),
    ] if col is None]
    if missing:
        print(f"[ScenePerception] Missing required template columns: {missing}")
        raise ValueError(f"Missing required template columns: {missing}")

    template_configs: Dict[int, TemplateConfig] = {}
    for _, row in templates_df.iterrows():
        raw_template_id = row[template_id_col]
        if pd.isna(raw_template_id) or str(raw_template_id).strip() == "":
            continue
        try:
            template_id = int(float(raw_template_id))
        except (TypeError, ValueError):
            continue
        if template_id not in template_ids:
            continue

        question_template = str(row[question_template_col])
        layer_id = str(row[layer_id_col]).strip() if layer_id_col else ""
        answer_type = str(row[answer_type_col]) if answer_type_col else "bool"
        commands_raw = row[commands_col] if commands_col else None
        commands_list = _parse_list_field(commands_raw)
        benchmark_layer = str(row[benchmark_layer_col]).strip() if benchmark_layer_col else ""
        context = str(row[context_col]).strip() if context_col else None
        subtask = str(row[subtask_col]).strip() if subtask_col else None

        template_configs[template_id] = TemplateConfig(
            template_id=template_id,
            layer_id=layer_id,
            question_template=question_template,
            answer_type=answer_type,
            commands=commands_list,
            benchmark_layer=benchmark_layer,
            context=context,
            subtask=subtask,
        )

    return template_configs


def load_scenes(viewpoint_df: pd.DataFrame) -> List[SceneRecord]:
    scene_id_col = _resolve_column(viewpoint_df, ["scene_id", "scene", "id"])
    file_path_col = _resolve_column(viewpoint_df, ["file_path", "filepath", "image_path", "path"])
    object_type_col = _resolve_column(viewpoint_df, ["object_type", "objects"])
    feature_name_col = _resolve_column(viewpoint_df, ["feature_name", "features"])
    template_id_col = _resolve_column(viewpoint_df, ["template_id", "template_ids", "templates", "template"])
    material_col = _resolve_column(viewpoint_df, ["material"])

    missing = [name for name, col in [
        ("scene_id", scene_id_col),
        ("file_path", file_path_col),
        ("object_type", object_type_col),
        ("template_id", template_id_col),
    ] if col is None]
    if missing:
        print(f"[ScenePerception] Missing required scene columns: {missing}")
        raise ValueError(f"Missing required scene columns: {missing}")

    records: List[SceneRecord] = []
    for _, row in viewpoint_df.iterrows():
        scene_id = _normalize_text(row[scene_id_col])
        file_path = _normalize_text(row[file_path_col])
        objects_raw = row[object_type_col]
        objects = [_clean_label(o.strip()) for o in str(objects_raw).split(",") if o.strip()] if pd.notna(objects_raw) else []
        features_map = parse_feature_mapping(row[feature_name_col]) if feature_name_col else {}
        material_map = _clean_material_map(parse_material_field(row[material_col])) if material_col else {"texture": {}, "material": {}, "color": {}}
        template_ids = parse_template_id_list(row[template_id_col]) if template_id_col else []
        fields: Dict[str, str] = {}
        for col in viewpoint_df.columns:
            key = _normalize_key(col)
            value = row[col]
            if pd.isna(value):
                continue
            value_str = str(value).strip()
            if value_str and key not in fields:
                fields[key] = value_str

        records.append(SceneRecord(
            scene_id=scene_id,
            file_path=file_path,
            objects=objects,
            features_map=features_map,
            template_ids=template_ids,
            fields=fields,
            material_map=material_map,
        ))

    return records


def build_global_object_candidates(
    scenes: Sequence[SceneRecord],
    template_commands: Sequence[str],
) -> List[str]:
    candidates = set(template_commands)
    for scene in scenes:
        candidates.update(scene.objects)
    return sorted({c.strip() for c in candidates if str(c).strip()})


def build_global_feature_candidates(
    scenes: Sequence[SceneRecord],
    template_commands: Sequence[str],
) -> List[str]:
    candidates = set(template_commands)
    for scene in scenes:
        for feats in scene.features_map.values():
            candidates.update(feats)
    return sorted({c.strip() for c in candidates if str(c).strip()})


def _choose_many(rng: random.Random, items: Sequence[str], count: int) -> List[str]:
    if count <= 0 or not items:
        return []
    if count >= len(items):
        return list(items)
    return rng.sample(list(items), count)


def generate_object_detection_questions(
    template: TemplateConfig,
    scenes: Sequence[SceneRecord],
    global_object_candidates: Sequence[str],
    positives_per_scene: Optional[int],
    negatives_per_scene: int,
    rng: random.Random,
) -> List[Dict]:
    if OBJECT_PLACEHOLDER not in template.question_template:
        raise ValueError("Object detection template missing {object_type} placeholder")

    results: List[Dict] = []
    # negatives_per_scene is ignored; negatives are derived as 40% of positives
    _ = negatives_per_scene

    for scene in scenes:
        if not scene.objects:
            continue

        # Positives: at least all true objects (optionally cap)
        true_objects = list(dict.fromkeys(scene.objects))
        if positives_per_scene is None or positives_per_scene >= len(true_objects):
            selected_pos = true_objects
        else:
            selected_pos = _choose_many(rng, true_objects, positives_per_scene)

        for obj in selected_pos:
            question_text = template.question_template.replace(OBJECT_PLACEHOLDER, obj)
            results.append({
                "template_id": template.template_id,
                "layer_id": template.layer_id,
                "scene_id": scene.scene_id,
                "file_path": scene.file_path,
                "question_text": question_text,
                "object_type_filled": obj,
                "feature_name_filled": None,
                "ground_truth_answer": "",
                "answer_type": template.answer_type or "bool",
            })

        # Negatives: objects not present in scene (40% of positives)
        negatives_pool = [c for c in global_object_candidates if c not in set(true_objects)]
        if negatives_pool and selected_pos:
            target_negatives = max(1, int(math.ceil(len(selected_pos) * 0.4)))
            selected_neg = _choose_many(rng, negatives_pool, min(target_negatives, len(negatives_pool)))
            for obj in selected_neg:
                question_text = template.question_template.replace(OBJECT_PLACEHOLDER, obj)
                results.append({
                    "template_id": template.template_id,
                    "layer_id": template.layer_id,
                    "scene_id": scene.scene_id,
                    "file_path": scene.file_path,
                    "question_text": question_text,
                    "object_type_filled": obj,
                    "feature_name_filled": None,
                    "ground_truth_answer": "",
                    "answer_type": template.answer_type or "bool",
                })

    return results


def generate_template_2_questions(
    template: TemplateConfig,
    scenes: Sequence[SceneRecord],
    global_feature_candidates: Sequence[str],
    negatives_per_scene: int,
    rng: random.Random,
) -> Tuple[List[Dict], int]:
    if OBJECT_PLACEHOLDER not in template.question_template or FEATURE_PLACEHOLDER not in template.question_template:
        raise ValueError("Template 2 question template missing {object_type} and/or {feature_name} placeholders")

    results: List[Dict] = []
    skipped_no_features = 0
    # negatives_per_scene is ignored; negatives are derived as 40% of positives
    _ = negatives_per_scene

    for scene in scenes:
        objects_with_features = {obj: feats for obj, feats in scene.features_map.items() if feats}
        objects_in_scene = list(dict.fromkeys(scene.objects))

        # Positives: all true object-feature pairs
        pos_candidates: List[Tuple[str, str]] = []
        for obj, feats in objects_with_features.items():
            for feat in feats:
                pos_candidates.append((obj, feat))

        if pos_candidates:
            for obj, feat in pos_candidates:
                question_text = template.question_template
                question_text = question_text.replace(OBJECT_PLACEHOLDER, obj)
                question_text = question_text.replace(FEATURE_PLACEHOLDER, feat)
                results.append({
                    "template_id": template.template_id,
                    "layer_id": template.layer_id,
                    "scene_id": scene.scene_id,
                    "file_path": scene.file_path,
                    "question_text": question_text,
                    "object_type_filled": obj,
                    "feature_name_filled": feat,
                    "ground_truth_answer": "",
                    "answer_type": template.answer_type or "bool",
                })
        else:
            skipped_no_features += 1

        # Negatives: choose a true object, but a wrong feature from template comments
        # plus features seen in other scenes
        if not objects_in_scene or not global_feature_candidates:
            continue

        pos_count = len(pos_candidates)
        if pos_count > 0:
            target_negatives = max(1, int(math.ceil(pos_count * 0.4)))
        else:
            target_negatives = max(1, int(math.ceil(len(objects_in_scene) * 0.4)))

        negative_candidates: List[Tuple[str, str]] = []
        for obj in objects_in_scene:
            true_feats = set(scene.features_map.get(obj, []))
            for feat in global_feature_candidates:
                if feat and feat not in true_feats:
                    negative_candidates.append((obj, feat))

        if negative_candidates:
            selected_neg = _choose_many(rng, negative_candidates, min(target_negatives, len(negative_candidates)))
            for obj, feat in selected_neg:
                question_text = template.question_template
                question_text = question_text.replace(OBJECT_PLACEHOLDER, obj)
                question_text = question_text.replace(FEATURE_PLACEHOLDER, feat)
                results.append({
                    "template_id": template.template_id,
                    "layer_id": template.layer_id,
                    "scene_id": scene.scene_id,
                    "file_path": scene.file_path,
                    "question_text": question_text,
                    "object_type_filled": obj,
                    "feature_name_filled": feat,
                    "ground_truth_answer": "",
                    "answer_type": template.answer_type or "bool",
                })

    return results, skipped_no_features


def generate_scene_questions(
    templates_df: pd.DataFrame,
    scenes_df: pd.DataFrame,
    positives_per_scene_t1: Optional[int],
    negatives_per_scene_t1: int,
    negatives_per_scene_t2: int,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    rng = random.Random(seed)

    scenes = load_scenes(scenes_df)
    template_ids_in_scenes = sorted({tid for scene in scenes for tid in scene.template_ids})
    templates = load_templates(templates_df, template_ids=template_ids_in_scenes)

    if not templates:
        raise ValueError("No matching templates found for the template_id values in ViewportScenes.")

    template_id_col = _resolve_column(templates_df, ["template_id"])
    benchmark_layer_col = _resolve_column(templates_df, ["benchmark_layer"])
    all_scene_perc_template_ids: set[int] = set()
    if template_id_col is None:
        print("[ScenePerception] Missing required template column: template_id")
    else:
        if benchmark_layer_col is None:
            print("[ScenePerception] Missing template column: benchmark_layer (logging all template_ids).")
        for _, row in templates_df.iterrows():
            raw_id = row[template_id_col]
            if pd.isna(raw_id) or str(raw_id).strip() == "":
                continue
            try:
                template_id = int(float(raw_id))
            except (TypeError, ValueError):
                continue
            if benchmark_layer_col:
                layer_value = str(row[benchmark_layer_col]).strip().lower()
                if layer_value != "scene_perception":
                    continue
            all_scene_perc_template_ids.add(template_id)

    global_object_candidates = build_global_object_candidates(
        scenes,
        [],
    )
    feature_commands: List[str] = []
    for template in templates.values():
        if template and is_feature_detection_template(template):
            feature_commands.extend(template.commands)
    global_feature_candidates = build_global_feature_candidates(
        scenes,
        feature_commands,
    )

    scenes_by_template: Dict[int, List[SceneRecord]] = {}
    for scene in scenes:
        for tid in scene.template_ids:
            scenes_by_template.setdefault(tid, []).append(scene)

    t1_questions: List[Dict] = []
    object_detection_template_ids: List[int] = []
    for template_id, template in templates.items():
        if not template:
            continue
        if is_object_detection_template(template):
            object_detection_template_ids.append(template_id)
            t1_questions.extend(
                generate_object_detection_questions(
                    template=template,
                    scenes=scenes_by_template.get(template_id, []),
                    global_object_candidates=global_object_candidates,
                    positives_per_scene=positives_per_scene_t1,
                    negatives_per_scene=negatives_per_scene_t1,
                    rng=rng,
                )
            )

    t2_questions: List[Dict] = []
    skipped_no_features = 0
    feature_detection_template_ids: List[int] = []
    for template_id, template in templates.items():
        if not template:
            continue
        if is_feature_detection_template(template):
            feature_detection_template_ids.append(template_id)
            generated, skipped = generate_template_2_questions(
                template=template,
                scenes=scenes_by_template.get(template_id, []),
                global_feature_candidates=global_feature_candidates,
                negatives_per_scene=negatives_per_scene_t2,
                rng=rng,
            )
            t2_questions.extend(generated)
            skipped_no_features += skipped

    material_questions: List[Dict] = []
    location_questions: List[Dict] = []
    ocr_questions: List[Dict] = []
    dimension_questions: List[Dict] = []
    static_questions: List[Dict] = []
    skip_reasons: Dict[int, Dict[str, int]] = {}
    skipped_missing_templates: List[int] = []
    unsupported_templates: List[int] = []
    multi_view_dimension_budget = 12

    handled_template_ids = set(object_detection_template_ids + feature_detection_template_ids)
    for template_id in template_ids_in_scenes:
        if template_id in handled_template_ids:
            continue
        template = templates.get(template_id)
        route, reason = route_scene_perception_template(template)
        if route == ROUTE_MISSING:
            skipped_missing_templates.append(template_id)
            continue
        if route == ROUTE_UNSUPPORTED:
            unsupported_templates.append(template_id)
            continue

        if is_dimension_context(template):
            template_scenes = scenes_by_template.get(template_id, [])
            # Dimension OCR templates can be enriched from multi-view annotations
            # even when template_id is not explicitly listed in the scene row.
            template_scenes = _merge_unique_scenes(
                template_scenes,
                [scene for scene in scenes if _has_multi_view_dimensions(scene)],
            )
            dimension_generated = generate_dimension_questions(
                template,
                template_scenes,
                skip_reasons=skip_reasons,
                rng=rng,
                max_multiview_extra_questions=multi_view_dimension_budget,
            )
            if dimension_generated:
                dimension_questions.extend(dimension_generated)
                used_multi_view = sum(
                    1 for row in dimension_generated
                    if str(row.get("dimension_source") or "").strip() == "multi_view_dimensions"
                )
                multi_view_dimension_budget = max(0, multi_view_dimension_budget - used_multi_view)
            else:
                if multi_view_dimension_budget <= 0:
                    _record_skip(
                        skip_reasons,
                        template_id,
                        "Template skipped: multi_view_dimensions budget exhausted",
                    )
                else:
                    unsupported_templates.append(template_id)
            continue

        if route == ROUTE_MATERIAL_COLOR_TEXTURE:
            material_questions.extend(
                generate_material_questions(template, scenes_by_template.get(template_id, []))
            )
            continue

        if route == ROUTE_OCR:
            ocr_generated = generate_ocr_questions(
                template,
                scenes_by_template.get(template_id, []),
                skip_reasons=skip_reasons,
            )
            if ocr_generated:
                ocr_questions.extend(ocr_generated)
            else:
                unsupported_templates.append(template_id)
            continue

        if route == ROUTE_STATIC and reason == "location_context":
            location_questions.extend(
                generate_location_context_questions(template, scenes_by_template.get(template_id, []))
            )
            continue

        if route == ROUTE_STATIC:
            static_generated = generate_static_questions(template, scenes_by_template.get(template_id, []))
            if static_generated:
                static_questions.extend(static_generated)
            else:
                unsupported_templates.append(template_id)
            continue

        if route in (ROUTE_OBJ_BOOL, ROUTE_FEATURE_BOOL):
            unsupported_templates.append(template_id)

    all_questions = (
        t1_questions
        + t2_questions
        + material_questions
        + location_questions
        + ocr_questions
        + dimension_questions
        + static_questions
    )
    for idx, row in enumerate(all_questions):
        row["generated_question_id"] = idx

    t1_pos = sum(1 for q in t1_questions if q.get("ground_truth_answer") == "yes")
    t1_neg = sum(1 for q in t1_questions if q.get("ground_truth_answer") == "no")
    t2_pos = sum(1 for q in t2_questions if q.get("ground_truth_answer") == "yes")
    t2_neg = sum(1 for q in t2_questions if q.get("ground_truth_answer") == "no")

    print("Scene Perception Question Generation Summary")
    print("-" * 60)
    print(f"Scenes processed: {len(scenes)}")
    print(f"Template 1 questions: {len(t1_questions)} (yes={t1_pos}, no={t1_neg})")
    print(f"Template 2 questions: {len(t2_questions)} (yes={t2_pos}, no={t2_neg})")
    if material_questions:
        print(f"Template 4/5/7 questions: {len(material_questions)}")
    if location_questions:
        print(f"Template 8 questions: {len(location_questions)}")
    if ocr_questions:
        print(f"OCR questions: {len(ocr_questions)}")
    if dimension_questions:
        print(f"Dimension questions: {len(dimension_questions)}")
    if static_questions:
        print(f"Static scene_perception questions: {len(static_questions)}")
    if skipped_no_features:
        print(f"Scenes without feature mapping (template 2 positives skipped): {skipped_no_features}")
    if skipped_missing_templates:
        print(f"Skipped templates not found in templates table: {sorted(set(skipped_missing_templates))}")
    if unsupported_templates:
        print(f"Templates present in scenes but not supported yet: {sorted(set(unsupported_templates))}")
    print("-" * 60)

    for template_id in sorted(all_scene_perc_template_ids - set(template_ids_in_scenes)):
        _record_skip(
            skip_reasons,
            template_id,
            "Template skipped: template_id not included in scene template_id list",
        )

    template_question_counts: Dict[int, int] = {}
    template_scene_ids: Dict[int, set] = {}
    for row in all_questions:
        tid = row.get("template_id")
        scene_id = row.get("scene_id")
        if tid is None:
            continue
        template_question_counts[tid] = template_question_counts.get(tid, 0) + 1
        if scene_id is not None:
            template_scene_ids.setdefault(tid, set()).add(scene_id)

    all_template_ids = sorted(set(all_scene_perc_template_ids) | set(template_question_counts) | set(scenes_by_template))
    if all_template_ids:
        print("Template Generation Details")
        print("-" * 60)
        for tid in all_template_ids:
            scenes_considered = len(scenes_by_template.get(tid, []))
            scenes_with_questions = len(template_scene_ids.get(tid, set()))
            questions_generated = template_question_counts.get(tid, 0)
            skipped_scenes = max(0, scenes_considered - scenes_with_questions)
            print(
                f"Template {tid}: scenes_considered={scenes_considered}, "
                f"scenes_with_questions={scenes_with_questions}, "
                f"questions_generated={questions_generated}, "
                f"scenes_skipped={skipped_scenes}"
            )
            for reason, count in skip_reasons.get(tid, {}).items():
                print(f"  - {reason}: {count}")
        print("-" * 60)

    df = pd.DataFrame(all_questions)
    if df.empty:
        return df

    column_order = [
        "generated_question_id",
        "template_id",
        "layer_id",
        "scene_id",
        "file_path",
        "question_text",
        "object_type_filled",
        "feature_name_filled",
        "ground_truth_answer",
        "answer_type",
    ]
    other_columns = [col for col in df.columns if col not in column_order]
    df = df[column_order + other_columns]
    return df


def write_outputs(
    df: pd.DataFrame,
    output_csv: Path,
    output_sqlite: Optional[Path] = None,
    table_name: str = "generated_scene_questions",
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"CSV written: {output_csv}")

    if output_sqlite:
        conn = setup_database(output_sqlite)
        try:
            save_dataframe_to_db(df, table_name, conn)
            print_database_summary(conn)
        finally:
            conn.close()
        print(f"SQLite written: {output_sqlite} (table: {table_name})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Scene Perception questions (template 1 & 2).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    script_dir = Path(__file__).resolve().parent
    backend_dir = script_dir.parent

    templates_matches = list(backend_dir.glob("*VQA_Rules_Scenes_Templates(templates).csv"))
    if not templates_matches:
        raise FileNotFoundError(
            "No templates CSV matching *VQA_Rules_Scenes_Templates(templates).csv"
        )
    default_templates = max(templates_matches, key=lambda p: p.stat().st_mtime)

    scenes_matches = list(backend_dir.glob("*VQA_Rules_Scenes_Templates(viewpoint_scenes).csv"))
    if not scenes_matches:
        raise FileNotFoundError(
            "No viewpoint scenes CSV matching *VQA_Rules_Scenes_Templates(viewpoint_scenes).csv"
        )
    default_scenes = max(scenes_matches, key=lambda p: p.stat().st_mtime)

    default_output = backend_dir / "scene_perception_questions.csv"

    parser.add_argument(
        "--templates",
        type=Path,
        default=default_templates,
        help="Path to templates CSV (default: latest *VQA_Rules_Scenes_Templates(templates).csv)",
    )
    parser.add_argument(
        "--scenes",
        type=Path,
        default=default_scenes,
        help="Path to viewpoint scenes CSV (default: latest *VQA_Rules_Scenes_Templates(viewpoint_scenes).csv)",
    )
    parser.add_argument("--output-csv", type=Path, default=default_output, help="Output CSV path")
    parser.add_argument("--output-sqlite", type=Path, default=None, help="Optional SQLite output path")
    parser.add_argument("--table-name", type=str, default="generated_scene_questions", help="SQLite table name")
    parser.add_argument("--positives-per-scene-t1", type=int, default=None,
                        help="Optional cap for template 1 positives per scene (default: all true objects)")
    parser.add_argument(
        "--negatives-per-scene-t1",
        type=int,
        default=None,
        help="Ignored for template 1 (negatives are 40%% of positives).",
    )
    parser.add_argument(
        "--positives-per-scene-t2",
        type=int,
        default=None,
        help="Ignored for template 2 (all positives are generated by design)",
    )
    parser.add_argument(
        "--negatives-per-scene-t2",
        type=int,
        default=None,
        help="Ignored for template 2 (negatives are 40%% of positives).",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.templates.exists():
        raise FileNotFoundError(f"Templates not found: {args.templates}")
    if not args.scenes.exists():
        raise FileNotFoundError(f"Scenes not found: {args.scenes}")

    templates_df = load_csv(args.templates)
    scenes_df = load_csv(args.scenes)

    if args.positives_per_scene_t2 is not None:
        print("Note: positives_per_scene_t2 is ignored (all template 2 positives are generated).")
    if args.negatives_per_scene_t1 is not None:
        print("Note: negatives_per_scene_t1 is ignored (template 1 negatives are 40% of positives).")
    if args.negatives_per_scene_t2 is not None:
        print("Note: negatives_per_scene_t2 is ignored (template 2 negatives are 40% of positives).")

    generated_df = generate_scene_questions(
        templates_df=templates_df,
        scenes_df=scenes_df,
        positives_per_scene_t1=args.positives_per_scene_t1,
        negatives_per_scene_t1=args.negatives_per_scene_t1 if args.negatives_per_scene_t1 is not None else 0,
        negatives_per_scene_t2=args.negatives_per_scene_t2 if args.negatives_per_scene_t2 is not None else 0,
        seed=args.seed,
    )

    if generated_df.empty:
        print("No questions generated.")
        return

    write_outputs(
        df=generated_df,
        output_csv=args.output_csv,
        output_sqlite=args.output_sqlite,
        table_name=args.table_name,
    )


if __name__ == "__main__":
    main()
