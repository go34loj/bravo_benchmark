from __future__ import annotations

"""
Generate compliance_reasoning question instances from database.db.

Usage:
  python compliance_question_gen.py --db database.db
"""

import argparse
import re
import sqlite3
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from utils.data_parsers import (
        detect_rule_text_placeholder,
        find_in_dict,
        normalize_id,
        parse_requires_rule,
        parse_template_id_list,
    )
except ImportError:
    from backend.utils.data_access_layer.data_parsers import (
        detect_rule_text_placeholder,
        find_in_dict,
        normalize_id,
        parse_requires_rule,
        parse_template_id_list,
    )


TARGET_TEMPLATE_IDS = {38, 39, 41, 42}
TEMPLATE39_TARGET_POSITIVE_SHARE = 0.55
TEMPLATE39_TARGET_NEGATIVE_SHARE = 0.45


def _normalize_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def _resolve_column(columns: Sequence[str], candidates: Iterable[str]) -> Optional[str]:
    normalized_map = {_normalize_key(col): col for col in columns}
    for cand in candidates:
        key = _normalize_key(cand)
        if key in normalized_map:
            return normalized_map[key]
    for cand in candidates:
        key = _normalize_key(cand)
        for col in columns:
            if key and key in _normalize_key(col):
                return col
    return None


def _list_tables(conn: sqlite3.Connection) -> List[str]:
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    return [row[0] for row in rows]


def _table_columns(conn: sqlite3.Connection, table_name: str) -> List[str]:
    rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    return [row[1] for row in rows]


def _find_table(
    conn: sqlite3.Connection,
    preferred_names: Iterable[str],
    required_groups: Sequence[Tuple[str, Sequence[str], bool]],
    label: str,
) -> Tuple[str, Dict[str, Optional[str]]]:
    tables = _list_tables(conn)
    preferred_norm = {_normalize_key(name) for name in preferred_names}
    candidates: List[Tuple[int, int, str, Dict[str, Optional[str]]]] = []

    for table in tables:
        columns = _table_columns(conn, table)
        resolved: Dict[str, Optional[str]] = {}
        missing_required = False
        score = 0
        for group_name, group_candidates, required in required_groups:
            col = _resolve_column(columns, group_candidates)
            resolved[group_name] = col
            if col:
                score += 1
            elif required:
                missing_required = True
                break
        if missing_required:
            continue
        name_score = 2 if _normalize_key(table) in preferred_norm else 0
        candidates.append((name_score, score, table, resolved))

    if not candidates:
        required_names = [g[0] for g in required_groups if g[2]]
        raise ValueError(f"Could not find {label} table with required columns: {required_names}")

    candidates.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    _, _, table_name, resolved_cols = candidates[0]
    return table_name, resolved_cols


def _coerce_int(value: object) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(float(str(value).strip()))
    except (ValueError, TypeError):
        return None


def _dedupe_preserve(values: Iterable[object]) -> List[object]:
    seen = set()
    result: List[object] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _parse_id_list(raw_value: object) -> List[str]:
    values = [str(val).strip() for val in parse_requires_rule(raw_value)]
    cleaned: List[str] = []
    for value in values:
        if not value:
            continue
        norm, _, orig = normalize_id(value)
        cleaned.append(norm if norm else orig)
    seen = set()
    return [val for val in cleaned if not (val in seen or seen.add(val))]


def _is_yes(value: object) -> bool:
    if value is None:
        return False
    text = str(value).strip().lower()
    return text in {"yes"}


def _resolve_atomic_rule(
    rule_id: str,
    rules: Dict[str, Dict[str, str]],
    figures: Dict[str, str],
) -> Optional[Dict[str, str]]:
    rule_norm, _, rule_orig = normalize_id(rule_id)
    matched_rule_data = find_in_dict(rules, rule_norm, rule_orig)
    if not matched_rule_data:
        return None

    rule_text_atomic = matched_rule_data.get("rule_text_atomic", "")
    if not rule_text_atomic:
        return None

    rule_figure_required = "yes" if _is_yes(matched_rule_data.get("figure_required")) else "no"
    rule_figure_id = matched_rule_data.get("figure_id", "").strip() if rule_figure_required == "yes" else ""
    rule_figure_asset = ""
    if rule_figure_id:
        fig_norm, _, fig_orig = normalize_id(rule_figure_id)
        rule_figure_asset = find_in_dict(figures, fig_norm, fig_orig) or ""

    return {
        "rule_id": rule_orig,
        "rule_text_atomic": rule_text_atomic,
        "classification": matched_rule_data.get("classification", ""),
        "ambiguity": matched_rule_data.get("ambiguity", ""),
        "rule_figure_required": rule_figure_required,
        "rule_figure_id": rule_figure_id,
        "rule_figure_asset": rule_figure_asset,
    }


def _resolve_scene_file_path(
    scene_id: object,
    file_path_value: Optional[object],
    scenes_dir: Path,
    repo_root: Path,
) -> str:
    if file_path_value is not None:
        file_path_text = str(file_path_value).strip()
        if file_path_text:
            return file_path_text

    if not scenes_dir.exists():
        return ""

    prefix = str(scene_id).strip()
    if not prefix:
        return ""

    matches = [p for p in scenes_dir.iterdir() if p.name.startswith(prefix)]
    if not matches:
        return ""

    matches.sort(key=lambda p: p.name)
    try:
        return str(matches[0].relative_to(repo_root))
    except ValueError:
        return str(matches[0])


def _load_templates(
    conn: sqlite3.Connection,
    table_name: str,
    cols: Dict[str, Optional[str]],
    target_ids: Iterable[int],
) -> Dict[int, Dict[str, str]]:
    template_id_col = cols["template_id"]
    question_col = cols["question_template"]
    answer_type_col = cols.get("answer_type")

    templates: Dict[int, Dict[str, str]] = {}
    rows = conn.execute(f"SELECT * FROM {table_name}").fetchall()
    for row in rows:
        template_id_raw = row[template_id_col]
        template_id = _coerce_int(template_id_raw)
        if template_id is None or template_id not in target_ids:
            continue
        question_template = str(row[question_col]) if row[question_col] is not None else ""
        answer_type = str(row[answer_type_col]) if answer_type_col and row[answer_type_col] is not None else "text"
        templates[template_id] = {
            "question_template": question_template,
            "answer_type": answer_type,
        }
    return templates


def _load_rules(
    conn: sqlite3.Connection,
    table_name: str,
    cols: Dict[str, Optional[str]],
) -> Dict[str, Dict[str, str]]:
    rule_id_col = cols["rule_id"]
    rule_text_col = cols["rule_text_atomic"]
    classification_col = cols.get("classification")
    ambiguity_col = cols.get("ambiguity")
    parent_rule_text_col = cols.get("parent_rule_text")
    classification_parent_col = cols.get("classification_parent")
    figure_required_col = cols.get("figure_required")
    figure_id_col = cols.get("figure_id")

    rules: Dict[str, Dict[str, str]] = {}
    rows = conn.execute(f"SELECT * FROM {table_name}").fetchall()
    for row in rows:
        rule_id_raw = row[rule_id_col]
        if rule_id_raw is None:
            continue
        rule_norm, _, rule_orig = normalize_id(rule_id_raw)

        rule_text_atomic = row[rule_text_col]
        if rule_text_atomic is None:
            continue
        rule_text_str = str(rule_text_atomic).strip()
        if not rule_text_str:
            continue

        rule_data = {"rule_text_atomic": rule_text_str}
        if parent_rule_text_col:
            parent_rule_text = row[parent_rule_text_col]
            if parent_rule_text is not None:
                rule_data["parent_rule_text"] = str(parent_rule_text).strip()
        if classification_col:
            classification = row[classification_col]
            if classification is not None:
                rule_data["classification"] = str(classification).strip()
        if ambiguity_col:
            ambiguity = row[ambiguity_col]
            if ambiguity is not None:
                rule_data["ambiguity"] = str(ambiguity).strip()
        if classification_parent_col:
            classification_parent = row[classification_parent_col]
            if classification_parent is not None:
                rule_data["classification_parent"] = str(classification_parent).strip()
        if figure_required_col:
            figure_required = row[figure_required_col]
            if figure_required is not None:
                rule_data["figure_required"] = str(figure_required).strip()
        if figure_id_col:
            figure_id = row[figure_id_col]
            if figure_id is not None:
                rule_data["figure_id"] = str(figure_id).strip()
        for key in {rule_norm, rule_orig}:
            if key and key not in rules:
                rules[key] = rule_data
    return rules


def _load_parent_rules(
    conn: sqlite3.Connection,
    table_name: str,
    cols: Dict[str, Optional[str]],
) -> Dict[str, Dict[str, str]]:
    parent_rule_id_col = cols.get("parent_rule_id")
    parent_rule_text_col = cols.get("parent_rule_text")
    classification_parent_col = cols.get("classification_parent")
    figure_required_col = cols.get("figure_required")
    figure_id_col = cols.get("figure_id")

    if not parent_rule_id_col or not parent_rule_text_col:
        return {}

    parent_rules: Dict[str, Dict[str, str]] = {}
    rows = conn.execute(f"SELECT * FROM {table_name}").fetchall()
    for row in rows:
        parent_ids = _parse_id_list(row[parent_rule_id_col])
        if not parent_ids:
            continue

        parent_rule_text = row[parent_rule_text_col]
        if parent_rule_text is None:
            continue
        parent_rule_text_str = str(parent_rule_text).strip()
        if not parent_rule_text_str:
            continue

        rule_data = {"parent_rule_text": parent_rule_text_str}
        if classification_parent_col:
            classification_parent = row[classification_parent_col]
            if classification_parent is not None:
                rule_data["classification_parent"] = str(classification_parent).strip()
        if figure_required_col:
            figure_required = row[figure_required_col]
            if figure_required is not None:
                rule_data["figure_required"] = str(figure_required).strip()
        if figure_id_col:
            figure_id = row[figure_id_col]
            if figure_id is not None:
                rule_data["figure_id"] = str(figure_id).strip()

        for parent_id in parent_ids:
            if parent_id and parent_id not in parent_rules:
                parent_rules[parent_id] = rule_data

    return parent_rules


def _load_cutouts(
    conn: sqlite3.Connection,
    table_name: str,
    cols: Dict[str, Optional[str]],
) -> Dict[str, List[str]]:
    cutout_id_col = cols["cutout_id"]
    parent_rule_id_col = cols["parent_rule_id"]

    cutouts: Dict[str, List[str]] = {}
    rows = conn.execute(f"SELECT * FROM {table_name}").fetchall()
    for row in rows:
        cutout_id_raw = row[cutout_id_col]
        if cutout_id_raw is None:
            continue
        cutout_norm, _, cutout_orig = normalize_id(cutout_id_raw)

        parent_rule_ids_raw = row[parent_rule_id_col]
        parent_rule_ids = [str(rule_id).strip() for rule_id in parse_requires_rule(parent_rule_ids_raw)]
        parent_rule_ids = [rule_id for rule_id in _dedupe_preserve(parent_rule_ids) if rule_id]
        if not parent_rule_ids:
            continue

        for key in {cutout_norm, cutout_orig}:
            if key and key not in cutouts:
                cutouts[key] = parent_rule_ids
    return cutouts


def _load_rule_figures(
    conn: sqlite3.Connection,
    table_name: str,
    cols: Dict[str, Optional[str]],
) -> Dict[str, str]:
    figure_id_col = cols["figure_id"]
    asset_col = cols.get("figure_asset")

    figures: Dict[str, str] = {}
    rows = conn.execute(f"SELECT * FROM {table_name}").fetchall()
    for row in rows:
        figure_id_raw = row[figure_id_col]
        if figure_id_raw is None:
            continue
        fig_norm, _, fig_orig = normalize_id(figure_id_raw)

        asset_value = ""
        if asset_col:
            asset_raw = row[asset_col]
            if asset_raw is not None:
                asset_value = str(asset_raw).strip()

        for key in {fig_norm, fig_orig}:
            if key and key not in figures:
                figures[key] = asset_value
    return figures


def _sort_scene_key(value: object) -> Tuple[int, str]:
    text = str(value).strip()
    try:
        num = float(text)
        return (0, f"{num:020.6f}")
    except (ValueError, TypeError):
        return (1, text.lower())


def _create_output_table(conn: sqlite3.Connection, table_name: str) -> None:
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            generated_question_id INTEGER PRIMARY KEY,
            scene_id TEXT,
            file_path TEXT,
            template_id INTEGER,
            question_text TEXT,
            rule_id TEXT,
            rule_text_atomic_used TEXT,
            parent_rule_id TEXT,
            parent_rule_text_used TEXT,
            classification TEXT,
            ambiguity TEXT,
            classification_parent TEXT,
            rule_figure_required TEXT,
            rule_figure_id TEXT,
            rule_figure_asset TEXT,
            answer_type TEXT
        )
        """
    )
    existing_cols = {row[1] for row in conn.execute(f"PRAGMA table_info({table_name})").fetchall()}
    for col_name in (
        "classification",
        "ambiguity",
        "parent_rule_id",
        "parent_rule_text_used",
        "classification_parent",
        "rule_figure_required",
        "rule_figure_id",
        "rule_figure_asset",
    ):
        if col_name not in existing_cols:
            conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {col_name} TEXT")


def _write_questions(
    conn: sqlite3.Connection,
    table_name: str,
    rows: Sequence[Dict[str, object]],
) -> None:
    conn.execute(f"DELETE FROM {table_name}")
    insert_sql = f"""
        INSERT INTO {table_name} (
            generated_question_id,
            scene_id,
            file_path,
            template_id,
            question_text,
            rule_id,
            rule_text_atomic_used,
            parent_rule_id,
            parent_rule_text_used,
            classification,
            ambiguity,
            classification_parent,
            rule_figure_required,
            rule_figure_id,
            rule_figure_asset,
            answer_type
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    payload = [
        (
            row["generated_question_id"],
            row["scene_id"],
            row["file_path"],
            row["template_id"],
            row["question_text"],
            row["rule_id"],
            row["rule_text_atomic_used"],
            row.get("parent_rule_id", ""),
            row.get("parent_rule_text_used", ""),
            row.get("classification", ""),
            row.get("ambiguity", ""),
            row.get("classification_parent", ""),
            row.get("rule_figure_required", ""),
            row.get("rule_figure_id", ""),
            row.get("rule_figure_asset", ""),
            row["answer_type"],
        )
        for row in rows
    ]
    conn.executemany(insert_sql, payload)
    conn.commit()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate compliance_reasoning questions from SQLite database.",
    )
    parser.add_argument("--db", type=Path, required=True, help="Path to SQLite database (database.db)")
    parser.add_argument("--table-name", type=str, default="generated_compliance_questions", help="Output table name")
    parser.add_argument("--seed", type=int, default=None, help="Optional seed for reproducible ordering")
    parser.add_argument(
        "--scenes-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "scenes",
        help="Scenes directory used to resolve file_path when missing",
    )
    return parser.parse_args()

def generate_compliance_questions(
    conn: sqlite3.Connection,
    table_name: str = "generated_compliance_questions",
    scenes_dir: Optional[Path] = None,
    seed: Optional[int] = None,
    verbose: bool = True,
) -> int:
    if scenes_dir is None:
        scenes_dir = Path(__file__).resolve().parent.parent / "scenes"

    original_row_factory = conn.row_factory
    conn.row_factory = sqlite3.Row
    try:
        templates_table, template_cols = _find_table(
            conn,
            preferred_names=["templates"],
            required_groups=[
                ("template_id", ["template_id", "templateid", "id"], True),
                ("question_template", ["question_template", "template_text", "template", "question", "text"], True),
                ("answer_type", ["answer_type", "answer"], False),
            ],
            label="templates",
        )

        scenes_table, scenes_cols = _find_table(
            conn,
            preferred_names=["viewpoint_scenes", "viewpointscene", "scene_viewpoints"],
            required_groups=[
                ("scene_id", ["scene_id", "sceneid", "scene", "id"], True),
                ("template_id", ["template_id", "template_ids", "templates", "template"], True),
                ("rule_id", ["rule_id", "ruleid", "atomic_rule_id", "rule"], True),
                ("not_sufficient", ["not_sufficient", "not sufficient", "notsufficient"], False),
                ("cutout_id", ["cutout_id", "cutoutid", "cutout"], False),
                ("file_path", ["file_path", "filepath", "path", "image_path"], False),
            ],
            label="viewpoint_scenes",
        )

        rules_table, rules_cols = _find_table(
            conn,
            preferred_names=["rules", "atomic_rules", "rule"],
            required_groups=[
                ("rule_id", ["rule_id", "ruleid", "id"], True),
                ("rule_text_atomic", ["rule_text_atomic", "atomic_rule_text", "rule_text", "text_atomic"], True),
                ("parent_rule_id", ["parent_rule_id", "parent_ruleid", "parent_id"], False),
                ("parent_rule_text", ["parent_rule_text", "parentruletext"], False),
                ("classification", ["classification", "class"], False),
                ("ambiguity", ["ambiguity", "ambiguous", "ambiguity_label"], False),
                ("classification_parent", ["classification_parent", "parent_classification"], False),
                ("figure_required", ["figure_required", "figure_req", "requires_figure"], False),
                ("figure_id", ["figure_id", "figureid"], False),
            ],
            label="rules",
        )

        templates = _load_templates(conn, templates_table, template_cols, TARGET_TEMPLATE_IDS)
        rules = _load_rules(conn, rules_table, rules_cols)
        parent_rules = _load_parent_rules(conn, rules_table, rules_cols)
        template_placeholders: Dict[int, Tuple[Optional[str], str]] = {}
        parent_template_ids = set()
        for template_id, template in templates.items():
            placeholder, text_type = detect_rule_text_placeholder(template["question_template"])
            template_placeholders[template_id] = (placeholder, text_type)
            if text_type == "parent_rule_text":
                parent_template_ids.add(template_id)

        cutouts: Dict[str, List[str]] = {}
        if parent_template_ids:
            cutout_table, cutout_cols = _find_table(
                conn,
                preferred_names=["cutout", "cutouts"],
                required_groups=[
                    ("cutout_id", ["cutout_id", "cutoutid", "cutout", "id"], True),
                    ("parent_rule_id", ["parent_rule_id", "parent_rule_ids", "parentruleid"], True),
                ],
                label="Cutout",
            )
            cutouts = _load_cutouts(conn, cutout_table, cutout_cols)

        figures: Dict[str, str] = {}
        try:
            figures_table, figures_cols = _find_table(
                conn,
                preferred_names=["rule_figures"],
                required_groups=[
                    ("figure_id", ["figure_id"], True),
                    ("figure_asset", ["file_path"], False),
                ],
                label="rule_figures",
            )
            figures = _load_rule_figures(conn, figures_table, figures_cols)
        except ValueError:
            figures = {}

        scene_rows = conn.execute(f"SELECT * FROM {scenes_table}").fetchall()
        scene_rows = sorted(scene_rows, key=lambda r: _sort_scene_key(r[scenes_cols["scene_id"]]))

        total_scenes = len(scene_rows)
        questions: List[Dict[str, object]] = []
        skipped_missing_rule_id = 0
        skipped_missing_rule_text = 0
        skipped_missing_template = 0
        skipped_no_relevant_templates = 0
        skipped_missing_placeholder = 0
        skipped_missing_cutout_id = 0
        skipped_missing_parent_rule_id = 0
        skipped_missing_parent_rule_text = 0
        templates_missing_in_db: List[int] = []
        parent_questions_required = 0
        parent_questions_generated = 0
        parent_expansion_extra = 0
        figures_attached = 0
        template39_positive_questions = 0
        template39_positive_yes_base = 0
        template39_positive_no_base = 0
        template39_negative_questions = 0
        template39_negative_candidates = 0
        template39_negative_skipped_by_ratio = 0
        template39_negative_cap = 0
        skipped_template39_negative_missing_rule_text = 0
        template39_negative_candidate_rows: List[Dict[str, object]] = []

        for scene_row in scene_rows:
            scene_id = scene_row[scenes_cols["scene_id"]]
            template_ids = parse_template_id_list(scene_row[scenes_cols["template_id"]])
            selected_templates = [tid for tid in template_ids if tid in TARGET_TEMPLATE_IDS]
            if not selected_templates:
                skipped_no_relevant_templates += 1
                continue
            has_template_39 = 39 in selected_templates

            rule_ids_raw = scene_row[scenes_cols["rule_id"]]
            rule_ids = [str(rule_id).strip() for rule_id in parse_requires_rule(rule_ids_raw)]
            rule_ids = [rule_id for rule_id in _dedupe_preserve(rule_ids) if rule_id]
            if not rule_ids and not has_template_39:
                skipped_missing_rule_id += 1
                continue

            scene_rule_keys = set()
            for rule_id in rule_ids:
                rule_norm, _, rule_orig = normalize_id(rule_id)
                key = rule_norm if rule_norm else rule_orig
                if key:
                    scene_rule_keys.add(key)

            resolved_rules: List[Dict[str, str]] = []
            for rule_id in rule_ids:
                resolved_rule = _resolve_atomic_rule(rule_id, rules, figures)
                if not resolved_rule:
                    skipped_missing_rule_text += 1
                    continue
                resolved_rules.append(resolved_rule)

            if not resolved_rules and not has_template_39:
                continue

            template39_negative_rules: List[Dict[str, str]] = []
            template39_not_sufficient_keys = set()
            if has_template_39:
                not_sufficient_col = scenes_cols.get("not_sufficient")
                not_sufficient_rule_ids_raw = scene_row[not_sufficient_col] if not_sufficient_col else None
                not_sufficient_rule_ids = [
                    str(rule_id).strip() for rule_id in parse_requires_rule(not_sufficient_rule_ids_raw)
                ]
                not_sufficient_rule_ids = [rule_id for rule_id in _dedupe_preserve(not_sufficient_rule_ids) if rule_id]

                extra_negative_keys = set()
                for rule_id in not_sufficient_rule_ids:
                    rule_norm, _, rule_orig = normalize_id(rule_id)
                    rule_key = rule_norm if rule_norm else rule_orig
                    if not rule_key:
                        continue
                    template39_not_sufficient_keys.add(rule_key)
                    if rule_key in scene_rule_keys:
                        continue
                    if rule_key in extra_negative_keys:
                        continue
                    extra_negative_keys.add(rule_key)

                    resolved_rule = _resolve_atomic_rule(rule_id, rules, figures)
                    if not resolved_rule:
                        skipped_template39_negative_missing_rule_text += 1
                        continue
                    template39_negative_rules.append(resolved_rule)

            file_path = _resolve_scene_file_path(
                scene_id,
                scene_row[scenes_cols["file_path"]] if scenes_cols.get("file_path") else None,
                scenes_dir,
                Path(__file__).resolve().parent.parent,
            )

            parent_rule_ids_by_cutout: Optional[List[Tuple[str, List[str]]]] = None

            for template_id in sorted(selected_templates):
                template = templates.get(template_id)
                if not template:
                    skipped_missing_template += 1
                    templates_missing_in_db.append(template_id)
                    continue

                template_text = template["question_template"]
                placeholder, text_type = template_placeholders.get(template_id, (None, ""))
                if not placeholder:
                    skipped_missing_placeholder += 1
                    continue

                if text_type == "parent_rule_text":
                    parent_questions_required += 1
                    if parent_rule_ids_by_cutout is None:
                        cutout_col = scenes_cols.get("cutout_id")
                        if not cutout_col:
                            skipped_missing_cutout_id += 1
                            parent_rule_ids_by_cutout = []
                        else:
                            cutout_ids_raw = scene_row[cutout_col]
                            cutout_ids = [str(cid).strip() for cid in parse_requires_rule(cutout_ids_raw)]
                            cutout_ids = [cid for cid in _dedupe_preserve(cutout_ids) if cid]
                            if not cutout_ids:
                                skipped_missing_cutout_id += 1
                                parent_rule_ids_by_cutout = []
                            else:
                                collected: List[Tuple[str, List[str]]] = []
                                for cutout_id in cutout_ids:
                                    cutout_norm, _, cutout_orig = normalize_id(cutout_id)
                                    parent_ids = find_in_dict(cutouts, cutout_norm, cutout_orig) or []
                                    parent_ids = [pid for pid in _dedupe_preserve(parent_ids) if pid]
                                    if parent_ids:
                                        collected.append((cutout_orig, parent_ids))
                                parent_rule_ids_by_cutout = collected
                                if not parent_rule_ids_by_cutout:
                                    skipped_missing_parent_rule_id += 1

                    if not parent_rule_ids_by_cutout:
                        continue

                    for _, parent_rule_ids in parent_rule_ids_by_cutout:
                        extra = max(0, len(parent_rule_ids) - 1)
                        if extra:
                            parent_expansion_extra += extra
                        for parent_rule_id in parent_rule_ids:
                            parent_norm, _, parent_orig = normalize_id(parent_rule_id)
                            parent_rule_data = find_in_dict(parent_rules, parent_norm, parent_orig)
                            if not parent_rule_data:
                                skipped_missing_parent_rule_text += 1
                                continue
                            parent_rule_text = parent_rule_data.get("parent_rule_text", "")
                            if not parent_rule_text:
                                skipped_missing_parent_rule_text += 1
                                continue

                            parent_figure_required = (
                                "yes" if _is_yes(parent_rule_data.get("figure_required")) else "no"
                            )
                            parent_figure_id = (
                                parent_rule_data.get("figure_id", "").strip()
                                if parent_figure_required == "yes"
                                else ""
                            )
                            parent_figure_asset = ""
                            if parent_figure_id:
                                fig_norm, _, fig_orig = normalize_id(parent_figure_id)
                                parent_figure_asset = find_in_dict(figures, fig_norm, fig_orig) or ""

                            question_text = str(template_text).replace(
                                placeholder, str(parent_rule_text).strip()
                            )
                            question_row = {
                                "generated_question_id": len(questions) + 1,
                                "scene_id": str(scene_id),
                                "file_path": file_path,
                                "template_id": template_id,
                                "question_text": question_text,
                                "rule_id": "",
                                "rule_text_atomic_used": "",
                                "parent_rule_id": parent_orig,
                                "parent_rule_text_used": parent_rule_text,
                                "classification": "",
                                "ambiguity": "",
                                "classification_parent": parent_rule_data.get("classification_parent", ""),
                                "rule_figure_required": parent_figure_required,
                                "rule_figure_id": parent_figure_id,
                                "rule_figure_asset": parent_figure_asset,
                                "answer_type": template.get("answer_type", "text"),
                            }
                            questions.append(question_row)
                            parent_questions_generated += 1
                            if (
                                question_row.get("rule_figure_required") == "yes"
                                and (question_row.get("rule_figure_id") or question_row.get("rule_figure_asset"))
                            ):
                                figures_attached += 1
                    continue

                for resolved_rule in resolved_rules:
                    question_text = str(template_text).replace(
                        placeholder, str(resolved_rule["rule_text_atomic"]).strip()
                    )
                    question_row = {
                        "generated_question_id": len(questions) + 1,
                        "scene_id": str(scene_id),
                        "file_path": file_path,
                        "template_id": template_id,
                        "question_text": question_text,
                        "rule_id": resolved_rule["rule_id"],
                        "rule_text_atomic_used": resolved_rule["rule_text_atomic"],
                        "parent_rule_id": "",
                        "parent_rule_text_used": "",
                        "classification": resolved_rule.get("classification", ""),
                        "ambiguity": resolved_rule.get("ambiguity", ""),
                        "classification_parent": "",
                        "rule_figure_required": resolved_rule.get("rule_figure_required", ""),
                        "rule_figure_id": resolved_rule.get("rule_figure_id", ""),
                        "rule_figure_asset": resolved_rule.get("rule_figure_asset", ""),
                        "answer_type": template.get("answer_type", "text"),
                    }
                    questions.append(question_row)
                    if (
                        question_row.get("rule_figure_required") == "yes"
                        and (question_row.get("rule_figure_id") or question_row.get("rule_figure_asset"))
                    ):
                        figures_attached += 1
                    if template_id == 39:
                        template39_positive_questions += 1
                        rule_norm, _, rule_orig = normalize_id(resolved_rule["rule_id"])
                        rule_key = rule_norm if rule_norm else rule_orig
                        if rule_key in template39_not_sufficient_keys:
                            template39_positive_no_base += 1
                        else:
                            template39_positive_yes_base += 1

                if template_id == 39:
                    for resolved_rule in template39_negative_rules:
                        question_text = str(template_text).replace(
                            placeholder, str(resolved_rule["rule_text_atomic"]).strip()
                        )
                        question_row = {
                            "generated_question_id": len(questions) + 1,
                            "scene_id": str(scene_id),
                            "file_path": file_path,
                            "template_id": template_id,
                            "question_text": question_text,
                            "rule_id": resolved_rule["rule_id"],
                            "rule_text_atomic_used": resolved_rule["rule_text_atomic"],
                            "parent_rule_id": "",
                            "parent_rule_text_used": "",
                            "classification": resolved_rule.get("classification", ""),
                            "ambiguity": resolved_rule.get("ambiguity", ""),
                            "classification_parent": "",
                            "rule_figure_required": resolved_rule.get("rule_figure_required", ""),
                            "rule_figure_id": resolved_rule.get("rule_figure_id", ""),
                            "rule_figure_asset": resolved_rule.get("rule_figure_asset", ""),
                            "answer_type": template.get("answer_type", "text"),
                        }
                        template39_negative_candidate_rows.append(question_row)

        template39_negative_candidates = len(template39_negative_candidate_rows)
        if template39_positive_yes_base > 0:
            max_total_no = int(
                template39_positive_yes_base
                * TEMPLATE39_TARGET_NEGATIVE_SHARE
                / TEMPLATE39_TARGET_POSITIVE_SHARE
            )
        else:
            max_total_no = 0
        template39_negative_cap = max(0, max_total_no - template39_positive_no_base)
        template39_negative_questions = min(template39_negative_candidates, template39_negative_cap)
        template39_negative_skipped_by_ratio = template39_negative_candidates - template39_negative_questions

        for question_row in template39_negative_candidate_rows[:template39_negative_questions]:
            question_row["generated_question_id"] = len(questions) + 1
            questions.append(question_row)
            if (
                question_row.get("rule_figure_required") == "yes"
                and (question_row.get("rule_figure_id") or question_row.get("rule_figure_asset"))
            ):
                figures_attached += 1

        _create_output_table(conn, table_name)
        _write_questions(conn, table_name, questions)

        if verbose:
            print("Compliance Reasoning Question Generation Summary")
            print("-" * 60)
            print(f"Scenes processed: {total_scenes}")
            print(f"Questions generated: {len(questions)}")
            print(f"Skipped rows (missing rule_id): {skipped_missing_rule_id}")
            print(f"Skipped rows (missing rule text): {skipped_missing_rule_text}")
            print(f"Skipped rows (no relevant templates): {skipped_no_relevant_templates}")
            print(f"Skipped templates missing in templates table: {skipped_missing_template}")
            print(f"Skipped templates without rule placeholder: {skipped_missing_placeholder}")
            print(f"Skipped rows (missing cutout_id): {skipped_missing_cutout_id}")
            print(f"Skipped rows (missing parent_rule_id): {skipped_missing_parent_rule_id}")
            print(f"Skipped rows (missing parent rule text): {skipped_missing_parent_rule_text}")
            print(f"Questions requiring parent_rule_text: {parent_questions_required}")
            print(f"Questions generated with parent_rule_text: {parent_questions_generated}")
            print(f"Parent rule expansions (extra questions): {parent_expansion_extra}")
            template39_total_questions = template39_positive_questions + template39_negative_questions
            print(f"Template 39 questions from rule_id (positive source): {template39_positive_questions}")
            print(f"Template 39 base answers from rule_id: yes={template39_positive_yes_base}, no={template39_positive_no_base}")
            print(f"Template 39 not-sufficient candidates: {template39_negative_candidates}")
            print(
                "Template 39 not-sufficient cap by ratio "
                f"({int(TEMPLATE39_TARGET_POSITIVE_SHARE * 100)}/"
                f"{int(TEMPLATE39_TARGET_NEGATIVE_SHARE * 100)}): {template39_negative_cap}"
            )
            print(f"Template 39 questions from not sufficient (negative source): {template39_negative_questions}")
            if template39_negative_skipped_by_ratio:
                print(f"Template 39 not-sufficient skipped by ratio cap: {template39_negative_skipped_by_ratio}")
            print(f"Template 39 questions total: {template39_total_questions}")
            if skipped_template39_negative_missing_rule_text:
                print(
                    "Skipped template 39 not-sufficient rules (missing rule text): "
                    f"{skipped_template39_negative_missing_rule_text}"
                )
            print(f"Questions with figures attached: {figures_attached}")
            if templates_missing_in_db:
                print(f"Missing template IDs in templates table: {sorted(set(templates_missing_in_db))}")
            print(f"Output table: {table_name}")
            print("-" * 60)

        _ = seed
        return len(questions)
    finally:
        conn.row_factory = original_row_factory


def main() -> None:
    args = _parse_args()
    if not args.db.exists():
        raise FileNotFoundError(f"Database not found: {args.db}")

    conn = sqlite3.connect(str(args.db))
    try:
        generate_compliance_questions(
            conn=conn,
            table_name=args.table_name,
            scenes_dir=args.scenes_dir,
            seed=args.seed,
            verbose=True,
        )
    finally:
        conn.close()


if __name__ == "__main__":
    main()
