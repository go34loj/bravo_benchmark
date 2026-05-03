from __future__ import annotations

import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))
"""
Generate ground-truth boolean answers for generated_compliance_questions.

Usage:
  python compliance_bool_gt_gen.py --db unified_database.db
"""

import argparse
import re
import sqlite3
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from utils.data_access_layer.data_parsers import normalize_id, parse_requires_rule
except ImportError:
    from data_access_layer.data_parsers import normalize_id, parse_requires_rule


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


def _ensure_column(conn: sqlite3.Connection, table_name: str, column_name: str) -> None:
    existing_cols = {row[1] for row in conn.execute(f"PRAGMA table_info({table_name})").fetchall()}
    if column_name not in existing_cols:
        conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} TEXT")


def _coerce_int(value: object) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(float(str(value).strip()))
    except (ValueError, TypeError):
        return None


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


def _build_not_compliant_map(
    conn: sqlite3.Connection,
    table_name: str,
    cols: Dict[str, Optional[str]],
) -> Dict[str, List[str]]:
    scene_col = cols["scene_id"]
    not_compliant_col = cols.get("not_compliant")
    if not not_compliant_col:
        return {}

    mapping: Dict[str, List[str]] = {}
    rows = conn.execute(f"SELECT * FROM {table_name}").fetchall()
    for row in rows:
        scene_raw = row[scene_col]
        if scene_raw is None:
            continue
        scene_norm, _, scene_orig = normalize_id(scene_raw)
        not_compliant_raw = row[not_compliant_col]
        ids = _parse_id_list(not_compliant_raw)
        for key in {scene_norm, scene_orig}:
            if key and key not in mapping:
                mapping[key] = ids
    return mapping


def _build_not_sufficient_map(
    conn: sqlite3.Connection,
    table_name: str,
    cols: Dict[str, Optional[str]],
) -> Dict[str, List[str]]:
    scene_col = cols["scene_id"]
    not_sufficient_col = cols.get("not_sufficient")
    if not not_sufficient_col:
        return {}

    mapping: Dict[str, List[str]] = {}
    rows = conn.execute(f"SELECT * FROM {table_name}").fetchall()
    for row in rows:
        scene_raw = row[scene_col]
        if scene_raw is None:
            continue
        scene_norm, _, scene_orig = normalize_id(scene_raw)
        not_sufficient_raw = row[not_sufficient_col]
        ids = _parse_id_list(not_sufficient_raw)
        for key in {scene_norm, scene_orig}:
            if key and key not in mapping:
                mapping[key] = ids
    return mapping


def _build_parent_to_atomic_map(
    conn: sqlite3.Connection,
    table_name: str,
    cols: Dict[str, Optional[str]],
) -> Dict[str, List[str]]:
    rule_id_col = cols["rule_id"]
    parent_rule_id_col = cols["parent_rule_id"]

    mapping: Dict[str, List[str]] = {}
    rows = conn.execute(f"SELECT * FROM {table_name}").fetchall()
    for row in rows:
        atomic_raw = row[rule_id_col]
        parent_raw = row[parent_rule_id_col]
        if atomic_raw is None or parent_raw is None:
            continue

        atomic_norm, _, atomic_orig = normalize_id(atomic_raw)
        atomic_id = atomic_norm if atomic_norm else atomic_orig
        if not atomic_id:
            continue

        parent_ids = _parse_id_list(parent_raw)
        if not parent_ids:
            continue

        for parent_id in parent_ids:
            mapping.setdefault(parent_id, [])
            if atomic_id not in mapping[parent_id]:
                mapping[parent_id].append(atomic_id)
    return mapping


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate ground_truth_answer for generated_compliance_questions.",
    )
    default_db = Path(__file__).resolve().parent.parent / "unified_database.db"
    parser.add_argument(
        "--db",
        type=Path,
        default=default_db,
        help=f"Path to SQLite database (default: {default_db})",
    )
    parser.add_argument(
        "--table-name",
        type=str,
        default="generated_compliance_questions",
        help="Generated questions table name",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not args.db.exists():
        raise FileNotFoundError(f"Database not found: {args.db}")

    conn = sqlite3.connect(str(args.db))
    conn.row_factory = sqlite3.Row
    try:
        questions_table, questions_cols = _find_table(
            conn,
            preferred_names=[args.table_name],
            required_groups=[
                ("scene_id", ["scene_id", "scene", "sceneid"], True),
                ("rule_id", ["rule_id", "ruleid"], True),
                ("template_id", ["template_id", "templateid", "template"], False),
                ("parent_rule_id", ["parent_rule_id", "parent_rule_id_used", "parent_rule"], False),
                ("parent_rule_text", ["parent_rule_text_used", "parent_rule_text"], False),
                ("question_id", ["generated_question_id", "id", "question_id"], False),
            ],
            label="generated_compliance_questions",
        )

        scenes_table, scenes_cols = _find_table(
            conn,
            preferred_names=["viewpoint_scenes", "viewpointscene", "scene_viewpoints"],
            required_groups=[
                ("scene_id", ["scene_id", "scene", "sceneid"], True),
                ("not_compliant", ["not_compliant", "non_compliant", "notcompliant"], True),
                ("not_sufficient", ["not_sufficient", "not sufficient", "notsufficient"], False),
            ],
            label="viewpoint_scenes",
        )

        rules_table, rules_cols = _find_table(
            conn,
            preferred_names=["rules"],
            required_groups=[
                ("rule_id", ["rule_id"], True),
                ("parent_rule_id", ["parent_rule_id"], True),
            ],
            label="rules",
        )

        _ensure_column(conn, questions_table, "ground_truth_answer")

        not_compliant_map = _build_not_compliant_map(conn, scenes_table, scenes_cols)
        not_sufficient_map = _build_not_sufficient_map(conn, scenes_table, scenes_cols)
        parent_to_atomic = _build_parent_to_atomic_map(conn, rules_table, rules_cols)

        if questions_cols.get("question_id"):
            select_sql = f"SELECT * FROM {questions_table}"
            id_col = questions_cols["question_id"]
        else:
            select_sql = f"SELECT rowid as _rowid_, * FROM {questions_table}"
            id_col = "_rowid_"

        rows = conn.execute(select_sql).fetchall()

        updates: List[Tuple[str, int]] = []
        processed = 0
        yes_count = 0
        no_count = 0
        skipped_missing_rule = 0
        skipped_missing_scene = 0
        skipped_other = 0

        parent_processed = 0
        parent_yes = 0
        parent_no = 0
        parent_skipped_missing_id = 0
        parent_skipped_missing_scene = 0
        parent_skipped_missing_mapping = 0
        parent_skipped_other = 0
        template39_yes = 0
        template39_no = 0

        for row in rows:
            processed += 1
            scene_raw = row[questions_cols["scene_id"]]
            if scene_raw is None:
                skipped_missing_scene += 1
                continue
            scene_norm, _, scene_orig = normalize_id(scene_raw)
            template_id = _coerce_int(row[questions_cols["template_id"]]) if questions_cols.get("template_id") else None
            # Template 39 asks about sufficiency, so its GT uses viewpoint_scenes.not_sufficient.
            if template_id == 39:
                scene_rule_ids = not_sufficient_map.get(scene_norm) or not_sufficient_map.get(scene_orig)
                if scene_rule_ids is None and (scene_norm in not_compliant_map or scene_orig in not_compliant_map):
                    scene_rule_ids = []
            else:
                scene_rule_ids = not_compliant_map.get(scene_norm) or not_compliant_map.get(scene_orig)
            if scene_rule_ids is None:
                skipped_missing_scene += 1
                continue

            parent_rule_present = False
            parent_rule_id_col = questions_cols.get("parent_rule_id")
            parent_rule_text_col = questions_cols.get("parent_rule_text")
            parent_rule_id_val = ""
            if parent_rule_id_col and row[parent_rule_id_col]:
                parent_rule_id_val = str(row[parent_rule_id_col]).strip()
                parent_rule_present = bool(parent_rule_id_val)
            if not parent_rule_present and parent_rule_text_col:
                parent_rule_present = bool(str(row[parent_rule_text_col]).strip()) if row[parent_rule_text_col] else False

            if parent_rule_present:
                parent_processed += 1
                if not parent_rule_id_val:
                    parent_skipped_missing_id += 1
                    continue
                parent_norm, _, parent_orig = normalize_id(parent_rule_id_val)
                parent_key = parent_norm if parent_norm else parent_orig
                if not parent_key:
                    parent_skipped_missing_id += 1
                    continue

                atomic_ids = parent_to_atomic.get(parent_key)
                if not atomic_ids:
                    parent_skipped_missing_mapping += 1
                    continue

                if any(aid in scene_rule_ids for aid in atomic_ids):
                    answer = "no"
                    parent_no += 1
                else:
                    answer = "yes"
                    parent_yes += 1

                updates.append((answer, row[id_col]))
                if template_id == 39:
                    if answer == "yes":
                        template39_yes += 1
                    else:
                        template39_no += 1
                continue

            rule_raw = row[questions_cols["rule_id"]]
            if rule_raw is None or not str(rule_raw).strip():
                skipped_missing_rule += 1
                continue

            rule_norm, _, rule_orig = normalize_id(rule_raw)
            if rule_norm in scene_rule_ids or rule_orig in scene_rule_ids:
                answer = "no"
                no_count += 1
            else:
                answer = "yes"
                yes_count += 1

            updates.append((answer, row[id_col]))
            if template_id == 39:
                if answer == "yes":
                    template39_yes += 1
                else:
                    template39_no += 1

        if updates:
            conn.executemany(
                f"UPDATE {questions_table} SET ground_truth_answer = ? WHERE {id_col} = ?",
                updates,
            )
            conn.commit()

        skipped_other = processed - (yes_count + no_count + skipped_missing_rule + skipped_missing_scene + parent_processed)
        parent_skipped_other = parent_processed - (
            parent_yes + parent_no + parent_skipped_missing_id + parent_skipped_missing_scene + parent_skipped_missing_mapping
        )

        print("Compliance Ground-Truth Generation Summary")
        print("-" * 60)
        print(f"Rows processed: {processed}")
        print(f"Answers set: yes={yes_count}, no={no_count}")
        print(f"Skipped (missing rule_id): {skipped_missing_rule}")
        print(f"Skipped (missing scene mapping): {skipped_missing_scene}")
        if skipped_other:
            print(f"Skipped (other): {skipped_other}")
        print("-" * 60)
        print("Parent-Rule Summary")
        print("-" * 60)
        print(f"Parent rows processed: {parent_processed}")
        print(f"Parent answers set: yes={parent_yes}, no={parent_no}")
        print(f"Skipped parent (missing parent_rule_id): {parent_skipped_missing_id}")
        print(f"Skipped parent (missing scene mapping): {parent_skipped_missing_scene}")
        print(f"Skipped parent (no atomic mapping): {parent_skipped_missing_mapping}")
        if parent_skipped_other:
            print(f"Skipped parent (other): {parent_skipped_other}")
        print("-" * 60)
        template39_total = template39_yes + template39_no
        print("Template 39 Summary")
        print("-" * 60)
        print(f"Template 39 answers set: yes={template39_yes}, no={template39_no}")
        if template39_total > 0:
            yes_share = (template39_yes / template39_total) * 100
            no_share = (template39_no / template39_total) * 100
            print(f"Template 39 yes/no share: {yes_share:.1f}% / {no_share:.1f}%")
        else:
            print("Template 39 yes/no share: n/a")
        print("-" * 60)
    finally:
        conn.close()


if __name__ == "__main__":
    main()



