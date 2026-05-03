from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

try:
    from utils.data_parsers import detect_rule_text_placeholder, normalize_id, _resolve_column, parse_requires_rule
except ImportError:
    from backend.utils.data_access_layer.data_parsers import (
        detect_rule_text_placeholder,
        normalize_id,
        _resolve_column,
        parse_requires_rule,
    )
try:
    from utils.file_operations import load_csv
except ImportError:
    from backend.utils.data_access_layer.file_operations import load_csv

TEMPLATE_COLUMNS_TO_KEEP = (
    "template_id",
    "context",
    "layer_id",
    "benchmark_layer",
    "subtask",
    "answer_type",
    "metrics",
)

LEGACY_TEMPLATE_COLUMNS_TO_DROP = {
    "answer_template",
    "question_template",
    "requires_rule",
    "requires_figure",
    "requires_view",
    "rule_filter_tags",
    "view_filter_tags",
    "comments",
}


def _build_template_payload(template_row: pd.Series) -> Dict[str, object]:
    row_dict = template_row.to_dict()
    lower_to_original = {str(k).strip().lower(): k for k in row_dict.keys()}
    payload: Dict[str, object] = {}
    for col in TEMPLATE_COLUMNS_TO_KEEP:
        source_col = lower_to_original.get(col.lower())
        payload[col] = row_dict.get(source_col, "") if source_col is not None else ""
    return payload


def _drop_legacy_template_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    to_drop = [col for col in df.columns if str(col).startswith("Unnamed:")]
    to_drop.extend([col for col in LEGACY_TEMPLATE_COLUMNS_TO_DROP if col in df.columns])
    if to_drop:
        return df.drop(columns=to_drop, errors="ignore")
    return df


def _load_table(conn: sqlite3.Connection, table_name: str) -> pd.DataFrame:
    return pd.read_sql(f"SELECT * FROM {table_name}", conn)


def _detect_placeholder(template_text: str) -> Optional[str]:
    placeholder, _ = detect_rule_text_placeholder(template_text)
    if placeholder:
        return placeholder
    if "Parent Rule Text" in str(template_text):
        return "Parent Rule Text"
    return None


def _sort_rule_id(value: object) -> Tuple[int, object]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return 1, ""
    text = str(value).strip()
    try:
        return 0, float(text)
    except (ValueError, TypeError):
        return 1, text


def _build_parent_groups(
    rules_df: pd.DataFrame,
    parent_rule_id_col: str,
    parent_rule_text_col: str,
    rule_id_col: str,
    rule_text_atomic_col: str,
) -> Dict[str, Dict[str, object]]:
    groups: Dict[str, Dict[str, object]] = {}
    for _, row in rules_df.iterrows():
        raw_parent_id = row.get(parent_rule_id_col)
        if pd.isna(raw_parent_id) or str(raw_parent_id).strip() == "":
            continue
        parent_id = normalize_id(raw_parent_id)[0]
        parent_text = row.get(parent_rule_text_col, "")
        atomic_text = row.get(rule_text_atomic_col, "")
        rule_id = row.get(rule_id_col)

        group = groups.setdefault(parent_id, {"parent_text": "", "rules": []})
        if not group["parent_text"] and pd.notna(parent_text) and str(parent_text).strip():
            group["parent_text"] = str(parent_text).strip()

        if pd.notna(atomic_text) and str(atomic_text).strip():
            group["rules"].append({
                "rule_id": rule_id,
                "rule_text_atomic": str(atomic_text).strip(),
            })

    return groups


def _format_ground_truth(rules: List[Dict[str, object]]) -> str:
    sorted_rules = sorted(rules, key=lambda r: _sort_rule_id(r.get("rule_id")))
    lines = []
    for idx, rule in enumerate(sorted_rules, start=1):
        text = str(rule.get("rule_text_atomic", "")).strip()
        if not text:
            continue
        lines.append(f"{idx}. {text}")
    return "\n".join(lines)


def _merge_with_existing(
    conn: sqlite3.Connection,
    new_df: pd.DataFrame,
    table_name: str,
) -> None:
    new_df = _drop_legacy_template_columns(new_df)
    try:
        existing_df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
    except Exception:
        existing_df = None

    existing_df = _drop_legacy_template_columns(existing_df) if existing_df is not None else None

    if existing_df is None or existing_df.empty:
        new_df.to_sql(table_name, conn, if_exists="replace", index=False)
        return

    all_cols = list(existing_df.columns)
    for col in new_df.columns:
        if col not in all_cols:
            all_cols.append(col)
    existing_df = existing_df.reindex(columns=all_cols)
    new_df = new_df.reindex(columns=all_cols)
    combined = pd.concat([existing_df, new_df], ignore_index=True)
    combined.to_sql(table_name, conn, if_exists="replace", index=False)


def get_next_generated_question_id(conn: sqlite3.Connection) -> int:
    try:
        existing_df = pd.read_sql("SELECT generated_question_id FROM generated_questions", conn)
        if not existing_df.empty:
            existing_df = existing_df.dropna()
            if not existing_df.empty:
                return int(existing_df["generated_question_id"].max()) + 1
    except Exception:
        pass
    return 0


def _filter_templates_by_context(
    templates_df: pd.DataFrame,
    context_value: str,
) -> pd.DataFrame:
    context_col = _resolve_column(templates_df, ["context", "content_type", "task", "type"])
    if context_col:
        return templates_df[
            templates_df[context_col].astype(str).str.strip().str.lower() == context_value.strip().lower()
        ]
    return templates_df


def generate_rule_understanding_questions(db_path: Path) -> None:
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    conn = sqlite3.connect(db_path)
    try:
        templates_df = _load_table(conn, "templates")
        rules_df = _load_table(conn, "rules")

        template_id_col = _resolve_column(templates_df, ["template_id", "template"])
        template_text_col = _resolve_column(templates_df, ["content", "question_template", "template_text", "template"])
        parent_rule_id_col = _resolve_column(rules_df, ["parent_rule_id"])
        parent_rule_text_col = _resolve_column(rules_df, ["parent_rule_text"])
        rule_id_col = _resolve_column(rules_df, ["rule_id"])
        rule_text_atomic_col = _resolve_column(rules_df, ["rule_text_atomic"])

        missing_template_cols = [name for name, col in [
            ("template_id", template_id_col),
            ("content", template_text_col),
        ] if col is None]
        if missing_template_cols:
            raise ValueError(f"Missing required template columns: {missing_template_cols}")

        missing_rules_cols = [name for name, col in [
            ("parent_rule_id", parent_rule_id_col),
            ("parent_rule_text", parent_rule_text_col),
            ("rule_id", rule_id_col),
            ("rule_text_atomic", rule_text_atomic_col),
        ] if col is None]
        if missing_rules_cols:
            raise ValueError(f"Missing required rules columns: {missing_rules_cols}")

        template_candidates = _filter_templates_by_context(templates_df, "rule text atomisation")

        selected_templates: List[Tuple[pd.Series, str]] = []
        for _, row in template_candidates.iterrows():
            template_text = str(row.get(template_text_col, ""))
            placeholder = _detect_placeholder(template_text)
            if not placeholder:
                continue
            selected_templates.append((row, placeholder))

        if not selected_templates:
            print("No templates found with placeholder 'Parent Rule Text' (and context 'rule text atomisation' if present).")
            return

        groups = _build_parent_groups(
            rules_df=rules_df,
            parent_rule_id_col=parent_rule_id_col,
            parent_rule_text_col=parent_rule_text_col,
            rule_id_col=rule_id_col,
            rule_text_atomic_col=rule_text_atomic_col,
        )

        if not groups:
            print("No parent rules found in rules table.")
            return

        start_id = get_next_generated_question_id(conn)

        generated_rows: List[Dict[str, object]] = []
        skipped_parents: List[str] = []
        processed_parents = 0
        question_id = start_id

        for parent_id, group in groups.items():
            parent_text = str(group.get("parent_text", "")).strip()
            if not parent_text:
                skipped_parents.append(parent_id)
                continue
            ground_truth = _format_ground_truth(group.get("rules", []))
            if not ground_truth:
                skipped_parents.append(parent_id)
                continue
            processed_parents += 1

            for template_row, placeholder in selected_templates:
                question_text = str(template_row.get(template_text_col, "")).replace(placeholder, parent_text)
                row = _build_template_payload(template_row)
                row["generated_question_id"] = question_id
                row["parent_rule_id"] = parent_id
                row["rule_id"] = ""
                row["question_text"] = question_text
                row["ground_truth_answer"] = ground_truth
                generated_rows.append(row)
                question_id += 1

        if not generated_rows:
            print("No questions generated.")
            return

        questions_df = pd.DataFrame(generated_rows)
        _merge_with_existing(conn, questions_df, "generated_questions")

        print("\n" + "=" * 60)
        print("Rule Text Atomisation Q&A Generation Summary")
        print("=" * 60)
        print(f"Templates used: {len(selected_templates)}")
        print(f"Parent rules processed: {processed_parents}")
        print(f"Total Q&A pairs created: {len(generated_rows)}")
        if skipped_parents:
            preview = ", ".join(sorted(set(skipped_parents))[:10])
            print(f"Skipped parent rules (missing text or atomic rules): {len(set(skipped_parents))}")
            print(f"Examples: {preview}")
        print("=" * 60)
    finally:
        conn.close()


def _detect_parent_placeholder(template_text: str) -> Optional[str]:
    placeholder, text_type = detect_rule_text_placeholder(template_text)
    if placeholder and text_type == "parent_rule_text":
        return placeholder
    template_str = str(template_text)
    if "{parent_rule_text}" in template_str:
        return "{parent_rule_text}"
    if "[parent_rule_text]" in template_str:
        return "[parent_rule_text]"
    if "Parent Rule Text" in template_str:
        return "Parent Rule Text"
    return None


def _detect_rase_csv(backend_dir: Path) -> Optional[Path]:
    candidates = list(backend_dir.glob("*RASE*.csv"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _normalize_rase(value: object) -> str:
    return str(value).strip().lower()


def _parse_applies_to(value: object) -> List[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    return parse_requires_rule(str(value), tolerant=True)


def _build_ground_truth_rase(
    rows: List[Dict[str, object]],
    applicability_rows: List[Dict[str, object]],
    target_rule_id: str,
) -> str:
    lines: List[str] = []
    applicability_texts: List[str] = []
    for row in _sort_applicability_rows(applicability_rows):
        text = str(row.get("rule_text_atomic", "")).strip()
        if text:
            applicability_texts.append(text)
    if not applicability_texts:
        return ""
    for idx, text in enumerate(applicability_texts, start=1):
        lines.append(f"{idx}. Applicability: {text}")

    rase_priority = [
        ("requirement", "Requirement"),
        ("selection", "Selection"),
        ("exception", "Exception"),
    ]

    counter = len(lines) + 1
    matching_rows = [row for row in rows if row.get("rule_id_norm") == target_rule_id]
    if not matching_rows:
        return ""
    for key, label in rase_priority:
        for row in matching_rows:
            if key in _normalize_rase(row.get("rase", "")):
                text = str(row.get("rule_text_atomic", "")).strip()
                if text:
                    lines.append(f"{counter}. {label}: {text}")
                    counter += 1
                break
    return "\n".join(lines)


def _extract_requirement_text(ground_truth: str) -> Optional[str]:
    for line in str(ground_truth).splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        normalized = stripped
        if "." in stripped:
            head, tail = stripped.split(".", 1)
            if head.strip().isdigit():
                normalized = tail.strip()
        if normalized.lower().startswith("requirement:"):
            return normalized.split(":", 1)[1].strip()
    return None


def generate_rase_atomic_questions(
    conn: sqlite3.Connection,
    rase_csv_path: Optional[Path] = None,
    debug: bool = False,
    start_id: Optional[int] = None,
) -> int:
    backend_dir = Path(__file__).resolve().parents[1]
    if rase_csv_path is None:
        rase_csv_path = _detect_rase_csv(backend_dir)
    if rase_csv_path is None or not rase_csv_path.exists():
        print("RASE CSV not found; skipping RASE-for-atomic generation.")
        return 0

    templates_df = _load_table(conn, "templates")
    try:
        figures_df = _load_table(conn, "rule_figures")
    except Exception:
        figures_df = None

    template_id_col = _resolve_column(templates_df, ["template_id", "template"])
    question_template_col = _resolve_column(templates_df, ["question_template", "content", "template_text", "template"])
    missing_template_cols = [name for name, col in [
        ("template_id", template_id_col),
        ("question_template", question_template_col),
    ] if col is None]
    if missing_template_cols:
        raise ValueError(f"Missing required template columns: {missing_template_cols}")

    templates_filtered = _filter_templates_by_context(templates_df, "rase for atomic")

    selected_templates: List[Tuple[pd.Series, str]] = []
    for _, row in templates_filtered.iterrows():
        template_text = str(row.get(question_template_col, ""))
        placeholder = _detect_parent_placeholder(template_text)
        if not placeholder:
            continue
        selected_templates.append((row, placeholder))

    if not selected_templates:
        print("No templates found for context 'RASE for atomic' with parent_rule_text placeholder.")
        return 0

    rase_df = load_csv(rase_csv_path)
    print(f"RASE CSV: {rase_csv_path}")
    print(f"RASE CSV rows loaded: {len(rase_df)}")
    print("RASE generation uses CSV ONLY; DB used only for rule_figures mapping.")
    parent_rule_id_col = _resolve_column(rase_df, ["parent_rule_id", "parent_rule", "parent rule id"])
    parent_rule_text_col = _resolve_column(rase_df, ["parent_rule_text", "parent rule text", "parent_text"])
    rule_id_col = _resolve_column(rase_df, ["rule_id", "rule id"])
    rule_text_atomic_col = _resolve_column(rase_df, ["rule_text_atomic", "rule text atomic", "atomic_rule_text"])
    rase_col = _resolve_column(rase_df, ["rase", "rase_label", "rase type"])
    applies_to_col = _resolve_column(rase_df, ["applies_to_rule", "applies to rule", "applies_to", "applies to"])
    figure_required_col = _resolve_column(rase_df, ["figure_required", "figure required", "fig_required"])
    figure_id_col = _resolve_column(rase_df, ["figure_id", "figure id"])

    missing_csv_cols = [name for name, col in [
        ("parent_rule_id", parent_rule_id_col),
        ("parent_rule_text", parent_rule_text_col),
        ("rule_id", rule_id_col),
        ("rule_text_atomic", rule_text_atomic_col),
        ("rase", rase_col),
    ] if col is None]
    if missing_csv_cols:
        raise ValueError(f"Missing required RASE CSV columns: {missing_csv_cols}")

    figure_map: Dict[str, Dict[str, str]] = {}
    if figures_df is not None and not figures_df.empty:
        fig_id_col = _resolve_column(figures_df, ["figure_id", "figure id"])
        fig_path_col = _resolve_column(figures_df, ["file_path", "figure_path", "path"])
        if fig_id_col and fig_path_col:
            for _, row in figures_df.iterrows():
                fig_id_raw = row.get(fig_id_col)
                if pd.isna(fig_id_raw) or str(fig_id_raw).strip() == "":
                    continue
                fig_id = normalize_id(fig_id_raw)[0]
                figure_map[fig_id] = {
                    "figure_file_path": str(row.get(fig_path_col, "")).strip(),
                }

    grouped: Dict[str, Dict[str, object]] = {}
    for idx, row in rase_df.iterrows():
        parent_raw = row.get(parent_rule_id_col)
        if pd.isna(parent_raw) or str(parent_raw).strip() == "":
            continue
        parent_id = normalize_id(parent_raw)[0]
        group = grouped.setdefault(parent_id, {
            "parent_rule_text": "",
            "figure_required": "",
            "figure_id": "",
            "rows": [],
        })

        parent_text = row.get(parent_rule_text_col)
        if not group["parent_rule_text"] and pd.notna(parent_text) and str(parent_text).strip():
            group["parent_rule_text"] = str(parent_text).strip()

        if figure_required_col:
            fig_req = row.get(figure_required_col)
            if pd.notna(fig_req) and str(fig_req).strip() and not group["figure_required"]:
                group["figure_required"] = str(fig_req).strip().lower()
        if figure_id_col:
            fig_id = row.get(figure_id_col)
            if pd.notna(fig_id) and str(fig_id).strip() and not group["figure_id"]:
                group["figure_id"] = normalize_id(fig_id)[0]

        rule_id = row.get(rule_id_col)
        rule_text_atomic = row.get(rule_text_atomic_col)
        rase_value = row.get(rase_col)
        applies_to_rule = row.get(applies_to_col) if applies_to_col else None
        rule_id_norm = normalize_id(rule_id)[0] if pd.notna(rule_id) and str(rule_id).strip() else ""

        group["rows"].append({
            "rule_id": rule_id,
            "rule_id_norm": rule_id_norm,
            "rule_text_atomic": str(rule_text_atomic).strip() if pd.notna(rule_text_atomic) else "",
            "rase": str(rase_value).strip() if pd.notna(rase_value) else "",
            "applies_to_rule": applies_to_rule,
            "_order_index": idx,
        })

    if start_id is None:
        start_id = get_next_generated_question_id(conn)

    generated_rows: List[Dict[str, object]] = []
    skipped_parents: Dict[str, str] = {}
    skipped_cases: List[Tuple[str, str]] = []
    processed_parent_ids: set[str] = set()
    question_id = start_id
    debug_emitted = 0
    applicability_rows_found = 0
    expansions_created = 0
    unique_targets_created = 0
    requirement_filled = 0
    requirement_missing = 0

    for parent_id, group in grouped.items():
        parent_text = str(group.get("parent_rule_text", "")).strip()
        if not parent_text:
            skipped_parents[parent_id] = "missing parent_rule_text"
            skipped_cases.append((parent_id, "missing parent_rule_text"))
            continue

        rows = group.get("rows", [])
        applicability_rows = [
            row for row in rows if "applicability" in _normalize_rase(row.get("rase", ""))
        ]
        if not applicability_rows:
            skipped_parents[parent_id] = "missing applicability row"
            skipped_cases.append((parent_id, "missing applicability row"))
            continue
        applicability_rows_found += len(applicability_rows)

        target_to_applicability: Dict[str, List[Dict[str, object]]] = {}
        for applicability_row in applicability_rows:
            applies_to_ids = _parse_applies_to(applicability_row.get("applies_to_rule"))
            if not applies_to_ids:
                skipped_cases.append((parent_id, "missing applies_to_rule"))
                continue
            normalized_ids = [normalize_id(val)[0] for val in applies_to_ids]
            for target_rule_id in normalized_ids:
                expansions_created += 1
                target_to_applicability.setdefault(target_rule_id, []).append(applicability_row)

        if not target_to_applicability:
            skipped_parents[parent_id] = "no target rules derived from applicability rows"
            skipped_cases.append((parent_id, "no target rules derived from applicability rows"))
            continue

        figure_required = str(group.get("figure_required", "")).strip().lower()
        figure_id = str(group.get("figure_id", "")).strip()
        figure_file_path = ""
        if figure_required == "yes" and figure_id:
            figure_file_path = figure_map.get(figure_id, {}).get("figure_file_path", "")
        else:
            figure_required = "no" if figure_required else ""
            figure_id = ""

        any_created_for_parent = False
        for target_rule_id in sorted(target_to_applicability.keys(), key=_sort_rule_id):
            applicability_group = target_to_applicability.get(target_rule_id, [])
            ground_truth = _build_ground_truth_rase(rows, applicability_group, target_rule_id)
            if not ground_truth:
                skipped_cases.append((parent_id, f"missing R/S/E rows for target_rule_id {target_rule_id}"))
                continue

            processed_parent_ids.add(parent_id)
            unique_targets_created += 1
            any_created_for_parent = True

            applicability_ids_used = []
            for row in applicability_group:
                rule_id_val = row.get("rule_id")
                if pd.notna(rule_id_val) and str(rule_id_val).strip():
                    applicability_ids_used.append(str(rule_id_val).strip())
            applicability_ids_used_str = ",".join(dict.fromkeys(applicability_ids_used))

            for template_row, placeholder in selected_templates:
                question_template = str(template_row.get(question_template_col, ""))
                question_text = question_template.replace(placeholder, parent_text)
                requirement_text = None
                if "{requirement_text}" in question_template:
                    requirement_text = _extract_requirement_text(ground_truth)
                    if not requirement_text:
                        requirement_missing += 1
                        print(
                            f"Warning: requirement_text missing for parent_rule_id={parent_id}, "
                            f"target_rule_id={target_rule_id}"
                        )
                        continue
                    question_text = question_text.replace("{requirement_text}", requirement_text)
                    requirement_filled += 1
                row = _build_template_payload(template_row)
                row["generated_question_id"] = question_id
                row["parent_rule_id"] = parent_id
                row["rule_id"] = ""
                row["question_text"] = question_text
                row["ground_truth_answer"] = ground_truth
                row["parent_rule_text_used"] = parent_text
                row["figure_required"] = figure_required
                row["figure_id"] = figure_id
                row["figure_path"] = figure_file_path
                row["target_rule_id_used"] = target_rule_id
                row["applicability_rule_ids_used"] = applicability_ids_used_str
                row["requirement_text_used"] = requirement_text or ""
                generated_rows.append(row)
                question_id += 1

            if debug and debug_emitted < 2:
                print("\n[RASE Debug]")
                print(f"parent_rule_id: {parent_id}")
                print(f"parent_rule_text (CSV): {parent_text}")
                print(f"target_rule_id_used: {target_rule_id}")
                print(f"applicability_rule_ids_used: {applicability_ids_used_str}")
                print(
                    f"figure_required (CSV): {figure_required}, figure_id (CSV): {figure_id}, "
                    f"figure_file_path (DB): {figure_file_path}"
                )
                debug_emitted += 1

        if not any_created_for_parent and parent_id not in skipped_parents:
            skipped_parents[parent_id] = "no Q&A created for parent rule"
            skipped_cases.append((parent_id, "no Q&A created for parent rule"))

    if not generated_rows:
        print("No RASE-for-atomic questions generated.")
        return 0

    questions_df = pd.DataFrame(generated_rows)
    _merge_with_existing(conn, questions_df, "generated_questions")

    print("\n" + "=" * 60)
    print("RASE-for-Atomic Q&A Generation Summary")
    print("=" * 60)
    print(f"Applicability rows found: {applicability_rows_found}")
    print(f"Applicability × target expansions: {expansions_created}")
    print(f"Unique target_rule_ids produced: {unique_targets_created}")
    print(f"Parent rules processed: {len(processed_parent_ids)}")
    print(f"Q&A pairs created: {len(generated_rows)}")
    print(f"Requirement text filled: {requirement_filled}")
    print(f"Missing requirement text: {requirement_missing}")
    if skipped_cases:
        print(f"Skipped cases: {len(skipped_cases)}")
        for parent_id, reason in skipped_cases[:10]:
            print(f"  {parent_id}: {reason}")
        if len(skipped_cases) > 10:
            print(f"  ... and {len(skipped_cases) - 10} more")
    print("=" * 60)

    return len(generated_rows)


def _is_requirement_or_exception(label: object) -> bool:
    text = _normalize_rase(label)
    return "requirement" in text or "exception" in text


def _sort_rule_rows_for_flagging(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    def _key(item: Dict[str, object]) -> Tuple[int, object, int]:
        rule_id = item.get("rule_id")
        order_index = item.get("_order_index", 0)
        if rule_id is None or (isinstance(rule_id, float) and pd.isna(rule_id)) or str(rule_id).strip() == "":
            return 1, order_index, order_index
        return 0, _sort_rule_id(rule_id)[1], order_index

    return sorted(rows, key=_key)


def _sort_applicability_rows(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    def _key(item: Dict[str, object]) -> Tuple[int, object, int]:
        rule_id = item.get("rule_id")
        order_index = item.get("_order_index", 0)
        if rule_id is None or (isinstance(rule_id, float) and pd.isna(rule_id)) or str(rule_id).strip() == "":
            return 1, order_index, order_index
        return 0, _sort_rule_id(rule_id)[1], order_index

    return sorted(rows, key=_key)


def generate_rase_requirement_flagging_questions(
    conn: sqlite3.Connection,
    rase_csv_path: Optional[Path] = None,
    start_id: Optional[int] = None,
) -> int:
    backend_dir = Path(__file__).resolve().parents[1]
    if rase_csv_path is None:
        rase_csv_path = _detect_rase_csv(backend_dir)
    if rase_csv_path is None or not rase_csv_path.exists():
        print("RASE CSV not found; skipping RASErequirement flagging generation.")
        return 0

    templates_df = _load_table(conn, "templates")

    template_id_col = _resolve_column(templates_df, ["template_id", "template"])
    question_template_col = _resolve_column(templates_df, ["question_template", "content", "template_text", "template"])
    missing_template_cols = [name for name, col in [
        ("template_id", template_id_col),
        ("question_template", question_template_col),
    ] if col is None]
    if missing_template_cols:
        raise ValueError(f"Missing required template columns: {missing_template_cols}")

    templates_filtered = _filter_templates_by_context(templates_df, "raserequirement flagging")

    selected_templates: List[Tuple[pd.Series, str]] = []
    for _, row in templates_filtered.iterrows():
        template_text = str(row.get(question_template_col, ""))
        placeholder = _detect_parent_placeholder(template_text)
        if not placeholder:
            continue
        selected_templates.append((row, placeholder))

    if not selected_templates:
        print("No templates found for context 'RASErequirement flagging' with parent_rule_text placeholder.")
        return 0

    rase_df = load_csv(rase_csv_path)
    parent_rule_id_col = _resolve_column(rase_df, ["parent_rule_id", "parent_rule", "parent rule id"])
    parent_rule_text_col = _resolve_column(rase_df, ["parent_rule_text", "parent rule text", "parent_text"])
    rule_id_col = _resolve_column(rase_df, ["rule_id", "rule id"])
    rule_text_atomic_col = _resolve_column(rase_df, ["rule_text_atomic", "rule text atomic", "atomic_rule_text"])
    rase_col = _resolve_column(rase_df, ["rase", "rase_label", "rase type"])

    missing_csv_cols = [name for name, col in [
        ("parent_rule_id", parent_rule_id_col),
        ("parent_rule_text", parent_rule_text_col),
        ("rule_text_atomic", rule_text_atomic_col),
        ("rase", rase_col),
    ] if col is None]
    if missing_csv_cols:
        raise ValueError(f"Missing required RASE CSV columns: {missing_csv_cols}")

    grouped: Dict[str, Dict[str, object]] = {}
    for idx, row in rase_df.iterrows():
        parent_raw = row.get(parent_rule_id_col)
        if pd.isna(parent_raw) or str(parent_raw).strip() == "":
            continue
        parent_id = normalize_id(parent_raw)[0]
        group = grouped.setdefault(parent_id, {
            "parent_rule_text": "",
            "rows": [],
        })

        parent_text = row.get(parent_rule_text_col)
        if not group["parent_rule_text"] and pd.notna(parent_text) and str(parent_text).strip():
            group["parent_rule_text"] = str(parent_text).strip()

        if not _is_requirement_or_exception(row.get(rase_col)):
            continue

        rule_id = row.get(rule_id_col) if rule_id_col else None
        rule_text_atomic = row.get(rule_text_atomic_col)
        rase_value = row.get(rase_col)
        group["rows"].append({
            "rule_id": rule_id,
            "rule_text_atomic": str(rule_text_atomic).strip() if pd.notna(rule_text_atomic) else "",
            "rase": str(rase_value).strip() if pd.notna(rase_value) else "",
            "_order_index": idx,
        })

    if start_id is None:
        start_id = get_next_generated_question_id(conn)

    generated_rows: List[Dict[str, object]] = []
    skipped_parents: Dict[str, str] = {}
    processed_parent_ids: set[str] = set()
    question_id = start_id

    for parent_id, group in grouped.items():
        parent_text = str(group.get("parent_rule_text", "")).strip()
        if not parent_text:
            skipped_parents[parent_id] = "missing parent_rule_text"
            continue

        filtered_rows = _sort_rule_rows_for_flagging(group.get("rows", []))
        filtered_rows = [row for row in filtered_rows if row.get("rule_text_atomic")]
        if not filtered_rows:
            skipped_parents[parent_id] = "no Requirement/Exception rows"
            continue

        ground_truth_lines = []
        for idx, row in enumerate(filtered_rows, start=1):
            label = "Requirement"
            if "exception" in _normalize_rase(row.get("rase", "")):
                label = "Requirement (within Exception)"
            ground_truth_lines.append(f"{idx}. {label}: {row.get('rule_text_atomic')}")
        ground_truth = "\n".join(ground_truth_lines)
        if not ground_truth:
            skipped_parents[parent_id] = "empty ground_truth_answer"
            continue

        processed_parent_ids.add(parent_id)
        for template_row, placeholder in selected_templates:
            question_text = str(template_row.get(question_template_col, "")).replace(placeholder, parent_text)
            row = _build_template_payload(template_row)
            row["generated_question_id"] = question_id
            row["parent_rule_id"] = parent_id
            row["rule_id"] = ""
            row["question_text"] = question_text
            row["ground_truth_answer"] = ground_truth
            row["parent_rule_text_used"] = parent_text
            generated_rows.append(row)
            question_id += 1

    if not generated_rows:
        print("No RASErequirement flagging questions generated.")
        return 0

    questions_df = pd.DataFrame(generated_rows)
    _merge_with_existing(conn, questions_df, "generated_questions")

    print("\n" + "=" * 60)
    print("RASErequirement Flagging Q&A Generation Summary")
    print("=" * 60)
    print(f"Parent rules processed: {len(processed_parent_ids)}")
    print(f"Q&A pairs created: {len(generated_rows)}")
    if skipped_parents:
        print(f"Skipped parent rules: {len(skipped_parents)}")
        for parent_id, reason in list(skipped_parents.items())[:10]:
            print(f"  {parent_id}: {reason}")
        if len(skipped_parents) > 10:
            print(f"  ... and {len(skipped_parents) - 10} more")
    print("=" * 60)

    return len(generated_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Q&A pairs for rule text atomisation and RASE-for-atomic from unified_database.db.",
    )
    script_dir = Path(__file__).resolve().parent
    default_db = script_dir.parent / "unified_database.db"
    parser.add_argument(
        "--db",
        type=Path,
        default=default_db,
        help="Path to unified_database.db (default: backend/unified_database.db)",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Path to RASE CSV (optional; required to run RASE-for-atomic from CLI).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print debug details for the first 1-2 parent rules in RASE generation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_rule_understanding_questions(args.db)
    if args.csv:
        conn = sqlite3.connect(args.db)
        try:
            start_id = get_next_generated_question_id(conn)
            created = generate_rase_atomic_questions(
                conn,
                rase_csv_path=args.csv,
                debug=args.debug,
                start_id=start_id,
            )
            start_id += created
            generate_rase_requirement_flagging_questions(
                conn,
                rase_csv_path=args.csv,
                start_id=start_id,
            )
        finally:
            conn.close()


if __name__ == "__main__":
    main()
