from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path
from typing import Optional

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
DEFAULT_DATASET_DIR = REPO_ROOT / "dataset"
DEFAULT_DB_PATH = REPO_ROOT / "scripts" / "unified_database.db"

DB_TABLES = [
    "generated_compliance_questions",
    "generated_questions",
    "generated_scene_questions",
    "generated_scene_understanding_questions",
]


def _normalize_template_id(value: object) -> Optional[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        num = float(text)
    except (TypeError, ValueError):
        return text
    if num.is_integer():
        return str(int(num))
    return text


def _normalize_yes_no(value: object) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text == "yes":
        return "yes"
    if text == "no":
        return "no"
    return None


def _has_non_empty_value(value: object) -> bool:
    if value is None:
        return False
    text = str(value).strip()
    if not text:
        return False
    return text.lower() not in {"none", "null", "nan"}


def _normalize_layer_id(value: object) -> Optional[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip().lower()
    if not text or text in {"none", "null", "nan"}:
        return None
    text = text.replace(" ", "")
    text = text.replace("-", "_").replace(".", "_")
    return text


def _detect_template_id_col(df: pd.DataFrame) -> Optional[str]:
    exact = {"template_id", "templateid", "template"}
    for col in df.columns:
        if col.lower() in exact:
            return col
    for col in df.columns:
        norm = col.lower().replace(" ", "_").replace("-", "_")
        if norm == "template_id":
            return col
    return None


def _print_template_stats(path: Path) -> None:
    df = pd.read_csv(path, dtype=object)
    template_col = _detect_template_id_col(df)
    if not template_col:
        print(f"{path.name}: template_id column not found, skipping.")
        return

    normalized = df[template_col].map(_normalize_template_id).dropna()
    if normalized.empty:
        print(f"{path.name}: no template_id values found.")
        return

    counts = normalized.value_counts()
    unique_ids = counts.index.tolist()

    print(f"{path.name}")
    print(f"  rows: {len(df)}")
    print(f"  template_id column: {template_col}")
    print(f"  unique template_ids: {len(unique_ids)}")
    print(f"  ids: {', '.join(unique_ids)}")
    print("  counts:")
    for template_id, count in counts.items():
        print(f"    {template_id}: {count}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print template_id usage for CSV files in a directory."
    )
    parser.add_argument(
        "--dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help="Directory containing CSV files.",
    )
    parser.add_argument(
        "--pattern",
        default="*.csv",
        help="Glob pattern for CSV files (default: *.csv).",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB_PATH,
        help="Path to unified_database.db",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = args.dir
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Directory not found: {dataset_dir}")

    files = sorted(dataset_dir.glob(args.pattern))
    if not files:
        print(f"No files found in {dataset_dir} matching {args.pattern}")
        return

    for path in files:
        _print_template_stats(path)

    _print_db_template_stats(args.db)


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,),
    ).fetchone()
    return row is not None


def _resolve_db_column(cursor: sqlite3.Cursor, table: str, candidates: list[str]) -> Optional[str]:
    cursor.execute(f"PRAGMA table_info({table})")
    columns = [row[1] for row in cursor.fetchall()]
    lower_map = {col.lower(): col for col in columns}
    for cand in candidates:
        key = cand.lower()
        if key in lower_map:
            return lower_map[key]
        key_norm = key.replace("-", "_")
        if key_norm in lower_map:
            return lower_map[key_norm]
    return None


def _print_yes_no_breakdown(label: str, answers: list[Optional[str]]) -> None:
    yes_count = sum(1 for answer in answers if answer == "yes")
    no_count = sum(1 for answer in answers if answer == "no")
    known_total = yes_count + no_count
    missing_count = len(answers) - known_total

    print(f"  {label}")
    print(f"    rows: {len(answers)}")
    print(f"    yes: {yes_count}")
    print(f"    no: {no_count}")

    if known_total > 0:
        yes_share = (yes_count / known_total) * 100
        no_share = (no_count / known_total) * 100
        ratio = f"{yes_count}:{no_count}" if no_count > 0 else f"{yes_count}:0"
        print(f"    yes/no ratio: {ratio}")
        print(f"    yes/no share: {yes_share:.1f}% / {no_share:.1f}%")
    else:
        print("    yes/no ratio: n/a (no yes/no answers)")

    if missing_count > 0:
        print(f"    non yes/no or empty: {missing_count}")


def _print_db_template_stats(db_path: Path) -> None:
    if not db_path.exists():
        print(f"DB not found: {db_path}")
        return

    print("\nDatabase template usage")
    print("-----------------------")

    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        for table in DB_TABLES:
            if not _table_exists(conn, table):
                print(f"{table}: table not found, skipping.")
                continue

            template_col = _resolve_db_column(cursor, table, ["template_id", "template"])
            if not template_col:
                print(f"{table}: template_id column not found, skipping.")
                continue

            gt_col = _resolve_db_column(cursor, table, ["ground_truth_answer"])
            figure_col = None
            layer_col = None
            if table.lower() == "generated_questions":
                figure_col = _resolve_db_column(
                    cursor,
                    table,
                    ["figure_path", "figurepath", "rule_figure_path", "rule_figure"],
                )
                layer_col = _resolve_db_column(cursor, table, ["layer_id", "layer", "layerid"])

            select_cols = [template_col]
            if gt_col:
                select_cols.append(gt_col)
            if figure_col:
                select_cols.append(figure_col)
            if layer_col:
                select_cols.append(layer_col)
            cursor.execute(f"SELECT {', '.join(select_cols)} FROM {table}")
            rows = cursor.fetchall()
            if not rows:
                print(f"{table}: no rows.")
                continue

            counts: dict[str, int] = {}
            gt_counts: dict[str, int] = {}
            template_answers: dict[str, list[Optional[str]]] = {}
            figure_stats: list[tuple[Optional[str], bool]] = []
            for row in rows:
                idx = 0
                template_id = row[idx]
                idx += 1
                gt_value = row[idx] if gt_col else None
                if gt_col:
                    idx += 1
                figure_value = row[idx] if figure_col else None
                if figure_col:
                    idx += 1
                layer_value = row[idx] if layer_col else None

                if table.lower() == "generated_questions":
                    figure_stats.append(
                        (_normalize_layer_id(layer_value), _has_non_empty_value(figure_value))
                    )

                tid = _normalize_template_id(template_id)
                if tid is None:
                    continue
                counts[tid] = counts.get(tid, 0) + 1
                if gt_col:
                    template_answers.setdefault(tid, []).append(_normalize_yes_no(gt_value))
                if gt_col is not None and gt_value is not None and str(gt_value).strip() != "":
                    gt_counts[tid] = gt_counts.get(tid, 0) + 1

            print(f"{table}")
            print(f"  rows: {len(rows)}")
            print(f"  template_id column: {template_col}")
            if gt_col:
                print(f"  answer column: {gt_col}")
            else:
                print("  answer column: not found")

            if not counts:
                print("  no template_id values found.")
                continue

            def _sort_key(value: str) -> tuple[int, object]:
                try:
                    return (0, float(value))
                except ValueError:
                    return (1, value)

            for tid in sorted(counts, key=_sort_key):
                total = counts[tid]
                gt = gt_counts.get(tid, 0)
                if gt_col:
                    print(f"    {tid}: {total} rows, answer filled: {gt}")
                else:
                    print(f"    {tid}: {total} rows")

            if table.lower() == "generated_compliance_questions" and gt_col:
                print("  compliance answer breakdown:")
                template_39_answers = template_answers.get("39", [])
                _print_yes_no_breakdown("template_id=39", template_39_answers)

                combined_41_42_answers = (
                    template_answers.get("41", []) + template_answers.get("42", [])
                )
                _print_yes_no_breakdown("template_id in (41, 42) combined", combined_41_42_answers)

            if table.lower() == "generated_questions":
                print("  rule-understanding figure coverage:")
                if not figure_col:
                    print("    figure_path column not found.")
                    continue

                total_with_figure = sum(1 for _, has_figure in figure_stats if has_figure)
                print(f"    with figure_path: {total_with_figure}")
                print(f"    without figure_path: {len(figure_stats) - total_with_figure}")

                layer_23_rows = [has for layer, has in figure_stats if layer == "2_3"]
                layer_21_22_rows = [
                    has for layer, has in figure_stats if layer in {"2_1", "2_2"}
                ]
                layer_23_with_figure = sum(1 for has in layer_23_rows if has)
                layer_21_22_with_figure = sum(1 for has in layer_21_22_rows if has)
                print(
                    "    layer_id == 2_3 (Ambiguity Questions) with figure_path: "
                    f"{layer_23_with_figure}"
                )
                print(
                    "    layer_id in (2_1, 2_2) (Rule Classification Questions) with figure_path: "
                    f"{layer_21_22_with_figure}"
                )
    finally:
        conn.close()


if __name__ == "__main__":
    main()
