from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Dict, Optional, List

import pandas as pd

try:
    from utils.data_parsers import detect_rule_text_placeholder, normalize_id, parse_requires_rule, find_in_dict, _resolve_column
except ImportError:
    from backend.utils.data_access_layer.data_parsers import detect_rule_text_placeholder, normalize_id, parse_requires_rule, find_in_dict, _resolve_column
try:
    from utils.rule_under_classific_QA import generate_rase_atomic_questions
except ImportError:
    from backend.utils.rule_under_classific_QA import generate_rase_atomic_questions
try:
    from utils.rule_under_classific_QA import generate_rase_requirement_flagging_questions
except ImportError:
    from backend.utils.rule_under_classific_QA import generate_rase_requirement_flagging_questions
try:
    from utils.rule_under_classific_QA import get_next_generated_question_id
except ImportError:
    from backend.utils.rule_under_classific_QA import get_next_generated_question_id
try:
    from utils.rule_under_classific_QA import generate_rule_understanding_questions as generate_atomisation_questions
except ImportError:
    from backend.utils.rule_under_classific_QA import generate_rule_understanding_questions as generate_atomisation_questions

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


def _build_template_payload(template_row: pd.Series) -> Dict:
    row_dict = template_row.to_dict()
    lower_to_original = {str(k).strip().lower(): k for k in row_dict.keys()}
    payload: Dict = {}
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


def generate_question_for_rule(
    template_row: pd.Series,
    question_template: str,
    rule_data: Dict,
    placeholder: str,
    text_type: str,
    rule_id: Optional[str] = None,
    parent_rule_id: Optional[str] = None,
    question_id: int = 0,
    figures_dict: Optional[Dict] = None,
) -> Optional[Dict]:
    """Build one question row. Returns None if rule text is empty."""
    rule_text = rule_data.get(text_type, "")
    if not rule_text or not str(rule_text).strip():
        return None

    question_text = str(question_template).replace(placeholder, str(rule_text).strip())
    row = _build_template_payload(template_row)
    row["generated_question_id"] = question_id
    row["rule_id"] = rule_id if rule_id else ""
    row["parent_rule_id"] = parent_rule_id if parent_rule_id else ""
    row["question_text"] = question_text

    figure_id = rule_data.get("figure_id")
    if figure_id and figures_dict and figure_id in figures_dict:
        figure_info = figures_dict[figure_id]
        row["figure_id"] = figure_id
        row["figure_path"] = figure_info.get("file_path", "")
        row["figure_caption"] = figure_info.get("caption", "")

    return row


def generate_questions(
    templates_df: pd.DataFrame,
    rules_df: pd.DataFrame,
    conn: sqlite3.Connection,
    figures_df: Optional[pd.DataFrame] = None,
) -> None:
    """Generate questions by replacing placeholders in templates with rule text."""
    print("\n" + "=" * 60)
    print("Generating Question Instances")
    print("=" * 60)


def generate_rule_questions(
    templates_df: pd.DataFrame,
    rules_df: pd.DataFrame,
    conn: sqlite3.Connection,
    figures_df: Optional[pd.DataFrame] = None,
) -> None:
    """Backward-compatible wrapper for generate_questions."""
    generate_questions(
        templates_df=templates_df,
        rules_df=rules_df,
        conn=conn,
        figures_df=figures_df,
    )

    # Normalize column names (case-insensitive)
    templates_cols_lower = {col.lower(): col for col in templates_df.columns}
    rules_cols_lower = {col.lower(): col for col in rules_df.columns}
    template_id_col = templates_cols_lower.get("template_id", "template_id")
    question_template_col = templates_cols_lower.get("question_template", "question_template")
    requires_rule_col = templates_cols_lower.get("requires_rule", "requires_rule")
    context_col = _resolve_column(
        templates_df,
        ["context"],
    )

    rule_id_col = rules_cols_lower.get("rule_id", "rule_id")
    rule_text_atomic_col = rules_cols_lower.get("rule_text_atomic", "rule_text_atomic")
    parent_rule_text_col = rules_cols_lower.get("parent_rule_text", "parent_rule_text")
    parent_rule_id_col = rules_cols_lower.get("parent_rule_id", "parent_rule_id")
    figure_required_col = rules_cols_lower.get("figure_required", "figure_required")
    figure_id_col = rules_cols_lower.get("figure_id", "figure_id")

    has_figure_required = figure_required_col in rules_df.columns
    has_figure_id = figure_id_col in rules_df.columns

    # Validate required columns exist
    required_template_cols = [template_id_col, question_template_col, requires_rule_col]
    required_rules_cols = [rule_id_col, rule_text_atomic_col]

    # Check which columns are available
    has_parent_rule_text = parent_rule_text_col in rules_df.columns
    has_parent_rule_id = parent_rule_id_col in rules_df.columns

    missing_template_cols = [col for col in required_template_cols if col not in templates_df.columns]
    missing_rules_cols = [col for col in required_rules_cols if col not in rules_df.columns]

    if missing_template_cols:
        print(f"Missing columns: {missing_template_cols}")
        return

    if missing_rules_cols:
        print(f"Missing columns: {missing_rules_cols}")
        return

    if not has_parent_rule_text:
        print("Warning: parent_rule_text column not found, skipping templates with {parent_rule_text}")

    def _merge_with_existing_table(new_df: pd.DataFrame, table_name: str = "generated_questions") -> None:
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

    # Build lookup dictionaries for rules
    rules_dict: Dict[str, Dict] = {}
    parent_rule_dict: Dict[str, Dict] = {}
    unique_rule_ids = set()

    rule_ids = rules_df[rule_id_col].to_numpy()
    parent_rule_ids = rules_df[parent_rule_id_col].to_numpy() if has_parent_rule_id else None
    rule_text_atomics = rules_df[rule_text_atomic_col].to_numpy()
    parent_rule_texts = rules_df[parent_rule_text_col].to_numpy() if has_parent_rule_text else None
    figure_requireds = rules_df[figure_required_col].to_numpy() if has_figure_required else None
    figure_ids = rules_df[figure_id_col].to_numpy() if has_figure_id else None

    figures_dict = {}
    if figures_df is not None:
        figures_cols_lower = {col.lower(): col for col in figures_df.columns}
        figure_id_fig_col = figures_cols_lower.get("figure_id", "figure_id")
        file_path_col = figures_cols_lower.get("file_path", "file_path")
        caption_col = figures_cols_lower.get("caption", "caption")

        if figure_id_fig_col in figures_df.columns:
            for _, fig_row in figures_df.iterrows():
                fig_id = fig_row[figure_id_fig_col]
                if pd.notna(fig_id):
                    fig_id_str = str(int(fig_id)) if pd.notna(fig_id) and str(fig_id).replace(".", "").isdigit() else str(fig_id)
                    figures_dict[fig_id_str] = {
                        "file_path": fig_row.get(file_path_col, "") if file_path_col in figures_df.columns else "",
                        "caption": fig_row.get(caption_col, "") if caption_col in figures_df.columns else "",
                    }

    for idx in range(len(rules_df)):
        rule_id_raw = rule_ids[idx]
        if pd.isna(rule_id_raw):
            continue

        rule_norm, _, rule_orig = normalize_id(rule_id_raw)

        parent_rule_id = None
        if has_parent_rule_id:
            parent_val = parent_rule_ids[idx]
            if not pd.isna(parent_val):
                parent_rule_id = normalize_id(parent_val)[0]

        figure_id = None
        if has_figure_required and has_figure_id:
            figure_req = figure_requireds[idx] if figure_requireds is not None else None
            if pd.notna(figure_req) and str(figure_req).strip().lower() == "yes":
                fig_id_val = figure_ids[idx] if figure_ids is not None else None
                if pd.notna(fig_id_val):
                    try:
                        fig_id_int = int(float(fig_id_val))
                        if fig_id_int > 0:
                            figure_id = str(fig_id_int)
                    except (ValueError, TypeError):
                        pass

        rule_data = {
            "rule_text_atomic": rule_text_atomics[idx] if pd.notna(rule_text_atomics[idx]) else "",
            "parent_rule_text": parent_rule_texts[idx] if has_parent_rule_text and pd.notna(parent_rule_texts[idx]) else "",
            "parent_rule_id": parent_rule_id,
            "figure_id": figure_id,
        }

        # Store with multiple key formats for flexible matching
        unique_rule_ids.add(rule_orig)
        for key in {rule_norm, rule_orig}:
            rules_dict[key] = rule_data

        if parent_rule_id:
            if parent_rule_id not in parent_rule_dict:
                parent_rule_dict[parent_rule_id] = rule_data
                parent_norm, _, parent_orig = normalize_id(parent_rule_id)
                for pkey in {parent_norm, parent_orig}:
                    parent_rule_dict.setdefault(pkey, rule_data)

    print(f"Loaded {len(unique_rule_ids)} rules")
    generated_questions = []
    skipped_templates = 0
    missing_rules = set()
    skipped_empty_text = 0

    for template_idx, template_row in templates_df.iterrows():
        template_id = template_row[template_id_col]
        question_template = template_row[question_template_col]
        requires_rule = template_row[requires_rule_col]

        # Skip if no requires_rule
        rule_ids = parse_requires_rule(requires_rule)
        if not rule_ids:
            skipped_templates += 1
            continue

        placeholder, text_type = detect_rule_text_placeholder(question_template)
        if not placeholder:
            skipped_templates += 1
            continue

        # If using parent_rule_text, requires_rule already contains parent_rule_ids
        if text_type == "parent_rule_text":
            seen_parents: Dict[str, Dict] = {}
            for parent_rule_id_raw in rule_ids:
                parent_norm, _, parent_orig = normalize_id(parent_rule_id_raw)
                matched_rule_data = find_in_dict(parent_rule_dict, parent_norm, parent_orig)
                if matched_rule_data:
                    seen_parents[parent_orig] = matched_rule_data
                else:
                    missing_rules.add(str(parent_rule_id_raw).strip())

            for parent_rule_id, rule_data in seen_parents.items():
                question_row = generate_question_for_rule(
                    template_row=template_row,
                    question_template=question_template,
                    rule_data=rule_data,
                    placeholder=placeholder,
                    text_type=text_type,
                    parent_rule_id=parent_rule_id,
                    question_id=len(generated_questions),
                    figures_dict=figures_dict,
                )
                if question_row is None:
                    skipped_empty_text += 1
                    continue
                generated_questions.append(question_row)
            continue

        for rule_id in rule_ids:
            rule_norm, _, rule_orig = normalize_id(rule_id)
            matched_rule_data = find_in_dict(rules_dict, rule_norm, rule_orig)

            if not matched_rule_data:
                missing_rules.add(str(rule_id).strip())
                continue

            rule_id_str = str(rule_id).strip()
            parent_rule_id_from_data = matched_rule_data.get("parent_rule_id")
            question_row = generate_question_for_rule(
                template_row=template_row,
                question_template=question_template,
                rule_data=matched_rule_data,
                placeholder=placeholder,
                text_type=text_type,
                rule_id=rule_id_str,
                parent_rule_id=parent_rule_id_from_data if parent_rule_id_from_data else None,
                question_id=len(generated_questions),
                figures_dict=figures_dict,
            )
            if question_row is None:
                skipped_empty_text += 1
                continue
            generated_questions.append(question_row)

    if generated_questions:
        questions_df = pd.DataFrame(generated_questions)

        # Put important columns first
        column_order = ["generated_question_id", template_id_col, "rule_id", "parent_rule_id", "question_text"]
        other_columns = [col for col in questions_df.columns if col not in column_order]
        questions_df = questions_df[column_order + other_columns]
        _merge_with_existing_table(questions_df, "generated_questions")

        print(f"Generated {len(generated_questions)} questions")
    else:
        print("No questions generated")
    print(f"Skipped {skipped_templates} templates")

    if missing_rules:
        print(f"Warning: {len(missing_rules)} rule IDs not found:")
        for rule_id in sorted(missing_rules)[:10]:
            print(f"  {rule_id}")
        if len(missing_rules) > 10:
            print(f"  ... and {len(missing_rules) - 10} more")

    if skipped_empty_text > 0:
        print(f"Skipped {skipped_empty_text} questions (empty text)")

    print("=" * 60)

    if figures_df is not None and context_col:
        figure_templates = templates_df[
            templates_df[context_col].astype(str).str.strip().str.lower() == "rule figures understanding"
        ]
    else:
        figure_templates = pd.DataFrame()

    if not figure_templates.empty and figures_df is not None:
        figure_id_fig_col = _resolve_column(figures_df, ["figure_id", "figure id"])
        file_path_col = _resolve_column(figures_df, ["file_path", "figure_path", "path"])
        caption_col = _resolve_column(figures_df, ["caption", "figure_caption", "caption_text"])

        inserted_rows: List[Dict] = []
        start_id = get_next_generated_question_id(conn)
        question_id = start_id

        for _, template_row in figure_templates.iterrows():
            template_text = str(template_row.get(question_template_col, ""))
            for _, fig_row in figures_df.iterrows():
                fig_id_val = fig_row.get(figure_id_fig_col, "") if figure_id_fig_col else ""
                fig_path_val = fig_row.get(file_path_col, "") if file_path_col else ""
                caption_val = fig_row.get(caption_col, "") if caption_col else ""
                question_text = template_text
                if "{figure_id}" in question_text:
                    question_text = question_text.replace("{figure_id}", str(fig_id_val))
                if "{figure_path}" in question_text:
                    question_text = question_text.replace("{figure_path}", str(fig_path_val))

                row = _build_template_payload(template_row)
                row["generated_question_id"] = question_id
                row["rule_id"] = ""
                row["parent_rule_id"] = ""
                row["question_text"] = question_text
                row["ground_truth_answer"] = caption_val
                row["figure_id"] = fig_id_val
                row["figure_path"] = fig_path_val
                inserted_rows.append(row)
                question_id += 1

        if inserted_rows:
            fig_df = pd.DataFrame(inserted_rows)
            _merge_with_existing_table(fig_df, "generated_questions")
            print(f"Rule figures processed: {len(figures_df)}")
            print(f"Rule figure questions inserted: {len(inserted_rows)}")
    try:
        start_id = get_next_generated_question_id(conn)
        created = generate_rase_atomic_questions(conn, start_id=start_id)
        start_id += created
    except Exception as exc:
        print(f"Warning: RASE-for-atomic generation failed: {exc}")
    try:
        generate_rase_requirement_flagging_questions(conn, start_id=start_id)
    except Exception as exc:
        print(f"Warning: RASErequirement flagging generation failed: {exc}")
    try:
        db_path = None
        try:
            rows = conn.execute("PRAGMA database_list").fetchall()
            for _, name, file_path in rows:
                if name == "main" and file_path:
                    db_path = Path(file_path)
                    break
        except Exception:
            db_path = None
        if db_path is None:
            raise RuntimeError("Could not resolve database path from SQLite connection.")
        generate_atomisation_questions(db_path)
    except Exception as exc:
        print(f"Warning: rule text atomisation generation failed: {exc}")
