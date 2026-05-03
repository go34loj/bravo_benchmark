"""
Data manipulation script for VQA Rules Scenes Templates dataset.

This script handles data transformations and linking between different CSV files.

Usage:
    # Link ambiguous rules to templates
    python scripts/data_manipulation.py \
        --templates "xx_xx_VQA_Rules_Scenes_Templates(templates).csv" \
        --rules "xx_xx_VQA_Rules_Scenes_Templates(Rules_sort).csv" \
        --output "templates_updated.csv"
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from io import StringIO
from typing import Optional

import pandas as pd

try:
    from utils.data_access_layer.data_parsers import (
        _normalize_scene_id,
        _resolve_column,
        format_template_id_list,
        parse_template_id_list,
    )
except ImportError:
    from data_access_layer.data_parsers import (
        _normalize_scene_id,
        _resolve_column,
        format_template_id_list,
        parse_template_id_list,
    )


def load_csv(path: Path, delimiter: str = ";") -> pd.DataFrame:
    """Read a CSV file with semicolon delimiter and robust encoding handling."""
    encodings_to_try = ["utf-8", "utf-8-sig", "windows-1252", "latin1", "cp1252", "iso-8859-1"]

    last_error: Exception | None = None
    for encoding in encodings_to_try:
        try:
            df = pd.read_csv(path, delimiter=delimiter, encoding=encoding)
            df.columns = df.columns.str.strip()
            df = df.dropna(how="all")
            return df
        except UnicodeDecodeError as e:
            last_error = e
            continue
        except Exception as e:
            last_error = e
            continue

    try:
        raw_bytes = path.read_bytes()
        text = raw_bytes.decode("cp1252", errors="replace")
        df = pd.read_csv(StringIO(text), delimiter=delimiter)
        df.columns = df.columns.str.strip()
        df = df.dropna(how="all")
        return df
    except Exception as e:
        last_error = e

    raise last_error if last_error else Exception(f"Could not load CSV file: {path}")


def _normalize_rule_id(value: str) -> str:
    value = str(value).strip()
    if not value:
        return ""
    try:
        number = float(value)
        if number.is_integer():
            return str(int(number))
        return str(number)
    except ValueError:
        return value


def _build_scene_path_index(scenes_dir: Path) -> dict[str, list[str]]:
    index: dict[str, list[str]] = {}
    if not scenes_dir.exists():
        return index
    for path in scenes_dir.iterdir():
        if not path.is_file():
            continue
        match = re.match(r"^\s*(\d+)", path.name)
        if not match:
            continue
        scene_id = match.group(1)
        index.setdefault(scene_id, []).append(str(path))
    return index


def update_viewpoint_scene_paths(viewpoints_path: Path, scenes_dir: Path) -> None:
    """
    Update file_path in viewpoint_scenes CSV based on scene_id and image filenames.
    Overwrites the same CSV file and logs updates to the terminal.
    """
    if not viewpoints_path.exists():
        print(f"Error: Viewpoint scenes file not found: {viewpoints_path}")
        return
    if not scenes_dir.exists():
        print(f"Error: Scenes directory not found: {scenes_dir}")
        return

    df = load_csv(viewpoints_path)
    cols_lower = {col.lower(): col for col in df.columns}
    scene_id_col = cols_lower.get("scene_id", "scene_id")
    file_path_col = cols_lower.get("file_path", "file_path")

    if scene_id_col not in df.columns or file_path_col not in df.columns:
        print(f"Error: Required columns not found in viewpoint scenes CSV: {viewpoints_path}")
        print(f"  Columns present: {list(df.columns)}")
        return

    index = _build_scene_path_index(scenes_dir)
    if not index:
        print(f"Error: No scene images found in {scenes_dir}")
        return

    missing_ids: list[str] = []
    ambiguous_ids: list[str] = []

    updated_paths: list[str] = []
    for raw_id in df[scene_id_col].tolist():
        scene_id = _normalize_scene_id(raw_id)
        paths = index.get(scene_id, [])
        if not scene_id or not paths:
            missing_ids.append(scene_id or "<empty>")
            updated_paths.append("")
            continue
        if len(paths) > 1:
            ambiguous_ids.append(scene_id)
        updated_paths.append(sorted(paths)[0])

    df[file_path_col] = updated_paths
    viewpoints_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(viewpoints_path, index=False, sep=";", encoding="utf-8")

    total = len(df)
    missing_count = len([sid for sid in missing_ids if sid and sid != "<empty>"])
    ambiguous_unique = sorted(set(ambiguous_ids))

    print(f"OK Viewpoints updated: {total} rows processed, {total - missing_count} paths assigned.")
    if missing_ids:
        preview = ", ".join(missing_ids[:10])
        print(f"Human Review Required: missing scene_id matches: {missing_count}. Examples: {preview}")
    if ambiguous_unique:
        preview = ", ".join(ambiguous_unique[:10])
        print(f"Human Review Required: multiple image matches for scene_id(s): {preview}")


def add_text_recognition_templates_to_viewpoints(
    viewpoints_path: Path,
    templates_path: Path,
) -> None:
    """
    For viewpoint scenes that contain any text in Text or Multi-View Dimensions,
    ensure all templates with context == "text recognition" are included in template_id list.
    """
    if not viewpoints_path.exists():
        print(f"Error: Viewpoint scenes file not found: {viewpoints_path}")
        return
    if not templates_path.exists():
        print(f"Error: Templates file not found: {templates_path}")
        return

    viewpoints_df = load_csv(viewpoints_path)
    templates_df = load_csv(templates_path)

    template_id_col = _resolve_column(viewpoints_df, ["template_id"])
    text_col = _resolve_column(viewpoints_df, ["text"])
    mvd_col = _resolve_column(
        viewpoints_df,
        [
            "multi-view_dimensions",
        ],
    )

    if template_id_col is None:
        print(f"Error: template_id column not found in viewpoint scenes CSV: {viewpoints_path}")
        print(f"  Columns present: {list(viewpoints_df.columns)}")
        return
    if text_col is None or mvd_col is None:
        print(f"Error: Text or Multi-View Dimensions column not found in viewpoint scenes CSV: {viewpoints_path}")
        print(f"  Columns present: {list(viewpoints_df.columns)}")
        return

    templates_id_col = _resolve_column(templates_df, ["template_id", "template"])
    context_col = _resolve_column(templates_df, ["context"])
    if templates_id_col is None or context_col is None:
        print(f"Error: template_id/context column not found in templates CSV: {templates_path}")
        print(f"  Columns present: {list(templates_df.columns)}")
        return

    text_recognition_template_ids: list[int] = []
    any_text_template_ids: list[int] = []
    for _, row in templates_df.iterrows():
        raw_id = row.get(templates_id_col)
        if pd.isna(raw_id) or str(raw_id).strip() == "":
            continue
        context = str(row.get(context_col, "")).strip().lower()
        try:
            template_id = int(float(raw_id))
        except (TypeError, ValueError):
            continue
        if context == "text recognition":
            text_recognition_template_ids.append(template_id)
        elif context == "any text in the picture":
            any_text_template_ids.append(template_id)

    all_ocr_template_ids = sorted(set(text_recognition_template_ids + any_text_template_ids))
    if not all_ocr_template_ids:
        print("Warning: No templates with context == 'text recognition' or 'any text in the picture' found.")
        return

    updated_rows = 0
    eligible_rows = 0
    for idx, row in viewpoints_df.iterrows():
        text_value = row.get(text_col)
        mvd_value = row.get(mvd_col)
        has_text = not pd.isna(text_value) and str(text_value).strip() != ""
        has_mvd = not pd.isna(mvd_value) and str(mvd_value).strip() != ""
        if not (has_text or has_mvd):
            continue
        eligible_rows += 1

        original_value = row.get(template_id_col)
        existing_ids = parse_template_id_list(original_value)
        updated_ids = list(existing_ids)
        for tid in all_ocr_template_ids:
            if tid not in updated_ids:
                updated_ids.append(tid)
        if set(updated_ids) != set(existing_ids):
            viewpoints_df.loc[idx, template_id_col] = format_template_id_list(original_value, updated_ids)
            updated_rows += 1

    viewpoints_path.parent.mkdir(parents=True, exist_ok=True)
    viewpoints_df.to_csv(viewpoints_path, index=False, sep=";", encoding="utf-8")

    print(
        "OK Added text recognition templates to viewpoint scenes: "
        f"{updated_rows} rows updated out of {eligible_rows} eligible rows."
    )


def _parse_rule_list(value: object) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    raw = str(value).strip()
    if not raw:
        return []
    if raw.startswith("[") and raw.endswith("]"):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return [str(v).strip() for v in parsed if str(v).strip()]
        except (json.JSONDecodeError, ValueError):
            pass
    return [v.strip() for v in raw.split(",") if v.strip()]


def _format_rule_list(original: object, rules: list[str]) -> str:
    raw = "" if original is None else str(original).strip()
    if raw.startswith("[") and raw.endswith("]"):
        return json.dumps(rules)
    return ",".join(rules)


def filter_requires_rule_by_figure(
    templates_df: pd.DataFrame,
    rules_df: pd.DataFrame,
) -> None:
    templates_cols_lower = {col.lower(): col for col in templates_df.columns}
    rules_cols_lower = {col.lower(): col for col in rules_df.columns}

    requires_rule_col = templates_cols_lower.get("requires_rule")
    requires_figure_col = templates_cols_lower.get("requires_figure")
    question_template_col = templates_cols_lower.get("question_template")
    rule_id_col = rules_cols_lower.get("rule_id")
    parent_rule_id_col = rules_cols_lower.get("parent_rule_id")
    figure_id_col = rules_cols_lower.get("figure_id")

    if not requires_figure_col:
        print("Info: requires_figure column not found in templates; skipping figure filtering.")
        return
    if not requires_rule_col:
        print("Info: requires_rule column not found in templates; skipping figure filtering.")
        return
    if not rule_id_col or not figure_id_col:
        print("Info: rule_id/figure_id column not found in rules; skipping figure filtering.")
        return
    if not question_template_col:
        print("Info: question_template column not found in templates; skipping figure filtering.")
        return

    valid_rule_ids: set[str] = set()
    valid_parent_rule_ids: set[str] = set()
    for _, row in rules_df.iterrows():
        rid_raw = row.get(rule_id_col)
        prid_raw = row.get(parent_rule_id_col) if parent_rule_id_col else None
        fig_raw = row.get(figure_id_col)
        if pd.isna(fig_raw):
            continue
        rid = _normalize_rule_id(rid_raw) if not pd.isna(rid_raw) else ""
        prid = _normalize_rule_id(prid_raw) if not pd.isna(prid_raw) else ""
        try:
            fig_val = float(fig_raw)
        except (ValueError, TypeError):
            continue
        if fig_val > 0:
            if rid:
                valid_rule_ids.add(rid)
            if prid:
                valid_parent_rule_ids.add(prid)

    if not valid_rule_ids and not valid_parent_rule_ids:
        print("Warning: No rules with figure_id > 0 found; figure filtering will remove all requires_rule entries.")

    filtered_templates = 0
    removed_total = 0

    for idx, row in templates_df.iterrows():
        requires_figure = str(row.get(requires_figure_col, "")).strip().lower()
        if requires_figure != "yes":
            continue

        question_template = str(row.get(question_template_col, ""))
        uses_parent_rule_text = "{parent_rule_text}" in question_template

        original_value = row.get(requires_rule_col)
        rule_ids = _parse_rule_list(original_value)
        if not rule_ids:
            continue

        normalized_ids = [_normalize_rule_id(r) for r in rule_ids]
        if uses_parent_rule_text:
            filtered_ids = [rid for rid in normalized_ids if rid in valid_parent_rule_ids]
        else:
            filtered_ids = [rid for rid in normalized_ids if rid in valid_rule_ids]

        removed = len(rule_ids) - len(filtered_ids)
        if removed > 0:
            removed_total += removed
            filtered_templates += 1
            templates_df.loc[idx, requires_rule_col] = _format_rule_list(original_value, filtered_ids)

    print(
        f"OK Figure filtering: updated {filtered_templates} templates; removed {removed_total} rule_id entries without figure_id."
    )


def link_ambiguous_rules_to_templates(
    templates_path: Path,
    rules_path: Path,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Link ambiguous rules to templates with subtask='ambiguity'.

    For rules where Ambiguity='yes', add their rule_ids (comma-separated)
    to the requires_rule column in templates where subtask='ambiguity'.

    Args:
        templates_path: Path to the templates CSV file
        rules_path: Path to the rules CSV file
        output_path: Optional path to save the updated templates CSV

    Returns:
        Updated templates DataFrame
    """
    print("Linking ambiguous rules to templates...")
    print(f"  Templates file: {templates_path}")
    print(f"  Rules file: {rules_path}")
    print()

    # Load rules CSV
    print("Loading rules file...")
    rules_df = load_csv(rules_path)

    # Normalize column names (case-insensitive matching)
    rules_cols_lower = {col.lower(): col for col in rules_df.columns}
    ambiguity_col = rules_cols_lower.get("ambiguity", "Ambiguity")
    rule_id_col = rules_cols_lower.get("rule_id", "rule_id")
    parent_rule_id_col = rules_cols_lower.get("parent_rule_id", "parent_rule_id")

    if ambiguity_col not in rules_df.columns or rule_id_col not in rules_df.columns:
        print(f"  Error: Could not find 'Ambiguity' or 'rule_id' columns in rules file")
        print(f"    Available columns: {list(rules_df.columns)}")
        raise ValueError("Required columns not found in rules file")

    # Check for duplicate rule_id in the rules file
    rule_ids_all = rules_df[rule_id_col].dropna().astype(str).tolist()
    rule_ids_unique = list(set(rule_ids_all))
    if len(rule_ids_all) != len(rule_ids_unique):
        duplicates = [rule_id for rule_id in rule_ids_unique if rule_ids_all.count(rule_id) > 1]
        print("  Warning: Found duplicate rule_id values in rules file:")
        for dup_id in duplicates:
            count = rule_ids_all.count(dup_id)
            print(f"    - rule_id '{dup_id}' appears {count} times")
        print()

    # Filter rules where Ambiguity = "yes"
    ambiguous_rules = rules_df[
        rules_df[ambiguity_col].astype(str).str.strip().str.lower() == "yes"
    ]
    ambiguous_rule_ids = ambiguous_rules[rule_id_col].dropna().astype(str).tolist()
    
    # Get parent_rule_ids for ambiguous rules
    ambiguous_parent_rule_ids = ambiguous_rules[parent_rule_id_col].dropna().astype(str).unique().tolist()

    print(f"  OK Found {len(ambiguous_rule_ids)} rules with Ambiguity='yes'")
    if ambiguous_rule_ids:
        print(f"  Rule IDs: {', '.join(map(str, ambiguous_rule_ids))}")
    if ambiguous_parent_rule_ids:
        print(f"  Parent Rule IDs: {', '.join(map(str, ambiguous_parent_rule_ids))}")
    print()

    # Load templates CSV
    print("Loading templates file...")
    templates_df = load_csv(templates_path)

    # Normalize column names
    templates_cols_lower = {col.lower(): col for col in templates_df.columns}
    subtask_col = templates_cols_lower.get("subtask", "subtask")
    requires_rule_col = templates_cols_lower.get("requires_rule", "requires_rule")
    question_template_col = templates_cols_lower.get("question_template", "question_template")
    answer_type_col = templates_cols_lower.get("answer_type", "answer_type")

    if subtask_col not in templates_df.columns or requires_rule_col not in templates_df.columns:
        print(f"  Error: Could not find 'subtask' or 'requires_rule' columns in templates file")
        print(f"    Available columns: {list(templates_df.columns)}")
        raise ValueError("Required columns not found in templates file")

    # Find templates where subtask = "ambiguity"
    ambiguity_mask = templates_df[subtask_col].astype(str).str.strip().str.lower() == "ambiguity"
    ambiguity_templates_count = ambiguity_mask.sum()

    print(f"  OK Found {ambiguity_templates_count} templates with subtask='ambiguity'")
    print()

    if ambiguity_templates_count == 0:
        print("  Warning: No templates with subtask='ambiguity' found")
        return templates_df

    if not ambiguous_rule_ids:
        print("  Warning: No ambiguous rules found")
        return templates_df

    # Ensure requires_rule column is of object type (string) to store JSON
    if requires_rule_col in templates_df.columns:
        templates_df[requires_rule_col] = templates_df[requires_rule_col].astype(object)

    # Convert rule IDs to list of strings for JSON storage
    ambiguous_rule_ids_str = [str(rid) for rid in ambiguous_rule_ids]
    ambiguous_parent_rule_ids_str = [str(rid) for rid in ambiguous_parent_rule_ids]
    all_rule_ids_str = [str(rid) for rid in rules_df[rule_id_col].dropna().astype(str).tolist()]
    all_parent_rule_ids_str = (
        [str(rid) for rid in rules_df[parent_rule_id_col].dropna().astype(str).unique().tolist()]
        if parent_rule_id_col in rules_df.columns
        else []
    )

    # Check which templates use parent_rule_text vs rule_text
    ambiguity_indices = templates_df[ambiguity_mask].index
    
    # Check if question_template contains {parent_rule_text} for each ambiguity template
    uses_parent_rule_text_dict = {}
    if question_template_col in templates_df.columns:
        for idx in ambiguity_indices:
            question_template = str(templates_df.loc[idx, question_template_col])
            uses_parent_rule_text_dict[idx] = "{parent_rule_text}" in question_template
    else:
        uses_parent_rule_text_dict = {idx: False for idx in ambiguity_indices}

    # Update requires_rule column for ambiguity templates
    # Store as JSON string for better SQLite compatibility
    def update_requires_rule(idx):
        value = templates_df.loc[idx, requires_rule_col]
        
        # Determine which IDs to use based on template type and parent_rule_text usage
        answer_type = str(templates_df.loc[idx, answer_type_col]) if answer_type_col in templates_df.columns else ""
        if answer_type.strip().lower() == "bool":
            if uses_parent_rule_text_dict.get(idx, False):
                ids_to_add = all_parent_rule_ids_str
            else:
                ids_to_add = all_rule_ids_str
        else:
            if uses_parent_rule_text_dict.get(idx, False):
                ids_to_add = ambiguous_parent_rule_ids_str
            else:
                ids_to_add = ambiguous_rule_ids_str
        
        if pd.isna(value) or str(value).strip() == "":
            # New value: store as JSON array
            return json.dumps(ids_to_add)
        else:
            # Parse existing value (could be JSON or comma-separated string)
            existing_str = str(value).strip()
            try:
                # Try to parse as JSON first
                existing_ids = json.loads(existing_str)
                if not isinstance(existing_ids, list):
                    # If not a list, treat as comma-separated string
                    existing_ids = [r.strip() for r in existing_str.split(",") if r.strip()]
            except (json.JSONDecodeError, ValueError):
                # Fallback: treat as comma-separated string
                existing_ids = [r.strip() for r in existing_str.split(",") if r.strip()]

            # Avoid duplicates and merge with new IDs
            all_ids = list(existing_ids)
            for new_id in ids_to_add:
                if new_id not in all_ids:
                    all_ids.append(new_id)

            # Return as JSON string
            return json.dumps(all_ids)

    # Apply update function for each ambiguity template
    for idx in ambiguity_indices:
        templates_df.loc[idx, requires_rule_col] = update_requires_rule(idx)

    filter_requires_rule_by_figure(templates_df=templates_df, rules_df=rules_df)

    # Count how many templates use parent_rule_id vs rule_id
    parent_rule_count = sum(uses_parent_rule_text_dict.values())
    rule_id_count = ambiguity_templates_count - parent_rule_count
    
    print(f"  OK Updated {ambiguity_templates_count} templates with rule IDs")
    if parent_rule_count > 0:
        print(f"    - {parent_rule_count} templates using parent_rule_id: {', '.join(ambiguous_parent_rule_ids_str)}")
    if rule_id_count > 0:
        print(f"    - {rule_id_count} templates using rule_id: {', '.join(ambiguous_rule_ids_str)}")
    print()

    # Save updated templates if output path is provided
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        templates_df.to_csv(output_path, index=False, sep=";", encoding="utf-8")
        print(f"  OK Saved updated templates to: {output_path}")
    else:
        print("  Info: No output path specified, returning updated DataFrame only")

    return templates_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Data manipulation utilities for VQA Rules Scenes Templates dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Define default paths relative to script directory
    script_dir = Path(__file__).parent

    templates_glob = list(script_dir.glob("*VQA_Rules_Scenes_Templates(templates).csv"))
    if not templates_glob:
        raise FileNotFoundError(
            "No templates CSV matching *VQA_Rules_Scenes_Templates(templates).csv"
        )
    default_templates = max(templates_glob, key=lambda p: p.stat().st_mtime)

    rules_glob = list(script_dir.glob("*VQA_Rules_Scenes_Templates(Rules_sort).csv"))
    if not rules_glob:
        raise FileNotFoundError(
            "No rules CSV matching *VQA_Rules_Scenes_Templates(Rules_sort).csv"
        )
    default_rules = max(rules_glob, key=lambda p: p.stat().st_mtime)

    default_output = script_dir / "templates_updated.csv"

    viewpoints_glob = list(script_dir.glob("*VQA_Rules_Scenes_Templates(viewpoint_scenes).csv"))
    default_viewpoints = max(viewpoints_glob, key=lambda p: p.stat().st_mtime) if viewpoints_glob else None

    parser.add_argument(
        "--templates",
        type=Path,
        default=default_templates,
        help="Path to the templates CSV file. Default: latest *VQA_Rules_Scenes_Templates(templates).csv",
    )
    parser.add_argument(
        "--rules",
        type=Path,
        default=default_rules,
        help="Path to the rules CSV file. Default: latest *VQA_Rules_Scenes_Templates(Rules_sort).csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output,
        help="Output path for the updated templates CSV file. Default: templates_updated.csv",
    )
    parser.add_argument(
        "--viewpoints",
        type=Path,
        default=default_viewpoints,
        help="Path to the viewpoint_scenes CSV file for updating scene paths.",
    )
    parser.add_argument(
        "--scenes-dir",
        type=Path,
        default=script_dir.parent / "dataset" / "BIM_design_scenes",
        help="Directory containing scene images (default: ../dataset/BIM_design_scenes).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Validate input files exist
    if not args.templates.exists():
        print(f"Error: Templates file not found: {args.templates}")
        return

    if not args.rules.exists():
        print(f"Error: Rules file not found: {args.rules}")
        return

    try:
        link_ambiguous_rules_to_templates(
            templates_path=args.templates,
            rules_path=args.rules,
            output_path=args.output,
        )
        if args.viewpoints:
            update_viewpoint_scene_paths(args.viewpoints, args.scenes_dir)
            add_text_recognition_templates_to_viewpoints(args.viewpoints, args.templates)
        print("\nOK Data manipulation completed successfully!")
    except Exception as e:
        print(f"\nError during data manipulation: {e}")
        raise


if __name__ == "__main__":
    main()

