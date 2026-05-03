"""
Generates questions by injecting rule text into templates.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml

from utils.rule_question_processing import generate_rule_questions
from utils.compliance_question_gen import generate_compliance_questions
from utils.scene_perc_quest_gen import generate_scene_questions
from utils.scene_underst_QA_gen import (
    generate_adjacency_flagging_rows,
    generate_connectivity_flagging_rows,
    generate_connectivity_relation_rows,
    generate_functional_grouping_rows,
    generate_multiview_dimension_rows,
    load_scene_understanding_scenes,
)
from backend.utils.data_access_layer.database_utils import print_database_summary
from backend.utils.data_access_layer.file_operations import detect_csv_files, load_csv
from utils.scene_perc_gen_gt import update_ground_truth as update_scene_perc_ground_truth
from utils.database_schema import ensure_schema, setup_connection, validate_integrity


TEXT_ID_COLUMNS = {
    "template_id",
    "rule_id",
    "parent_rule_id",
    "scene_id",
    "figure_id",
    "cutout_id",
}


def _normalize_id_value(value: object) -> Optional[str]:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        number = float(text)
        if number.is_integer():
            return str(int(number))
    except (ValueError, TypeError):
        pass
    return text


def _quote_ident(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def _prepare_df_for_table(df: pd.DataFrame) -> pd.DataFrame:
    prepared = df.copy()
    for col in prepared.columns:
        key = str(col).strip().lower().replace(" ", "_").replace("-", "_")
        if col == "generated_question_id":
            prepared[col] = pd.to_numeric(prepared[col], errors="coerce").astype("Int64")
        elif key in TEXT_ID_COLUMNS:
            prepared[col] = prepared[col].map(_normalize_id_value)
    return prepared


def _ensure_table_columns(conn, table_name: str, columns: list[str]) -> None:
    info = conn.execute(f'PRAGMA table_info({_quote_ident(table_name)})').fetchall()
    existing = {row[1] for row in info}
    for col in columns:
        if col in existing:
            continue
        col_type = "INTEGER" if col == "generated_question_id" else "TEXT"
        conn.execute(f'ALTER TABLE {_quote_ident(table_name)} ADD COLUMN {_quote_ident(col)} {col_type}')


def _replace_table_rows(conn, table_name: str, df: pd.DataFrame) -> None:
    prepared = _prepare_df_for_table(df)
    _ensure_table_columns(conn, table_name, list(prepared.columns))
    conn.execute(f'DELETE FROM {_quote_ident(table_name)}')
    prepared.to_sql(table_name, conn, if_exists="append", index=False)


def create_database(
    templates_path: Path,
    rules_path: Path,
    output_sqlite: Path,
    figures_path: Optional[Path] = None,
    scene_scenes_path: Optional[Path] = None,
    scenes_dir: Optional[Path] = None,
    cutouts_path: Optional[Path] = None,
    run_scene_perception: bool = True,
    run_scene_understanding: bool = True,
    run_rule_understanding: bool = True,
    run_compliance_reasoning: bool = True,
    run_scene_perception_gt: bool = False,
    run_compliance_gt: bool = False,
    seed: Optional[int] = None,
) -> None:
    """Load CSVs, generate questions, save to database."""

    if not templates_path.exists():
        raise FileNotFoundError(f"Templates not found: {templates_path}")
    if not rules_path.exists():
        raise FileNotFoundError(f"Rules not found: {rules_path}")

    conn = setup_connection(output_sqlite)
    ensure_schema(conn)

    try:
        print(f"Loading {templates_path.name}...")
        templates_df = load_csv(templates_path)
        _replace_table_rows(conn, "templates", templates_df)
        print(f"Loaded {len(templates_df)} rows")

        print(f"\nLoading {rules_path.name}...")
        rules_df = load_csv(rules_path)
        _replace_table_rows(conn, "rules", rules_df)
        print(f"  OK Loaded {len(rules_df)} rows with {len(rules_df.columns)} columns")

        figures_df = None
        if figures_path and figures_path.exists():
            print(f"\nLoading {figures_path.name}...")
            figures_df = load_csv(figures_path)
            _replace_table_rows(conn, "rule_figures", figures_df)
            print(f"Loaded {len(figures_df)} figures")
        elif figures_path:
            print("\nRule figures file not found; skipping rule figures load.")

        if cutouts_path and cutouts_path.exists():
            print(f"\nLoading {cutouts_path.name}...")
            cutouts_df = load_csv(cutouts_path)
            _replace_table_rows(conn, "cutout", cutouts_df)
            print(f"Loaded {len(cutouts_df)} cutouts")
        elif cutouts_path:
            print("\nCutouts file not found; skipping cutout load.")

        if run_rule_understanding:
            generate_rule_questions(templates_df=templates_df, rules_df=rules_df, conn=conn, figures_df=figures_df)
        else:
            print("\nSkipping rule understanding generation (run_rule_understanding=false).")

        if scene_scenes_path and scene_scenes_path.exists():
            print(f"\nUsing unified templates for scene perception (source: {templates_path.name})...")
            scene_templates_df = templates_df

            print(f"\nLoading {scene_scenes_path.name}...")
            scenes_df = load_csv(scene_scenes_path)
            _replace_table_rows(conn, "viewpoint_scenes", scenes_df)
            print(f"Loaded {len(scenes_df)} rows")

            if run_scene_perception:
                scene_questions_df = generate_scene_questions(
                    templates_df=scene_templates_df,
                    scenes_df=scenes_df,
                    positives_per_scene_t1=None,
                    negatives_per_scene_t1=0,
                    negatives_per_scene_t2=0,
                    seed=seed,
                )
                if not scene_questions_df.empty:
                    _replace_table_rows(conn, "generated_scene_questions", scene_questions_df)
                    print(f"Generated {len(scene_questions_df)} scene perception questions")
            else:
                print("Skipping scene perception generation (run_scene_perception=false).")

            if run_scene_understanding:
                print("\nGenerating scene understanding questions...")
                scene_underst_records = load_scene_understanding_scenes(scenes_df)
                connectivity_df = generate_connectivity_flagging_rows(
                    templates_df=scene_templates_df,
                    scenes=scene_underst_records,
                    start_id=0,
                )
                not_direct_df = generate_connectivity_relation_rows(
                    templates_df=scene_templates_df,
                    scenes=scene_underst_records,
                    relation_type="not_direct_connectivity",
                    start_id=len(connectivity_df) if not connectivity_df.empty else 0,
                )
                adjacency_df = generate_adjacency_flagging_rows(
                    templates_df=scene_templates_df,
                    scenes=scene_underst_records,
                    start_id=(
                        len(connectivity_df)
                        + (len(not_direct_df) if not not_direct_df.empty else 0)
                    ),
                )
                functional_df = generate_functional_grouping_rows(
                    templates_df=scene_templates_df,
                    scenes=scene_underst_records,
                    cutouts_df=cutouts_df,
                    start_id=(
                        len(connectivity_df)
                        + (len(not_direct_df) if not not_direct_df.empty else 0)
                        + (len(adjacency_df) if not adjacency_df.empty else 0)
                    ),
                )
                multiview_df = generate_multiview_dimension_rows(
                    templates_df=scene_templates_df,
                    viewpoint_df=scenes_df,
                    start_id=(
                        len(connectivity_df)
                        + (len(not_direct_df) if not not_direct_df.empty else 0)
                        + (len(adjacency_df) if not adjacency_df.empty else 0)
                        + (len(functional_df) if not functional_df.empty else 0)
                    ),
                )
                scene_underst_df = pd.concat(
                    [df for df in [connectivity_df, not_direct_df, adjacency_df, functional_df, multiview_df] if not df.empty],
                    ignore_index=True,
                )
                if not scene_underst_df.empty:
                    _replace_table_rows(conn, "generated_scene_understanding_questions", scene_underst_df)
                    print(f"Generated {len(scene_underst_df)} scene understanding questions")
                else:
                    print("No scene understanding questions generated")
            else:
                print("Skipping scene understanding generation (run_scene_understanding=false).")

            if run_compliance_reasoning:
                print("\nGenerating compliance reasoning questions...")
                compliance_count = generate_compliance_questions(
                    conn=conn,
                    table_name="generated_compliance_questions",
                    scenes_dir=scenes_dir,
                    seed=seed,
                    verbose=True,
                )
                if compliance_count:
                    print(f"Generated {compliance_count} compliance reasoning questions")
                else:
                    print("No compliance reasoning questions generated")
            else:
                print("Skipping compliance reasoning generation (run_compliance_reasoning=false).")
        else:
            if scene_scenes_path:
                print("\nScene perception files not found; skipping scene perception question generation.")

        if run_scene_perception_gt:
            print("\nRunning scene perception GT generation...")
            update_scene_perc_ground_truth(output_sqlite, seed=seed)
            print("Scene perception GT generation completed.")
        else:
            print("\nSkipping scene perception GT generation (run_scene_perception_gt=false).")

        if run_compliance_gt:
            print("\nRunning compliance GT generation...")
            script_path = Path(__file__).parent / "utils" / "compliance_bool_gt_gen.py"
            subprocess.run(
                [sys.executable, str(script_path), "--db", str(output_sqlite), "--table-name", "generated_compliance_questions"],
                check=True,
            )
            print("Compliance GT generation completed.")
        else:
            print("Skipping compliance GT generation (run_compliance_gt=false).")

        validate_integrity(conn)
        print_database_summary(conn)



    finally:
        conn.close()

    print(f"\nDatabase created: {output_sqlite}")



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate questions from templates and rules.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    script_dir = Path(__file__).parent

    parser.add_argument(
        "--config",
        type=Path,
        default=script_dir / "config.yaml",
        help="Path to backend YAML config (default: backend/config.yaml).",
    )
    parser.add_argument(
        "--templates",
        type=Path,
        default=None,
        help="Path to templates CSV file. Default: auto-detect latest templates_updated.csv or *templates*.csv",
    )
    parser.add_argument(
        "--rules",
        type=Path,
        default=None,
        help="Path to rules CSV file. Default: auto-detect latest *Rules_sort*.csv",
    )
    parser.add_argument(
        "--backend-dir",
        type=Path,
        default=script_dir,
        help="Backend directory to search for CSV files (default: script directory).",
    )
    parser.add_argument(
        "--output-sqlite",
        type=Path,
        default=script_dir / "unified_database.db",
        help="Output SQLite database path (default: backend/unified_database.db).",
    )
    parser.add_argument(
        "--figures",
        type=Path,
        default=None,
        help="Path to rule figures CSV file. Default: auto-detect latest *rule_figure*.csv",
    )
    parser.add_argument(
        "--scene-scenes",
        type=Path,
        default=None,
        help="Path to viewpoint scenes CSV file. Default: auto-detect latest *viewpoint_scenes*.csv",
    )
    parser.add_argument(
        "--cutouts",
        type=Path,
        default=None,
        help="Path to cutouts CSV file. Default: auto-detect latest *cutouts*.csv",
    )
    return parser.parse_args()

def load_yaml_config(config_path: Optional[Path]) -> dict:
    if not config_path or not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Invalid config format in {config_path}: root must be a mapping")
    return data


def _to_bool(value: object, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


def _cfg_path(base: Path, value: Optional[str]) -> Optional[Path]:
    if not value:
        return None
    p = Path(value)
    return p if p.is_absolute() else (base / p)


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)

    backend_cfg = config.get("backend", {})
    inputs_cfg = config.get("inputs", {})
    execution_cfg = config.get("execution", {})
    randomness_cfg = config.get("randomness", {})

    cfg_base_dir = _cfg_path(Path(__file__).parent, backend_cfg.get("base_dir")) if backend_cfg else None
    cfg_backend_dir = cfg_base_dir if cfg_base_dir else args.backend_dir

    csv_files = detect_csv_files(cfg_backend_dir)

    if args.templates:
        templates_path = args.templates
    else:
        cfg_templates = _cfg_path(cfg_backend_dir, inputs_cfg.get("templates_csv")) if inputs_cfg else None
        if cfg_templates and cfg_templates.exists():
            templates_path = cfg_templates
        elif "templates" in csv_files:
            templates_path = csv_files["templates"]
        else:
            print(f"Templates not found in {args.backend_dir}")
            print("Use --templates or add templates_updated.csv to backend directory")
            return

    if args.rules:
        rules_path = args.rules
    else:
        cfg_rules = _cfg_path(cfg_backend_dir, inputs_cfg.get("rules_csv")) if inputs_cfg else None
        if cfg_rules and cfg_rules.exists():
            rules_path = cfg_rules
        elif "rules" in csv_files:
            rules_path = csv_files["rules"]
        else:
            print(f"Rules not found in {args.backend_dir}")
            print("Use --rules or add *Rules_sort*.csv to backend directory")
            return

    if args.figures:
        figures_path = args.figures
    else:
        cfg_figures = _cfg_path(cfg_backend_dir, inputs_cfg.get("rule_figures_csv")) if inputs_cfg else None
        figures_path = cfg_figures if cfg_figures and cfg_figures.exists() else csv_files.get("figures")

    if args.scene_scenes:
        scene_scenes_path = args.scene_scenes
    else:
        cfg_scenes = _cfg_path(cfg_backend_dir, inputs_cfg.get("viewpoint_scenes_csv")) if inputs_cfg else None
        scene_scenes_path = cfg_scenes if cfg_scenes and cfg_scenes.exists() else csv_files.get("scenes")

    if args.cutouts:
        cutouts_path = args.cutouts
    else:
        cfg_cutouts = _cfg_path(cfg_backend_dir, inputs_cfg.get("cutouts_csv")) if inputs_cfg else None
        cutouts_path = cfg_cutouts if cfg_cutouts and cfg_cutouts.exists() else csv_files.get("cutouts")

    run_scene_perception = _to_bool(execution_cfg.get("run_scene_perception"), True)
    run_scene_perception_gt = _to_bool(execution_cfg.get("run_scene_perception_gt"), False)
    run_scene_understanding = _to_bool(execution_cfg.get("run_scene_understanding"), True)
    run_rule_understanding = _to_bool(execution_cfg.get("run_rule_understanding"), True)
    run_compliance_reasoning = _to_bool(execution_cfg.get("run_compliance_reasoning"), True)
    run_compliance_gt = _to_bool(execution_cfg.get("run_compliance_gt"), False)
    seed = randomness_cfg.get("seed")
    try:
        seed = int(seed) if seed is not None else None
    except (TypeError, ValueError):
        seed = None

    output_sqlite = args.output_sqlite
    if output_sqlite == Path(__file__).parent / "unified_database.db":
        cfg_output = _cfg_path(cfg_backend_dir, backend_cfg.get("output_sqlite")) if backend_cfg else None
        if cfg_output:
            output_sqlite = cfg_output

    print("=" * 60)
    print("Question Generation from Templates and Rules")
    print("=" * 60)
    if args.config and args.config.exists():
        print(f"Config: {args.config}")
    print(f"Templates: {templates_path.name}")
    print(f"Rules: {rules_path.name}")
    if figures_path:
        print(f"Figures: {figures_path.name}")
    if scene_scenes_path and scene_scenes_path.exists():
        print(f"Scene Scenes: {scene_scenes_path.name}")
    if cutouts_path and cutouts_path.exists():
        print(f"Cutouts: {cutouts_path.name}")
    print(f"Output: {output_sqlite}")
    print()

    create_database(
        templates_path=templates_path,
        rules_path=rules_path,
        output_sqlite=output_sqlite,
        figures_path=figures_path,
        scene_scenes_path=scene_scenes_path,
        scenes_dir=cfg_backend_dir / "scenes",
        cutouts_path=cutouts_path,
        run_scene_perception=run_scene_perception,
        run_scene_understanding=run_scene_understanding,
        run_rule_understanding=run_rule_understanding,
        run_compliance_reasoning=run_compliance_reasoning,
        run_scene_perception_gt=run_scene_perception_gt,
        run_compliance_gt=run_compliance_gt,
        seed=seed,
    )


if __name__ == "__main__":
    main()
