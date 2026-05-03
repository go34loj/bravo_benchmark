from __future__ import annotations

import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))
import sqlite3
from pathlib import Path
from typing import Iterable, Sequence


BASE_TABLE_DDL: dict[str, str] = {
    "templates": """
        CREATE TABLE IF NOT EXISTS templates (
            template_id TEXT PRIMARY KEY,
            layer_id TEXT,
            benchmark_layer TEXT,
            subtask TEXT,
            answer_type TEXT,
            answer_template TEXT,
            question_template TEXT,
            context TEXT,
            requires_rule TEXT,
            requires_figure TEXT,
            requires_view TEXT,
            metrics TEXT
        )
    """,
    "rules": """
        CREATE TABLE IF NOT EXISTS rules (
            rule_id TEXT PRIMARY KEY,
            parent_rule_id TEXT,
            parent_rule_text TEXT,
            rule_text_atomic TEXT,
            classification TEXT,
            ambiguity TEXT,
            benchmark_layer TEXT,
            figure_required TEXT,
            figure_id TEXT
        )
    """,
    "viewpoint_scenes": """
        CREATE TABLE IF NOT EXISTS viewpoint_scenes (
            scene_id TEXT PRIMARY KEY,
            cutout_id TEXT,
            rule_id TEXT,
            template_id TEXT,
            file_path TEXT,
            caption TEXT,
            "not compliant" TEXT,
            "not sufficient" TEXT,
            object_type TEXT,
            feature_name TEXT,
            location_context TEXT,
            space_naming TEXT,
            material TEXT,
            text TEXT,
            "multi-view_dimensions" TEXT,
            "spatial relations" TEXT
        )
    """,
    "rule_figures": """
        CREATE TABLE IF NOT EXISTS rule_figures (
            figure_id TEXT PRIMARY KEY,
            file_path TEXT,
            source_ref TEXT,
            caption TEXT
        )
    """,
    "cutout": """
        CREATE TABLE IF NOT EXISTS cutout (
            cutout_id TEXT PRIMARY KEY,
            file_path TEXT,
            cutout_title TEXT,
            parent_rule_id TEXT,
            compliant TEXT
        )
    """,
}


GENERATED_TABLE_DDL: dict[str, str] = {
    "generated_scene_questions": """
        CREATE TABLE IF NOT EXISTS generated_scene_questions (
            generated_question_id INTEGER PRIMARY KEY,
            template_id TEXT NOT NULL,
            scene_id TEXT NOT NULL,
            question_text TEXT,
            file_path TEXT,
            answer_type TEXT,
            ground_truth_answer TEXT,
            FOREIGN KEY (template_id) REFERENCES templates(template_id),
            FOREIGN KEY (scene_id) REFERENCES viewpoint_scenes(scene_id)
        )
    """,
    "generated_scene_understanding_questions": """
        CREATE TABLE IF NOT EXISTS generated_scene_understanding_questions (
            generated_question_id INTEGER PRIMARY KEY,
            template_id TEXT,
            scene_id TEXT,
            question_text TEXT,
            file_path TEXT,
            answer_type TEXT,
            ground_truth_answer TEXT,
            relation_type TEXT,
            FOREIGN KEY (template_id) REFERENCES templates(template_id),
            FOREIGN KEY (scene_id) REFERENCES viewpoint_scenes(scene_id)
        )
    """,
    "generated_questions": """
        CREATE TABLE IF NOT EXISTS generated_questions (
            generated_question_id INTEGER PRIMARY KEY,
            template_id TEXT,
            rule_id TEXT,
            parent_rule_id TEXT,
            question_text TEXT,
            answer_type TEXT,
            ground_truth_answer TEXT,
            figure_id TEXT,
            figure_path TEXT,
            figure_caption TEXT,
            FOREIGN KEY (template_id) REFERENCES templates(template_id),
            FOREIGN KEY (rule_id) REFERENCES rules(rule_id),
            FOREIGN KEY (figure_id) REFERENCES rule_figures(figure_id)
        )
    """,
    "generated_compliance_questions": """
        CREATE TABLE IF NOT EXISTS generated_compliance_questions (
            generated_question_id INTEGER PRIMARY KEY,
            scene_id TEXT,
            file_path TEXT,
            template_id TEXT,
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
            figure_path TEXT,
            answer_type TEXT,
            ground_truth_answer TEXT,
            FOREIGN KEY (template_id) REFERENCES templates(template_id),
            FOREIGN KEY (scene_id) REFERENCES viewpoint_scenes(scene_id),
            FOREIGN KEY (rule_id) REFERENCES rules(rule_id)
        )
    """,
}


INDEXES: Sequence[str] = (
    "CREATE INDEX IF NOT EXISTS idx_rules_parent_rule_id ON rules(parent_rule_id)",
    "CREATE INDEX IF NOT EXISTS idx_viewpoint_scenes_cutout_id ON viewpoint_scenes(cutout_id)",
    "CREATE INDEX IF NOT EXISTS idx_generated_scene_questions_template_id ON generated_scene_questions(template_id)",
    "CREATE INDEX IF NOT EXISTS idx_generated_scene_questions_scene_id ON generated_scene_questions(scene_id)",
    "CREATE INDEX IF NOT EXISTS idx_generated_scene_understanding_questions_template_id ON generated_scene_understanding_questions(template_id)",
    "CREATE INDEX IF NOT EXISTS idx_generated_scene_understanding_questions_scene_id ON generated_scene_understanding_questions(scene_id)",
    "CREATE INDEX IF NOT EXISTS idx_generated_questions_template_id ON generated_questions(template_id)",
    "CREATE INDEX IF NOT EXISTS idx_generated_questions_rule_id ON generated_questions(rule_id)",
    "CREATE INDEX IF NOT EXISTS idx_generated_questions_parent_rule_id ON generated_questions(parent_rule_id)",
    "CREATE INDEX IF NOT EXISTS idx_generated_questions_figure_id ON generated_questions(figure_id)",
    "CREATE INDEX IF NOT EXISTS idx_generated_compliance_questions_template_id ON generated_compliance_questions(template_id)",
    "CREATE INDEX IF NOT EXISTS idx_generated_compliance_questions_scene_id ON generated_compliance_questions(scene_id)",
    "CREATE INDEX IF NOT EXISTS idx_generated_compliance_questions_rule_id ON generated_compliance_questions(rule_id)",
    "CREATE INDEX IF NOT EXISTS idx_generated_compliance_questions_parent_rule_id ON generated_compliance_questions(parent_rule_id)",
)


REQUIRED_PK: dict[str, Sequence[str]] = {
    "templates": ("template_id",),
    "rules": ("rule_id",),
    "viewpoint_scenes": ("scene_id",),
    "rule_figures": ("figure_id",),
    "cutout": ("cutout_id",),
    "generated_scene_questions": ("generated_question_id",),
    "generated_scene_understanding_questions": ("generated_question_id",),
    "generated_questions": ("generated_question_id",),
    "generated_compliance_questions": ("generated_question_id",),
}


TEXT_ID_COLUMNS: set[str] = {
    "template_id",
    "rule_id",
    "parent_rule_id",
    "scene_id",
    "figure_id",
    "cutout_id",
}


def setup_connection(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def create_base_tables(conn: sqlite3.Connection) -> None:
    _ensure_table_group(conn, BASE_TABLE_DDL)


def create_generated_tables(conn: sqlite3.Connection) -> None:
    _ensure_table_group(conn, GENERATED_TABLE_DDL)


def ensure_schema(conn: sqlite3.Connection) -> None:
    create_base_tables(conn)
    create_generated_tables(conn)
    for sql in INDEXES:
        conn.execute(sql)
    conn.commit()


def validate_integrity(conn: sqlite3.Connection) -> None:
    violations = conn.execute("PRAGMA foreign_key_check;").fetchall()
    if violations:
        preview = "\n".join(
            f"- table={row[0]}, rowid={row[1]}, parent={row[2]}, fk_index={row[3]}"
            for row in violations[:20]
        )
        more = f"\n... and {len(violations) - 20} more" if len(violations) > 20 else ""
        raise ValueError(f"Foreign key violations detected:\n{preview}{more}")

    for table, pk_cols in REQUIRED_PK.items():
        if not _table_exists(conn, table):
            continue
        _assert_pk_uniqueness(conn, table, pk_cols)


def _ensure_table_group(conn: sqlite3.Connection, ddl_map: dict[str, str]) -> None:
    for table_name, ddl in ddl_map.items():
        if _table_exists(conn, table_name) and not _has_required_pk(conn, table_name, REQUIRED_PK[table_name]):
            _migrate_table_with_constraints(conn, table_name, ddl)
        else:
            conn.execute(ddl)


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name = ?",
        (table_name,),
    ).fetchone()
    return row is not None


def _has_required_pk(conn: sqlite3.Connection, table_name: str, pk_cols: Sequence[str]) -> bool:
    rows = conn.execute(f'PRAGMA table_info("{table_name}")').fetchall()
    current_pk = [row[1] for row in sorted(rows, key=lambda r: r[5]) if row[5] > 0]
    return tuple(current_pk) == tuple(pk_cols)


def _migrate_table_with_constraints(conn: sqlite3.Connection, table_name: str, ddl: str) -> None:
    new_table = f"new_{table_name}"
    old_table = f"old_{table_name}"
    conn.execute("PRAGMA foreign_keys = OFF;")
    try:
        conn.execute(f'DROP TABLE IF EXISTS "{new_table}"')
        conn.execute(ddl.replace(f"CREATE TABLE IF NOT EXISTS {table_name}", f'CREATE TABLE "{new_table}"'))

        old_cols = _table_columns(conn, table_name)
        new_cols = _table_columns(conn, new_table)
        common_cols = [col for col in old_cols if col in new_cols]
        if common_cols:
            select_expr = ", ".join(_copy_expr(col) for col in common_cols)
            target_cols = ", ".join(_q(col) for col in common_cols)
            conn.execute(
                f'INSERT INTO "{new_table}" ({target_cols}) '
                f'SELECT {select_expr} FROM "{table_name}"'
            )

        conn.execute(f'ALTER TABLE "{table_name}" RENAME TO "{old_table}"')
        conn.execute(f'ALTER TABLE "{new_table}" RENAME TO "{table_name}"')
        conn.execute(f'DROP TABLE "{old_table}"')
        conn.commit()
    finally:
        conn.execute("PRAGMA foreign_keys = ON;")


def _table_columns(conn: sqlite3.Connection, table_name: str) -> list[str]:
    rows = conn.execute(f'PRAGMA table_info("{table_name}")').fetchall()
    return [row[1] for row in rows]


def _copy_expr(col: str) -> str:
    qcol = _q(col)
    if col == "generated_question_id":
        return f"CAST(NULLIF(TRIM({qcol}), '') AS INTEGER) AS {qcol}"
    if col in TEXT_ID_COLUMNS:
        return (
            f"CASE WHEN {qcol} IS NULL OR TRIM(CAST({qcol} AS TEXT)) = '' "
            f"THEN NULL ELSE TRIM(CAST({qcol} AS TEXT)) END AS {qcol}"
        )
    return qcol


def _assert_pk_uniqueness(conn: sqlite3.Connection, table_name: str, pk_cols: Iterable[str]) -> None:
    cols = list(pk_cols)
    if not cols:
        return
    cols_expr = ", ".join(_q(c) for c in cols)
    row = conn.execute(
        f'SELECT COUNT(*) FROM (SELECT {cols_expr}, COUNT(*) c FROM "{table_name}" '
        f"GROUP BY {cols_expr} HAVING c > 1)"
    ).fetchone()
    dup_count = int(row[0]) if row else 0
    if dup_count > 0:
        raise ValueError(f"Primary key uniqueness violation in table '{table_name}': {dup_count} duplicate key groups")


def _q(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


