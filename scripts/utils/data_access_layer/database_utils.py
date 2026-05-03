from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd


def setup_database(output_sqlite: Path) -> sqlite3.Connection:
    output_sqlite.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(output_sqlite)


def save_dataframe_to_db(df: pd.DataFrame, table_name: str, conn: sqlite3.Connection) -> None:
    df.to_sql(table_name, conn, if_exists="replace", index=False)


def print_database_summary(conn: sqlite3.Connection) -> None:
    print("\n" + "=" * 60)
    print("Database Summary:")
    print("=" * 60)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    for (table_name,) in tables:
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        count = cursor.fetchone()[0]
        print(f"  {table_name}: {count} rows")
    print("=" * 60)
