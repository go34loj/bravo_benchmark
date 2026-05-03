import sqlite3
from pathlib import Path

import pandas as pd


def _normalize_scene_id(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).strip()
    if not text:
        return ""
    try:
        num = float(text)
        if num.is_integer():
            return str(int(num))
        return str(num)
    except (ValueError, TypeError):
        return text


def _load_db_pairs(db_path: Path) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql(
            "SELECT * FROM generated_scene_understanding_questions",
            conn,
        )
    finally:
        conn.close()
    df["scene_id_norm"] = df["scene_id"].apply(_normalize_scene_id)
    df["main_entity_norm"] = df["main_entity"].fillna("").astype(str)
    df["related_entity_norm"] = df["related_entity"].fillna("").astype(str)
    return df


def _load_csv_pairs(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.attrs["original_columns"] = list(df.columns)
    required_cols = {"scene_id", "main_entity", "related_entity", "ground_truth_answer"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"CSV missing columns: {sorted(missing_cols)}")
    df["scene_id_norm"] = df["scene_id"].apply(_normalize_scene_id)
    df["main_entity_norm"] = df["main_entity"].fillna("").astype(str)
    df["related_entity_norm"] = df["related_entity"].fillna("").astype(str)
    df["ground_truth_norm"] = (
        df["ground_truth_answer"].fillna("").astype(str).str.strip().str.lower()
    )
    return df


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    db_path = repo_root / "unified_database.db"
    csv_path = repo_root / "notebooks" / "1_2_clean_claude-opus-4.6.csv"

    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    db_df = _load_db_pairs(db_path)
    csv_df = _load_csv_pairs(csv_path)

    db_pairs = set(
        zip(
            db_df["scene_id_norm"],
            db_df["main_entity_norm"],
            db_df["related_entity_norm"],
        )
    )
    csv_pairs = set(
        zip(
            csv_df["scene_id_norm"],
            csv_df["main_entity_norm"],
            csv_df["related_entity_norm"],
        )
    )

    total_rows = len(csv_df)
    matches = 0
    missing_rows = []

    for idx, row in csv_df.iterrows():
        key = (
            row["scene_id_norm"],
            row["main_entity_norm"],
            row["related_entity_norm"],
        )
        if key in db_pairs:
            matches += 1
        else:
            missing_rows.append({
                "csv_row_number": idx + 2,
                "scene_id": row["scene_id_norm"],
                "main_entity": row["main_entity_norm"],
                "related_entity": row["related_entity_norm"],
            })

    db_only = db_df[
        ~db_df[["scene_id_norm", "main_entity_norm", "related_entity_norm"]]
        .apply(tuple, axis=1)
        .isin(csv_pairs)
    ].copy()

    print("=" * 60)
    print("Scene Understanding Quick Check (main_entity + related_entity)")
    print("=" * 60)
    print(f"Total CSV rows checked: {total_rows}")
    print(f"Exact pair matches in DB: {matches}")
    print(f"Missing in DB: {len(missing_rows)}")
    print("=" * 60)

    if missing_rows:
        print("\nMissing pairs (CSV -> DB):")
        for item in missing_rows:
            print(
                f"[MISSING] row={item['csv_row_number']} "
                f"scene_id={item['scene_id']} "
                f"main_entity={item['main_entity']} "
                f"related_entity={item['related_entity']}"
            )

    # Replace non-binary ground_truth rows with DB rows for the same pairs.
    non_binary = csv_df[~csv_df["ground_truth_norm"].isin(["yes", "no"])].copy()
    non_binary_keys = set(
        zip(
            non_binary["scene_id_norm"],
            non_binary["main_entity_norm"],
            non_binary["related_entity_norm"],
        )
    )
    db_for_non_binary = db_df[
        db_df[["scene_id_norm", "main_entity_norm", "related_entity_norm"]]
        .apply(tuple, axis=1)
        .isin(non_binary_keys)
    ].copy()

    db_export = pd.concat([db_only, db_for_non_binary], ignore_index=True)

    # Match CSV schema for the export.
    export_cols = csv_df.attrs.get("original_columns", list(csv_df.columns))
    for col in export_cols:
        if col not in db_export.columns:
            db_export[col] = ""
    db_export = db_export[export_cols]

    output_path = csv_path.parent / "1_2_scene_underst_dataset_missing_items_claude.csv"
    db_export.to_csv(output_path, index=False)
    print(f"\nDB-only + non-binary replacements saved to: {output_path}")


if __name__ == "__main__":
    main()
