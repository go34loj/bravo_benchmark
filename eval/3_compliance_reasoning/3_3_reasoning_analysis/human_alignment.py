from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_AUTOMATIC_CSV = Path(__file__).with_name("3_2_to_3_5_gt_coverage_summary_automatic.csv")
DEFAULT_REVIEW_CSV = Path(__file__).with_name("3_2_to_3_5_gt_coverage_judge_review_LLM.csv")
DEFAULT_OUTPUT_CSV = Path(__file__).with_name("3_2_to_3_5_human_alignment_metrics_LLM.csv")

REVIEW_COLS = [
    "judge_coverage_structure",
    "judge_coverage_quality",
    "judge_best_pred_ids",
    "judge_notes",
]


def _is_blank_series(sr: pd.Series) -> pd.Series:
    return sr.isna() | (sr.astype(str).str.strip() == "")


def _series_or_blank(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        return df[col]
    return pd.Series([""] * len(df), index=df.index)


def _parse_pred_ids(value: Any) -> list[int]:
    s = "" if pd.isna(value) else str(value).strip()
    if not s:
        return []
    out: list[int] = []
    for token in s.split(","):
        t = token.strip().upper()
        if t.startswith("P_"):
            t = t[2:]
        elif t.startswith("P"):
            t = t[1:]
        if t.isdigit():
            out.append(int(t))
    return out


def _parse_id_set(value: Any) -> set[int]:
    return set(_parse_pred_ids(value))


def _jaccard(a: set[int], b: set[int]) -> float:
    if not a and not b:
        return np.nan
    union = a | b
    if not union:
        return np.nan
    return float(len(a & b) / len(union))


def _merge_review_fields(auto_gt_df: pd.DataFrame, review_df: pd.DataFrame) -> pd.DataFrame:
    out = auto_gt_df.copy()
    for c in REVIEW_COLS:
        if c not in out.columns:
            out[c] = ""

    if review_df.empty:
        return out

    prev = review_df.copy()
    for c in REVIEW_COLS:
        if c not in prev.columns:
            prev[c] = ""

    for c in ["generated_question_id", "row_index", "gt_step_id", "gt_step_position"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").astype("Int64")
        if c in prev.columns:
            prev[c] = pd.to_numeric(prev[c], errors="coerce").astype("Int64")

    strict_keys = [
        c for c in ["model_name", "generated_question_id", "row_index", "gt_step_id"]
        if c in out.columns and c in prev.columns
    ]
    if len(strict_keys) >= 2:
        prev_keep = prev[strict_keys + REVIEW_COLS].drop_duplicates(subset=strict_keys)
        merged = out.merge(prev_keep, on=strict_keys, how="left", suffixes=("", "__prev"))
        for c in REVIEW_COLS:
            cp = f"{c}__prev"
            if cp in merged.columns:
                left = merged[c].fillna("").astype(str).str.strip()
                right = merged[cp].fillna("").astype(str).str.strip()
                merged[c] = np.where(left != "", left, right)
                merged.drop(columns=[cp], inplace=True)
        out = merged

    fallback_keys = []
    if "gt_step_position" in out.columns and "gt_step_position" in prev.columns:
        fallback_keys = [
            c for c in ["model_name", "generated_question_id", "gt_step_position"]
            if c in out.columns and c in prev.columns
        ]
    if len(fallback_keys) >= 3:
        prev_fb = prev[fallback_keys + REVIEW_COLS].drop_duplicates(subset=fallback_keys)
        fb = out.merge(prev_fb, on=fallback_keys, how="left", suffixes=("", "__fb"))
        for c in REVIEW_COLS:
            cfb = f"{c}__fb"
            if cfb in fb.columns:
                left = fb[c].fillna("").astype(str).str.strip()
                right = fb[cfb].fillna("").astype(str).str.strip()
                fb[c] = np.where(left != "", left, right)
                fb.drop(columns=[cfb], inplace=True)
        out = fb

    return out


def _reviewed_mask(df_gt: pd.DataFrame) -> pd.Series:
    if df_gt.empty:
        return pd.Series([], dtype=bool)
    present_review_cols = [c for c in REVIEW_COLS if c in df_gt.columns]
    if not present_review_cols:
        return pd.Series([False] * len(df_gt), index=df_gt.index)
    mask = pd.Series([False] * len(df_gt), index=df_gt.index)
    for c in present_review_cols:
        mask = mask | (~_is_blank_series(df_gt[c]))
    return mask


def _alignment_summary(df_gt: pd.DataFrame) -> dict[str, float | int]:
    if df_gt.empty:
        return {
            "matched_id_overlap": np.nan,
            "unchanged_percentage": np.nan,
            "n_reviewed_steps": 0,
            "n_reviewed_questions": 0,
        }

    reviewed = _reviewed_mask(df_gt)
    n_reviewed_steps = int(reviewed.sum())
    if n_reviewed_steps == 0:
        return {
            "matched_id_overlap": np.nan,
            "unchanged_percentage": np.nan,
            "n_reviewed_steps": 0,
            "n_reviewed_questions": 0,
        }

    sub = df_gt[reviewed].copy()

    human_struct = _series_or_blank(sub, "final_coverage_structure").fillna("").astype(str).str.strip().str.lower()
    human_quality = _series_or_blank(sub, "final_coverage_quality").fillna("").astype(str).str.strip().str.lower()
    judge_struct = _series_or_blank(sub, "judge_coverage_structure").fillna("").astype(str).str.strip().str.lower()
    judge_quality = _series_or_blank(sub, "judge_coverage_quality").fillna("").astype(str).str.strip().str.lower()

    # Keep representation comparable with final_* convention from human table.
    judge_quality = pd.Series(
        np.where(judge_struct == "missing", "", judge_quality),
        index=sub.index,
    )

    unchanged_parts: list[pd.Series] = []
    if "final_coverage_structure" in sub.columns and "judge_coverage_structure" in sub.columns:
        unchanged_parts.append((human_struct == judge_struct).astype(float))
    if "final_coverage_quality" in sub.columns and "judge_coverage_quality" in sub.columns:
        unchanged_parts.append((human_quality == judge_quality).astype(float))

    unchanged_pct = np.nan
    if unchanged_parts:
        row_level = pd.concat(unchanged_parts, axis=1).mean(axis=1)
        unchanged_pct = float(row_level.mean())

    overlaps: list[float] = []
    if {"final_best_pred_ids", "judge_best_pred_ids"}.issubset(sub.columns):
        for _, row in sub.iterrows():
            a = _parse_id_set(row.get("final_best_pred_ids", ""))
            b = _parse_id_set(row.get("judge_best_pred_ids", ""))
            j = _jaccard(a, b)
            if not pd.isna(j):
                overlaps.append(float(j))
    overlap_mean = float(np.mean(overlaps)) if overlaps else np.nan

    if "sample_key" in sub.columns:
        n_reviewed_questions = int(sub["sample_key"].nunique())
    elif "generated_question_id" in sub.columns:
        n_reviewed_questions = int(sub["generated_question_id"].nunique())
    else:
        n_reviewed_questions = 0

    return {
        "matched_id_overlap": overlap_mean,
        "unchanged_percentage": unchanged_pct,
        "n_reviewed_steps": n_reviewed_steps,
        "n_reviewed_questions": n_reviewed_questions,
    }


def compute_human_alignment_metrics(
    automatic_csv: Path = DEFAULT_AUTOMATIC_CSV,
    review_csv: Path = DEFAULT_REVIEW_CSV,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    auto_df = pd.read_csv(automatic_csv)
    review_df = pd.read_csv(review_csv)

    merged = _merge_review_fields(auto_df, review_df)

    rows: list[dict[str, Any]] = []
    model_names = sorted(merged["model_name"].dropna().astype(str).unique().tolist()) if "model_name" in merged.columns else ["unknown"]
    for model_name in model_names:
        sub_model = merged[merged["model_name"].astype(str) == model_name].copy() if "model_name" in merged.columns else merged.copy()
        if "gt_step_position" in sub_model.columns:
            step_values = sorted(pd.to_numeric(sub_model["gt_step_position"], errors="coerce").dropna().astype(int).unique().tolist())
        else:
            step_values = []

        for step_idx in step_values:
            sub_step = sub_model[pd.to_numeric(sub_model["gt_step_position"], errors="coerce") == step_idx].copy()
            a = _alignment_summary(sub_step)
            rows.append(
                {
                    "model_name": model_name,
                    "scope": f"step_{step_idx}",
                    "gt_step_position": step_idx,
                    "Matched ID Overlap": a.get("matched_id_overlap"),
                    "Unchanged Percentage": a.get("unchanged_percentage"),
                    "n_reviewed_steps": int(a.get("n_reviewed_steps", 0) or 0),
                    "n_questions": int(a.get("n_reviewed_questions", 0) or 0),
                }
            )

        overall = _alignment_summary(sub_model)
        rows.append(
            {
                "model_name": model_name,
                "scope": "overall",
                "gt_step_position": np.nan,
                "Matched ID Overlap": overall.get("matched_id_overlap"),
                "Unchanged Percentage": overall.get("unchanged_percentage"),
                "n_reviewed_steps": int(overall.get("n_reviewed_steps", 0) or 0),
                "n_questions": int(overall.get("n_reviewed_questions", 0) or 0),
            }
        )

    alignment_df = pd.DataFrame(rows)
    return alignment_df, merged


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute judge-vs-human alignment metrics from automatic GT coverage and judge-review tables.",
    )
    parser.add_argument("--automatic-csv", type=Path, default=DEFAULT_AUTOMATIC_CSV)
    parser.add_argument("--review-csv", type=Path, default=DEFAULT_REVIEW_CSV)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    alignment_df, _ = compute_human_alignment_metrics(
        automatic_csv=args.automatic_csv,
        review_csv=args.review_csv,
    )
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    alignment_df.to_csv(args.output_csv, index=False, encoding="utf-8-sig")

    pd.set_option("display.max_columns", 50)
    print(f"Saved: {args.output_csv}")
    print(alignment_df.to_string(index=False))


if __name__ == "__main__":
    main()
