from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Rule-only baseline analysis (Claude + GPT).")
    parser.add_argument("--dataset", type=Path, default=base_dir / "3_2_to_3_5_(2)_rule_onl_subset.csv")
    parser.add_argument(
        "--claude-merged-v1", type=Path, default=base_dir / "3_2_to_3_5_rule-only_claude-opus-4.6_merged80_V1.csv"
    )
    parser.add_argument("--claude-main", type=Path, default=base_dir / "3_2_to_3_5_rule-only_claude-opus-4.6.csv")
    parser.add_argument(
        "--claude-preview", type=Path, default=base_dir / "3_2_to_3_5_preview_rule_onl_claude-opus-4.6.csv"
    )
    parser.add_argument("--gpt-main", type=Path, default=base_dir / "3_2_to_3_5_rule-only_gpt-5_2.csv")
    parser.add_argument("--gemini-main", type=Path, default=base_dir / "3_2_to_3_5_clean_rule-only_gemini.csv")
    parser.add_argument(
        "--gemini-preview", type=Path, default=base_dir / "3_2_to_3_5_preview_rule-only_gemini.csv"
    )
    parser.add_argument(
        "--claude-merged-out", type=Path, default=base_dir / "3_2_to_3_5_rule-only_claude-opus-4.6_merged80.csv"
    )
    parser.add_argument(
        "--gemini-merged-out", type=Path, default=base_dir / "3_2_to_3_5_rule-only_gemini_3_pro_prev_merged80.csv"
    )
    parser.add_argument(
        "--metrics-out", type=Path, default=base_dir / "3_2_to_3_5_rule_only_baseline_metrics.csv"
    )
    return parser.parse_args()


def is_filled(value: object) -> bool:
    return value is not None and (not pd.isna(value)) and str(value).strip() != ""


def normalize_ambiguity(value: object) -> str:
    s = "" if value is None or pd.isna(value) else str(value).strip().lower()
    return "yes" if s in {"yes", "y", "true", "1"} else "no"


def normalize_label(value: object) -> str:
    s = "" if value is None or pd.isna(value) else str(value).strip().lower()
    if s in {"yes", "true"}:
        return "yes"
    if s in {"no", "false"}:
        return "no"
    return ""


def extract_pred_yes_no(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    if not text:
        return ""

    at_hits = re.findall(r"@\s*(YES|NO)\b", text, flags=re.IGNORECASE)
    if at_hits:
        return at_hits[-1].lower()

    line_hits = re.findall(r"(?mi)^\s*(yes|no)\s*$", text)
    if line_hits:
        return line_hits[-1].lower()

    t = text.lower().strip()
    if t in {"yes", "no"}:
        return t
    return ""


def f1_for_label(y_true: list[str], y_pred: list[str], label: str) -> float:
    tp = sum((t == label) and (p == label) for t, p in zip(y_true, y_pred))
    fp = sum((t != label) and (p == label) for t, p in zip(y_true, y_pred))
    fn = sum((t == label) and (p != label) for t, p in zip(y_true, y_pred))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0


def macro_f1_binary(y_true: list[str], y_pred: list[str]) -> float:
    return (f1_for_label(y_true, y_pred, "yes") + f1_for_label(y_true, y_pred, "no")) / 2.0


def answer_score(value: object) -> int:
    text = "" if value is None or pd.isna(value) else str(value)
    if not text.strip():
        return 0
    if re.search(r"@\s*(YES|NO)\b", text, flags=re.IGNORECASE):
        return 3
    if re.search(r"(?mi)^\s*(yes|no)\s*$", text):
        return 2
    if re.search(r"\b(yes|no)\b", text, flags=re.IGNORECASE):
        return 1
    return 0


def pick_best_answers(df: pd.DataFrame, source_priority: int) -> pd.DataFrame:
    out = df.copy()
    if "generated_question_id" not in out.columns:
        raise ValueError("Missing generated_question_id in model output table.")
    out["model_answer"] = out.get("model_answer", "").fillna("")
    out["_score"] = out["model_answer"].apply(answer_score)
    out["_src_priority"] = source_priority
    keep_cols = ["generated_question_id", "model_answer", "openrouter_request_id", "error_reason", "model", "_score", "_src_priority"]
    for col in keep_cols:
        if col not in out.columns:
            out[col] = ""
    return out[keep_cols]


def build_model_table(dataset_df: pd.DataFrame, answers_df: pd.DataFrame, fallback_model: str) -> pd.DataFrame:
    ranked = answers_df.sort_values(
        ["generated_question_id", "_score", "_src_priority"], ascending=[True, False, False]
    )
    best = ranked.drop_duplicates(subset=["generated_question_id"], keep="first").copy()
    merge_cols = ["generated_question_id", "model_answer", "openrouter_request_id", "error_reason", "model"]
    merged = dataset_df.merge(best[merge_cols], on="generated_question_id", how="left")
    merged["model"] = merged["model"].fillna("").replace("", fallback_model)
    return merged


def add_section_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["gt_norm"] = out["correct_answer"].apply(normalize_label)
    out["pred_raw"] = out["model_answer"].apply(extract_pred_yes_no)
    out["pred_bin"] = out["pred_raw"].apply(lambda x: x if x in {"yes", "no"} else "no")
    out["_amb_yes"] = out["Ambiguity"].apply(normalize_ambiguity).eq("yes") if "Ambiguity" in out.columns else False
    out["_atomic"] = out["RuleID"].apply(is_filled) if "RuleID" in out.columns else False
    out["_parent"] = out["ParentRuleID"].apply(is_filled) if "ParentRuleID" in out.columns else False
    return out


def compute_metrics(sub: pd.DataFrame) -> dict:
    s = sub[sub["gt_norm"].isin(["yes", "no"])].copy()
    n = len(s)
    if n == 0:
        return {
            "n": 0,
            "gt_yes": 0,
            "gt_no": 0,
            "gt_yes_share": 0.0,
            "gt_no_share": 0.0,
            "label_distribution": "",
            "pred_yes": 0,
            "pred_no": 0,
            "pred_invalid": 0,
            "answered_total": 0,
            "n_correct": 0,
            "accuracy": 0.0,
            "f1_macro": 0.0,
            "f1_yes": 0.0,
            "f1_no": 0.0,
            "majority_baseline": 0.0,
            "baseline_f1_macro": 0.0,
            "baseline_f1_yes": 0.0,
            "baseline_f1_no": 0.0,
            "tp": 0,
            "tn": 0,
            "fp": 0,
            "fn": 0,
        }

    y_true = s["gt_norm"].tolist()
    y_pred = s["pred_bin"].tolist()
    pred_raw = s["pred_raw"].tolist()

    gt_yes = sum(t == "yes" for t in y_true)
    gt_no = sum(t == "no" for t in y_true)
    pred_yes = sum(p == "yes" for p in pred_raw)
    pred_no = sum(p == "no" for p in pred_raw)
    pred_invalid = n - pred_yes - pred_no
    answered_total = pred_yes + pred_no

    tp = sum((t == "yes") and (p == "yes") for t, p in zip(y_true, y_pred))
    tn = sum((t == "no") and (p == "no") for t, p in zip(y_true, y_pred))
    fp = sum((t == "no") and (p == "yes") for t, p in zip(y_true, y_pred))
    fn = sum((t == "yes") and (p == "no") for t, p in zip(y_true, y_pred))

    n_correct = tp + tn
    accuracy = n_correct / n
    f1_yes = f1_for_label(y_true, y_pred, "yes")
    f1_no = f1_for_label(y_true, y_pred, "no")
    f1_macro = (f1_yes + f1_no) / 2.0

    majority_label = "yes" if gt_yes >= gt_no else "no"
    majority_baseline = max(gt_yes, gt_no) / n
    y_base = [majority_label] * n
    baseline_f1_yes = f1_for_label(y_true, y_base, "yes")
    baseline_f1_no = f1_for_label(y_true, y_base, "no")
    baseline_f1_macro = (baseline_f1_yes + baseline_f1_no) / 2.0

    return {
        "n": n,
        "gt_yes": gt_yes,
        "gt_no": gt_no,
        "gt_yes_share": gt_yes / n,
        "gt_no_share": gt_no / n,
        "label_distribution": f"yes:{gt_yes}, no:{gt_no}",
        "pred_yes": pred_yes,
        "pred_no": pred_no,
        "pred_invalid": pred_invalid,
        "answered_total": answered_total,
        "n_correct": n_correct,
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "f1_yes": f1_yes,
        "f1_no": f1_no,
        "majority_baseline": majority_baseline,
        "baseline_f1_macro": baseline_f1_macro,
        "baseline_f1_yes": baseline_f1_yes,
        "baseline_f1_no": baseline_f1_no,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def evaluate_sections(df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    sections = [
        ("Overall", pd.Series([True] * len(df), index=df.index)),
        ("Ambiguity=yes", df["_amb_yes"]),
        ("Ambiguity!=yes", ~df["_amb_yes"]),
        ("RuleType=Parent Rule text", df["_parent"]),
        ("RuleType=Atomic Rule text", df["_atomic"]),
    ]
    rows = []
    for section_name, mask in sections:
        metrics = compute_metrics(df[mask])
        rows.append({"model_name": model_name, "section": section_name, **metrics})
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()

    dataset_df = pd.read_csv(args.gemini_merged_out)
    if "generated_question_id" not in dataset_df.columns:
        raise ValueError("Dataset must contain generated_question_id.")
    dataset_df = dataset_df.drop(
        columns=[c for c in ["model", "openrouter_request_id", "model_answer", "error_reason"] if c in dataset_df.columns]
    )

    claude_merged_v1 = pd.read_csv(args.claude_merged_v1)
    claude_main = pd.read_csv(args.claude_main)
    claude_preview = pd.read_csv(args.claude_preview)
    gpt_main = pd.read_csv(args.gpt_main)
    gemini_main = pd.read_csv(args.gemini_main)
    gemini_preview = pd.read_csv(args.gemini_preview)

    claude_answers = pd.concat(
        [
            pick_best_answers(claude_merged_v1, source_priority=3),
            pick_best_answers(claude_main, source_priority=2),
            pick_best_answers(claude_preview, source_priority=1),
        ],
        ignore_index=True,
    )
    claude_df = build_model_table(dataset_df, claude_answers, fallback_model="anthropic/claude-opus-4.6")
    gpt_df = build_model_table(dataset_df, pick_best_answers(gpt_main, source_priority=2), fallback_model="openai/gpt-5.2")
    gemini_answers = pd.concat(
        [
            pick_best_answers(gemini_main, source_priority=2),
            pick_best_answers(gemini_preview, source_priority=1),
        ],
        ignore_index=True,
    )
    gemini_df = build_model_table(
        dataset_df,
        gemini_answers,
        fallback_model="google/gemini-3.1-pro-preview",
    )

    claude_df = add_section_columns(claude_df)
    gpt_df = add_section_columns(gpt_df)
    gemini_df = add_section_columns(gemini_df)

    if len(claude_df) != len(dataset_df):
        raise ValueError(f"Claude merged rows={len(claude_df)} but dataset rows={len(dataset_df)}.")
    if len(gpt_df) != len(dataset_df):
        raise ValueError(f"GPT merged rows={len(gpt_df)} but dataset rows={len(dataset_df)}.")
    if len(gemini_df) != len(dataset_df):
        raise ValueError(f"Gemini merged rows={len(gemini_df)} but dataset rows={len(dataset_df)}.")

    args.claude_merged_out.parent.mkdir(parents=True, exist_ok=True)
    cols_to_save = [c for c in dataset_df.columns] + ["model", "openrouter_request_id", "model_answer", "error_reason"]
    for col in cols_to_save:
        if col not in claude_df.columns:
            claude_df[col] = ""
    claude_df[cols_to_save].to_csv(args.claude_merged_out, index=False)
    for col in cols_to_save:
        if col not in gemini_df.columns:
            gemini_df[col] = ""
    args.gemini_merged_out.parent.mkdir(parents=True, exist_ok=True)
    gemini_df[cols_to_save].to_csv(args.gemini_merged_out, index=False)

    report = pd.concat(
        [
            evaluate_sections(claude_df, "Claude Opus"),
            evaluate_sections(gpt_df, "GPT-5.2"),
            evaluate_sections(gemini_df, "Gemini 3 Pro Preview"),
        ],
        ignore_index=True,
    )
    report = report.sort_values(["section", "model_name"]).reset_index(drop=True)

    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.metrics_out, index=False)

    float_cols = [
        "accuracy",
        "f1_macro",
        "f1_yes",
        "f1_no",
        "majority_baseline",
        "baseline_f1_macro",
        "baseline_f1_yes",
        "baseline_f1_no",
        "gt_yes_share",
        "gt_no_share",
    ]
    print(
        report.assign(**{c: report[c].map(lambda v: f"{v:.4f}") for c in float_cols}).to_string(index=False)
    )
    print(f"\nSaved Claude merged table: {args.claude_merged_out}")
    print(f"Saved Gemini merged table: {args.gemini_merged_out}")
    print(f"Saved rule-only baseline metrics: {args.metrics_out}")


if __name__ == "__main__":
    main()
