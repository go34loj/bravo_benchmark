#!/usr/bin/env python3
"""
Dataset quality audit for CSV files.

Usage:
  python qualitycheck.py --input path/to/data.csv --write-csv --output-dir qc_outputs
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

try:
    from rapidfuzz import fuzz  # type: ignore

    RAPIDFUZZ_AVAILABLE = True
except Exception:
    from difflib import SequenceMatcher

    RAPIDFUZZ_AVAILABLE = False


COLUMN_HINTS: Dict[str, List[str]] = {
    "text": ["question_text", "question", "prompt", "input", "query"],
    "label": ["ground_truth_answer", "label", "answer", "target", "gold"],
    "answer_type": ["answer_type", "task_type", "type"],
    "options": ["options", "choices", "answer_choices", "answer_options", "choice", "mc_options"],
    "parent_rule": ["parent_rule", "rule_parent", "parent"],
    "child_rule": ["child_rule", "rule_child", "child"],
    "object": ["object_type_filled", "object_type", "object"],
    "feature": ["feature_name_filled", "feature_name", "feature", "attribute"],
    "id": ["id", "question_id", "generated_question_id", "uid"],
}

BOOL_TRUE = {"yes", "true", "1", "y", "t"}
BOOL_FALSE = {"no", "false", "0", "n", "f"}

ANSWER_INSTRUCTION_RE = re.compile(
    r"(?:^|\n)\s*answer\s+.*$", flags=re.IGNORECASE | re.DOTALL
)

TEMPLATE_ARTIFACT_RE = re.compile(r"(\{\{.*?\}\}|\{.*?\}|<.*?>|\[\[.*?\]\])")

PUNCT_REPEAT_RE = re.compile(r"[?!]{3,}")


@dataclass
class QCConfig:
    text_col: Optional[str] = None
    label_col: Optional[str] = None
    answer_type_col: Optional[str] = None
    options_col: Optional[str] = None
    parent_rule_col: Optional[str] = None
    child_rule_col: Optional[str] = None
    object_col: Optional[str] = None
    feature_col: Optional[str] = None
    id_col: Optional[str] = None

    near_dup_threshold: int = 90
    max_pairwise_rows: int = 2000
    max_near_dup_pairs: int = 50000
    min_question_len: int = 12
    top_template_threshold: float = 0.2
    label_imbalance_threshold: float = 0.9
    purity_threshold: float = 0.95
    min_group_size: int = 5
    label_word_leak_threshold: float = 0.6


def pick_column(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    for name in candidates:
        if name in df.columns:
            return name
    for col in df.columns:
        col_l = col.lower()
        for name in candidates:
            if name in col_l:
                return col
    return None


def resolve_columns(df: pd.DataFrame, cfg: QCConfig) -> QCConfig:
    if cfg.text_col is None:
        cfg.text_col = pick_column(df, COLUMN_HINTS["text"])
    if cfg.label_col is None:
        cfg.label_col = pick_column(df, COLUMN_HINTS["label"])
    if cfg.answer_type_col is None:
        cfg.answer_type_col = pick_column(df, COLUMN_HINTS["answer_type"])
    if cfg.options_col is None:
        cfg.options_col = pick_column(df, COLUMN_HINTS["options"])
    if cfg.parent_rule_col is None:
        cfg.parent_rule_col = pick_column(df, COLUMN_HINTS["parent_rule"])
    if cfg.child_rule_col is None:
        cfg.child_rule_col = pick_column(df, COLUMN_HINTS["child_rule"])
    if cfg.object_col is None:
        cfg.object_col = pick_column(df, COLUMN_HINTS["object"])
    if cfg.feature_col is None:
        cfg.feature_col = pick_column(df, COLUMN_HINTS["feature"])
    if cfg.id_col is None:
        cfg.id_col = pick_column(df, COLUMN_HINTS["id"])
    return cfg


def is_empty(value: object) -> bool:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    return False


def norm_text(text: object) -> str:
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    s = str(text).strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def strip_answer_instructions(text: str) -> str:
    return ANSWER_INSTRUCTION_RE.sub("", text)


def similarity(a: str, b: str) -> float:
    if RAPIDFUZZ_AVAILABLE:
        return float(fuzz.token_set_ratio(a, b))
    return SequenceMatcher(None, a, b).ratio() * 100.0


def parse_options(raw: object) -> List[str]:
    if is_empty(raw):
        return []
    s = str(raw).strip()
    if not s:
        return []
    if (s.startswith("[") and s.endswith("]")) or (s.startswith("{") and s.endswith("}")):
        try:
            parsed = json.loads(s)
            if isinstance(parsed, dict):
                return [str(v).strip() for v in parsed.values()]
            if isinstance(parsed, list):
                return [str(v).strip() for v in parsed]
        except Exception:
            pass
    if "\n" in s:
        parts = s.splitlines()
    elif "|" in s:
        parts = s.split("|")
    elif ";" in s:
        parts = s.split(";")
    elif "," in s:
        parts = s.split(",")
    else:
        parts = [s]
    return [p.strip() for p in parts if p.strip()]


def display_columns(df: pd.DataFrame, cfg: QCConfig) -> List[str]:
    preferred = [
        cfg.id_col,
        cfg.text_col,
        cfg.label_col,
        cfg.answer_type_col,
        cfg.options_col,
        cfg.object_col,
        cfg.feature_col,
        cfg.parent_rule_col,
        cfg.child_rule_col,
    ]
    cols = [c for c in preferred if c and c in df.columns]
    if not cols:
        cols = list(df.columns[:6])
    return cols


def check_empty_cells(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    empty_mask = df.applymap(is_empty)
    rows = df[empty_mask.any(axis=1)].copy()
    by_col = empty_mask.sum().sort_values(ascending=False)
    return rows, by_col


def check_duplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    dup_mask = df.duplicated(keep=False)
    return df[dup_mask].copy()


def check_near_duplicate_texts(
    df: pd.DataFrame,
    text_col: Optional[str],
    cfg: QCConfig,
) -> pd.DataFrame:
    if text_col is None or text_col not in df.columns:
        return pd.DataFrame()
    texts = df[text_col].fillna("").astype(str).tolist()
    n = len(texts)
    if n > cfg.max_pairwise_rows:
        return pd.DataFrame()
    pairs: List[Dict[str, object]] = []
    normalized = [norm_text(t) for t in texts]
    for i in range(n):
        a = normalized[i]
        if not a:
            continue
        for j in range(i + 1, n):
            b = normalized[j]
            if not b:
                continue
            if abs(len(a) - len(b)) > 200:
                continue
            score = similarity(a, b)
            if score >= cfg.near_dup_threshold:
                pairs.append(
                    {
                        "index_a": i,
                        "index_b": j,
                        "score": round(score, 2),
                        "text_a": texts[i],
                        "text_b": texts[j],
                    }
                )
            if len(pairs) >= cfg.max_near_dup_pairs:
                return pd.DataFrame(pairs)
    return pd.DataFrame(pairs)


def check_inconsistent_labels_exact(
    df: pd.DataFrame, text_col: Optional[str], label_col: Optional[str]
) -> pd.DataFrame:
    if text_col is None or label_col is None:
        return pd.DataFrame()
    if text_col not in df.columns or label_col not in df.columns:
        return pd.DataFrame()
    norm = df[text_col].map(norm_text)
    label = df[label_col].map(norm_text)
    tmp = df.copy()
    tmp["_norm_text"] = norm
    tmp["_norm_label"] = label
    label_counts = tmp.groupby("_norm_text")["_norm_label"].nunique()
    inconsistent_texts = label_counts[label_counts > 1].index
    return tmp[tmp["_norm_text"].isin(inconsistent_texts)].drop(columns=["_norm_text", "_norm_label"])


def check_inconsistent_labels_near(
    df: pd.DataFrame,
    near_dups: pd.DataFrame,
    label_col: Optional[str],
) -> pd.DataFrame:
    if near_dups.empty or label_col is None or label_col not in df.columns:
        return pd.DataFrame()
    rows: List[Dict[str, object]] = []
    for _, row in near_dups.iterrows():
        i = int(row["index_a"])
        j = int(row["index_b"])
        label_a = df.iloc[i][label_col]
        label_b = df.iloc[j][label_col]
        if norm_text(label_a) and norm_text(label_b) and norm_text(label_a) != norm_text(label_b):
            rows.append(
                {
                    "index_a": i,
                    "index_b": j,
                    "label_a": label_a,
                    "label_b": label_b,
                    "score": row["score"],
                    "text_a": row["text_a"],
                    "text_b": row["text_b"],
                }
            )
    return pd.DataFrame(rows)


def check_malformed_questions(
    df: pd.DataFrame, text_col: Optional[str], cfg: QCConfig
) -> pd.DataFrame:
    if text_col is None or text_col not in df.columns:
        return pd.DataFrame()
    issues: List[Dict[str, object]] = []
    for idx, raw in df[text_col].items():
        if is_empty(raw):
            continue
        text = str(raw).strip()
        problems: List[str] = []
        if len(text) < cfg.min_question_len:
            problems.append("too_short")
        if "?" not in text:
            problems.append("missing_question_mark")
        if TEMPLATE_ARTIFACT_RE.search(text):
            problems.append("template_artifact")
        if PUNCT_REPEAT_RE.search(text):
            problems.append("excessive_punctuation")
        if text.endswith("..."):
            problems.append("truncated_ellipsis")
        if re.search(r"\b(TODO|FIXME|TEMPLATE)\b", text, flags=re.IGNORECASE):
            problems.append("placeholder_tokens")
        if re.search(r"\bAnswer\b", text, flags=re.IGNORECASE) and "?" not in text.splitlines()[0]:
            problems.append("question_instruction_mixed")
        if problems:
            issues.append({"index": idx, "problem": ";".join(sorted(set(problems))), text_col: raw})
    return pd.DataFrame(issues)


def check_answer_format(
    df: pd.DataFrame,
    cfg: QCConfig,
) -> pd.DataFrame:
    if cfg.label_col is None or cfg.label_col not in df.columns:
        return pd.DataFrame()
    if cfg.answer_type_col is None or cfg.answer_type_col not in df.columns:
        return pd.DataFrame()
    issues: List[Dict[str, object]] = []
    for idx, row in df.iterrows():
        answer = row[cfg.label_col]
        answer_type = row[cfg.answer_type_col]
        if is_empty(answer_type) or is_empty(answer):
            continue
        atype = norm_text(answer_type)
        ans = norm_text(answer)
        if atype in {"bool", "boolean", "yes_no", "binary"}:
            if ans not in BOOL_TRUE | BOOL_FALSE:
                issues.append({"index": idx, "problem": "invalid_bool", cfg.label_col: answer, cfg.answer_type_col: answer_type})
        elif atype in {"int", "integer", "float", "number", "numeric"}:
            try:
                float(str(answer))
            except Exception:
                issues.append({"index": idx, "problem": "invalid_numeric", cfg.label_col: answer, cfg.answer_type_col: answer_type})
        elif atype in {"choice", "multiple_choice", "mc", "select"}:
            if cfg.options_col and cfg.options_col in df.columns:
                options = parse_options(row[cfg.options_col])
                if options and str(answer).strip() not in options:
                    issues.append(
                        {
                            "index": idx,
                            "problem": "answer_not_in_options",
                            cfg.label_col: answer,
                            cfg.options_col: row[cfg.options_col],
                        }
                    )
        elif atype in {"text", "freeform", "string"}:
            if is_empty(answer):
                issues.append({"index": idx, "problem": "empty_text_answer"})
    return pd.DataFrame(issues)


def check_question_answer_type_mismatch(
    df: pd.DataFrame, cfg: QCConfig
) -> pd.DataFrame:
    if cfg.text_col is None or cfg.answer_type_col is None:
        return pd.DataFrame()
    if cfg.text_col not in df.columns or cfg.answer_type_col not in df.columns:
        return pd.DataFrame()
    issues: List[Dict[str, object]] = []
    for idx, row in df.iterrows():
        text = norm_text(row[cfg.text_col])
        atype = norm_text(row[cfg.answer_type_col])
        if not text or not atype:
            continue
        if "yes or no" in text and atype not in {"bool", "boolean", "yes_no", "binary"}:
            issues.append({"index": idx, "problem": "question_suggests_bool", cfg.answer_type_col: row[cfg.answer_type_col]})
        if text.startswith("how many") and atype not in {"int", "integer", "float", "number", "numeric"}:
            issues.append({"index": idx, "problem": "question_suggests_numeric", cfg.answer_type_col: row[cfg.answer_type_col]})
    return pd.DataFrame(issues)


def check_related_columns(
    df: pd.DataFrame,
    cfg: QCConfig,
) -> pd.DataFrame:
    issues: List[Dict[str, object]] = []
    if cfg.parent_rule_col and cfg.parent_rule_col in df.columns and cfg.child_rule_col and cfg.child_rule_col in df.columns:
        for idx, row in df.iterrows():
            parent = norm_text(row[cfg.parent_rule_col])
            child = norm_text(row[cfg.child_rule_col])
            if parent and child and child not in parent:
                issues.append(
                    {"index": idx, "problem": "child_not_in_parent_rule", cfg.parent_rule_col: row[cfg.parent_rule_col], cfg.child_rule_col: row[cfg.child_rule_col]}
                )
    if cfg.object_col and cfg.object_col in df.columns and cfg.text_col and cfg.text_col in df.columns:
        for idx, row in df.iterrows():
            obj = norm_text(row[cfg.object_col])
            text = norm_text(row[cfg.text_col])
            if obj and text and obj not in text:
                issues.append({"index": idx, "problem": "object_not_mentioned", cfg.object_col: row[cfg.object_col], cfg.text_col: row[cfg.text_col]})
    if cfg.feature_col and cfg.feature_col in df.columns and cfg.text_col and cfg.text_col in df.columns:
        for idx, row in df.iterrows():
            feat = norm_text(row[cfg.feature_col])
            text = norm_text(row[cfg.text_col])
            if feat and text and feat not in text:
                issues.append({"index": idx, "problem": "feature_not_mentioned", cfg.feature_col: row[cfg.feature_col], cfg.text_col: row[cfg.text_col]})
    return pd.DataFrame(issues)


def check_answer_leakage(
    df: pd.DataFrame,
    cfg: QCConfig,
) -> pd.DataFrame:
    if cfg.text_col is None or cfg.label_col is None:
        return pd.DataFrame()
    if cfg.text_col not in df.columns or cfg.label_col not in df.columns:
        return pd.DataFrame()
    issues: List[Dict[str, object]] = []
    for idx, row in df.iterrows():
        answer = norm_text(row[cfg.label_col])
        text_raw = row[cfg.text_col]
        if is_empty(text_raw) or not answer:
            continue
        text = strip_answer_instructions(norm_text(text_raw))
        if answer in BOOL_TRUE | BOOL_FALSE:
            if re.search(r"\b" + re.escape(answer) + r"\b", text):
                issues.append({"index": idx, "problem": "bool_answer_in_prompt", cfg.text_col: row[cfg.text_col], cfg.label_col: row[cfg.label_col]})
        else:
            if len(answer) >= 3 and re.search(r"\b" + re.escape(answer) + r"\b", text):
                issues.append({"index": idx, "problem": "answer_in_prompt", cfg.text_col: row[cfg.text_col], cfg.label_col: row[cfg.label_col]})
    return pd.DataFrame(issues)


def check_template_frequency(
    df: pd.DataFrame, text_col: Optional[str], cfg: QCConfig
) -> Optional[pd.DataFrame]:
    if text_col is None or text_col not in df.columns:
        return None
    norm = df[text_col].map(norm_text)
    counts = norm.value_counts()
    if counts.empty:
        return None
    top_ratio = counts.iloc[0] / max(len(norm), 1)
    if top_ratio < cfg.top_template_threshold:
        return None
    return counts.head(20).reset_index().rename(columns={"index": "normalized_text", text_col: "count"})


def check_label_imbalance(
    df: pd.DataFrame, label_col: Optional[str], cfg: QCConfig
) -> Optional[pd.Series]:
    if label_col is None or label_col not in df.columns:
        return None
    counts = df[label_col].value_counts(dropna=False)
    if counts.empty:
        return None
    top_ratio = counts.iloc[0] / max(counts.sum(), 1)
    if top_ratio >= cfg.label_imbalance_threshold:
        return counts
    return counts


def check_label_words_in_questions(
    df: pd.DataFrame, cfg: QCConfig
) -> Optional[pd.DataFrame]:
    if cfg.text_col is None or cfg.label_col is None:
        return None
    if cfg.text_col not in df.columns or cfg.label_col not in df.columns:
        return None
    labels = (
        df[cfg.label_col]
        .dropna()
        .astype(str)
        .map(str.strip)
        .unique()
        .tolist()
    )
    records: List[Dict[str, object]] = []
    for label in labels:
        label_norm = norm_text(label)
        if len(label_norm) < 3:
            continue
        mask = df[cfg.text_col].fillna("").astype(str).str.lower().str.contains(r"\b" + re.escape(label_norm) + r"\b")
        ratio = mask.mean()
        if ratio >= cfg.label_word_leak_threshold:
            records.append({"label": label, "ratio": round(ratio, 3), "count": int(mask.sum())})
    if not records:
        return None
    return pd.DataFrame(records).sort_values(by="ratio", ascending=False)


def check_answer_position_bias(
    df: pd.DataFrame, cfg: QCConfig
) -> Optional[pd.DataFrame]:
    if cfg.options_col is None or cfg.label_col is None:
        return None
    if cfg.options_col not in df.columns or cfg.label_col not in df.columns:
        return None
    positions: List[int] = []
    for _, row in df.iterrows():
        options = parse_options(row[cfg.options_col])
        if not options:
            continue
        answer = str(row[cfg.label_col]).strip()
        if answer in options:
            positions.append(options.index(answer))
    if not positions:
        return None
    pos_counts = pd.Series(positions).value_counts().sort_index()
    total = pos_counts.sum()
    data = []
    for pos, count in pos_counts.items():
        data.append({"position": int(pos), "count": int(count), "ratio": round(count / total, 3)})
    return pd.DataFrame(data)


def check_label_purity_by_column(
    df: pd.DataFrame, cfg: QCConfig, columns: List[str]
) -> Optional[pd.DataFrame]:
    if cfg.label_col is None or cfg.label_col not in df.columns:
        return None
    records: List[Dict[str, object]] = []
    for col in columns:
        if col not in df.columns:
            continue
        groups = df.groupby(col)[cfg.label_col]
        for value, series in groups:
            if is_empty(value):
                continue
            count = series.shape[0]
            if count < cfg.min_group_size:
                continue
            top = series.value_counts(dropna=False).iloc[0]
            purity = top / count
            if purity >= cfg.purity_threshold:
                dominant = series.value_counts(dropna=False).index[0]
                records.append(
                    {
                        "column": col,
                        "value": value,
                        "count": int(count),
                        "dominant_label": dominant,
                        "purity": round(purity, 3),
                    }
                )
    if not records:
        return None
    return pd.DataFrame(records).sort_values(by="purity", ascending=False)


def print_section(title: str) -> None:
    print("\n" + title)
    print("-" * len(title))


def print_issue_sample(df: pd.DataFrame, cfg: QCConfig, sample_size: int = 5) -> None:
    if df.empty:
        print("No issues found.")
        return
    cols = display_columns(df, cfg)
    sample = df.head(sample_size)
    print(sample[cols].to_string(index=False))


def write_issue_csv(df: pd.DataFrame, output_dir: str, name: str) -> Optional[str]:
    if df.empty:
        return None
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{name}.csv")
    df.to_csv(path, index=False)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit CSV datasets for data quality issues.")
    parser.add_argument("--input", "-i", required=True, help="Path to CSV file.")
    parser.add_argument("--output-dir", "-o", default="qualitycheck_outputs", help="Output directory for CSV reports.")
    parser.add_argument("--write-csv", action="store_true", help="Write CSV files for flagged rows.")
    parser.add_argument("--sample-size", type=int, default=5, help="Sample size to show in console.")
    parser.add_argument("--near-dup-threshold", type=int, default=90, help="Near-duplicate similarity threshold.")
    parser.add_argument("--max-pairwise-rows", type=int, default=2000, help="Max rows for pairwise near-duplicate check.")

    parser.add_argument("--text-col", default=None)
    parser.add_argument("--label-col", default=None)
    parser.add_argument("--answer-type-col", default=None)
    parser.add_argument("--options-col", default=None)
    parser.add_argument("--parent-rule-col", default=None)
    parser.add_argument("--child-rule-col", default=None)
    parser.add_argument("--object-col", default=None)
    parser.add_argument("--feature-col", default=None)
    parser.add_argument("--id-col", default=None)

    args = parser.parse_args()

    df = pd.read_csv(args.input, dtype=object)

    cfg = QCConfig(
        text_col=args.text_col,
        label_col=args.label_col,
        answer_type_col=args.answer_type_col,
        options_col=args.options_col,
        parent_rule_col=args.parent_rule_col,
        child_rule_col=args.child_rule_col,
        object_col=args.object_col,
        feature_col=args.feature_col,
        id_col=args.id_col,
        near_dup_threshold=args.near_dup_threshold,
        max_pairwise_rows=args.max_pairwise_rows,
    )
    cfg = resolve_columns(df, cfg)

    print("Quality Check Report")
    print("====================")
    print(f"Rows: {len(df)}  Columns: {len(df.columns)}")
    print("Detected columns:")
    for name, col in [
        ("text", cfg.text_col),
        ("label", cfg.label_col),
        ("answer_type", cfg.answer_type_col),
        ("options", cfg.options_col),
        ("parent_rule", cfg.parent_rule_col),
        ("child_rule", cfg.child_rule_col),
        ("object", cfg.object_col),
        ("feature", cfg.feature_col),
        ("id", cfg.id_col),
    ]:
        print(f"  {name}: {col}")

    issues: Dict[str, pd.DataFrame] = {}
    summaries: Dict[str, pd.DataFrame] = {}

    empty_rows, empty_by_col = check_empty_cells(df)
    issues["empty_cells"] = empty_rows

    duplicates = check_duplicate_rows(df)
    issues["duplicate_rows"] = duplicates

    near_dups = check_near_duplicate_texts(df, cfg.text_col, cfg)
    issues["near_duplicate_questions"] = near_dups

    inconsistent_exact = check_inconsistent_labels_exact(df, cfg.text_col, cfg.label_col)
    issues["inconsistent_labels_exact"] = inconsistent_exact

    inconsistent_near = check_inconsistent_labels_near(df, near_dups, cfg.label_col)
    issues["inconsistent_labels_near"] = inconsistent_near

    malformed = check_malformed_questions(df, cfg.text_col, cfg)
    issues["malformed_questions"] = malformed

    invalid_answer = check_answer_format(df, cfg)
    issues["invalid_answer_format"] = invalid_answer

    qa_mismatch = check_question_answer_type_mismatch(df, cfg)
    issues["question_answer_type_mismatch"] = qa_mismatch

    related_mismatch = check_related_columns(df, cfg)
    issues["related_column_mismatch"] = related_mismatch

    leakage = check_answer_leakage(df, cfg)
    issues["answer_leakage"] = leakage

    template_freq = check_template_frequency(df, cfg.text_col, cfg)
    if template_freq is not None:
        summaries["template_frequency"] = template_freq

    label_imbalance = check_label_imbalance(df, cfg.label_col, cfg)
    if label_imbalance is not None:
        summaries["label_distribution"] = label_imbalance.reset_index().rename(columns={"index": "label", cfg.label_col or "label": "count"})

    label_words = check_label_words_in_questions(df, cfg)
    if label_words is not None:
        summaries["label_words_in_questions"] = label_words

    position_bias = check_answer_position_bias(df, cfg)
    if position_bias is not None:
        summaries["answer_position_bias"] = position_bias

    purity_cols = [c for c in [cfg.object_col, cfg.feature_col, "template_id", "layer_id", cfg.answer_type_col] if c]
    purity = check_label_purity_by_column(df, cfg, purity_cols)
    if purity is not None:
        summaries["label_purity_by_column"] = purity

    print_section("Missing/Empty Cells")
    print(f"Rows with empty cells: {len(empty_rows)}")
    print(empty_by_col.head(10).to_string())
    print_issue_sample(empty_rows, cfg, args.sample_size)

    print_section("Exact Duplicate Rows")
    print(f"Duplicate rows: {len(duplicates)}")
    print_issue_sample(duplicates, cfg, args.sample_size)

    print_section("Near-Duplicate Questions")
    if cfg.text_col is None or cfg.text_col not in df.columns:
        print("Skipped (text column not found).")
    elif len(df) > cfg.max_pairwise_rows:
        print(f"Skipped (row count {len(df)} exceeds max_pairwise_rows={cfg.max_pairwise_rows}).")
    else:
        print(f"Near-duplicate pairs: {len(near_dups)}")
        if not near_dups.empty:
            print(near_dups.head(args.sample_size).to_string(index=False))

    print_section("Inconsistent Labels (Exact Text)")
    print(f"Inconsistent label rows: {len(inconsistent_exact)}")
    print_issue_sample(inconsistent_exact, cfg, args.sample_size)

    print_section("Inconsistent Labels (Near-Duplicate Text)")
    print(f"Inconsistent near-duplicate pairs: {len(inconsistent_near)}")
    if not inconsistent_near.empty:
        print(inconsistent_near.head(args.sample_size).to_string(index=False))

    print_section("Malformed Questions")
    print(f"Malformed question rows: {len(malformed)}")
    if not malformed.empty:
        print(malformed.head(args.sample_size).to_string(index=False))

    print_section("Invalid Answer Format")
    print(f"Invalid answer format rows: {len(invalid_answer)}")
    if not invalid_answer.empty:
        print(invalid_answer.head(args.sample_size).to_string(index=False))

    print_section("Question/Answer-Type Mismatch")
    print(f"Mismatch rows: {len(qa_mismatch)}")
    if not qa_mismatch.empty:
        print(qa_mismatch.head(args.sample_size).to_string(index=False))

    print_section("Related Column Mismatches")
    print(f"Related mismatches: {len(related_mismatch)}")
    if not related_mismatch.empty:
        print(related_mismatch.head(args.sample_size).to_string(index=False))

    print_section("Answer Leakage")
    print(f"Leakage rows: {len(leakage)}")
    if not leakage.empty:
        print(leakage.head(args.sample_size).to_string(index=False))

    print_section("Distributions and Suspicious Patterns")
    if "label_distribution" in summaries:
        print("Label distribution:")
        print(summaries["label_distribution"].head(20).to_string(index=False))
    else:
        print("Label distribution: skipped (label column not found).")

    if "template_frequency" in summaries:
        print("\nTop repeated templates:")
        print(summaries["template_frequency"].head(10).to_string(index=False))

    if "label_words_in_questions" in summaries:
        print("\nLabel words appearing in questions:")
        print(summaries["label_words_in_questions"].to_string(index=False))

    if "answer_position_bias" in summaries:
        print("\nAnswer position bias:")
        print(summaries["answer_position_bias"].to_string(index=False))

    if "label_purity_by_column" in summaries:
        print("\nHigh-purity label correlations:")
        print(summaries["label_purity_by_column"].head(20).to_string(index=False))

    print_section("Descriptive Stats")
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if numeric_cols:
        print("Numeric columns:")
        print(df[numeric_cols].describe().to_string())
    else:
        print("No numeric columns detected.")
    print("\nCategorical columns (top 10):")
    for col in df.columns:
        if col in numeric_cols:
            continue
        counts = df[col].value_counts(dropna=False).head(10)
        print(f"\n{col}:")
        print(counts.to_string())

    if args.write_csv:
        print_section("Writing CSV Outputs")
        for name, issue_df in issues.items():
            path = write_issue_csv(issue_df, args.output_dir, name)
            if path:
                print(f"Wrote {name}: {path}")
        for name, summary_df in summaries.items():
            path = write_issue_csv(summary_df, args.output_dir, f"summary_{name}")
            if path:
                print(f"Wrote summary {name}: {path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
