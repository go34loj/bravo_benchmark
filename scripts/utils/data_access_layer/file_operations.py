from __future__ import annotations

from io import StringIO
from pathlib import Path
from typing import Dict

import pandas as pd


def detect_csv_files(backend_dir: Path) -> Dict[str, Path]:
    """Find CSV files in the directory."""
    csv_files = {}

    def _latest(matches: list[Path]) -> Path:
        return max(matches, key=lambda p: p.stat().st_mtime)

    templates_matches = list(backend_dir.glob("*templates*.csv"))
    if templates_matches:
        updated_match = [m for m in templates_matches if "updated" in m.name.lower()]
        if updated_match:
            csv_files["templates"] = _latest(updated_match)
        else:
            csv_files["templates"] = _latest(templates_matches)

    rules_matches = list(backend_dir.glob("*Rules_sort*.csv"))
    if rules_matches:
        csv_files["rules"] = _latest(rules_matches)

    figures_matches = list(backend_dir.glob("*rule_figure*.csv"))
    if figures_matches:
        csv_files["figures"] = _latest(figures_matches)

    scenes_matches = list(backend_dir.glob("*viewpoint_scenes*.csv"))
    if scenes_matches:
        csv_files["scenes"] = _latest(scenes_matches)

    cutouts_matches = list(backend_dir.glob("*cutouts*.csv"))
    if cutouts_matches:
        csv_files["cutouts"] = _latest(cutouts_matches)

    return csv_files


def detect_encoding(file_path: Path) -> str:
    """Detect the encoding of a file if chardet is available."""
    try:
        import chardet  # type: ignore
    except Exception:
        return "utf-8"

    try:
        with open(file_path, "rb") as file:
            raw_data = file.read(10000)  # Read first 10KB for detection
            result = chardet.detect(raw_data)
            return result["encoding"] if result["encoding"] else "utf-8"
    except Exception:
        return "utf-8"


def load_csv(path: Path, delim: str = ";") -> pd.DataFrame:
    """Load CSV file with automatic encoding detection."""
    # List of common encodings to try
    encodings_to_try = ["utf-8", "utf-8-sig", "windows-1252", "latin1", "cp1252", "iso-8859-1"]

    # First try to detect encoding
    detected_encoding = detect_encoding(path)
    if detected_encoding and detected_encoding not in encodings_to_try:
        encodings_to_try.insert(0, detected_encoding)

    last_error = None

    for encoding in encodings_to_try:
        try:
            # Load CSV and strip whitespace from column names
            raw_df = pd.read_csv(path, delimiter=delim, encoding=encoding)
            raw_df.columns = [col.strip() for col in raw_df.columns]

            # Remove rows that are entirely empty - these sometimes sneak in
            cleaned_df = raw_df.dropna(how="all")

            print(f"Successfully loaded {path} with encoding: {encoding}")
            return cleaned_df

        except UnicodeDecodeError as e:
            last_error = e
            print(f"Failed to load {path} with encoding {encoding}: {e}")
            continue
        except Exception as e:
            last_error = e
            print(f"Error loading {path} with encoding {encoding}: {e}")
            continue

    # Final fallback: decode with a permissive encoding to avoid hard failure
    try:
        raw_bytes = path.read_bytes()
        text = raw_bytes.decode("cp1252", errors="replace")
        raw_df = pd.read_csv(StringIO(text), delimiter=delim)
        raw_df.columns = [col.strip() for col in raw_df.columns]
        cleaned_df = raw_df.dropna(how="all")
        print(f"Loaded {path} with fallback decoding (cp1252, errors=replace).")
        return cleaned_df
    except Exception as e:
        last_error = e

    # If all encodings failed, raise the last error
    raise last_error if last_error else Exception(f"Could not load CSV file: {path}")
