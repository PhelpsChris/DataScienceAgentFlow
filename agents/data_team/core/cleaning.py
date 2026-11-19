"""Rule-based cleaning. Every change is logged in plain English for dataset_changes.txt.

Missing values are *not* filled here; the model pipeline imputes them (median for
numbers, most frequent for categories) so the imputation is learned on training
data only. We log that so the report can explain it.
"""
from __future__ import annotations

import re

import pandas as pd

def _is_text(s: pd.Series) -> bool:
    """True for text columns in both pandas 2 (object dtype) and pandas 3 (str dtype)."""
    return s.dtype == object or pd.api.types.is_string_dtype(s)


ID_NAME = re.compile(r"(^id$|_id$|^id_|uuid|^index$|^unnamed)", re.IGNORECASE)


def clean(df: pd.DataFrame, target: str, drop_columns: list[str] | None = None) -> tuple[pd.DataFrame, list[str]]:
    df = df.copy()
    log: list[str] = []
    n0 = len(df)

    # 1. Column names: strip and de-duplicate.
    df.columns = [str(c).strip() for c in df.columns]
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()]
        log.append("Removed duplicated column names (kept the first copy).")

    if target not in df.columns:
        raise KeyError(f"Target column '{target}' is not in the dataset.")

    # 2. Columns the user / agent asked to drop.
    for c in drop_columns or []:
        if c in df.columns and c != target:
            df = df.drop(columns=c)
            log.append(f"Dropped column '{c}' as requested.")

    # 3. Numbers stored as text (e.g. "5" or " 12.0 ").
    for c in df.columns:
        if _is_text(df[c]):
            conv = pd.to_numeric(df[c].astype(str).str.strip().replace({"": None, "nan": None}), errors="coerce")
            non_null = df[c].notna().sum()
            if non_null and conv.notna().sum() >= 0.95 * non_null:
                df[c] = conv
                log.append(f"Converted column '{c}' from text to numbers.")

    # 4. Useless columns.
    for c in list(df.columns):
        if c == target:
            continue
        s = df[c]
        if s.isna().all():
            df = df.drop(columns=c)
            log.append(f"Dropped column '{c}': it is completely empty.")
        elif s.nunique(dropna=True) <= 1:
            df = df.drop(columns=c)
            log.append(f"Dropped column '{c}': it has the same value in every row.")
        elif ID_NAME.search(c) and s.nunique() >= 0.9 * len(s):
            df = df.drop(columns=c)
            log.append(f"Dropped column '{c}': it looks like a row ID, which can't help predict anything.")
        elif _is_text(s) and s.nunique() > 50 and s.nunique() > 0.5 * len(s):
            df = df.drop(columns=c)
            log.append(f"Dropped column '{c}': free text with {s.nunique()} distinct values, too many to use as a category.")

    # 5. Rows.
    dups = int(df.duplicated().sum())
    if dups:
        df = df.drop_duplicates().reset_index(drop=True)
        log.append(f"Removed {dups} exact duplicate rows.")
    missing_target = int(df[target].isna().sum())
    if missing_target:
        df = df[df[target].notna()].reset_index(drop=True)
        log.append(f"Removed {missing_target} rows with no value for the target '{target}'.")

    # 6. Note remaining missing values (handled by the model pipeline).
    miss = df.drop(columns=target).isna().sum()
    miss = miss[miss > 0]
    if len(miss):
        detail = ", ".join(f"{c} ({n})" for c, n in miss.items())
        log.append(
            "Missing values remain in " + detail + "; the model fills these with the column median "
            "(numbers) or the most common value (categories), learned from the training data only."
        )

    if not log:
        log.append("No cleaning was needed: no duplicates, empty, constant or ID columns, and no missing values.")
    log.insert(0, f"Started with {n0} rows; finished with {len(df)} rows and {df.shape[1]} columns.")
    return df, log
