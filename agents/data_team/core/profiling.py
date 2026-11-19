"""Quick exploratory summary of a dataset (the Verify Agent's 'simple EDA').

The output is plain text so it can be saved as dataVerified.txt and read by an
LLM or a person.
"""
from __future__ import annotations

import re

import pandas as pd

TARGET_HINTS = re.compile(
    r"(target|label|class|outcome|result|grade|score|quality|price|species|survived|churn|g3|y$)",
    re.IGNORECASE,
)


def column_kind(s: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(s):
        return "boolean"
    if pd.api.types.is_numeric_dtype(s):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(s):
        return "datetime"
    return "categorical"


def candidate_targets(df: pd.DataFrame) -> list[str]:
    """Columns that look like something a user would want to predict."""
    hinted = [c for c in df.columns if TARGET_HINTS.search(str(c))]
    last = [df.columns[-1]] if len(df.columns) else []
    seen, out = set(), []
    for c in last + hinted:
        if c not in seen and c != "source_file":
            seen.add(c)
            out.append(c)
    return out[:5]


def profile(df: pd.DataFrame, load_notes: list[str] | None = None) -> dict:
    """Structured profile used by both the text summary and the tests."""
    cols = []
    for c in df.columns:
        s = df[c]
        info = {
            "name": str(c),
            "kind": column_kind(s),
            "missing_pct": round(float(s.isna().mean() * 100), 1),
            "unique": int(s.nunique(dropna=True)),
        }
        if info["kind"] == "numeric":
            d = s.describe()
            info.update(min=float(d["min"]), mean=round(float(d["mean"]), 3), max=float(d["max"]))
        else:
            top = s.astype(str).value_counts().head(4)
            info["top_values"] = {str(k): int(v) for k, v in top.items()}
        cols.append(info)
    return {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "duplicate_rows": int(df.duplicated().sum()),
        "total_missing_cells": int(df.isna().sum().sum()),
        "numeric_columns": [c["name"] for c in cols if c["kind"] == "numeric"],
        "categorical_columns": [c["name"] for c in cols if c["kind"] in ("categorical", "boolean")],
        "candidate_targets": candidate_targets(df),
        "column_details": cols,
        "load_notes": list(load_notes or []),
    }


def profile_to_text(p: dict) -> str:
    lines = [
        "DATASET VERIFIED",
        "================",
        *[f"- {n}" for n in p["load_notes"]],
        f"Rows: {p['rows']}   Columns: {p['columns']}",
        f"Duplicate rows: {p['duplicate_rows']}   Missing cells: {p['total_missing_cells']}",
        f"Numeric columns ({len(p['numeric_columns'])}): {', '.join(p['numeric_columns'])}",
        f"Categorical columns ({len(p['categorical_columns'])}): {', '.join(p['categorical_columns'])}",
        f"Likely target columns: {', '.join(p['candidate_targets'])}",
        "",
        "Column details:",
    ]
    for c in p["column_details"]:
        base = f"  {c['name']} [{c['kind']}] missing={c['missing_pct']}% unique={c['unique']}"
        if c["kind"] == "numeric":
            base += f" min={c['min']:g} mean={c['mean']:g} max={c['max']:g}"
        else:
            base += " top=" + ", ".join(f"{k}({v})" for k, v in c["top_values"].items())
        lines.append(base)
    return "\n".join(lines)
