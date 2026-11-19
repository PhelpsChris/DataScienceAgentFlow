"""Load .csv / .xlsx / .json datasets into a pandas DataFrame.

Handles the quirks we hit in practice:
- CSV delimiter is auto-detected (the UCI student files use ';').
- A UTF-8 byte-order mark is stripped (the Wine Quality CSV has one).
- Several files with the same columns are stacked into one table, with a
  `source_file` column saying where each row came from (student-mat + student-por).
"""
from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

ACCEPTED_EXTENSIONS = (".csv", ".xlsx", ".json")
_CANDIDATE_DELIMITERS = [",", ";", "\t", "|"]


class UnsupportedFileError(ValueError):
    """Raised when a file is not one of ACCEPTED_EXTENSIONS or cannot be parsed."""


@dataclass
class LoadResult:
    df: pd.DataFrame
    files: list[str]
    notes: list[str] = field(default_factory=list)


def _decode(raw: bytes) -> str:
    for enc in ("utf-8-sig", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    raise UnsupportedFileError("Could not decode the file as text.")


def detect_delimiter(text: str) -> str:
    """Pick the delimiter that splits the first lines into the most consistent columns."""
    sample = text[:65536]
    try:
        return csv.Sniffer().sniff(sample, delimiters="".join(_CANDIDATE_DELIMITERS)).delimiter
    except csv.Error:
        pass
    lines = [ln for ln in sample.splitlines() if ln.strip()][:20]
    best, best_score = ",", -1
    for d in _CANDIDATE_DELIMITERS:
        counts = [ln.count(d) for ln in lines]
        if not counts or counts[0] == 0:
            continue
        score = counts[0] if len(set(counts)) == 1 else 0
        if score > best_score:
            best, best_score = d, score
    return best


def read_one(path: str | Path) -> tuple[pd.DataFrame, str]:
    """Read a single file. Returns (dataframe, note describing how it was read)."""
    path = Path(path)
    ext = path.suffix.lower()
    if ext not in ACCEPTED_EXTENSIONS:
        raise UnsupportedFileError(
            f"'{path.name}' is a {ext or 'file with no extension'}; "
            f"accepted formats are {', '.join(ACCEPTED_EXTENSIONS)}."
        )
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if ext == ".csv":
        text = _decode(path.read_bytes())
        delim = detect_delimiter(text)
        df = pd.read_csv(io.StringIO(text), sep=delim)
        name = {",": "comma", ";": "semicolon", "\t": "tab", "|": "pipe"}[delim]
        return df, f"{path.name}: CSV, {name}-separated"
    if ext == ".xlsx":
        df = pd.read_excel(path, sheet_name=0)
        return df, f"{path.name}: Excel workbook, first sheet"
    # .json
    data = json.loads(_decode(path.read_bytes()))
    if isinstance(data, dict):
        lists = [v for v in data.values() if isinstance(v, list)]
        if len(lists) == 1:  # {"data": [...]} style
            data = lists[0]
        elif all(isinstance(v, (list, dict)) for v in data.values()):
            return pd.DataFrame(data), f"{path.name}: JSON, column-oriented"
    if not isinstance(data, list):
        raise UnsupportedFileError(f"'{path.name}' does not contain a table of records.")
    return pd.json_normalize(data), f"{path.name}: JSON, list of records"


def load_files(paths: list[str | Path]) -> LoadResult:
    """Load one or more files. Same-shaped files are stacked with a `source_file` column."""
    if not paths:
        raise UnsupportedFileError("No file was provided.")
    frames, notes, names = [], [], []
    for p in paths:
        df, note = read_one(p)
        df.columns = [str(c).strip() for c in df.columns]
        frames.append(df)
        notes.append(note)
        names.append(Path(p).stem)

    if len(frames) == 1:
        return LoadResult(frames[0], [str(p) for p in paths], notes)

    first_cols = list(frames[0].columns)
    if all(set(f.columns) == set(first_cols) for f in frames[1:]):
        stacked = []
        for name, f in zip(names, frames):
            f = f[first_cols].copy()
            f.insert(0, "source_file", name)
            stacked.append(f)
        merged = pd.concat(stacked, ignore_index=True)
        notes.append(
            f"Stacked {len(frames)} files with identical columns into one table "
            f"({len(merged)} rows); added a 'source_file' column."
        )
        return LoadResult(merged, [str(p) for p in paths], notes)

    notes.append(
        "The files have different columns, so only the first file was used: "
        f"{Path(paths[0]).name}."
    )
    return LoadResult(frames[0], [str(paths[0])], notes)
