"""The steps of the workflow, as plain functions over a run folder.

Each agent tool is a thin wrapper around one of these functions, and the
offline script (scripts/run_offline.py) calls them directly. The files written
here are the hand-offs ("payloads") between agents, exactly as in our design:

  Verify Agent  -> user_responses.txt, dataVerified.txt, usergoal.txt, dataset.csv
  DS Agent      -> dataset_changes.txt, model_selection_why.txt, metrics.json,
                   charts/, report_agent_package.txt
  Report Agent  -> report.pdf
"""
from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path

import pandas as pd

from .core import charts as charts_mod
from .core import cleaning, modeling, profiling
from .core.data_io import ACCEPTED_EXTENSIONS, load_files
from .core.pdf_report import SECTIONS, build_pdf

REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_DIR = Path(os.environ.get("SAMPLE_DATA_DIR", REPO_ROOT / "sample_data"))
RUNS_DIR = Path(os.environ.get("OUTPUT_DIR", REPO_ROOT / "runs"))

SAMPLES = {
    "student": (["student-mat.csv", "student-por.csv"],
                "UCI Student Performance, math + Portuguese classes stacked (semicolon-separated)"),
    "student-math": (["student-mat.csv"], "UCI Student Performance, math class only"),
    "student-portuguese": (["student-por.csv"], "UCI Student Performance, Portuguese class only"),
    "wine": (["winequality-red.csv"], "UCI Wine Quality (red), 1,599 wines, predict quality 3-8"),
    "penguins": (["penguins.json"], "Palmer Penguins (JSON), predict species; has missing values"),
}


# ---------------------------------------------------------------- run folder
def new_run_dir(tag: str = "") -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    d = RUNS_DIR / (f"{stamp}_{tag}" if tag else stamp)
    (d / "uploads").mkdir(parents=True, exist_ok=True)
    (d / "charts").mkdir(exist_ok=True)
    return d


def _write(run_dir: Path, name: str, text: str) -> None:
    (Path(run_dir) / name).write_text(text, encoding="utf-8")


def _read(run_dir: Path, name: str) -> str:
    p = Path(run_dir) / name
    return p.read_text(encoding="utf-8") if p.exists() else ""


def log_user_response(run_dir: Path, text: str, filenames: list[str] | None = None) -> None:
    """Append what the user said. File *names* only, never file contents."""
    stamp = datetime.now().strftime("%H:%M:%S")
    line = f"[{stamp}] USER: {text.strip()}" if text.strip() else f"[{stamp}] USER: (no text)"
    if filenames:
        line += f"  [attached: {', '.join(filenames)}]"
    with open(Path(run_dir) / "user_responses.txt", "a", encoding="utf-8") as f:
        f.write(line + "\n")


# ---------------------------------------------------------------- Verify Agent steps
def list_samples() -> dict:
    return {k: v[1] for k, v in SAMPLES.items()}


def resolve_sources(source: str, uploads: Path, latest_uploads: list[str] | None = None) -> list[Path]:
    """Turn what the user gave us into file paths: a sample name, an uploaded file
    name, 'uploaded' (= the files attached most recently), or a path on disk."""
    s = source.strip().strip("'\"")
    key = s.lower().replace(" ", "-")
    if key in SAMPLES:
        return [SAMPLE_DIR / f for f in SAMPLES[key][0]]
    uploads = Path(uploads)
    if key in ("uploaded", "upload", "uploads", "attached", "attachment", "file"):
        if latest_uploads:
            files = [uploads / n for n in latest_uploads if (uploads / n).exists()]
        else:
            files = sorted(p for p in uploads.iterdir() if p.is_file()) if uploads.exists() else []
        if not files:
            raise FileNotFoundError("No uploaded file was found. Please attach a file in the chat.")
        return files
    paths = []
    for part in re.split(r"\s*[,+]\s*|\s+and\s+", s):
        if not part:
            continue
        for cand in (uploads / Path(part).name, Path(part), SAMPLE_DIR / part):
            if cand.exists():
                paths.append(cand)
                break
        else:
            raise FileNotFoundError(f"Could not find '{part}'. Upload it, or pick a sample: {', '.join(SAMPLES)}.")
    return paths


def load_and_verify(run_dir: Path, source: str, uploads: Path | None = None,
                    latest_uploads: list[str] | None = None) -> dict:
    paths = resolve_sources(source, uploads or Path(run_dir) / "uploads", latest_uploads)
    bad = [p.name for p in paths if p.suffix.lower() not in ACCEPTED_EXTENSIONS]
    if bad:
        return {"ok": False, "message": f"Unsupported file type: {', '.join(bad)}. "
                f"Accepted formats: {', '.join(ACCEPTED_EXTENSIONS)}."}
    res = load_files(paths)
    if len(res.df) < 20 or res.df.shape[1] < 2:
        return {"ok": False, "message": f"The file has only {len(res.df)} rows and {res.df.shape[1]} columns; "
                "we need at least 20 rows and 2 columns to build a model."}
    res.df.to_csv(Path(run_dir) / "dataset.csv", index=False)
    prof = profiling.profile(res.df, res.notes)
    text = profiling.profile_to_text(prof)
    _write(run_dir, "dataVerified.txt", text)
    (Path(run_dir) / "profile.json").write_text(json.dumps(prof, indent=2))
    return {"ok": True, "files": [p.name for p in paths], "summary": text}


def _match_column(df: pd.DataFrame, name: str) -> str | None:
    if name in df.columns:
        return name
    norm = lambda x: re.sub(r"[^a-z0-9]", "", str(x).lower())  # noqa: E731
    for c in df.columns:
        if norm(c) == norm(name):
            return c
    return None


def check_goal(run_dir: Path, target_column: str = "") -> dict:
    """Deterministic feasibility check: does the target exist and is it usable?"""
    p = Path(run_dir) / "dataset.csv"
    if not p.exists():
        return {"ok": False, "message": "No dataset has been loaded yet."}
    df = pd.read_csv(p)
    if not target_column:
        target_column = profiling.candidate_targets(df)[0]
        defaulted = True
    else:
        defaulted = False
    col = _match_column(df, target_column)
    if col is None:
        return {"ok": False, "message": f"There is no column called '{target_column}'. "
                f"Columns are: {', '.join(map(str, df.columns))}."}
    y = df[col].dropna()
    if y.nunique() < 2:
        return {"ok": False, "message": f"'{col}' has only one value, so there is nothing to predict."}
    if len(y) < 20:
        return {"ok": False, "message": f"'{col}' has only {len(y)} non-empty values; too few to model."}
    task = modeling.detect_task(y)
    if task == "classification" and y.nunique() > 50:
        return {"ok": False, "message": f"'{col}' has {y.nunique()} different text values; too many categories to predict."}
    return {
        "ok": True,
        "target_column": col,
        "task": task,
        "defaulted_target": defaulted,
        "message": (f"'{col}' can be predicted as a {task} problem "
                    f"({y.nunique()} distinct values, {len(y)} rows)."),
    }


def confirm_goal(run_dir: Path, goal: str, target_column: str) -> dict:
    chk = check_goal(run_dir, target_column)
    if not chk["ok"]:
        return chk
    data = {"goal": goal.strip() or "Explain the whole dataset", "target_column": chk["target_column"],
            "task": chk["task"], "confirmed_at": datetime.now().isoformat(timespec="seconds")}
    (Path(run_dir) / "usergoal.json").write_text(json.dumps(data, indent=2))
    _write(run_dir, "usergoal.txt", f"User goal: {data['goal']}\nTarget column: {data['target_column']}\n"
           f"Task type: {data['task']}\n")
    _write(run_dir, "verify_payload.json", json.dumps(
        {"payload": ["user_responses.txt", "dataVerified.txt", "usergoal.txt", "dataset.csv"]}, indent=2))
    return {"ok": True, **data}


# ---------------------------------------------------------------- Data Science Agent steps
def read_verify_payload(run_dir: Path) -> dict:
    return {f: _read(run_dir, f) for f in ("dataVerified.txt", "usergoal.txt", "user_responses.txt")}


def goal(run_dir: Path) -> dict:
    return json.loads(_read(run_dir, "usergoal.json") or "{}")


def clean_dataset(run_dir: Path, drop_columns: list[str] | None = None) -> dict:
    g = goal(run_dir)
    df = pd.read_csv(Path(run_dir) / "dataset.csv")
    clean_df, log = cleaning.clean(df, g["target_column"], drop_columns)
    clean_df.to_csv(Path(run_dir) / "dataset_clean.csv", index=False)
    _write(run_dir, "dataset_changes.txt", "DATASET CHANGES\n===============\n" + "\n".join(f"- {l}" for l in log))
    suspects = modeling.leakage_suspects(clean_df, g["target_column"])
    return {"changes": log, "rows": len(clean_df), "columns": clean_df.shape[1], "possible_leakage": suspects}


def _metrics(run_dir: Path) -> dict:
    return json.loads(_read(run_dir, "metrics.json") or '{"scenarios": {}}')


def train_models(run_dir: Path, scenario_name: str = "all_features", exclude_columns: list[str] | None = None) -> dict:
    g = goal(run_dir)
    df = pd.read_csv(Path(run_dir) / "dataset_clean.csv")
    res = modeling.train_and_compare(df, g["target_column"], g.get("task", "auto"), exclude_columns)
    scenario_name = re.sub(r"[^A-Za-z0-9_\-]+", "_", scenario_name.strip()) or "scenario"
    m = _metrics(run_dir)
    m["scenarios"][scenario_name] = res
    (Path(run_dir) / "metrics.json").write_text(json.dumps(m, indent=2, default=str))

    why = [f"[{scenario_name}] target={res['target']} task={res['task']} excluded={res['excluded_columns'] or 'none'}",
           f"Split: {res['split']}", "Validation scores:"]
    why += [f"  - {c['model']}: {c['validation']}" for c in res["candidates"]]
    why += [f"Selected: {res['selected_model']}. {res['selection_reason']}", ""]
    with open(Path(run_dir) / "model_selection_why.txt", "a", encoding="utf-8") as f:
        f.write("\n".join(why) + "\n")

    public = {k: v for k, v in res.items() if not k.startswith("_")}
    public["feature_importance"] = res["feature_importance"][:8]
    public.pop("confusion_matrix", None)
    return public


def make_charts(run_dir: Path) -> list[str]:
    m = _metrics(run_dir)
    if not m["scenarios"]:
        raise RuntimeError("Train at least one model before making charts.")
    df = pd.read_csv(Path(run_dir) / "dataset_clean.csv")
    paths = charts_mod.make_all(df, m["scenarios"], Path(run_dir) / "charts")
    m["charts"] = [str(Path(p).relative_to(run_dir)) for p in paths]
    (Path(run_dir) / "metrics.json").write_text(json.dumps(m, indent=2, default=str))
    return m["charts"]


def bundle_report_package(run_dir: Path, analyst_notes: str = "") -> dict:
    m = _metrics(run_dir)
    lean = {}
    for name, s in m["scenarios"].items():
        lean[name] = {k: v for k, v in s.items() if not k.startswith("_") and k != "confusion_matrix"}
    if analyst_notes.strip():
        with open(Path(run_dir) / "model_selection_why.txt", "a", encoding="utf-8") as f:
            f.write("Analyst notes:\n" + analyst_notes.strip() + "\n")
    parts = [
        "REPORT AGENT PACKAGE",
        "====================",
        "## User goal", _read(run_dir, "usergoal.txt"),
        "## Dataset summary (from Verify Agent)", _read(run_dir, "dataVerified.txt"),
        "## Data cleaning", _read(run_dir, "dataset_changes.txt"),
        "## Model selection reasoning", _read(run_dir, "model_selection_why.txt"),
        "## Metrics by scenario (JSON)", json.dumps(lean, indent=1, default=str),
        "## Charts available", "\n".join(m.get("charts", [])),
    ]
    _write(run_dir, "report_agent_package.txt", "\n".join(parts))
    payload = ["user_responses.txt", "dataVerified.txt", "usergoal.txt", "dataset.csv", "dataset_changes.txt",
               "model_selection_why.txt", "metrics.json", "charts/", "report_agent_package.txt"]
    _write(run_dir, "report_payload.json", json.dumps({"payload": payload}, indent=2))
    return {"package_file": "report_agent_package.txt", "payload": payload, "scenarios": list(lean),
            "charts": m.get("charts", [])}


# ---------------------------------------------------------------- Report Agent steps
def read_report_package(run_dir: Path) -> str:
    return _read(run_dir, "report_agent_package.txt")


def write_pdf(run_dir: Path, title: str, sections: dict[str, str]) -> str:
    m = _metrics(run_dir)
    g = goal(run_dir)
    missing = [s for s in SECTIONS if not sections.get(s, "").strip()]
    if missing:
        raise ValueError(f"These report sections are empty: {', '.join(missing)}")
    scen = {k: v for k, v in m["scenarios"].items()}
    charts = [str(Path(run_dir) / c) for c in m.get("charts", [])]
    for s in SECTIONS:
        _write(run_dir, f"report_{s.lower().replace(' ', '_')}.md", sections[s])
    return build_pdf(Path(run_dir) / "report.pdf", title, g.get("goal", ""), sections, scen, charts)


def upload_run_to_gcs(run_dir: Path) -> str | None:
    """Copy the run folder to gs://$OUTPUT_BUCKET/runs/<run>/ when OUTPUT_BUCKET is set."""
    bucket_name = os.environ.get("OUTPUT_BUCKET")
    if not bucket_name:
        return None
    from google.cloud import storage  # optional dependency, only needed on Cloud Run

    bucket = storage.Client().bucket(bucket_name)
    run_dir = Path(run_dir)
    for p in run_dir.rglob("*"):
        if p.is_file():
            bucket.blob(f"runs/{run_dir.name}/{p.relative_to(run_dir)}").upload_from_filename(str(p))
    return f"gs://{bucket_name}/runs/{run_dir.name}/"
