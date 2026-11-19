"""Unit tests for the deterministic tools, run on the real datasets in sample_data/."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data_team import pipeline
from data_team.core import cleaning, data_io, modeling, profiling

SAMPLES = Path(__file__).resolve().parents[1] / "sample_data"
STUDENT = [SAMPLES / "student-mat.csv", SAMPLES / "student-por.csv"]


# ---------------------------------------------------------------- loading
def test_student_files_are_semicolon_and_stack():
    res = data_io.load_files(STUDENT)
    assert len(res.df) == 395 + 649
    assert res.df.shape[1] == 33 + 1 and res.df.columns[0] == "source_file"
    assert any("semicolon" in n for n in res.notes)
    assert pd.api.types.is_numeric_dtype(res.df["G1"])  # quoted "5" values parse as numbers


def test_bom_is_stripped():
    df, note = data_io.read_one(SAMPLES / "winequality-red.csv")
    assert df.columns[0] == "fixed acidity" and "comma" in note


def test_xlsx_and_json(tmp_path):
    df = pd.DataFrame({"a": range(30), "b": ["x", "y"] * 15})
    df.to_excel(tmp_path / "t.xlsx", index=False)
    assert data_io.read_one(tmp_path / "t.xlsx")[0].shape == (30, 2)
    (tmp_path / "t.json").write_text(json.dumps({"data": df.to_dict(orient="records")}))
    assert data_io.read_one(tmp_path / "t.json")[0].shape == (30, 2)
    assert data_io.read_one(SAMPLES / "penguins.json")[0].shape == (344, 7)


@pytest.mark.parametrize("delim", [",", ";", "\t", "|"])
def test_delimiters(tmp_path, delim):
    p = tmp_path / "d.csv"
    p.write_text(delim.join(["a", "b", "c"]) + "\n" + "\n".join(delim.join(["1", "2", "x"]) for _ in range(5)))
    assert list(data_io.read_one(p)[0].columns) == ["a", "b", "c"]


@pytest.mark.parametrize("name", ["model.r", "notebook.ipynb", "data.txt"])
def test_unsupported_types_rejected(tmp_path, name):
    (tmp_path / name).write_text("x")
    with pytest.raises(data_io.UnsupportedFileError):
        data_io.read_one(tmp_path / name)


def test_different_columns_uses_first(tmp_path):
    pd.DataFrame({"a": [1]}).to_csv(tmp_path / "1.csv", index=False)
    pd.DataFrame({"b": [1]}).to_csv(tmp_path / "2.csv", index=False)
    res = data_io.load_files([tmp_path / "1.csv", tmp_path / "2.csv"])
    assert list(res.df.columns) == ["a"] and "different columns" in res.notes[-1]


# ---------------------------------------------------------------- profiling & goal
def test_profile_finds_g3_as_target():
    df = data_io.load_files(STUDENT).df
    p = profiling.profile(df)
    assert p["candidate_targets"][0] == "G3"
    assert "Rows: 1044" in profiling.profile_to_text(p)


def test_check_goal(tmp_path):
    run = tmp_path
    assert pipeline.load_and_verify(run, "student")["ok"]
    assert pipeline.check_goal(run, "g3")["target_column"] == "G3"  # case-insensitive
    assert pipeline.check_goal(run, "")["defaulted_target"] is True
    assert not pipeline.check_goal(run, "millionaire")["ok"]


def test_too_small_dataset_rejected(tmp_path):
    pd.DataFrame({"a": range(5), "b": range(5)}).to_csv(tmp_path / "uploads.csv", index=False)
    (tmp_path / "uploads").mkdir()
    (tmp_path / "uploads" / "tiny.csv").write_text((tmp_path / "uploads.csv").read_text())
    assert not pipeline.load_and_verify(tmp_path, "uploaded")["ok"]


# ---------------------------------------------------------------- cleaning
def test_cleaning_rules():
    df = pd.DataFrame({
        "id": range(100), "const": 1, "empty": np.nan, "num_text": [str(i) for i in range(100)],
        "x": np.r_[np.arange(98), [np.nan, np.nan]], "y": [i % 3 for i in range(100)],
    })
    df = pd.concat([df, df.iloc[:5]], ignore_index=True)
    out, log = cleaning.clean(df, "y")
    assert set(out.columns) == {"num_text", "x", "y"}
    assert pd.api.types.is_numeric_dtype(out["num_text"])
    assert len(out) == 100
    text = " ".join(log)
    assert "duplicate" in text and "row ID" in text and "Missing values remain in x" in text


# ---------------------------------------------------------------- modeling
def test_detect_task():
    assert modeling.detect_task(pd.Series(range(20))) == "regression"
    assert modeling.detect_task(pd.Series([3, 4, 5, 6] * 5)) == "classification"
    assert modeling.detect_task(pd.Series(["a", "b"] * 5)) == "classification"


def test_leakage_flags_g1_g2():
    df = data_io.load_files(STUDENT).df
    flagged = [d["feature"] for d in modeling.leakage_suspects(df, "G3")]
    assert flagged == ["G2", "G1"]


def test_tie_goes_to_simpler_model():
    rng = np.random.default_rng(0)
    x = rng.normal(size=600)
    df = pd.DataFrame({"x": x, "noise": rng.normal(size=600), "y": 3 * x + rng.normal(scale=0.01, size=600)})
    res = modeling.train_and_compare(df, "y")
    assert res["task"] == "regression"
    assert res["selected_model"] == "Linear Regression (Ridge)"
    assert "tie rule" in res["selection_reason"] or "best validation" in res["selection_reason"]


def test_training_is_deterministic():
    df = pd.read_csv(SAMPLES / "winequality-red.csv", encoding="utf-8-sig")
    a = modeling.train_and_compare(df, "quality")
    b = modeling.train_and_compare(df, "quality")
    assert a["selected_model"] == b["selected_model"] and a["test_scores"] == b["test_scores"]


def test_student_with_and_without_g1_g2():
    df = data_io.load_files(STUDENT).df
    full = modeling.train_and_compare(df, "G3")
    without = modeling.train_and_compare(df, "G3", exclude_columns=["G1", "G2"])
    assert full["test_scores"]["r2"] > 0.7
    assert without["test_scores"]["r2"] < full["test_scores"]["r2"]
    assert full["test_scores"]["mae"] < full["baseline"]["mae"]
    assert "G1" not in [f["feature"] for f in without["feature_importance"]]


# ---------------------------------------------------------------- whole pipeline
@pytest.mark.parametrize("sample,target", [("student", "G3"), ("wine", "quality"), ("penguins", "species")])
def test_pipeline_end_to_end(tmp_path, sample, target):
    run = tmp_path
    assert pipeline.load_and_verify(run, sample)["ok"]
    assert pipeline.confirm_goal(run, f"Predict {target}", target)["ok"]
    cleaned = pipeline.clean_dataset(run)
    pipeline.train_models(run, "all_features")
    leaks = [d["feature"] for d in cleaned["possible_leakage"]]
    if leaks:
        pipeline.train_models(run, "without_leaky", leaks)
    charts = pipeline.make_charts(run)
    assert all((run / c).exists() for c in charts)
    pipeline.bundle_report_package(run, "- note")
    pkg = pipeline.read_report_package(run)
    assert "## Metrics by scenario" in pkg and "_test_truth" not in pkg
    pdf = pipeline.write_pdf(run, "Test report", {s: "Text with <angle> & **bold**\n- bullet" for s in pipeline.SECTIONS})
    assert Path(pdf).read_bytes()[:4] == b"%PDF"


def test_pdf_refuses_empty_section(tmp_path):
    pipeline.load_and_verify(tmp_path, "wine")
    pipeline.confirm_goal(tmp_path, "g", "quality")
    pipeline.clean_dataset(tmp_path)
    pipeline.train_models(tmp_path)
    with pytest.raises(ValueError):
        pipeline.write_pdf(tmp_path, "t", {"Executive Summary": "x"})
