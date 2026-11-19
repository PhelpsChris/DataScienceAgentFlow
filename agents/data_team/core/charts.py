"""Static PNG charts for the PDF report (matplotlib, no display needed)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# Validated categorical palette (slots in fixed order) and neutral inks.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "font.size": 9,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK_MUTED,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlecolor": INK,
        "axes.titlelocation": "left",
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
    }
)


def _save(fig, path: Path) -> str:
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    return str(path)


def target_distribution(df: pd.DataFrame, target: str, task: str, out: Path) -> str:
    fig, ax = plt.subplots(figsize=(6, 3.2))
    if task == "regression":
        ax.hist(df[target].dropna(), bins=20, color=SERIES[0], edgecolor="white", linewidth=1)
        ax.set_xlabel(target)
        ax.set_ylabel("Rows")
    else:
        counts = df[target].astype(str).value_counts().sort_index()
        ax.bar(counts.index, counts.values, color=SERIES[0], width=0.6)
        ax.set_xlabel(target)
        ax.set_ylabel("Rows")
        ax.grid(axis="x", visible=False)
    ax.set_title(f"Distribution of {target}")
    return _save(fig, out / "target_distribution.png")


def correlation_heatmap(df: pd.DataFrame, target: str, out: Path, top: int = 10) -> str | None:
    num = df.select_dtypes("number")
    if num.shape[1] < 2:
        return None
    if target in num.columns:
        order = num.corr()[target].abs().sort_values(ascending=False).index[: top + 1]
        num = num[order]
    else:
        num = num.iloc[:, : top + 1]
    corr = num.corr()
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr)), corr.columns)
    ax.grid(False)
    for i in range(len(corr)):
        for j in range(len(corr)):
            v = corr.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                    color="white" if abs(v) > 0.6 else INK)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Correlation")
    ax.set_title("Correlation between numeric columns")
    return _save(fig, out / "correlation_heatmap.png")


def feature_importance(result: dict, out: Path, scenario: str) -> str:
    imp = result["feature_importance"][:10][::-1]
    fig, ax = plt.subplots(figsize=(6, 0.3 * len(imp) + 1.2))
    vals = [d["importance"] for d in imp]
    ax.barh([d["feature"] for d in imp], vals, xerr=[d["std"] for d in imp],
            color=SERIES[0], height=0.6, error_kw={"ecolor": INK_MUTED, "lw": 0.8})
    ax.set_xlabel(f"Drop in test {result['primary_metric']} when the feature is shuffled")
    ax.grid(axis="y", visible=False)
    ax.set_title(f"Most important features ({scenario})")
    return _save(fig, out / f"feature_importance_{scenario}.png")


def model_comparison(scenarios: dict, out: Path) -> str:
    names = list(scenarios)
    first = scenarios[names[0]]
    metric = first["primary_metric"]
    models = [c["model"] for c in first["candidates"]]
    fig, ax = plt.subplots(figsize=(6, 3.2))
    width = 0.8 / len(names)
    x = np.arange(len(models))
    for i, n in enumerate(names):
        vals = [c["validation"][metric] for c in scenarios[n]["candidates"]]
        bars = ax.bar(x + i * width - 0.4 + width / 2, vals, width * 0.92, color=SERIES[i % 3], label=n)
        ax.bar_label(bars, fmt="%.2f", fontsize=7, color=INK_MUTED, padding=2)
    ax.set_xticks(x, models)
    ax.set_ylabel(f"Validation {metric}")
    ax.grid(axis="x", visible=False)
    top = max(c["validation"][metric] for n in names for c in scenarios[n]["candidates"])
    ax.set_ylim(min(0, ax.get_ylim()[0]), max(1.0, top) * 1.22)
    if len(names) > 1:
        ax.legend(frameon=False, fontsize=8, loc="upper center", ncol=len(names))
    ax.set_title("Candidate models compared")
    return _save(fig, out / "model_comparison.png")


def prediction_quality(result: dict, out: Path, scenario: str) -> str:
    truth, pred = result["_test_truth"], result["_test_pred"]
    if result["task"] == "regression":
        fig, ax = plt.subplots(figsize=(4.8, 4.2))
        ax.scatter(truth, pred, s=16, color=SERIES[0], alpha=0.7, edgecolor="white", linewidth=0.5)
        lo, hi = min(min(truth), min(pred)), max(max(truth), max(pred))
        ax.plot([lo, hi], [lo, hi], color=INK_MUTED, lw=1, ls="--", label="perfect prediction")
        ax.set_xlabel(f"Actual {result['target']}")
        ax.set_ylabel(f"Predicted {result['target']}")
        ax.legend(frameon=False, fontsize=8)
        ax.set_title(f"Predicted vs actual, test set ({scenario})")
        return _save(fig, out / f"predicted_vs_actual_{scenario}.png")
    labels, cm = result["class_labels"], np.array(result["confusion_matrix"])
    size = max(3.6, 0.45 * len(labels) + 2)
    fig, ax = plt.subplots(figsize=(size + 0.8, size))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.grid(False)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8,
                    color="white" if cm[i, j] > cm.max() / 2 else INK)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(f"Confusion matrix, test set ({scenario})")
    return _save(fig, out / f"confusion_matrix_{scenario}.png")


def scenario_comparison(scenarios: dict, out: Path) -> str | None:
    if len(scenarios) < 2:
        return None
    metric = next(iter(scenarios.values()))["primary_metric"]
    names = list(scenarios)
    vals = [scenarios[n]["test_scores"][metric] for n in names]
    base = next(iter(scenarios.values()))["baseline"][metric]
    fig, ax = plt.subplots(figsize=(6, 2.8))
    bars = ax.barh(names[::-1], vals[::-1], color=SERIES[0], height=0.5)
    ax.bar_label(bars, fmt="%.2f", fontsize=8, color=INK_MUTED, padding=3)
    ax.axvline(base, color=SERIES[1], lw=1.5, ls="--", label=f"baseline ({base:.2f})")
    ax.set_xlabel(f"Test {metric}")
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_title("Scenario comparison on the test set")
    return _save(fig, out / "scenario_comparison.png")


def make_all(df: pd.DataFrame, scenarios: dict, out_dir: Path) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    first = next(iter(scenarios.values()))
    paths = [
        target_distribution(df, first["target"], first["task"], out_dir),
        correlation_heatmap(df, first["target"], out_dir),
        model_comparison(scenarios, out_dir),
        scenario_comparison(scenarios, out_dir),
    ]
    for name, res in scenarios.items():
        paths.append(feature_importance(res, out_dir, name))
        paths.append(prediction_quality(res, out_dir, name))
    return [p for p in paths if p]
