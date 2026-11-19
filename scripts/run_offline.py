"""Run the whole workflow WITHOUT Gemini: same tools, same files, same PDF.

The agents' reasoning is replaced by fixed rules and the report text by a
template. Use it to test a new dataset before the demo, or as a backup if the
internet or the API key fails on stage.

    python scripts/run_offline.py --data student --target G3
    python scripts/run_offline.py --data path/to/file.csv            # target = most likely column
    python scripts/run_offline.py --data student --target G3 --exclude G1 G2
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agents"))

from data_team import pipeline  # noqa: E402


def _fmt(scores: dict) -> str:
    return ", ".join(f"{k.upper()} {v:.3f}" for k, v in scores.items())


def template_report(run: Path) -> tuple[str, dict[str, str]]:
    import json

    m = json.loads((run / "metrics.json").read_text())
    g = pipeline.goal(run)
    scen = m["scenarios"]
    names = list(scen)
    main = scen[names[0]]
    metric = main["primary_metric"]
    top = [f["feature"] for f in main["feature_importance"][:3]]
    changes = [l.strip("- ") for l in (run / "dataset_changes.txt").read_text().splitlines()[2:]]

    summary = (
        f"Goal: {g['goal']}. We treated this as a {main['task']} problem predicting '{main['target']}'. "
        f"The selected model, {main['selected_model']}, reached test {_fmt(main['test_scores'])}, "
        f"compared with a naive baseline ({main['baseline']['description']}) at {metric.upper()} "
        f"{main['baseline'][metric]:.3f}. The most influential features were {', '.join(top)}."
    )
    if len(names) > 1:
        alt = scen[names[1]]
        summary += (f" Without {', '.join(alt['excluded_columns'])}, the score falls to {metric.upper()} "
                    f"{alt['test_scores'][metric]:.3f}, and the top features become "
                    f"{', '.join(f['feature'] for f in alt['feature_importance'][:3])}.")
    methodology = "\n".join(
        [f"- Data: {main['n_rows']} rows and {main['n_features']} features after cleaning."]
        + [f"- Cleaning: {c}" for c in changes]
        + [f"- Split: {main['split']['train']} train / {main['split']['validation']} validation / "
           f"{main['split']['test']} test rows, fixed random seed.",
           "- Candidates: " + "; ".join(f"{c['model']} ({c['how_it_works']})" for c in main["candidates"]) + ".",
           f"- Selection: best validation {metric.upper()}; models within 0.01 count as a tie and the simpler one wins. "
           + main["selection_reason"],
           "- Feature importance: permutation importance on the test set (how much the score drops when a "
           "feature is shuffled)."]
    )
    findings = []
    for n, s in scen.items():
        findings.append(f"- **{n}**: {s['selected_model']}, test {_fmt(s['test_scores'])}. Top features: "
                        + ", ".join(f"{f['feature']} ({f['importance']:.3f})" for f in s["feature_importance"][:5]) + ".")
    limitations = "\n".join([
        f"- Sample size: {main['n_rows']} rows; results may not generalise to other populations.",
        "- Feature importance shows association, not cause and effect.",
        "- Only three standard model families were tried, with default settings.",
    ] + ([f"- {', '.join(scen[names[1]]['excluded_columns'])} are very strongly correlated with the target, "
          "so the full model partly restates information that is close to the answer."] if len(names) > 1 else []))
    recommendations = "\n".join([
        f"- Focus on the strongest drivers identified: {', '.join(top)}.",
        "- Collect more data and re-run the analysis to confirm the findings.",
        "- Use the model as a screening aid, not as the sole basis for decisions.",
    ] + (["- For early intervention, use the model without the near-answer features, since those are not "
          "known early."] if len(names) > 1 else []))
    return f"Analysis report: {g['goal']}", {
        "Executive Summary": summary, "Methodology": methodology, "Findings": "\n".join(findings),
        "Limitations": limitations, "Recommendations": recommendations,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="sample name (student, wine, penguins, ...) or file path(s), comma-separated")
    ap.add_argument("--target", default="", help="column to predict (default: most likely target)")
    ap.add_argument("--goal", default="", help="goal sentence for the report")
    ap.add_argument("--exclude", nargs="*", default=None,
                    help="also run a second scenario without these columns (default: auto, from the leakage check)")
    args = ap.parse_args()

    run = pipeline.new_run_dir("offline")
    print(f"Run folder: {run}")
    res = pipeline.load_and_verify(run, args.data)
    if not res["ok"]:
        sys.exit(res["message"])
    print(res["summary"].split("Column details")[0])
    chk = pipeline.check_goal(run, args.target)
    if not chk["ok"]:
        sys.exit(chk["message"])
    goal = args.goal or (f"Predict {chk['target_column']}" if args.target else "Explain the whole dataset")
    pipeline.confirm_goal(run, goal, chk["target_column"])
    cleaned = pipeline.clean_dataset(run)
    print("Cleaning:", *cleaned["changes"], sep="\n  ")
    r = pipeline.train_models(run, "all_features")
    print(f"all_features: {r['selected_model']} {r['test_scores']}  ({r['selection_reason']})")
    exclude = args.exclude if args.exclude is not None else [d["feature"] for d in cleaned["possible_leakage"]]
    if exclude:
        name = "without_" + "_".join(exclude)
        r = pipeline.train_models(run, name, exclude)
        print(f"{name}: {r['selected_model']} {r['test_scores']}")
    pipeline.make_charts(run)
    pipeline.bundle_report_package(run, "(offline run: no analyst notes)")
    title, sections = template_report(run)
    print("PDF:", pipeline.write_pdf(run, title, sections))


if __name__ == "__main__":
    main()
