"""ADK tools: thin wrappers that connect the agents to the tested pipeline steps.

Each tool returns a small dict the LLM can read. Two tools also *hand off*
control to the next agent (confirm_goal -> DS Agent, bundle_report_package ->
Report Agent) by setting `tool_context.actions.transfer_to_agent`, so the
hand-offs happen in code rather than depending on the model remembering to do it.
"""
from __future__ import annotations

from pathlib import Path

from google.adk.tools import ToolContext
from google.genai import types

from . import pipeline
from .session_files import current_run_dir, new_analysis_dir, session_dir


def _err(e: Exception) -> dict:
    return {"ok": False, "error": f"{type(e).__name__}: {e}"}


def _run(tool_context: ToolContext) -> Path:
    run = current_run_dir(tool_context.state)
    if run is None:
        raise RuntimeError("No dataset has been loaded in this session yet.")
    return run


# ------------------------------------------------------------------ Verify Agent
def list_sample_datasets() -> dict:
    """Lists the built-in sample datasets the user can analyse without uploading anything."""
    return {"samples": pipeline.list_samples(), "accepted_upload_formats": [".csv", ".xlsx", ".json"]}


def load_dataset(source: str, tool_context: ToolContext) -> dict:
    """Loads and verifies a dataset, and writes dataVerified.txt.

    Args:
      source: "uploaded" for the file(s) the user just attached, a sample name from
        list_sample_datasets (e.g. "student"), or a file name.
    """
    try:
        state = tool_context.state
        sdir = session_dir(state, tool_context.session.id)
        run = new_analysis_dir(state, tool_context.session.id)
        res = pipeline.load_and_verify(run, source, sdir / "uploads", state.get("latest_uploads"))
        if res["ok"]:
            state["dataset_loaded"] = True
            state["goal_confirmed"] = False
        return res
    except Exception as e:  # noqa: BLE001 - errors go back to the LLM as text
        return _err(e)


def check_goal(target_column: str, tool_context: ToolContext) -> dict:
    """Checks whether a target column exists and can be modelled.

    Args:
      target_column: the column the user's goal is about. Pass "" when the user has
        no specific goal, to use the default (most likely) target.
    """
    try:
        return pipeline.check_goal(_run(tool_context), target_column)
    except Exception as e:  # noqa: BLE001
        return _err(e)


def confirm_goal(goal: str, target_column: str, tool_context: ToolContext) -> dict:
    """Saves the agreed goal to usergoal.txt and hands off to the Data Science Agent.

    Args:
      goal: the user's goal in one sentence.
      target_column: the column to predict (as returned by check_goal).
    """
    try:
        res = pipeline.confirm_goal(_run(tool_context), goal, target_column)
    except Exception as e:  # noqa: BLE001
        return _err(e)
    if res.get("ok"):
        tool_context.state["goal_confirmed"] = True
        tool_context.actions.transfer_to_agent = "data_science_agent"
    return res


# ------------------------------------------------------------------ Data Science Agent
def read_verify_payload(tool_context: ToolContext) -> dict:
    """Reads the Verify Agent's payload: dataVerified.txt, usergoal.txt, user_responses.txt."""
    try:
        return pipeline.read_verify_payload(_run(tool_context))
    except Exception as e:  # noqa: BLE001
        return _err(e)


def clean_dataset(tool_context: ToolContext, drop_columns: list[str] | None = None) -> dict:
    """Cleans a copy of the dataset with fixed rules and logs every change to dataset_changes.txt.

    Args:
      drop_columns: optional extra columns to remove (identifiers or columns unknown at prediction time).
    """
    try:
        return pipeline.clean_dataset(_run(tool_context), drop_columns or [])
    except Exception as e:  # noqa: BLE001
        return _err(e)


def train_models(tool_context: ToolContext, scenario_name: str = "all_features",
                 exclude_columns: list[str] | None = None) -> dict:
    """Trains three candidate models, picks the best (ties go to the simpler model),
    and records metrics and feature importance in metrics.json and model_selection_why.txt.

    Args:
      scenario_name: a short label for this run, e.g. "all_features" or "without_G1_G2".
      exclude_columns: feature columns to leave out for this scenario.
    """
    try:
        return pipeline.train_models(_run(tool_context), scenario_name, exclude_columns or [])
    except Exception as e:  # noqa: BLE001
        return _err(e)


def make_charts(tool_context: ToolContext) -> dict:
    """Saves the standard charts (distribution, correlations, model comparison,
    feature importance, prediction quality) as PNG files in charts/."""
    try:
        return {"ok": True, "charts": pipeline.make_charts(_run(tool_context))}
    except Exception as e:  # noqa: BLE001
        return _err(e)


def bundle_report_package(analyst_notes: str, tool_context: ToolContext) -> dict:
    """Writes report_agent_package.txt for the Report Agent and hands off to it.

    Args:
      analyst_notes: 4-8 short bullet points summarising the results for the report writer.
    """
    try:
        res = pipeline.bundle_report_package(_run(tool_context), analyst_notes)
    except Exception as e:  # noqa: BLE001
        return _err(e)
    tool_context.actions.transfer_to_agent = "report_agent"
    return {"ok": True, **res}


# ------------------------------------------------------------------ Report Agent
def read_report_package(tool_context: ToolContext) -> dict:
    """Reads report_agent_package.txt, the only input the Report Agent uses."""
    try:
        return {"package": pipeline.read_report_package(_run(tool_context))}
    except Exception as e:  # noqa: BLE001
        return _err(e)


async def build_pdf_report(
    title: str,
    executive_summary: str,
    methodology: str,
    findings: str,
    limitations: str,
    recommendations: str,
    tool_context: ToolContext,
) -> dict:
    """Renders the five report sections, the metrics table and all charts into report.pdf.

    Args:
      title: report title.
      executive_summary: Executive Summary section text.
      methodology: Methodology section text.
      findings: Findings section text.
      limitations: Limitations section text.
      recommendations: Recommendations section text.
    """
    try:
        run = _run(tool_context)
        path = pipeline.write_pdf(run, title, {
            "Executive Summary": executive_summary,
            "Methodology": methodology,
            "Findings": findings,
            "Limitations": limitations,
            "Recommendations": recommendations,
        })
        pdf = Path(path).read_bytes()
        try:
            await tool_context.save_artifact(
                f"report_{run.name}.pdf", types.Part.from_bytes(data=pdf, mime_type="application/pdf"))
        except Exception:  # noqa: BLE001 - no artifact service configured; file is still on disk
            pass
        gcs = None
        try:
            gcs = pipeline.upload_run_to_gcs(run)
        except Exception as e:  # noqa: BLE001
            gcs = f"upload failed: {e}"
        return {"ok": True, "pdf_path": str(path), "artifact": f"report_{run.name}.pdf", "cloud_storage": gcs}
    except Exception as e:  # noqa: BLE001
        return _err(e)
