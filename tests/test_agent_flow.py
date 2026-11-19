"""End-to-end test of the agent wiring with a scripted stand-in for Gemini.

The scripted model returns pre-written tool calls, so this checks everything
except the LLM's own judgement: uploads are saved, tools run, payload files are
written, hand-offs between the three agents happen, and the PDF is produced.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import AsyncGenerator

import pytest
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import InMemoryRunner
from google.genai import types

from data_team import agent as agent_mod
from data_team import pipeline

SAMPLES = Path(__file__).resolve().parents[1] / "sample_data"


def call(name, **args):
    return LlmResponse(content=types.Content(role="model", parts=[types.Part(function_call=types.FunctionCall(name=name, args=args))]))


def say(text):
    return LlmResponse(content=types.Content(role="model", parts=[types.Part(text=text)]))


class ScriptedLlm(BaseLlm):
    model: str = "scripted"
    script: list = []
    seen_requests: list = []

    async def generate_content_async(self, llm_request: LlmRequest, stream: bool = False) -> AsyncGenerator[LlmResponse, None]:
        self.seen_requests.append(llm_request)
        if not self.script:
            raise AssertionError(f"Scripted model ran out of responses: {self.model}")
        yield self.script.pop(0)


def _attach(path: Path, mime="text/csv"):
    return types.Part(inline_data=types.Blob(data=path.read_bytes(), mime_type=mime, display_name=path.name))


@pytest.fixture
def scripted(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "RUNS_DIR", tmp_path)
    verify = ScriptedLlm(model="verify", script=[
        say("Hi! Attach a .csv, .xlsx or .json file."),
        call("load_dataset", source="uploaded"), say("Loaded 1044 rows. What do you want to analyze?"),
        call("check_goal", target_column="g3"), call("confirm_goal", goal="Predict final grade G3", target_column="G3"),
        # after the report, the user starts over with a new dataset
        call("load_dataset", source="wine"), say("Loaded wine data."),
    ])
    ds = ScriptedLlm(model="ds", script=[
        call("read_verify_payload"), call("clean_dataset"),
        call("train_models", scenario_name="all_features"),
        call("train_models", scenario_name="without_G1_G2", exclude_columns=["G1", "G2"]),
        call("make_charts"),
        call("bundle_report_package", analyst_notes="- Random Forest won\n- G2 dominates"),
    ])
    rep = ScriptedLlm(model="rep", script=[
        call("read_report_package"),
        call("build_pdf_report", title="Student grades", executive_summary="We predicted G3.",
             methodology="- 60/20/20 split", findings="G2 matters most.", limitations="Small sample.",
             recommendations="- Track early grades"),
        say("Your report is ready."),
    ])
    monkeypatch.setattr(agent_mod.verify_agent, "model", verify)
    monkeypatch.setattr(agent_mod.data_science_agent, "model", ds)
    monkeypatch.setattr(agent_mod.report_agent, "model", rep)
    return verify, ds, rep


async def _send(runner, session, parts):
    events = []
    async for ev in runner.run_async(user_id="u", session_id=session.id,
                                     new_message=types.Content(role="user", parts=parts)):
        events.append(ev)
    return events


def test_full_flow(scripted, tmp_path):
    verify, ds, rep = scripted

    async def go():
        runner = InMemoryRunner(agent=agent_mod.root_agent, app_name="data_team")
        session = await runner.session_service.create_session(app_name="data_team", user_id="u")
        await _send(runner, session, [types.Part(text="hello")])
        await _send(runner, session, [types.Part(text="here is my data"),
                                      _attach(SAMPLES / "student-mat.csv"), _attach(SAMPLES / "student-por.csv")])
        evs = await _send(runner, session, [types.Part(text="predict the final grade G3")])
        authors = [e.author for e in evs]
        assert "data_science_agent" in authors and "report_agent" in authors
        s = await runner.session_service.get_session(app_name="data_team", user_id="u", session_id=session.id)
        run = Path(s.state["run_dir"])
        for f in ["user_responses.txt", "dataVerified.txt", "usergoal.txt", "dataset.csv", "dataset_changes.txt",
                  "model_selection_why.txt", "metrics.json", "report_agent_package.txt", "report.pdf"]:
            assert (run / f).exists(), f
        assert "Stacked 2 files" in (run / "dataVerified.txt").read_text()
        log = (run / "user_responses.txt").read_text()
        assert "student-mat.csv" in log and "G3" in log and "school;sex" not in log  # names, not contents
        arts = await runner.artifact_service.list_artifact_keys(app_name="data_team", user_id="u", session_id=session.id)
        assert any(a.endswith(".pdf") for a in arts)
        # Raw file bytes never reach the model.
        for req in verify.seen_requests:
            for c in req.contents:
                assert not any(p.inline_data for p in c.parts or [])
        # A later message starts a new analysis with the Verify Agent.
        evs = await _send(runner, session, [types.Part(text="now analyse the wine data")])
        assert any(e.author == "verify_agent" for e in evs)
        s = await runner.session_service.get_session(app_name="data_team", user_id="u", session_id=session.id)
        assert s.state["run_dir"].endswith("analysis_2")
        return evs

    asyncio.run(go())
    assert not verify.script and not ds.script and not rep.script
