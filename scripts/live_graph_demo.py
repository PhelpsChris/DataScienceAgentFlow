"""Preview the live agent graph WITHOUT Gemini (no API key, no credits used).

Runs the real agents and tools on the student data, but Gemini is replaced by a
scripted stand-in that pauses a little before each step so you can watch the
blue highlight move: verify_agent -> data_science_agent -> report_agent.

    python scripts/live_graph_demo.py            # then open http://localhost:8765
    python scripts/live_graph_demo.py --pause 2  # slower
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
import tempfile
import logging
import time
import webbrowser
from pathlib import Path
from typing import AsyncGenerator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agents"))
os.environ["LIVE_GRAPH"] = "0"  # we start the page ourselves below

from google.adk.models.base_llm import BaseLlm  # noqa: E402
from google.adk.models.llm_request import LlmRequest  # noqa: E402
from google.adk.models.llm_response import LlmResponse  # noqa: E402
from google.adk.runners import InMemoryRunner  # noqa: E402
from google.genai import types  # noqa: E402

from data_team import agent as agent_mod  # noqa: E402
from data_team import live_graph, pipeline  # noqa: E402

PAUSE = 1.0


def call(name, **args):
    return LlmResponse(content=types.Content(role="model", parts=[
        types.Part(function_call=types.FunctionCall(name=name, args=args))]))


def say(text):
    return LlmResponse(content=types.Content(role="model", parts=[types.Part(text=text)]))


class ScriptedLlm(BaseLlm):
    model: str = "scripted"
    script: list = []

    async def generate_content_async(self, llm_request: LlmRequest, stream: bool = False) -> AsyncGenerator[LlmResponse, None]:
        await asyncio.sleep(PAUSE)  # "thinking"
        yield self.script.pop(0)


def install_script():
    agent_mod.verify_agent.model = ScriptedLlm(script=[
        say("Hi! Attach a .csv, .xlsx or .json file, or ask for a sample."),
        call("list_sample_datasets"), call("load_dataset", source="student"),
        say("Loaded 1,044 students. What would you like to analyze?"),
        call("check_goal", target_column="G3"),
        call("confirm_goal", goal="Predict students' final grade G3", target_column="G3"),
    ])
    agent_mod.data_science_agent.model = ScriptedLlm(script=[
        call("read_verify_payload"), call("clean_dataset"),
        call("train_models", scenario_name="all_features"),
        call("train_models", scenario_name="without_G1_G2", exclude_columns=["G1", "G2"]),
        call("make_charts"),
        call("bundle_report_package", analyst_notes="- Random Forest won\n- G2 dominates"),
    ])
    agent_mod.report_agent.model = ScriptedLlm(script=[
        call("read_report_package"),
        call("build_pdf_report", title="Student grades (demo)", executive_summary="Scripted demo run.",
             methodology="- 60/20/20 split", findings="G2 matters most.", limitations="Demo text.",
             recommendations="- Track early grades"),
        say("Your report is ready."),
    ])


async def run_once():
    install_script()
    runner = InMemoryRunner(agent=agent_mod.root_agent, app_name="data_team")
    session = await runner.session_service.create_session(app_name="data_team", user_id="demo")
    for text in ["hi", "use the student sample", "predict the final grade G3"]:
        print(f"> {text}")
        async for ev in runner.run_async(user_id="demo", session_id=session.id,
                                         new_message=types.Content(role="user", parts=[types.Part(text=text)])):
            for part in (ev.content.parts if ev.content and ev.content.parts else []):
                if part.text:
                    print(f"  [{ev.author}] {part.text}", flush=True)
                elif part.function_call:  # show each step so slow ones don't look frozen
                    print(f"  [{ev.author}] running {part.function_call.name} ...", flush=True)
        await asyncio.sleep(PAUSE * 2)


def main():
    global PAUSE
    ap = argparse.ArgumentParser()
    ap.add_argument("--pause", type=float, default=1.0, help="seconds to wait before each agent step")
    ap.add_argument("--port", type=int, default=int(os.environ.get("LIVE_GRAPH_PORT", live_graph.DEFAULT_PORT)))
    ap.add_argument("--wait", type=float, default=6.0, help="seconds to wait for you to open the page first")
    ap.add_argument("--loop", action="store_true", help="repeat the demo until Ctrl+C")
    ap.add_argument("--no-browser", action="store_true", help="don't open the graph page automatically")
    a = ap.parse_args()
    PAUSE = a.pause
    # Hide ADK's harmless warnings about the scripted model (no token counts, no context cache).
    for name in ("google_adk", "google.adk"):
        logging.getLogger(name).setLevel(logging.ERROR)
    pipeline.RUNS_DIR = Path(tempfile.mkdtemp(prefix="graph_demo_"))
    if live_graph.start_server(a.port) is None:
        sys.exit(f"Port {a.port} is busy (is adk web already running?). Try --port 8766.")
    url = f"http://localhost:{a.port}"
    print(f"Opening {url} in your browser. Starting in {a.wait:.0f}s...", flush=True)
    if not a.no_browser:
        webbrowser.open(url)
    time.sleep(a.wait)
    try:
        while True:
            asyncio.run(run_once())
            if not a.loop:
                break
        print(f"Done. The graph page stays up at {url} until you press Ctrl+C.", flush=True)
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
