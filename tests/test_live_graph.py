"""The live agent graph: structure, the events the agents emit, and the web page."""
from __future__ import annotations

import asyncio
import json
import urllib.request

from google.adk.runners import InMemoryRunner
from google.genai import types

from data_team import agent as agent_mod
from data_team import live_graph
from test_agent_flow import SAMPLES, _attach, _send, scripted  # noqa: F401  (fixture)


def test_graph_has_every_agent_and_tool():
    g = live_graph.graph_from_agent(agent_mod.root_agent)
    ids = {n["id"] for n in g["nodes"]}
    assert g["root"] == "agent:verify_agent"
    for a in ["verify_agent", "data_science_agent", "report_agent"]:
        assert f"agent:{a}" in ids
    for t in ["load_dataset", "confirm_goal", "train_models", "bundle_report_package", "build_pdf_report"]:
        assert f"tool:{t}" in ids
    edges = {(e["from"], e["to"], e["kind"]) for e in g["edges"]}
    assert ("agent:verify_agent", "agent:data_science_agent", "agent") in edges
    assert ("agent:data_science_agent", "agent:report_agent", "agent") in edges
    assert ("agent:report_agent", "tool:build_pdf_report", "tool") in edges


def test_agents_report_what_is_running(scripted):  # noqa: F811
    live_graph.bus.clear()
    start = live_graph.bus.last_seq

    async def go():
        runner = InMemoryRunner(agent=agent_mod.root_agent, app_name="data_team")
        session = await runner.session_service.create_session(app_name="data_team", user_id="u")
        await _send(runner, session, [types.Part(text="hello")])
        await _send(runner, session, [_attach(SAMPLES / "student-mat.csv"), _attach(SAMPLES / "student-por.csv")])
        await _send(runner, session, [types.Part(text="predict the final grade G3")])

    asyncio.run(go())
    evs = live_graph.bus.since(start)
    seq = [(e["kind"], e["node"]) for e in evs]
    # every tool and Gemini call that started also finished
    for kind in ("tool", "model"):
        starts = sorted(n for k, n in seq if k == f"{kind}_start")
        ends = sorted(n for k, n in seq if k == f"{kind}_end")
        assert starts == ends, kind
    tools_run = [n for k, n in seq if k == "tool_start"]
    assert tools_run.index("tool:confirm_goal") < tools_run.index("tool:train_models") < tools_run.index("tool:build_pdf_report")
    # the tool event says which agent called it
    train = next(e for e in evs if e["kind"] == "tool_start" and e["node"] == "tool:train_models")
    assert train["agent"] == "agent:data_science_agent" and train["args"]["scenario_name"] == "all_features"
    # After a hand-off the parent never reports agent_end (ADK skips it), so the page
    # treats any agent_end as "this turn is over". Check that each turn ends with one.
    agent_evs = [(k, n) for k, n in seq if k.startswith("agent")]
    assert agent_evs[-1] == ("agent_end", "agent:report_agent")
    assert agent_evs.count(("agent_end", "agent:verify_agent")) == 2  # turns 1 and 2 (no hand-off)


def test_web_page_and_event_stream():
    live_graph.set_graph(agent_mod.root_agent)
    srv = live_graph.start_server(port=0)
    try:
        base = f"http://127.0.0.1:{srv.server_address[1]}"
        html = urllib.request.urlopen(base + "/", timeout=5).read().decode()
        assert "Live agent graph" in html
        g = json.loads(urllib.request.urlopen(base + "/graph.json", timeout=5).read())
        assert g["root"] == "agent:verify_agent"
        live_graph.bus.publish("tool_start", node="tool:train_models", agent="agent:data_science_agent")
        with urllib.request.urlopen(base + "/events", timeout=5) as r:
            assert r.headers["Content-Type"] == "text/event-stream"
            line = r.readline().decode()
            while not line.startswith("data: "):
                line = r.readline().decode()
            assert json.loads(line[6:])["kind"]
    finally:
        live_graph.stop_server()
