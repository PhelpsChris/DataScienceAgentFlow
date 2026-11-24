"""The AI data team: Verify Agent -> Data Science Agent -> Report Agent.

`adk web` looks for `root_agent` in this file.

The chain mirrors our design chart. Each agent is a sub-agent of the one before
it, and the hand-off happens inside a tool (confirm_goal, bundle_report_package)
once that agent's payload files are written.
"""
from __future__ import annotations

import os

from dotenv import load_dotenv
from google.adk.agents import LlmAgent

from . import live_graph, tools
from .prompts import DS_PROMPT, REPORT_PROMPT, VERIFY_PROMPT
from .session_files import save_uploads_and_log, strip_file_bytes

load_dotenv()
MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")

# Callbacks shared by all three agents. The live_graph ones only report what is
# running to the live agent graph page; they never change the agents' behaviour.
COMMON_CALLBACKS = dict(
    before_agent_callback=[live_graph.on_agent_start, save_uploads_and_log],
    after_agent_callback=live_graph.on_agent_end,
    before_model_callback=[strip_file_bytes, live_graph.on_model_start],
    after_model_callback=live_graph.on_model_end,
    before_tool_callback=live_graph.on_tool_start,
    after_tool_callback=live_graph.on_tool_end,
    on_tool_error_callback=live_graph.on_tool_error,
)

report_agent = LlmAgent(
    name="report_agent",
    model=MODEL,
    description="Turns the Data Science Agent's package into a five-section PDF report with charts.",
    instruction=REPORT_PROMPT,
    tools=[tools.read_report_package, tools.build_pdf_report],
    **COMMON_CALLBACKS,
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
)

data_science_agent = LlmAgent(
    name="data_science_agent",
    model=MODEL,
    description="Cleans the verified dataset, trains and compares models, and makes charts.",
    instruction=DS_PROMPT,
    tools=[tools.read_verify_payload, tools.clean_dataset, tools.train_models,
           tools.make_charts, tools.bundle_report_package],
    sub_agents=[report_agent],
    **COMMON_CALLBACKS,
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
)

verify_agent = LlmAgent(
    name="verify_agent",
    model=MODEL,
    description="Greets the user, loads and checks their dataset, and agrees on the analysis goal.",
    instruction=VERIFY_PROMPT,
    tools=[tools.list_sample_datasets, tools.load_dataset, tools.check_goal, tools.create_target,
           tools.confirm_goal],
    sub_agents=[data_science_agent],
    **COMMON_CALLBACKS,
)

root_agent = verify_agent

# Live agent graph page (http://localhost:8765). LIVE_GRAPH=0 turns it off.
live_graph.set_graph(root_agent)
live_graph.maybe_start_from_env()
