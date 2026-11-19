"""Start the ADK chat UI and the live agent graph together, with one command.

    python scripts/web_with_graph.py          # chat: http://localhost:8000   graph: http://localhost:8765
    python scripts/web_with_graph.py --port 8080

Plain `adk web agents` also shows the graph, but only after you pick data_team in
the chat UI (that is when ADK loads the agents). This script loads them first, so
the graph page is ready immediately. Any extra options are passed to `adk web`.
"""
from __future__ import annotations

import sys
import webbrowser
from pathlib import Path

AGENTS = Path(__file__).resolve().parents[1] / "agents"
sys.path.insert(0, str(AGENTS))

import data_team.agent  # noqa: E402,F401  (starts the graph page; adk web reuses this module)
from data_team import live_graph  # noqa: E402
from google.adk.cli.cli_tools_click import main  # noqa: E402

if __name__ == "__main__":
    extra = sys.argv[1:]
    port = extra[extra.index("--port") + 1] if "--port" in extra else "8000"
    if live_graph._server is not None:
        graph_url = f"http://localhost:{live_graph._server.server_address[1]}"
        chat = f"http://localhost:{port}/dev-ui/?app=data_team"
        url = graph_url + (f"/?split=1&chat={chat}" if port != "8000" else "/?split=1")
        print(f"  Chat + graph side by side: {url}\n")
        if "--no-browser" not in extra:
            webbrowser.open(url)
    extra = [a for a in extra if a != "--no-browser"]
    sys.exit(main(["web", str(AGENTS), "--no-reload", *extra]))
