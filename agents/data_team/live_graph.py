"""Live agent graph: a small web page that shows the team as a tree
(agents -> sub-agents -> tools) and puts a blue border on whatever is running.

How it works
- The agents call the callbacks below (agent start/end, Gemini call start/end,
  tool start/end). Each one appends an event to an in-memory, thread-safe log.
- A tiny web server (Python standard library, its own thread) serves the page
  at http://localhost:8765 and streams the events to it with Server-Sent Events.
  Because it runs in its own thread, the page keeps updating even while a slow
  tool (like train_models) is busy.

The server starts when agent.py is imported (that is, when `adk web` loads the
agent). Set LIVE_GRAPH=0 to turn it off, LIVE_GRAPH_PORT to change the port.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

log = logging.getLogger(__name__)

PAGE = Path(__file__).with_name("live_graph.html")
DEFAULT_PORT = 8765


# --------------------------------------------------------------------- event log
class EventBus:
    """Append-only event log that web-server threads can wait on."""

    def __init__(self, maxlen: int = 2000):
        self._events: deque[dict] = deque(maxlen=maxlen)
        self._seq = 0
        self._cond = threading.Condition()

    def publish(self, kind: str, **fields) -> dict:
        with self._cond:
            self._seq += 1
            ev = {"seq": self._seq, "t": time.time(), "kind": kind, **fields}
            self._events.append(ev)
            self._cond.notify_all()
        return ev

    def since(self, seq: int) -> list[dict]:
        with self._cond:
            return [e for e in self._events if e["seq"] > seq]

    def wait(self, seq: int, timeout: float) -> list[dict]:
        with self._cond:
            if self._seq <= seq:
                self._cond.wait(timeout)
            return [e for e in self._events if e["seq"] > seq]

    @property
    def last_seq(self) -> int:
        with self._cond:
            return self._seq

    def clear(self) -> None:
        with self._cond:
            self._events.clear()


bus = EventBus()


# --------------------------------------------------------------- graph structure
def _tool_name(tool) -> str:
    return getattr(tool, "name", None) or getattr(tool, "__name__", None) or type(tool).__name__


def graph_from_agent(root) -> dict:
    """Nodes and edges for the agent tree, in the same shape as the ADK dev UI graph."""
    nodes, edges, seen = [], [], set()

    def add(node_id, kind, label, description=""):
        if node_id not in seen:
            seen.add(node_id)
            nodes.append({"id": node_id, "kind": kind, "label": label, "description": description})

    def walk(agent):
        aid = f"agent:{agent.name}"
        add(aid, "agent", agent.name, getattr(agent, "description", "") or "")
        for tool in getattr(agent, "tools", None) or []:
            tid = f"tool:{_tool_name(tool)}"
            add(tid, "tool", _tool_name(tool), (getattr(tool, "__doc__", "") or "").strip().split("\n")[0])
            edges.append({"from": aid, "to": tid, "kind": "tool"})
        for sub in getattr(agent, "sub_agents", None) or []:
            edges.append({"from": aid, "to": f"agent:{sub.name}", "kind": "agent"})
            walk(sub)

    walk(root)
    return {"root": f"agent:{root.name}", "nodes": nodes, "edges": edges}


_graph: dict = {"root": None, "nodes": [], "edges": []}


def set_graph(root_agent) -> None:
    global _graph
    _graph = graph_from_agent(root_agent)


# --------------------------------------------------------------------- callbacks
# All return None so they never change what the agents do.
def _ids(ctx) -> dict:
    out = {}
    try:
        out["invocation"] = ctx.invocation_id
        out["session"] = ctx.session.id
    except Exception:  # noqa: BLE001 - visual only, never break the agent
        pass
    return out


def on_agent_start(callback_context) -> None:
    bus.publish("agent_start", node=f"agent:{callback_context.agent_name}", **_ids(callback_context))
    return None


def on_agent_end(callback_context) -> None:
    bus.publish("agent_end", node=f"agent:{callback_context.agent_name}", **_ids(callback_context))
    return None


def on_model_start(callback_context, llm_request) -> None:
    bus.publish("model_start", node=f"agent:{callback_context.agent_name}", **_ids(callback_context))
    return None


def on_model_end(callback_context, llm_response) -> None:
    bus.publish("model_end", node=f"agent:{callback_context.agent_name}", **_ids(callback_context))
    return None


def on_tool_start(tool, args, tool_context) -> None:
    bus.publish("tool_start", node=f"tool:{tool.name}", agent=f"agent:{tool_context.agent_name}",
                args={k: _short(v) for k, v in (args or {}).items()}, **_ids(tool_context))
    return None


def on_tool_end(tool, args, tool_context, tool_response) -> None:
    ok = not (isinstance(tool_response, dict) and tool_response.get("ok") is False)
    bus.publish("tool_end", node=f"tool:{tool.name}", agent=f"agent:{tool_context.agent_name}", ok=ok,
                **_ids(tool_context))
    return None


def on_tool_error(tool, args, tool_context, error) -> None:
    bus.publish("tool_end", node=f"tool:{tool.name}", agent=f"agent:{tool_context.agent_name}", ok=False,
                error=str(error)[:200], **_ids(tool_context))
    return None


def _short(v, n: int = 60) -> str:
    s = v if isinstance(v, str) else json.dumps(v, default=str)
    return s if len(s) <= n else s[: n - 1] + "…"


# -------------------------------------------------------------------- web server
class _Handler(BaseHTTPRequestHandler):
    server_version = "LiveGraph/1.0"

    def log_message(self, *args):  # keep the adk web console clean
        pass

    def _send(self, code: int, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802 - http.server naming
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
        elif path == "/graph.json":
            self._send(200, json.dumps(_graph).encode(), "application/json")
        elif path == "/events.json":
            self._send(200, json.dumps(bus.since(0)).encode(), "application/json")
        elif path == "/events":
            self._stream()
        else:
            self._send(404, b"not found", "text/plain")

    def _stream(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        # Replay this server's history so a page opened mid-run catches up.
        seq = 0
        try:
            while not getattr(self.server, "stopping", False):
                events = bus.wait(seq, timeout=10)
                if events:
                    for ev in events:
                        self.wfile.write(f"data: {json.dumps(ev)}\n\n".encode())
                    seq = events[-1]["seq"]
                else:
                    self.wfile.write(b": keep-alive\n\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
            return


_server: ThreadingHTTPServer | None = None


def start_server(port: int | None = None, host: str = "127.0.0.1") -> ThreadingHTTPServer | None:
    """Start the graph page in a background thread (once). Returns None if the port is busy."""
    global _server
    if _server is not None:
        return _server
    port = DEFAULT_PORT if port is None else port
    try:
        srv = ThreadingHTTPServer((host, port), _Handler)
    except OSError as e:
        log.warning("Live agent graph not started: port %s is in use (%s). Set LIVE_GRAPH_PORT.", port, e)
        return None
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, name="live-graph", daemon=True).start()
    _server = srv
    shown = "localhost" if host in ("127.0.0.1", "0.0.0.0") else host
    print(f"\n  Live agent graph: http://{shown}:{srv.server_address[1]}\n", flush=True)
    return srv


def stop_server() -> None:
    global _server
    if _server is not None:
        _server.stopping = True
        _server.shutdown()
        _server.server_close()
        _server = None


def maybe_start_from_env() -> None:
    if os.environ.get("LIVE_GRAPH", "1").strip().lower() in ("0", "false", "no", "off"):
        return
    start_server(int(os.environ.get("LIVE_GRAPH_PORT", DEFAULT_PORT)),
                 os.environ.get("LIVE_GRAPH_HOST", "127.0.0.1"))
