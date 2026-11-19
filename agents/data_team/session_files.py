"""Where each chat session keeps its files, plus the two callbacks that handle uploads.

runs/<date>_<session>/              one folder per chat session
    uploads/                        files the user attached in the chat
    user_responses.txt              everything the user typed (file names only)
    analysis_1/, analysis_2/ ...    one folder per dataset analysed (the payloads)
"""
from __future__ import annotations

import mimetypes
import re
import shutil
from datetime import datetime
from pathlib import Path

from google.genai import types

from . import pipeline

_EXT_FOR_MIME = {
    "text/csv": ".csv",
    "application/json": ".json",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
}


def session_dir(state, session_id: str = "") -> Path:
    if not state.get("session_dir"):
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        d = pipeline.RUNS_DIR / f"{stamp}_{(session_id or 'session')[:8]}"
        (d / "uploads").mkdir(parents=True, exist_ok=True)
        state["session_dir"] = str(d)
    return Path(state["session_dir"])


def new_analysis_dir(state, session_id: str = "") -> Path:
    sdir = session_dir(state, session_id)
    n = int(state.get("analysis_count", 0)) + 1
    state["analysis_count"] = n
    d = sdir / f"analysis_{n}"
    (d / "charts").mkdir(parents=True, exist_ok=True)
    log = sdir / "user_responses.txt"
    if log.exists():
        shutil.copy(log, d / "user_responses.txt")
    state["run_dir"] = str(d)
    return d


def current_run_dir(state) -> Path | None:
    return Path(state["run_dir"]) if state.get("run_dir") else None


def _safe_name(name: str, mime: str, idx: int) -> str:
    name = Path(name or "").name
    name = re.sub(r"[^A-Za-z0-9._\- ]+", "_", name).strip() or f"upload_{idx}"
    if not Path(name).suffix:
        name += _EXT_FOR_MIME.get(mime, mimetypes.guess_extension(mime or "") or "")
    return name


async def save_uploads_and_log(callback_context) -> None:
    """before_agent_callback: save attached files to disk and log what the user said.

    Runs once per user message (whichever agent receives it first)."""
    state = callback_context.state
    inv = callback_context.invocation_id
    if state.get("_logged_invocation") == inv:
        return None
    state["_logged_invocation"] = inv
    content = callback_context.user_content
    if not content or not content.parts:
        return None
    sdir = session_dir(state, callback_context.session.id)
    saved, texts = [], []
    for i, part in enumerate(content.parts):
        if part.inline_data and part.inline_data.data:
            name = _safe_name(part.inline_data.display_name, part.inline_data.mime_type, i)
            (sdir / "uploads" / name).write_bytes(part.inline_data.data)
            saved.append(name)
        elif part.text:
            texts.append(part.text)
    if saved:
        state["latest_uploads"] = saved
    text = " ".join(texts)
    pipeline.log_user_response(sdir, text, saved)
    run = current_run_dir(state)
    if run and run.exists():
        pipeline.log_user_response(run, text, saved)
    return None


def strip_file_bytes(callback_context, llm_request) -> None:
    """before_model_callback: never send raw file bytes to Gemini.

    The tools read the saved copy from disk instead. This keeps requests small and
    avoids errors for formats Gemini can't read (like .xlsx)."""
    for content in llm_request.contents or []:
        new_parts = []
        for part in content.parts or []:
            if part.inline_data:
                name = part.inline_data.display_name or part.inline_data.mime_type
                new_parts.append(types.Part(text=f"[User attached file: {name}. It has been saved; "
                                                 "call load_dataset('uploaded') to read it.]"))
            else:
                new_parts.append(part)
        content.parts = new_parts
    return None
