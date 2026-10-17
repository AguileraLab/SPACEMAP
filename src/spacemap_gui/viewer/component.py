"""The viewer as a Streamlit component, and where its data routes live."""

from __future__ import annotations

import socket
import threading
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from . import api

FRONTEND_DIR = Path(__file__).resolve().parent / "frontend"
_component = components.declare_component("spacemap_viewer", path=str(FRONTEND_DIR))

_FALLBACK = {"port": None}
_FALLBACK_LOCK = threading.Lock()


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("", 0))
        return probe.getsockname()[1]


def _start_fallback_server() -> int:
    """Serve the routes on a second port when SPACEMAP was not started by server.py."""
    with _FALLBACK_LOCK:
        if _FALLBACK["port"]:
            return _FALLBACK["port"]
        import uvicorn
        from starlette.applications import Starlette
        from starlette.middleware import Middleware
        from starlette.middleware.cors import CORSMiddleware
        from starlette.routing import Mount

        port = _free_port()
        fallback = Starlette(
            routes=[Mount(api.MOUNT_PATH, app=api.app)],
            middleware=[
                Middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"])
            ],
        )
        server = uvicorn.Server(uvicorn.Config(fallback, host="0.0.0.0", port=port, log_level="warning"))
        threading.Thread(target=server.run, name="spacemap-viewer-api", daemon=True).start()
        _FALLBACK["port"] = port
        return port


def api_location() -> dict:
    """Where the browser should send viewer requests."""
    if api.MOUNTED:
        base = str(st.get_option("server.baseUrlPath") or "").strip("/")
        return {"api": (f"/{base}" if base else "") + api.MOUNT_PATH}
    return {"api": api.MOUNT_PATH, "api_port": _start_fallback_server()}


def spacemap_viewer(spec: dict, *, key: str | None = None):
    """Draw the viewer. Returns the last cell clicked, if any."""
    return _component(spec={**api_location(), **spec}, key=key, default=None)
