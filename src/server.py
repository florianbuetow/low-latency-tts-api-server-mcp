"""Compatibility entry point for running the Kokoro FastAPI server."""

from __future__ import annotations

from low_latency_tts_service_mcp.server import run_server

if __name__ == "__main__":
    run_server()
