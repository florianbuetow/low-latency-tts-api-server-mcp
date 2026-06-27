"""Minimal uvicorn stubs used by this project."""

from __future__ import annotations

def run(app: str, *, host: str, port: int, log_level: str) -> None: ...
