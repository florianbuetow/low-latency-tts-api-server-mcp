"""Integration tests that exercise the real service and MCP protocol paths."""

from __future__ import annotations

import base64
import io
import json
import os
import queue
import socket
import subprocess
import threading
import time
import wave
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest
import uvicorn
from fastapi.testclient import TestClient

import low_latency_tts_service_mcp.tts as tts_module
from low_latency_tts_service_mcp.server import app


class _RecordingOutputStream:
    """sounddevice.OutputStream replacement that records audio writes."""

    writes: list[np.ndarray[Any, Any]] = []
    started_count = 0
    stopped_count = 0
    closed_count = 0

    def __init__(self, *, samplerate: int, channels: int, dtype: str) -> None:
        self.samplerate = samplerate
        self.channels = channels
        self.dtype = dtype

    @classmethod
    def reset(cls) -> None:
        cls.writes = []
        cls.started_count = 0
        cls.stopped_count = 0
        cls.closed_count = 0

    def start(self) -> object:
        type(self).started_count += 1
        return None

    def stop(self) -> object:
        type(self).stopped_count += 1
        return None

    def close(self) -> object:
        type(self).closed_count += 1
        return None

    def write(self, data: np.ndarray[Any, Any]) -> object:
        type(self).writes.append(np.array(data, copy=True))
        return None


class _McpStdioClient:
    """Small JSON-RPC line client for the MCP stdio transport."""

    def __init__(self, process: subprocess.Popen[str]) -> None:
        self.process = process
        self._next_id = 1
        self._stdout_lines: queue.Queue[str] = queue.Queue()
        self._stderr_lines: queue.Queue[str] = queue.Queue()
        if process.stdin is None or process.stdout is None or process.stderr is None:
            raise RuntimeError("MCP process was not started with stdio pipes")
        self._stdin = process.stdin
        self._start_reader(process.stdout, self._stdout_lines)
        self._start_reader(process.stderr, self._stderr_lines)

    def _start_reader(self, pipe: Any, target: queue.Queue[str]) -> None:
        def read_lines() -> None:
            for line in pipe:
                target.put(line)

        threading.Thread(target=read_lines, daemon=True).start()

    def initialize(self) -> dict[str, Any]:
        response = self.request(
            "initialize",
            {
                "protocolVersion": "2025-11-25",
                "capabilities": {},
                "clientInfo": {"name": "pytest", "version": "0.1.0"},
            },
        )
        self.notify("notifications/initialized", {})
        return response

    def request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        request_id = self._next_id
        self._next_id += 1
        self._write({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params})
        return self._read_response(request_id)

    def notify(self, method: str, params: dict[str, Any]) -> None:
        self._write({"jsonrpc": "2.0", "method": method, "params": params})

    def close(self) -> None:
        self.process.terminate()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=5)

    def _write(self, message: dict[str, Any]) -> None:
        self._stdin.write(json.dumps(message) + "\n")
        self._stdin.flush()

    def _read_response(self, request_id: int) -> dict[str, Any]:
        deadline = time.monotonic() + 10.0
        ignored: list[dict[str, Any]] = []
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise AssertionError(f"MCP process exited early: {self._drain_stderr()}")
            try:
                line = self._stdout_lines.get(timeout=0.1)
            except queue.Empty:
                continue
            message = json.loads(line)
            if message.get("id") == request_id:
                if "error" in message:
                    raise AssertionError(f"MCP request failed: {message['error']}")
                return message
            ignored.append(message)
        raise AssertionError(f"Timed out waiting for MCP response {request_id}; ignored={ignored}; stderr={self._drain_stderr()}")

    def _drain_stderr(self) -> list[str]:
        lines: list[str] = []
        while True:
            try:
                lines.append(self._stderr_lines.get_nowait().rstrip())
            except queue.Empty:
                return lines


def _wav_payload_base64() -> str:
    buffer = io.BytesIO()
    samples = (np.array([0.0, 0.25, -0.25, 0.5, -0.5], dtype=np.float32) * 32767).astype(np.int16)
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(24000)
        wav_file.writeframes(samples.tobytes())
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _write_fake_tts_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    call_log = tmp_path / "fake-tts-calls.log"
    monkeypatch.setenv("KOKORO_FAKE_TTS_LOG", str(call_log))
    tts_cli = tmp_path / "tts-cli"
    payload = _wav_payload_base64()
    tts_cli.write_text(
        f"""#!/usr/bin/env bash
set -euo pipefail
save_path=""
voice=""
prompt=""
model_path=""
while [ "$#" -gt 0 ]; do
    case "$1" in
        --save-path)
            save_path="$2"
            shift 2
            ;;
        --voice)
            voice="$2"
            shift 2
            ;;
        --prompt)
            prompt="$2"
            shift 2
            ;;
        --model-path)
            model_path="$2"
            shift 2
            ;;
        --n-threads)
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done
if [ -z "$save_path" ]; then
    echo "missing --save-path" >&2
    exit 11
fi
if [ -z "${{KOKORO_FAKE_TTS_LOG:-}}" ]; then
    echo "missing KOKORO_FAKE_TTS_LOG" >&2
    exit 12
fi
{{
    printf 'voice=%s\\n' "$voice"
    printf 'prompt=%s\\n' "$prompt"
    printf 'model=%s\\n' "$model_path"
    printf 'save=%s\\n' "$save_path"
    printf -- '---\\n'
}} >> "$KOKORO_FAKE_TTS_LOG"
mkdir -p "$(dirname "$save_path")"
node - "$save_path" <<'NODE'
const fs = require("fs");
const output = process.argv[2];
const payload = Buffer.from("{payload}", "base64");
fs.writeFileSync(output, payload);
NODE
""",
        encoding="utf-8",
    )
    tts_cli.chmod(0o755)
    return tts_cli, call_log


def _write_config(tmp_path: Path, tts_cli: Path, host: str, port: int) -> Path:
    model_path = tmp_path / "Kokoro_no_espeak.gguf"
    model_path.write_bytes(b"fake model")
    # Every word in these tests is in misaki's lexicon, so the fallback must never run.
    phonemize_cli = tmp_path / "phonemize"
    phonemize_cli.write_text("#!/bin/sh\necho 'unexpected phonemize fallback call' >&2\nexit 1\n", encoding="utf-8")
    phonemize_cli.chmod(0o755)
    output_dir = tmp_path / "output"
    config_path = tmp_path / "config.yaml"
    lines = [
        f"tts_cli: {json.dumps(str(tts_cli))}",
        f"phonemize_cli: {json.dumps(str(phonemize_cli))}",
        f"model: {json.dumps(str(model_path))}",
        f"output_dir: {json.dumps(str(output_dir))}",
        "sample_rate: 24000",
        "lead_silence_ms: 10",
        "default_voice: af_heart",
        "save_wav: true",
        "simplify_punctuation: true",
        "n_threads: 2",
        "timeout_seconds: 10",
        f"host: {host}",
        f"port: {port}",
    ]
    config_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return config_path


def _prepare_service(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, host: str, port: int) -> tuple[Path, Path]:
    tts_cli, call_log = _write_fake_tts_cli(tmp_path, monkeypatch)
    config_path = _write_config(tmp_path, tts_cli, host, port)
    monkeypatch.setattr(tts_module, "CONFIG_PATH", config_path)
    _RecordingOutputStream.reset()
    monkeypatch.setattr("low_latency_tts_service_mcp.tts.sd.OutputStream", _RecordingOutputStream)
    return config_path, call_log


def _wait_for_completed_status(client: TestClient, message_id: str) -> dict[str, Any]:
    deadline = time.monotonic() + 10.0
    seen: list[str] = []
    while time.monotonic() < deadline:
        response = client.get(f"/status/{message_id}")
        assert response.status_code == 200
        body = response.json()
        seen.append(body["status"])
        if body["status"] == "completed":
            return body
        if body["status"] == "error":
            raise AssertionError(body)
        time.sleep(0.05)
    raise AssertionError(f"Timed out waiting for completion; seen statuses: {seen}")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _start_uvicorn(port: int) -> tuple[uvicorn.Server, threading.Thread]:
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning", lifespan="on")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10.0
    url = f"http://127.0.0.1:{port}/health"
    while time.monotonic() < deadline:
        try:
            response = httpx.get(url, timeout=0.2)
            if response.status_code == 200:
                return server, thread
        except httpx.HTTPError:
            time.sleep(0.05)
    server.should_exit = True
    thread.join(timeout=5)
    raise AssertionError(f"Uvicorn server did not become healthy at {url}")


def _tool_result_text(result: dict[str, Any]) -> dict[str, Any]:
    content = result["content"]
    assert isinstance(content, list)
    assert content[0]["type"] == "text"
    return json.loads(content[0]["text"])


def _call_tool(client: _McpStdioClient, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    response = client.request("tools/call", {"name": name, "arguments": arguments})
    return response["result"]


def _wait_for_mcp_completed_status(client: _McpStdioClient, message_id: str) -> dict[str, Any]:
    deadline = time.monotonic() + 10.0
    seen: list[str] = []
    while time.monotonic() < deadline:
        result = _call_tool(client, "get_status", {"message_id": message_id})
        status_body = _tool_result_text(result)
        seen.append(status_body["status"])
        if status_body["status"] == "completed":
            return status_body
        if status_body["status"] == "error":
            raise AssertionError(status_body)
        time.sleep(0.05)
    raise AssertionError(f"Timed out waiting for MCP completion; seen statuses: {seen}")


def test_fastapi_lifespan_generates_wav_and_plays_audio(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _config_path, call_log = _prepare_service(tmp_path, monkeypatch, "127.0.0.1", 12000)

    with TestClient(app) as client:
        assert client.get("/health").json() == {"status": "ok"}
        voices = client.get("/voices").json()
        assert voices["default_voice"] == "af_heart"
        assert {"af_heart", "bm_george"} <= set(voices["voices"])

        response = client.post("/say", json={"text": "  Hello, service integration!  ", "voice": "bm_george"})
        assert response.status_code == 202
        message_id = response.json()["message_id"]
        completed = _wait_for_completed_status(client, message_id)

    audio_file = completed["audio_file"]
    assert audio_file is not None
    assert Path(audio_file).is_file()
    assert "voice=bm_george" in call_log.read_text(encoding="utf-8")
    assert "prompt=həlˈO sˈɜɹvəs ˌɪntəɡɹˈAʃən." in call_log.read_text(encoding="utf-8")
    assert _RecordingOutputStream.started_count == 1
    assert _RecordingOutputStream.closed_count == 1
    assert len(_RecordingOutputStream.writes) == 2
    assert _RecordingOutputStream.writes[0].shape == (240, 1)
    assert _RecordingOutputStream.writes[1].shape == (5, 1)


def test_mcp_stdio_tools_drive_the_real_service(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    port = _free_port()
    config_path, call_log = _prepare_service(tmp_path, monkeypatch, "127.0.0.1", port)
    server, thread = _start_uvicorn(port)
    mcp_dir = Path.cwd() / "mcp"
    tsx = mcp_dir / "node_modules" / ".bin" / "tsx"
    if not tsx.is_file():
        raise AssertionError("MCP dependencies are missing; run `just mcp-install`")

    env = os.environ.copy()
    env["KOKORO_TTS_CONFIG_PATH"] = str(config_path)
    process = subprocess.Popen(
        [str(tsx), "tts-mcp.ts"],
        cwd=mcp_dir,
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    mcp_client = _McpStdioClient(process)
    try:
        initialize = mcp_client.initialize()
        assert initialize["result"]["serverInfo"]["name"] == "kokoro-tts-mcp"

        tools = mcp_client.request("tools/list", {})["result"]["tools"]
        tool_names = {tool["name"] for tool in tools}
        assert {"get_voices", "say", "get_status"} <= tool_names

        voices_result = _call_tool(mcp_client, "get_voices", {})
        voices_body = _tool_result_text(voices_result)
        assert voices_body["default_voice"] == "af_heart"
        assert "bm_george" in voices_body["voices"]

        bad_voice = _call_tool(mcp_client, "say", {"text": "Hello", "voice": "not_a_voice"})
        assert bad_voice["isError"] is True
        bad_voice_body = _tool_result_text(bad_voice)
        assert bad_voice_body["status_code"] == 400

        say_result = _call_tool(mcp_client, "say", {"text": "Hello, MCP integration!", "voice": "bm_george"})
        say_body = _tool_result_text(say_result)
        assert say_body["status"] == "queued"
        completed = _wait_for_mcp_completed_status(mcp_client, say_body["message_id"])
    finally:
        mcp_client.close()
        server.should_exit = True
        thread.join(timeout=5)

    audio_file = completed["audio_file"]
    assert audio_file is not None
    assert Path(audio_file).is_file()
    log_text = call_log.read_text(encoding="utf-8")
    assert "voice=bm_george" in log_text
    assert "prompt=həlˈO ˌɛmsˌipˈi ˌɪntəɡɹˈAʃən." in log_text
