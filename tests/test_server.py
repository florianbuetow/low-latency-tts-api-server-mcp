"""Tests for the FastAPI Kokoro TTS server."""

from __future__ import annotations

import re
import threading
import time
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from low_latency_tts_service_mcp.server import (
    STATUS_TTL_SECONDS,
    MessageStatus,
    ServerState,
    WorkItem,
    router,
    server_audio_worker,
)
from low_latency_tts_service_mcp.tts import KokoroRuntimeConfig


def _runtime_config(tmp_path: Path) -> KokoroRuntimeConfig:
    return KokoroRuntimeConfig(
        tts_cli=tmp_path / "tts-cli",
        model_path=tmp_path / "Kokoro_no_espeak.gguf",
        n_threads=2,
        timeout_seconds=30,
    )


def _make_state(
    tmp_path: Path,
    voices: list[str],
    default_voice: str,
    simplify_punctuation_enabled: bool,
    save_wav: bool,
) -> ServerState:
    return ServerState(
        runtime=_runtime_config(tmp_path),
        output_dir=tmp_path / "output",
        voices=voices,
        default_voice=default_voice,
        sample_rate=24000,
        lead_silence_ms=200,
        simplify_punctuation_enabled=simplify_punctuation_enabled,
        save_wav=save_wav,
    )


def _make_app(state: ServerState) -> FastAPI:
    app = FastAPI()
    app.state.server = state
    app.include_router(router)
    return app


class _ImmediateAudioPlayer:
    """Synchronous fake for server worker tests."""

    playback_error: Exception | None = None

    def __init__(self, sample_rate: int, lead_silence_ms: int) -> None:
        self.sample_rate = sample_rate
        self.lead_silence_ms = lead_silence_ms

    def submit(self, job: Any) -> None:
        if _ImmediateAudioPlayer.playback_error is not None:
            job.on_error(_ImmediateAudioPlayer.playback_error)
            return
        job.on_complete(job.reported_path)

    def close(self) -> None:
        return


@pytest.fixture(autouse=True)
def _use_immediate_audio_player(monkeypatch: pytest.MonkeyPatch) -> None:
    _ImmediateAudioPlayer.playback_error = None
    monkeypatch.setattr("low_latency_tts_service_mcp.server.AudioPlayer", _ImmediateAudioPlayer)


def test_health_returns_ok(tmp_path: Path) -> None:
    state = _make_state(tmp_path, ["af_heart"], "af_heart", False, True)
    client = TestClient(_make_app(state))

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_voices_returns_available_kokoro_voices(tmp_path: Path) -> None:
    state = _make_state(tmp_path, ["af_heart", "bm_george"], "af_heart", False, True)
    client = TestClient(_make_app(state))

    response = client.get("/voices")

    assert response.status_code == 200
    assert response.json() == {"voices": ["af_heart", "bm_george"], "default_voice": "af_heart"}


def test_message_id_format_and_increment(tmp_path: Path) -> None:
    state = _make_state(tmp_path, ["af_heart"], "af_heart", False, True)

    first = state.next_message_id()
    second = state.next_message_id()

    assert re.match(r"^msg_\d{8}_\d{6}_\d{3}$", first)
    assert int(second.rsplit("_", 1)[1]) == int(first.rsplit("_", 1)[1]) + 1


def test_say_queues_work_item_with_default_voice(tmp_path: Path) -> None:
    state = _make_state(tmp_path, ["af_heart", "bm_george"], "af_heart", False, True)
    client = TestClient(_make_app(state))

    response = client.post("/say", json={"text": "  Hello   world  "})

    assert response.status_code == 202
    data = response.json()
    assert data["status"] == "queued"
    item = state.work_queue.get_nowait()
    assert isinstance(item, WorkItem)
    assert item.text == "Hello world"
    assert item.voice == "af_heart"


def test_say_accepts_voice_override_and_simplifies_punctuation(tmp_path: Path) -> None:
    state = _make_state(tmp_path, ["af_heart", "bm_george"], "af_heart", True, True)
    client = TestClient(_make_app(state))

    response = client.post("/say", json={"text": "Hello, world!", "voice": "bm_george"})

    assert response.status_code == 202
    item = state.work_queue.get_nowait()
    assert isinstance(item, WorkItem)
    assert item.text == "Hello world."
    assert item.voice == "bm_george"


def test_say_rejects_empty_and_unknown_voice(tmp_path: Path) -> None:
    state = _make_state(tmp_path, ["af_heart"], "af_heart", False, True)
    client = TestClient(_make_app(state))

    empty = client.post("/say", json={"text": "   "})
    unknown = client.post("/say", json={"text": "Hello", "voice": "not_a_voice"})

    assert empty.status_code == 422
    assert unknown.status_code == 400
    assert "not_a_voice" in unknown.json()["detail"]


def test_status_returns_known_message_and_evicts_expired(tmp_path: Path) -> None:
    state = _make_state(tmp_path, ["af_heart"], "af_heart", False, True)
    expired_time = time.time() - STATUS_TTL_SECONDS - 1
    with state.status_lock:
        state.statuses["msg_known"] = MessageStatus(
            message_id="msg_known",
            status="completed",
            text="Hello",
            audio_file="out.wav",
            error=None,
            completed_at=time.time(),
        )
        state.statuses["msg_old"] = MessageStatus(
            message_id="msg_old",
            status="completed",
            text="Old",
            audio_file="old.wav",
            error=None,
            completed_at=expired_time,
        )
    client = TestClient(_make_app(state))

    known = client.get("/status/msg_known")
    old = client.get("/status/msg_old")

    assert known.status_code == 200
    assert known.json()["audio_file"] == "out.wav"
    assert old.status_code == 404


def test_server_audio_worker_generates_and_completes_message(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    state = _make_state(tmp_path, ["af_heart"], "af_heart", False, True)

    def fake_generate_wav(_runtime: KokoroRuntimeConfig, _text: str, _voice: str, output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"fake wav")
        return output_path

    monkeypatch.setattr("low_latency_tts_service_mcp.server.generate_wav", fake_generate_wav)

    msg_id = "msg_test_001"
    with state.status_lock:
        state.statuses[msg_id] = MessageStatus(
            message_id=msg_id,
            status="queued",
            text="Hello",
            audio_file=None,
            error=None,
            completed_at=None,
        )
    state.work_queue.put(WorkItem(message_id=msg_id, text="Hello", voice="af_heart"))
    state.work_queue.put(None)

    worker = threading.Thread(target=server_audio_worker, args=(state,))
    worker.start()
    worker.join(timeout=5)

    assert not worker.is_alive()
    with state.status_lock:
        assert state.statuses[msg_id].status == "completed"
        assert state.statuses[msg_id].audio_file is not None


def test_server_audio_worker_records_generation_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    state = _make_state(tmp_path, ["af_heart"], "af_heart", False, True)

    def fake_generate_wav(_runtime: KokoroRuntimeConfig, _text: str, _voice: str, _output_path: Path) -> Path:
        raise RuntimeError("generation failed")

    monkeypatch.setattr("low_latency_tts_service_mcp.server.generate_wav", fake_generate_wav)

    msg_id = "msg_test_002"
    with state.status_lock:
        state.statuses[msg_id] = MessageStatus(
            message_id=msg_id,
            status="queued",
            text="Hello",
            audio_file=None,
            error=None,
            completed_at=None,
        )
    state.work_queue.put(WorkItem(message_id=msg_id, text="Hello", voice="af_heart"))
    state.work_queue.put(None)

    worker = threading.Thread(target=server_audio_worker, args=(state,))
    worker.start()
    worker.join(timeout=5)

    assert not worker.is_alive()
    with state.status_lock:
        assert state.statuses[msg_id].status == "error"
        assert state.statuses[msg_id].error == "generation failed"
