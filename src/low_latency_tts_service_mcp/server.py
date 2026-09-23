"""FastAPI server that queues Kokoro TTS.cpp generation and playback."""

from __future__ import annotations

import contextlib
import dataclasses
import datetime
import json
import logging
import queue
import threading
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import APIRouter, FastAPI, HTTPException, Request
from pydantic import BaseModel

from low_latency_tts_service_mcp.tts import (
    AudioPlayer,
    KokoroRuntimeConfig,
    PlaybackJob,
    clean_text,
    generate_wav,
    kokoro_voices,
    load_config,
    make_output_path,
    simplify_punctuation,
    validate_runtime_config,
)

logger = logging.getLogger("kokoro-tts-server")

STATUS_TTL_SECONDS: int = 3600


@dataclasses.dataclass(frozen=True)
class WorkItem:
    """A queued TTS request for the audio worker."""

    message_id: str
    text: str
    voice: str


@dataclasses.dataclass
class MessageStatus:
    """Lifecycle record for one queued message."""

    message_id: str
    status: str
    text: str
    audio_file: str | None
    error: str | None
    completed_at: float | None


class ServerState:
    """Mutable server state shared between endpoints and the worker."""

    def __init__(
        self,
        runtime: KokoroRuntimeConfig,
        output_dir: Path,
        voices: list[str],
        default_voice: str,
        sample_rate: int,
        lead_silence_ms: int,
        simplify_punctuation_enabled: bool,
        save_wav: bool,
    ) -> None:
        """Initialize server state from fully validated settings."""
        self.runtime = runtime
        self.output_dir = output_dir
        self.voices = voices
        self.default_voice = default_voice
        self.sample_rate = sample_rate
        self.lead_silence_ms = lead_silence_ms
        self.simplify_punctuation_enabled = simplify_punctuation_enabled
        self.save_wav = save_wav
        self.work_queue: queue.Queue[WorkItem | None] = queue.Queue()
        self.statuses: dict[str, MessageStatus] = {}
        self.status_lock = threading.Lock()
        self._counter = 0
        self._counter_lock = threading.Lock()

    def next_message_id(self) -> str:
        """Generate a unique message ID."""
        with self._counter_lock:
            self._counter += 1
            counter = self._counter
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"msg_{timestamp}_{counter:03d}"

    def evict_expired(self) -> None:
        """Remove completed or failed status entries older than the status TTL."""
        now = time.time()
        with self.status_lock:
            expired = [
                message_id
                for message_id, status in self.statuses.items()
                if status.completed_at is not None and (now - status.completed_at) > STATUS_TTL_SECONDS
            ]
            for message_id in expired:
                del self.statuses[message_id]


class SayRequest(BaseModel):
    """Request body for POST /say."""

    text: str
    voice: str | None = None


class SayResponse(BaseModel):
    """Response body for POST /say."""

    message_id: str
    status: str
    queue_position: int


class StatusResponse(BaseModel):
    """Response body for GET /status/{message_id}."""

    message_id: str
    status: str
    text: str
    audio_file: str | None
    error: str | None


class VoicesResponse(BaseModel):
    """Response body for GET /voices."""

    voices: list[str]
    default_voice: str


class HealthResponse(BaseModel):
    """Response body for GET /health."""

    status: str


router = APIRouter()


@router.get("/health")
def health() -> HealthResponse:
    """Return server liveness."""
    return HealthResponse(status="ok")


@router.get("/voices")
def voices(request: Request) -> VoicesResponse:
    """List available Kokoro voices."""
    state: ServerState = request.app.state.server
    return VoicesResponse(voices=state.voices, default_voice=state.default_voice)


@router.post("/say", status_code=202)
def say(request: Request, body: SayRequest) -> SayResponse:
    """Queue text for Kokoro speech synthesis and playback."""
    state: ServerState = request.app.state.server

    cleaned = clean_text(body.text)
    if not cleaned:
        raise HTTPException(status_code=422, detail="Text is empty after cleaning")

    if state.simplify_punctuation_enabled:
        cleaned = simplify_punctuation(cleaned)

    voice = body.voice if body.voice else state.default_voice
    if voice not in state.voices:
        raise HTTPException(
            status_code=400,
            detail=f"Voice '{voice}' not available. Available voices: {', '.join(state.voices)}",
        )

    message_id = state.next_message_id()
    state.evict_expired()
    queue_position = state.work_queue.qsize()

    with state.status_lock:
        state.statuses[message_id] = MessageStatus(
            message_id=message_id,
            status="queued",
            text=cleaned,
            audio_file=None,
            error=None,
            completed_at=None,
        )

    state.work_queue.put(WorkItem(message_id=message_id, text=cleaned, voice=voice))

    logger.debug(
        "POST /say request:\n%s",
        json.dumps({"text": body.text, "voice": voice, "message_id": message_id}, indent=2),
    )

    return SayResponse(message_id=message_id, status="queued", queue_position=queue_position)


@router.get("/status/{message_id}")
def status(request: Request, message_id: str) -> StatusResponse:
    """Return the status of a queued, generating, playing, completed, or failed message."""
    state: ServerState = request.app.state.server
    state.evict_expired()

    with state.status_lock:
        message_status = state.statuses.get(message_id)

    if message_status is None:
        raise HTTPException(status_code=404, detail=f"Unknown message ID: {message_id}")

    return StatusResponse(
        message_id=message_status.message_id,
        status=message_status.status,
        text=message_status.text,
        audio_file=message_status.audio_file,
        error=message_status.error,
    )


def _fail_item(state: ServerState, message_id: str, error: str) -> None:
    """Mark one queued item as failed."""
    with state.status_lock:
        message_status = state.statuses.get(message_id)
        if message_status is not None:
            message_status.status = "error"
            message_status.error = error
            message_status.completed_at = time.time()


def _start_playback(
    state: ServerState,
    player: AudioPlayer,
    pending: tuple[WorkItem, Path, Path | None, bool],
    playback_done: threading.Event | None,
) -> threading.Event:
    """Wait for prior playback, then queue the next generated WAV."""
    if playback_done is not None:
        playback_done.wait()

    work_item, wav_path, reported_path, delete_after_playback = pending
    done = threading.Event()

    def on_complete(output_path: Path | None) -> None:
        with state.status_lock:
            message_status = state.statuses[work_item.message_id]
            message_status.status = "completed"
            message_status.audio_file = str(output_path) if output_path is not None else None
            message_status.completed_at = time.time()
        logger.debug("Playback completed for %s -> %s", work_item.message_id, output_path)
        done.set()

    def on_error(error: Exception) -> None:
        logger.error("Playback failed for %s: %s", work_item.message_id, error)
        with state.status_lock:
            message_status = state.statuses[work_item.message_id]
            message_status.status = "error"
            message_status.error = str(error)
            message_status.completed_at = time.time()
        done.set()

    with state.status_lock:
        state.statuses[work_item.message_id].status = "playing"

    player.submit(
        PlaybackJob(
            wav_path=wav_path,
            reported_path=reported_path,
            delete_after_playback=delete_after_playback,
            on_complete=on_complete,
            on_error=on_error,
        )
    )
    return done


def _generate_item(state: ServerState, item: WorkItem) -> tuple[Path, Path | None, bool] | None:
    """Generate a WAV for one work item and return playback metadata."""
    output_path = make_output_path(state.output_dir)
    reported_path = output_path if state.save_wav else None
    delete_after_playback = not state.save_wav

    with state.status_lock:
        state.statuses[item.message_id].status = "generating"

    try:
        generated_path = generate_wav(state.runtime, item.text, item.voice, output_path)
        return generated_path, reported_path, delete_after_playback
    except (FileExistsError, RuntimeError, TimeoutError, ValueError) as error:
        logger.error("Kokoro generation failed for %s: %s", item.message_id, error)
        _fail_item(state, item.message_id, str(error))
        if delete_after_playback and output_path.exists():
            output_path.unlink()
        return None


def server_audio_worker(state: ServerState) -> None:
    """Background worker that generates Kokoro WAVs and plays them sequentially."""
    player = AudioPlayer(state.sample_rate, state.lead_silence_ms)
    pending: tuple[WorkItem, Path, Path | None, bool] | None = None
    playback_done: threading.Event | None = None

    try:
        while True:
            current_item: WorkItem | None = None
            try:
                if pending is not None:
                    playback_done = _start_playback(state, player, pending, playback_done)
                    pending = None

                item = state.work_queue.get()
                if item is None:
                    if playback_done is not None:
                        playback_done.wait()
                    break

                current_item = item
                generated = _generate_item(state, item)
                if generated is not None:
                    pending = (item, generated[0], generated[1], generated[2])

                state.work_queue.task_done()
            except Exception as error:
                logger.error("Audio worker caught unexpected error: %s", error, exc_info=True)
                if current_item is not None:
                    _fail_item(state, current_item.message_id, f"unexpected worker error: {error}")
                    with contextlib.suppress(ValueError):
                        state.work_queue.task_done()

        if pending is not None:
            playback_done = _start_playback(state, player, pending, playback_done)
            playback_done.wait()
    finally:
        player.close()


@dataclasses.dataclass(frozen=True)
class _ServerConfig:
    """Parsed server configuration from config.yaml."""

    runtime: KokoroRuntimeConfig
    output_dir: Path
    sample_rate: int
    default_voice: str
    simplify_punctuation_enabled: bool
    save_wav: bool
    lead_silence_ms: int
    host: str
    port: int


def _require(config: dict[str, object], key: str) -> object:
    """Fetch a required config key or raise a clear error."""
    if key not in config:
        raise ValueError(f"Missing required key '{key}' in config.yaml")
    value = config[key]
    if value is None:
        raise ValueError(f"Missing required key '{key}' in config.yaml")
    return value


def _require_str(config: dict[str, object], key: str) -> str:
    """Fetch a required string config key."""
    value = _require(config, key)
    if not isinstance(value, str):
        raise ValueError(f"'{key}' in config.yaml must be a string")
    return value


def _require_bool(config: dict[str, object], key: str) -> bool:
    """Fetch a required boolean config key."""
    value = _require(config, key)
    if not isinstance(value, bool):
        raise ValueError(f"'{key}' in config.yaml must be a boolean")
    return value


def _require_int(config: dict[str, object], key: str) -> int:
    """Fetch a required integer config key."""
    value = _require(config, key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"'{key}' in config.yaml must be an integer")
    return value


def _parse_server_config() -> _ServerConfig:
    """Load and validate server settings from config.yaml."""
    config = load_config()
    runtime = KokoroRuntimeConfig(
        tts_cli=Path(_require_str(config, "tts_cli")),
        phonemize_cli=Path(_require_str(config, "phonemize_cli")),
        model_path=Path(_require_str(config, "model")),
        n_threads=_require_int(config, "n_threads"),
        timeout_seconds=_require_int(config, "timeout_seconds"),
    )
    return _ServerConfig(
        runtime=runtime,
        output_dir=Path(_require_str(config, "output_dir")),
        sample_rate=_require_int(config, "sample_rate"),
        default_voice=_require_str(config, "default_voice"),
        simplify_punctuation_enabled=_require_bool(config, "simplify_punctuation"),
        save_wav=_require_bool(config, "save_wav"),
        lead_silence_ms=_require_int(config, "lead_silence_ms"),
        host=_require_str(config, "host"),
        port=_require_int(config, "port"),
    )


def _build_server_state(config: _ServerConfig) -> ServerState:
    """Build validated runtime state for the FastAPI app."""
    validate_runtime_config(config.runtime)
    voices = kokoro_voices()
    if config.default_voice not in voices:
        raise ValueError(f"default_voice '{config.default_voice}' not found. Available: {', '.join(voices)}")
    if config.sample_rate <= 0:
        raise ValueError(f"sample_rate must be > 0, got {config.sample_rate}")
    if config.lead_silence_ms < 0:
        raise ValueError(f"lead_silence_ms must be >= 0, got {config.lead_silence_ms}")

    return ServerState(
        runtime=config.runtime,
        output_dir=config.output_dir,
        voices=voices,
        default_voice=config.default_voice,
        sample_rate=config.sample_rate,
        lead_silence_ms=config.lead_silence_ms,
        simplify_punctuation_enabled=config.simplify_punctuation_enabled,
        save_wav=config.save_wav,
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    """Start the server worker and shut it down cleanly with the app."""
    state = _build_server_state(_parse_server_config())
    app.state.server = state
    worker = threading.Thread(target=server_audio_worker, args=(state,), daemon=True)
    worker.start()

    yield

    state.work_queue.put(None)
    worker.join(timeout=10)
    if worker.is_alive():
        raise RuntimeError("Kokoro audio worker did not shut down within 10 seconds")


app = FastAPI(lifespan=lifespan)
app.include_router(router)


def run_server() -> None:
    """Run the configured FastAPI app with uvicorn."""
    config = _parse_server_config()
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    import uvicorn

    uvicorn.run("low_latency_tts_service_mcp.server:app", host=config.host, port=config.port, log_level="debug")


if __name__ == "__main__":
    run_server()
