"""Kokoro TTS.cpp model processes, WAV playback, and text preparation."""

from __future__ import annotations

import collections
import dataclasses
import datetime
import functools
import os
import queue
import re
import subprocess
import threading
import wave
from collections.abc import Callable
from pathlib import Path
from typing import Protocol, cast

import numpy as np
import numpy.typing as npt
import sounddevice as sd
import yaml
from misaki import en
from misaki.token import MToken

CONFIG_PATH = Path("config.yaml")
OUTPUT_DIR = Path("data/output")

type FloatAudio = npt.NDArray[np.float32]

KOKORO_VOICES: tuple[str, ...] = (
    "af_alloy",
    "af_aoede",
    "af_bella",
    "af_heart",
    "af_jessica",
    "af_kore",
    "af_nicole",
    "af_nova",
    "af_river",
    "af_sarah",
    "af_sky",
    "am_adam",
    "am_echo",
    "am_eric",
    "am_fenrir",
    "am_liam",
    "am_michael",
    "am_onyx",
    "am_puck",
    "am_santa",
    "bf_alice",
    "bf_emma",
    "bf_isabella",
    "bf_lily",
    "bm_daniel",
    "bm_fable",
    "bm_george",
)
"""English Kokoro voice packs carried by the no-espeak GGUF."""


class AudioOutputStream(Protocol):
    """Runtime methods used from sounddevice.OutputStream."""

    def start(self) -> object:
        """Start the stream."""
        ...

    def stop(self) -> object:
        """Stop the stream."""
        ...

    def close(self) -> object:
        """Close the stream."""
        ...

    def write(self, data: FloatAudio) -> object:
        """Write audio frames to the stream."""
        ...


@dataclasses.dataclass(frozen=True)
class KokoroRuntimeConfig:
    """All settings needed to execute one Kokoro TTS.cpp generation."""

    tts_cli: Path
    phonemize_cli: Path
    model_path: Path
    n_threads: int
    timeout_seconds: int


@dataclasses.dataclass(frozen=True)
class PlaybackJob:
    """One generated WAV file queued for playback."""

    wav_path: Path
    reported_path: Path | None
    delete_after_playback: bool
    on_complete: Callable[[Path | None], None]
    on_error: Callable[[Exception], None]


def clean_text(text: str) -> str:
    """Strip text and collapse whitespace while preserving single newlines."""
    text = re.sub(r"(?i)\bmicrosoft\b", "Micro-soft", text)
    stripped = text.strip()
    stripped = re.sub(r"\t", " ", stripped)
    stripped = re.sub(r" {2,}", " ", stripped)
    stripped = re.sub(r"\n{2,}", "\n", stripped)
    return stripped.strip()


def simplify_punctuation(text: str) -> str:
    """Simplify punctuation for Kokoro prompts that perform better with plain marks."""
    simplified = text.replace(",", "")
    simplified = simplified.replace("\uff0c", "")
    simplified = simplified.replace("...", ".")
    simplified = simplified.replace("--", ".")
    simplified = simplified.replace("\u2019", "'")  # preserve right single quote as apostrophe

    for char in '!?;:()[]{}"`\u2014\u2013\u2026\u201c\u201d\u2018':
        simplified = simplified.replace(char, ".")

    simplified = re.sub(r"\.\s*(?:\.\s*)+", ".", simplified)
    simplified = re.sub(r"\s+\.", ".", simplified)
    simplified = re.sub(r"\.(?=[^\s.\d])", ". ", simplified)
    simplified = re.sub(r"^[\s.]+", "", simplified)
    return simplified.rstrip()


def load_config() -> dict[str, object]:
    """Load config.yaml as a mapping.

    Raises:
        FileNotFoundError: If config.yaml is absent.
        ValueError: If the file does not contain a mapping.
    """
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Configuration file not found: {CONFIG_PATH}")

    with CONFIG_PATH.open(encoding="utf-8") as config_file:
        loaded = yaml.safe_load(config_file)

    if not isinstance(loaded, dict):
        raise ValueError(f"Invalid config.yaml: expected a mapping, got {type(loaded).__name__}")

    return cast(dict[str, object], loaded)


def kokoro_voices() -> list[str]:
    """Return the built-in Kokoro no-espeak voice identifiers."""
    return list(KOKORO_VOICES)


def validate_voice(voice: str, voices: list[str]) -> None:
    """Raise when a requested voice is not present in the available voice list."""
    if voice not in voices:
        raise ValueError(f"Voice '{voice}' not available. Available voices: {', '.join(voices)}")


def validate_runtime_config(config: KokoroRuntimeConfig) -> None:
    """Validate that the configured TTS.cpp executable and Kokoro GGUF are usable."""
    if not config.tts_cli.is_file():
        raise FileNotFoundError(f"tts-cli binary not found: {config.tts_cli}")
    if not os.access(config.tts_cli, os.X_OK):
        raise PermissionError(f"tts-cli is not executable: {config.tts_cli}")
    if not config.phonemize_cli.is_file():
        raise FileNotFoundError(f"phonemize binary not found: {config.phonemize_cli}")
    if not os.access(config.phonemize_cli, os.X_OK):
        raise PermissionError(f"phonemize is not executable: {config.phonemize_cli}")
    if not config.model_path.is_file():
        raise FileNotFoundError(f"Kokoro GGUF model not found: {config.model_path}")
    if config.n_threads <= 0:
        raise ValueError(f"n_threads must be > 0, got {config.n_threads}")
    if config.timeout_seconds <= 0:
        raise ValueError(f"timeout_seconds must be > 0, got {config.timeout_seconds}")


def make_output_path(output_dir: Path) -> Path:
    """Build a unique timestamped WAV path under an output directory."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return output_dir / f"speech_{timestamp}.wav"


class ServeProcess:
    """A TTS.cpp binary started with --serve, keeping its model loaded between requests.

    Each request is one stdin line. The process answers it with one stdout line,
    "@@done<TAB><payload>" or "@@error<TAB><message>", and prints "@@ready" once loaded.
    Other stdout lines are log output and are ignored.
    """

    def __init__(self, command: tuple[str, ...], timeout_seconds: int) -> None:
        """Start the process and wait until its model is loaded."""
        self.command = command
        self._timeout_seconds = timeout_seconds
        self._lock = threading.Lock()
        self._responses: queue.Queue[str | None] = queue.Queue()
        self._stderr_tail: collections.deque[str] = collections.deque(maxlen=20)
        self._process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
        )
        self._stdout_reader = threading.Thread(target=self._read_stdout, daemon=True)
        self._stderr_reader = threading.Thread(target=self._read_stderr, daemon=True)
        self._stdout_reader.start()
        self._stderr_reader.start()
        ready = self._next_response()
        if ready != "@@ready":
            self.close()
            raise RuntimeError(f"{command[0]} did not report ready, got: {ready!r}")

    def _read_stdout(self) -> None:
        stdout = self._process.stdout
        if stdout is None:
            raise RuntimeError(f"{self.command[0]} has no stdout pipe")
        for line in stdout:
            if line.startswith("@@"):
                self._responses.put(line.rstrip("\n"))
        self._responses.put(None)

    def _read_stderr(self) -> None:
        stderr = self._process.stderr
        if stderr is None:
            raise RuntimeError(f"{self.command[0]} has no stderr pipe")
        for line in stderr:
            self._stderr_tail.append(line.rstrip("\n"))

    def is_alive(self) -> bool:
        """Return whether the process is still running."""
        return self._process.poll() is None

    def _next_response(self) -> str:
        try:
            response = self._responses.get(timeout=self._timeout_seconds)
        except queue.Empty as error:
            self._process.kill()
            self._process.wait()
            raise TimeoutError(f"{self.command[0]} did not answer within {self._timeout_seconds}s") from error
        if response is None:
            returncode = self._process.wait()
            self._stderr_reader.join(timeout=1)
            message = f"{self.command[0]} exited with status {returncode}"
            if self._stderr_tail:
                message += f" | stderr: {' | '.join(self._stderr_tail)}"
            raise RuntimeError(message)
        return response

    def request(self, line: str) -> str:
        """Send one request line and return the payload of the "@@done" answer.

        Raises:
            ValueError: If the request contains a newline.
            RuntimeError: If the process is not running, exits, or answers "@@error".
            TimeoutError: If no answer arrives within the timeout; the process is killed.
        """
        if "\n" in line:
            raise ValueError(f"Request must be a single line: {line!r}")
        with self._lock:
            stdin = self._process.stdin
            if stdin is None or not self.is_alive():
                raise RuntimeError(f"{self.command[0]} is not running")
            try:
                stdin.write(line + "\n")
                stdin.flush()
            except BrokenPipeError as error:
                raise RuntimeError(f"{self.command[0]} exited with status {self._process.wait()} before accepting the request") from error
            response = self._next_response()
        kind, _, payload = response.partition("\t")
        if kind == "@@done":
            return payload
        if kind == "@@error":
            raise RuntimeError(f"{self.command[0]} failed: {payload}")
        raise RuntimeError(f"Unexpected answer from {self.command[0]}: {response!r}")

    def close(self) -> None:
        """Close stdin so the process exits, killing it if it does not exit in time."""
        stdin = self._process.stdin
        if stdin is not None and not stdin.closed and self.is_alive():
            stdin.close()
        try:
            self._process.wait(timeout=self._timeout_seconds)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait()


class ServeProcessSlot:
    """Holds the one ServeProcess for a command, starting it on first use and after it exits."""

    def __init__(self, command: tuple[str, ...], timeout_seconds: int) -> None:
        """Create an empty slot; the process starts on the first ``get``."""
        self.command = command
        self._timeout_seconds = timeout_seconds
        self._process: ServeProcess | None = None
        self._lock = threading.Lock()

    def get(self) -> ServeProcess:
        """Return the running process, starting it if it is not running."""
        with self._lock:
            if self._process is None or not self._process.is_alive():
                self._process = ServeProcess(self.command, self._timeout_seconds)
            return self._process

    def close(self) -> None:
        """Close the process if one is running."""
        with self._lock:
            if self._process is not None:
                self._process.close()
                self._process = None


@functools.cache
def tts_model_slot(command: tuple[str, ...], timeout_seconds: int) -> ServeProcessSlot:
    """Return the process-wide slot holding the loaded Kokoro model for a tts-cli command."""
    return ServeProcessSlot(command, timeout_seconds)


def build_tts_serve_command(config: KokoroRuntimeConfig) -> tuple[str, ...]:
    """Build the tts-cli command that loads the Kokoro model once and serves requests."""
    return (
        str(config.tts_cli),
        "--serve",
        "--model-path",
        str(config.model_path),
        "--n-threads",
        str(config.n_threads),
    )


def load_tts_model(config: KokoroRuntimeConfig) -> ServeProcess:
    """Return the loaded Kokoro model process for this configuration, starting it if needed.

    The Kokoro GGUF carries every voice pack, so this one process serves all voices.
    """
    return tts_model_slot(build_tts_serve_command(config), config.timeout_seconds).get()


def unload_tts_model(config: KokoroRuntimeConfig) -> None:
    """Stop the loaded Kokoro model process for this configuration, if it is running."""
    tts_model_slot(build_tts_serve_command(config), config.timeout_seconds).close()


@functools.cache
def _g2p(phonemize_cli: Path, model_path: Path, timeout_seconds: int) -> en.G2P:
    """Build the misaki grapheme-to-phoneme converter once per configuration.

    misaki is the G2P library Kokoro was trained with. Words missing from its
    lexicons fall back to the TTS.cpp phonemizer built into the Kokoro GGUF,
    which runs as one persistent process so the GGUF is loaded only once.
    """
    phonemizer = ServeProcessSlot((str(phonemize_cli), "--serve", "--phonemizer-path", str(model_path)), timeout_seconds)

    @functools.cache
    def phonemize_word(word: str) -> str:
        # misaki can hand over a token spanning a line break ("Overview\nThis"); the
        # phonemizer treats newlines as spaces, and the request must be one line.
        return phonemizer.get().request(word.replace("\n", " ")).strip()

    def fallback(token: MToken) -> tuple[str, int]:
        return phonemize_word(token.text), 1

    return en.G2P(trf=False, british=False, fallback=fallback)


def text_to_phonemes(config: KokoroRuntimeConfig, text: str) -> str:
    """Convert text into the Kokoro phoneme string handed to tts-cli."""
    phonemes, _tokens = _g2p(config.phonemize_cli, config.model_path, config.timeout_seconds)(text)
    if not phonemes.strip():
        raise ValueError(f"Text produced no phonemes: {text!r}")
    return phonemes


def build_tts_request(phonemes: str, voice: str, output_path: Path) -> str:
    """Build the tts-cli serve request line for one Kokoro generation.

    Raises:
        ValueError: If a field contains a tab or newline, which the line protocol cannot carry.
    """
    fields = (voice, str(output_path), phonemes.replace("\n", " "))
    for field in fields:
        if "\t" in field or "\n" in field:
            raise ValueError(f"TTS request field must not contain tabs or newlines: {field!r}")
    return "\t".join(fields)


def generate_wav(config: KokoroRuntimeConfig, text: str, voice: str, output_path: Path) -> Path:
    """Generate a WAV file with the persistent, model-loaded TTS.cpp process."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_path}")

    request = build_tts_request(text_to_phonemes(config, text), voice, output_path)
    saved_path = load_tts_model(config).request(request)
    if Path(saved_path) != output_path:
        raise RuntimeError(f"TTS.cpp saved {saved_path} instead of {output_path}")
    if not output_path.is_file():
        raise RuntimeError(f"TTS.cpp reported success but produced no WAV at {output_path}")

    return output_path


def read_wav_mono_float32(wav_path: Path) -> tuple[FloatAudio, int]:
    """Read a mono PCM WAV file into float32 samples and return samples plus rate."""
    with wave.open(str(wav_path), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        frame_count = wav_file.getnframes()
        frames = wav_file.readframes(frame_count)

    if channels != 1:
        raise ValueError(f"Expected mono WAV, got {channels} channels in {wav_path}")
    if frame_count <= 0:
        raise ValueError(f"WAV file contains no audio frames: {wav_path}")

    if sample_width == 2:
        int16_audio = np.frombuffer(frames, dtype="<i2")
        audio = int16_audio.astype(np.float32) / np.float32(32768.0)
        return audio.astype(np.float32, copy=False), sample_rate
    if sample_width == 4:
        int32_audio = np.frombuffer(frames, dtype="<i4")
        audio = int32_audio.astype(np.float32) / np.float32(2147483648.0)
        return audio.astype(np.float32, copy=False), sample_rate

    raise ValueError(f"Unsupported WAV sample width {sample_width} bytes in {wav_path}")


def _write_lead_silence(stream: AudioOutputStream, sample_rate: int, lead_silence_ms: int) -> None:
    """Write configured lead silence to a newly opened stream."""
    if lead_silence_ms < 0:
        raise ValueError(f"lead_silence_ms must be >= 0, got {lead_silence_ms}")

    silence_frames = int(sample_rate * lead_silence_ms / 1000)
    if silence_frames == 0:
        return

    silence = np.zeros((silence_frames, 1), dtype=np.float32)
    stream.write(silence)


class AudioPlayer:
    """Persistent output stream that serializes generated WAV playback."""

    def __init__(self, sample_rate: int, lead_silence_ms: int) -> None:
        """Initialize the player with explicit audio-device settings."""
        if lead_silence_ms < 0:
            raise ValueError(f"lead_silence_ms must be >= 0, got {lead_silence_ms}")

        self._sample_rate = sample_rate
        self._lead_silence_ms = lead_silence_ms
        self._jobs: queue.Queue[PlaybackJob | None] = queue.Queue()
        self._unhandled_errors: queue.Queue[Exception] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._closed = False

    def submit(self, job: PlaybackJob) -> None:
        """Queue a playback job for serial playback."""
        if self._closed:
            raise RuntimeError("AudioPlayer is closed")
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        self._jobs.put(job)

    def close(self) -> None:
        """Drain queued playback and close the persistent stream."""
        self._closed = True
        if self._thread is None:
            return

        self._jobs.join()
        self._jobs.put(None)
        self._thread.join()
        self._thread = None
        if not self._unhandled_errors.empty():
            raise self._unhandled_errors.get()

    def _open_stream(self) -> AudioOutputStream:
        stream = cast(AudioOutputStream, sd.OutputStream(samplerate=self._sample_rate, channels=1, dtype="float32"))
        stream.start()
        _write_lead_silence(stream, self._sample_rate, self._lead_silence_ms)
        return stream

    def _close_stream(self, stream: AudioOutputStream | None) -> None:
        if stream is None:
            return
        try:
            stream.stop()
        finally:
            stream.close()

    def _delete_transient_file(self, job: PlaybackJob) -> None:
        if job.delete_after_playback:
            job.wav_path.unlink()

    def _ensure_stream(self, stream: AudioOutputStream | None) -> AudioOutputStream:
        if stream is not None:
            return stream
        return self._open_stream()

    def _handle_job(self, stream: AudioOutputStream | None, job: PlaybackJob) -> AudioOutputStream | None:
        stream_to_close = stream
        try:
            audio, sample_rate = read_wav_mono_float32(job.wav_path)
            if sample_rate != self._sample_rate:
                raise ValueError(f"Expected {self._sample_rate} Hz WAV, got {sample_rate} Hz in {job.wav_path}")
            active_stream = self._ensure_stream(stream)
            stream_to_close = active_stream
            active_stream.write(audio.reshape(-1, 1))
            self._delete_transient_file(job)
            job.on_complete(job.reported_path)
            return active_stream
        except Exception as error:
            close_error: Exception | None = None
            try:
                self._close_stream(stream_to_close)
            except Exception as stream_close_error:
                close_error = stream_close_error

            playback_error = error
            if close_error is not None:
                playback_error = RuntimeError(f"{error}; additionally failed to close audio stream: {close_error}")
            job.on_error(playback_error)
            return None

    def _run(self) -> None:
        stream: AudioOutputStream | None = None
        try:
            while True:
                job = self._jobs.get()
                try:
                    if job is None:
                        break
                    stream = self._handle_job(stream, job)
                finally:
                    self._jobs.task_done()
        finally:
            try:
                self._close_stream(stream)
            except Exception as error:
                self._unhandled_errors.put(error)
