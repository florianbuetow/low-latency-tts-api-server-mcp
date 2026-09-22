"""Tests for Kokoro TTS command execution and playback helpers."""

from __future__ import annotations

import subprocess
import wave
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from low_latency_tts_service_mcp.tts import (
    KOKORO_VOICES,
    AudioPlayer,
    KokoroRuntimeConfig,
    PlaybackJob,
    build_kokoro_command,
    clean_text,
    generate_wav,
    kokoro_voices,
    read_wav_mono_float32,
    simplify_punctuation,
    validate_runtime_config,
    validate_voice,
)


def _runtime_config(tmp_path: Path) -> KokoroRuntimeConfig:
    tts_cli = tmp_path / "tts-cli"
    tts_cli.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    tts_cli.chmod(0o755)
    model_path = tmp_path / "Kokoro_no_espeak.gguf"
    model_path.write_bytes(b"fake")
    return KokoroRuntimeConfig(
        tts_cli=tts_cli,
        model_path=model_path,
        n_threads=4,
        timeout_seconds=30,
    )


def _write_wav(path: Path, sample_rate: int) -> None:
    samples = (np.array([0.0, 0.25, -0.25, 0.5], dtype=np.float32) * 32767).astype(np.int16)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(samples.tobytes())


def test_kokoro_voice_registry_matches_no_espeak_pack() -> None:
    voices = kokoro_voices()

    assert len(voices) == 27
    assert voices == list(KOKORO_VOICES)
    assert {"af_heart", "af_sky", "am_adam", "bf_emma", "bm_george"} <= set(voices)


def test_validate_voice_rejects_unknown_voice() -> None:
    with pytest.raises(ValueError, match="not_a_voice"):
        validate_voice("not_a_voice", kokoro_voices())


def test_build_kokoro_command_uses_voice(tmp_path: Path) -> None:
    config = _runtime_config(tmp_path)
    output_path = tmp_path / "out.wav"

    command = build_kokoro_command(config, "hello world", "af_heart", output_path)

    assert command == (
        str(config.tts_cli),
        "--model-path",
        str(config.model_path),
        "--prompt",
        "hello world",
        "--save-path",
        str(output_path),
        "--n-threads",
        "4",
        "--voice",
        "af_heart",
    )


def test_validate_runtime_config_requires_files(tmp_path: Path) -> None:
    config = _runtime_config(tmp_path)

    validate_runtime_config(config)

    missing = KokoroRuntimeConfig(
        tts_cli=config.tts_cli,
        model_path=tmp_path / "missing.gguf",
        n_threads=config.n_threads,
        timeout_seconds=config.timeout_seconds,
    )
    with pytest.raises(FileNotFoundError, match="Kokoro GGUF"):
        validate_runtime_config(missing)


def test_generate_wav_runs_tts_cli_and_requires_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = _runtime_config(tmp_path)
    output_path = tmp_path / "generated.wav"
    calls: list[tuple[str, ...]] = []

    def fake_run(command: tuple[str, ...], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        assert kwargs["timeout"] == 30
        save_path = Path(command[command.index("--save-path") + 1])
        _write_wav(save_path, sample_rate=24000)
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    monkeypatch.setattr("low_latency_tts_service_mcp.tts.subprocess.run", fake_run)

    result = generate_wav(config, "hello", "af_heart", output_path)

    assert result == output_path
    assert output_path.is_file()
    assert calls[0][-2:] == ("--voice", "af_heart")


def test_generate_wav_surfaces_cli_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = _runtime_config(tmp_path)

    def fake_run(command: tuple[str, ...], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        raise subprocess.CalledProcessError(2, command, output="bad stdout", stderr="bad stderr")

    monkeypatch.setattr("low_latency_tts_service_mcp.tts.subprocess.run", fake_run)

    with pytest.raises(RuntimeError, match="bad stderr"):
        generate_wav(config, "hello", "af_heart", tmp_path / "missing.wav")


def test_read_wav_mono_float32_reads_pcm(tmp_path: Path) -> None:
    wav_path = tmp_path / "voice.wav"
    _write_wav(wav_path, sample_rate=24000)

    audio, sample_rate = read_wav_mono_float32(wav_path)

    assert sample_rate == 24000
    assert audio.dtype == np.float32
    assert audio.shape == (4,)
    assert float(np.max(audio)) > 0.0


def test_audio_player_writes_lead_silence_and_audio(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    wav_path = tmp_path / "voice.wav"
    _write_wav(wav_path, sample_rate=1000)
    mock_stream = MagicMock()
    mock_output_stream = MagicMock(return_value=mock_stream)
    monkeypatch.setattr("low_latency_tts_service_mcp.tts.sd.OutputStream", mock_output_stream)

    completed: list[Path | None] = []
    errors: list[Exception] = []
    player = AudioPlayer(sample_rate=1000, lead_silence_ms=200)
    player.submit(
        PlaybackJob(
            wav_path=wav_path,
            reported_path=wav_path,
            delete_after_playback=False,
            on_complete=completed.append,
            on_error=errors.append,
        )
    )
    player.close()

    mock_output_stream.assert_called_once_with(samplerate=1000, channels=1, dtype="float32")
    assert mock_stream.write.call_count == 2
    silence = mock_stream.write.call_args_list[0].args[0]
    assert silence.shape == (200, 1)
    assert completed == [wav_path]
    assert errors == []


def test_clean_text_and_simplify_punctuation() -> None:
    assert clean_text("  hello   world  ") == "hello world"
    assert clean_text("line1\n\nline2") == "line1\nline2"
    assert simplify_punctuation("Hello, world!") == "Hello world."
