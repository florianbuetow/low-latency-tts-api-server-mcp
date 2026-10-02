"""Tests for the persistent Kokoro TTS.cpp processes and playback helpers."""

from __future__ import annotations

import wave
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from low_latency_tts_service_mcp.tts import (
    KOKORO_VOICES,
    AudioPlayer,
    KokoroRuntimeConfig,
    PlaybackJob,
    build_tts_request,
    build_tts_serve_command,
    clean_text,
    generate_wav,
    kokoro_voices,
    load_tts_model,
    read_wav_mono_float32,
    simplify_punctuation,
    text_to_phonemes,
    unload_tts_model,
    validate_runtime_config,
    validate_voice,
)


def _fake_tts_cli(tmp_path: Path) -> Path:
    """Fake `tts-cli --serve`: logs each start and request; the voice selects error behaviour."""
    template = tmp_path / "template.wav"
    _write_wav(template, sample_rate=24000)
    log = tmp_path / "tts-cli.log"
    tts_cli = tmp_path / "tts-cli"
    tts_cli.write_text(
        f"""#!/bin/sh
echo "start $*" >> "{log}"
echo "@@ready"
while IFS='\t' read -r voice save_path prompt; do
    echo "request $voice $prompt" >> "{log}"
    case "$voice" in
        fail_voice) printf '@@error\\tsynthesis failed for %s\\n' "$voice"; continue ;;
        crash_voice) echo "fatal crash" >&2; exit 3 ;;
        hang_voice) exec sleep 60 ;;
    esac
    cp "{template}" "$save_path"
    printf '@@done\\t%s\\n' "$save_path"
done
""",
        encoding="utf-8",
    )
    tts_cli.chmod(0o755)
    return tts_cli


def _fake_phonemize(tmp_path: Path) -> Path:
    """Fake `phonemize --serve`: logs each start and answers every word with fixed phonemes."""
    log = tmp_path / "phonemize.log"
    phonemize_cli = tmp_path / "phonemize"
    phonemize_cli.write_text(
        f"""#!/bin/sh
echo "start" >> "{log}"
echo "@@ready"
while IFS= read -r word; do
    printf '@@done\\tfallback-phonemes\\n'
done
""",
        encoding="utf-8",
    )
    phonemize_cli.chmod(0o755)
    return phonemize_cli


def _runtime_config(tmp_path: Path, timeout_seconds: int = 30) -> KokoroRuntimeConfig:
    model_path = tmp_path / "Kokoro_no_espeak.gguf"
    model_path.write_bytes(b"fake")
    return KokoroRuntimeConfig(
        tts_cli=_fake_tts_cli(tmp_path),
        phonemize_cli=_fake_phonemize(tmp_path),
        model_path=model_path,
        n_threads=4,
        timeout_seconds=timeout_seconds,
    )


def _log_lines(path: Path, prefix: str) -> list[str]:
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line.startswith(prefix)]


@pytest.fixture
def config(tmp_path: Path) -> Iterator[KokoroRuntimeConfig]:
    runtime = _runtime_config(tmp_path)
    yield runtime
    unload_tts_model(runtime)


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


def test_build_tts_serve_command_loads_model_once_for_all_voices(config: KokoroRuntimeConfig) -> None:
    assert build_tts_serve_command(config) == (
        str(config.tts_cli),
        "--serve",
        "--model-path",
        str(config.model_path),
        "--n-threads",
        "4",
    )


def test_build_tts_request_is_one_tab_separated_line(tmp_path: Path) -> None:
    output_path = tmp_path / "out.wav"

    assert build_tts_request("həlˈO\nwˈɜɹld", "af_heart", output_path) == f"af_heart\t{output_path}\thəlˈO wˈɜɹld"


def test_build_tts_request_rejects_tabs(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="tabs or newlines"):
        build_tts_request("həlˈO", "af\theart", tmp_path / "out.wav")


def test_text_to_phonemes_uses_misaki_lexicon(config: KokoroRuntimeConfig) -> None:
    assert text_to_phonemes(config, "The tools are ready.") == "ðə tˈulz ɑɹ ɹˈɛdi."


def test_text_to_phonemes_falls_back_to_one_persistent_phonemizer(config: KokoroRuntimeConfig, tmp_path: Path) -> None:
    assert text_to_phonemes(config, "kubectl") == "fallback-phonemes"
    assert text_to_phonemes(config, "zorbleflax") == "fallback-phonemes"

    assert _log_lines(tmp_path / "phonemize.log", "start") == ["start"]


def test_text_to_phonemes_sends_tokens_spanning_a_line_break_as_one_line(config: KokoroRuntimeConfig) -> None:
    # misaki hands "Overview\nThis" to the fallback as a single unknown token.
    assert text_to_phonemes(config, "Overview\nThis document is fine.") == "fallback-phonemes dˈɑkjəmənt ɪz fˈIn."


def test_validate_runtime_config_requires_files(config: KokoroRuntimeConfig, tmp_path: Path) -> None:
    validate_runtime_config(config)

    missing = KokoroRuntimeConfig(
        tts_cli=config.tts_cli,
        phonemize_cli=config.phonemize_cli,
        model_path=tmp_path / "missing.gguf",
        n_threads=config.n_threads,
        timeout_seconds=config.timeout_seconds,
    )
    with pytest.raises(FileNotFoundError, match="Kokoro GGUF"):
        validate_runtime_config(missing)


def test_generate_wav_loads_model_once_for_every_voice(config: KokoroRuntimeConfig, tmp_path: Path) -> None:
    first = generate_wav(config, "hello", "af_heart", tmp_path / "out" / "first.wav")
    second = generate_wav(config, "hello", "bm_george", tmp_path / "out" / "second.wav")

    assert first.is_file()
    assert second.is_file()
    log = tmp_path / "tts-cli.log"
    assert _log_lines(log, "start") == [f"start --serve --model-path {config.model_path} --n-threads 4"]
    assert _log_lines(log, "request") == ["request af_heart həlˈO", "request bm_george həlˈO"]


def test_load_tts_model_preloads_the_model_used_by_generate_wav(config: KokoroRuntimeConfig, tmp_path: Path) -> None:
    load_tts_model(config)
    generate_wav(config, "hello", "af_heart", tmp_path / "out.wav")

    assert len(_log_lines(tmp_path / "tts-cli.log", "start")) == 1


def test_generate_wav_surfaces_error_answer_and_keeps_model_loaded(config: KokoroRuntimeConfig, tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="synthesis failed for fail_voice"):
        generate_wav(config, "hello", "fail_voice", tmp_path / "failed.wav")

    generate_wav(config, "hello", "af_heart", tmp_path / "ok.wav")
    assert len(_log_lines(tmp_path / "tts-cli.log", "start")) == 1


def test_generate_wav_reports_crash_and_restarts_model(config: KokoroRuntimeConfig, tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match=r"exited with status 3 \| stderr: fatal crash"):
        generate_wav(config, "hello", "crash_voice", tmp_path / "crashed.wav")

    generate_wav(config, "hello", "af_heart", tmp_path / "ok.wav")
    assert len(_log_lines(tmp_path / "tts-cli.log", "start")) == 2


def test_generate_wav_times_out_and_kills_model(tmp_path: Path) -> None:
    config = _runtime_config(tmp_path, timeout_seconds=1)
    try:
        hung_process = load_tts_model(config)
        with pytest.raises(TimeoutError, match="did not answer within 1s"):
            generate_wav(config, "hello", "hang_voice", tmp_path / "hung.wav")
        assert not hung_process.is_alive()

        generate_wav(config, "hello", "af_heart", tmp_path / "ok.wav")
        assert len(_log_lines(tmp_path / "tts-cli.log", "start")) == 2
    finally:
        unload_tts_model(config)


def test_generate_wav_refuses_to_overwrite(config: KokoroRuntimeConfig, tmp_path: Path) -> None:
    output_path = tmp_path / "exists.wav"
    output_path.write_bytes(b"existing")

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        generate_wav(config, "hello", "af_heart", output_path)


def test_unload_tts_model_stops_the_process(config: KokoroRuntimeConfig) -> None:
    process = load_tts_model(config)

    unload_tts_model(config)

    assert not process.is_alive()


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
