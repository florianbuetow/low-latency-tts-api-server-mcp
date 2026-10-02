"""Tests for the text-file-to-MP3 converter."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from low_latency_tts_service_mcp.tts import KokoroRuntimeConfig
from src.main import ChatConfig, convert_file, main, split_paragraphs

SAMPLE_RATE = 24000


def _chat_config(tmp_path: Path) -> ChatConfig:
    return ChatConfig(
        runtime=KokoroRuntimeConfig(
            tts_cli=tmp_path / "tts-cli",
            phonemize_cli=tmp_path / "phonemize",
            model_path=tmp_path / "model.gguf",
            n_threads=1,
            timeout_seconds=10,
        ),
        output_dir=tmp_path / "output",
        sample_rate=SAMPLE_RATE,
        lead_silence_ms=200,
        save_wav=False,
        simplify_punctuation_enabled=False,
    )


def _write_wav(path: Path, frames: int, sample_rate: int) -> None:
    samples = (np.sin(np.linspace(0, 200, frames)) * 10000).astype("<i2")
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(samples.tobytes())


def _accept_runtime(_config: KokoroRuntimeConfig) -> None:
    return None


def _patch_runtime(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, frames_by_text: dict[str, int], sample_rate: int) -> list[str]:
    calls: list[str] = []

    def fake_generate_wav(_runtime: KokoroRuntimeConfig, text: str, _voice: str, output_path: Path) -> Path:
        calls.append(text)
        _write_wav(output_path, frames_by_text[text], sample_rate)
        return output_path

    monkeypatch.setattr("src.main.load_chat_config", lambda: _chat_config(tmp_path))
    monkeypatch.setattr("src.main.validate_runtime_config", _accept_runtime)
    monkeypatch.setattr("src.main.generate_wav", fake_generate_wav)
    return calls


class TestSplitParagraphs:
    def test_splits_on_blank_lines_and_keeps_single_newlines(self) -> None:
        text = "  First line\nsecond  line \n\n \n\nSecond paragraph\n"

        assert split_paragraphs(text, simplify_punct=False) == ["First line\nsecond line", "Second paragraph"]

    def test_drops_empty_paragraphs(self) -> None:
        assert split_paragraphs("\n\n   \n\n", simplify_punct=False) == []

    def test_applies_punctuation_simplification(self) -> None:
        assert split_paragraphs("Hi, there!", simplify_punct=True) == ["Hi there."]


class TestConvert:
    def test_writes_concatenated_mp3(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        input_path = tmp_path / "in.txt"
        input_path.write_text("One.\n\nTwo.\n", encoding="utf-8")
        output_path = tmp_path / "nested" / "out.mp3"
        calls = _patch_runtime(monkeypatch, tmp_path, {"One.": SAMPLE_RATE, "Two.": SAMPLE_RATE * 2}, SAMPLE_RATE)

        convert_file("af_heart", input_path, output_path)

        assert calls == ["One.", "Two."]
        info = sf.info(str(output_path))
        assert info.format == "MP3"
        assert info.samplerate == SAMPLE_RATE
        assert abs(info.duration - 3.0) < 0.1

    def test_rejects_non_mp3_output(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match=r"\.mp3"):
            convert_file("af_heart", tmp_path / "in.txt", tmp_path / "out.wav")

    def test_rejects_missing_input(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="Input file not found"):
            convert_file("af_heart", tmp_path / "missing.txt", tmp_path / "out.mp3")

    def test_rejects_unknown_voice(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        input_path = tmp_path / "in.txt"
        input_path.write_text("One.", encoding="utf-8")
        _patch_runtime(monkeypatch, tmp_path, {}, SAMPLE_RATE)

        with pytest.raises(ValueError, match="not available"):
            convert_file("nope", input_path, tmp_path / "out.mp3")

    def test_rejects_empty_input(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        input_path = tmp_path / "in.txt"
        input_path.write_text("  \n\n ", encoding="utf-8")
        _patch_runtime(monkeypatch, tmp_path, {}, SAMPLE_RATE)

        with pytest.raises(ValueError, match="no text"):
            convert_file("af_heart", input_path, tmp_path / "out.mp3")

    def test_rejects_unexpected_sample_rate(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        input_path = tmp_path / "in.txt"
        input_path.write_text("One.", encoding="utf-8")
        output_path = tmp_path / "out.mp3"
        _patch_runtime(monkeypatch, tmp_path, {"One.": 100}, 16000)

        with pytest.raises(ValueError, match="Expected 24000 Hz"):
            convert_file("af_heart", input_path, output_path)
        assert not output_path.exists()


def _convert_argv(input_path: Path, output_path: Path) -> list[str]:
    return ["main", "--voice", "af_heart", "--input-file", str(input_path), "--output", str(output_path)]


class TestMain:
    def test_skips_existing_output(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        output_path = tmp_path / "out.mp3"
        output_path.write_bytes(b"existing")
        monkeypatch.setattr("sys.argv", _convert_argv(tmp_path / "in.txt", output_path))

        main()

        assert "Skipped" in capsys.readouterr().out
        assert output_path.read_bytes() == b"existing"

    def test_exits_1_when_convert_arguments_incomplete(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr("sys.argv", ["main", "--input-file", str(tmp_path / "in.txt"), "--output", str(tmp_path / "out.mp3")])

        with pytest.raises(SystemExit) as exc_info:
            main()

        assert exc_info.value.code == 1
        assert "must be given together" in capsys.readouterr().err

    def test_exits_1_on_error(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        monkeypatch.setattr("sys.argv", _convert_argv(tmp_path / "missing.txt", tmp_path / "out.mp3"))

        with pytest.raises(SystemExit) as exc_info:
            main()

        assert exc_info.value.code == 1
        assert "Input file not found" in capsys.readouterr().err
