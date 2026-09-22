"""Tests for CLI-specific functions in the chat REPL main module."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.main import (
    ChatConfig,
    create_argument_parser,
    list_outputs,
    load_chat_config,
    prepare_text,
    resolve_voice,
)


def _base_config() -> dict[str, object]:
    return {
        "tts_cli": "./vendor/TTS.cpp/build/bin/tts-cli",
        "model": "./data/models/Kokoro_no_espeak.gguf",
        "output_dir": "./data/output",
        "sample_rate": 24000,
        "lead_silence_ms": 200,
        "save_wav": True,
        "simplify_punctuation": False,
        "n_threads": 8,
        "timeout_seconds": 120,
    }


class TestLoadChatConfig:
    """Tests for chat config parsing."""

    def test_loads_full_config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("src.main.load_config", _base_config)

        config = load_chat_config()

        assert isinstance(config, ChatConfig)
        assert config.sample_rate == 24000
        assert config.lead_silence_ms == 200
        assert config.save_wav is True
        assert config.simplify_punctuation_enabled is False
        assert config.output_dir == Path("./data/output")
        assert config.runtime.model_path == Path("./data/models/Kokoro_no_espeak.gguf")
        assert config.runtime.n_threads == 8
        assert config.runtime.timeout_seconds == 120

    def test_raises_when_model_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        config = _base_config()
        del config["model"]
        monkeypatch.setattr("src.main.load_config", lambda: config)

        with pytest.raises(ValueError, match="model"):
            load_chat_config()

    def test_raises_when_sample_rate_wrong_type(self, monkeypatch: pytest.MonkeyPatch) -> None:
        config = _base_config()
        config["sample_rate"] = "24000"
        monkeypatch.setattr("src.main.load_config", lambda: config)

        with pytest.raises(ValueError, match="sample_rate"):
            load_chat_config()

    def test_raises_when_save_wav_not_bool(self, monkeypatch: pytest.MonkeyPatch) -> None:
        config = _base_config()
        config["save_wav"] = "yes"
        monkeypatch.setattr("src.main.load_config", lambda: config)

        with pytest.raises(ValueError, match="save_wav"):
            load_chat_config()


class TestResolveVoice:
    """Tests for the resolve_voice function."""

    def test_uses_cli_voice_when_available(self) -> None:
        assert resolve_voice("af_heart", ["af_heart", "am_adam"]) == "af_heart"

    def test_raises_when_cli_voice_unavailable(self) -> None:
        with pytest.raises(ValueError, match="not available"):
            resolve_voice("nonexistent_voice", ["af_heart", "am_adam"])


class TestPrepareText:
    """Tests for the prepare_text function."""

    def test_cleans_whitespace(self) -> None:
        assert prepare_text("  hello   world  ", simplify_punct=False) == "hello world"

    def test_simplifies_punctuation_when_enabled(self) -> None:
        result = prepare_text("hello, world!", simplify_punct=True)
        assert "," not in result
        assert "!" not in result

    def test_keeps_punctuation_when_disabled(self) -> None:
        assert prepare_text("hello, world!", simplify_punct=False) == "hello, world!"


class TestListOutputs:
    """Tests for the list_outputs function."""

    def test_prints_message_when_dir_missing(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        list_outputs(tmp_path / "nonexistent")

        out = capsys.readouterr().out
        assert "No output directory" in out

    def test_prints_message_when_no_wav_files(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        list_outputs(tmp_path)

        out = capsys.readouterr().out
        assert "No audio files" in out

    def test_lists_wav_files(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        (tmp_path / "speech_20260101_120000.wav").write_bytes(b"\x00" * 1000)
        (tmp_path / "speech_20260101_120030.wav").write_bytes(b"\x00" * 2000)

        list_outputs(tmp_path)

        out = capsys.readouterr().out
        assert "speech_20260101_120000.wav" in out
        assert "speech_20260101_120030.wav" in out


class TestCreateArgumentParser:
    """Tests for the create_argument_parser function."""

    def test_parses_text_argument(self) -> None:
        parser = create_argument_parser()
        args = parser.parse_args(["Hello world"])
        assert args.text == "Hello world"

    def test_parses_voice_flag(self) -> None:
        parser = create_argument_parser()
        args = parser.parse_args(["--voice", "af_heart"])
        assert args.voice == "af_heart"

    def test_parses_list_outputs_flag(self) -> None:
        parser = create_argument_parser()
        args = parser.parse_args(["--list-outputs"])
        assert args.list_outputs is True
