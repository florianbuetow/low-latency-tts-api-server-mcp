"""Interactive chat REPL for the Kokoro low-latency TTS service.

Reads text from the terminal line by line, synthesizes each submission with the
patched TTS.cpp Kokoro CLI, and plays it back. Generation of the next line
overlaps playback of the current one, so there is no gap between utterances.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import queue
import sys
import termios
import threading
import tty
from pathlib import Path

from low_latency_tts_service_mcp.tts import (
    OUTPUT_DIR,
    AudioPlayer,
    KokoroRuntimeConfig,
    PlaybackJob,
    SamplingParams,
    clean_text,
    generate_wav,
    kokoro_voices,
    load_config,
    make_output_path,
    simplify_punctuation,
    validate_runtime_config,
    validate_voice,
)


@dataclasses.dataclass(frozen=True)
class ChatConfig:
    """All settings the interactive chat REPL needs, loaded from config.yaml."""

    runtime: KokoroRuntimeConfig
    output_dir: Path
    sample_rate: int
    lead_silence_ms: int
    save_wav: bool
    simplify_punctuation_enabled: bool


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


def _require_float(config: dict[str, object], key: str) -> float:
    """Fetch a required numeric config key as float."""
    value = _require(config, key)
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError(f"'{key}' in config.yaml must be a number")
    return float(value)


def load_chat_config() -> ChatConfig:
    """Load and validate the chat REPL settings from config.yaml.

    Returns:
        Fully populated chat configuration.

    Raises:
        ValueError: If a required key is missing or has the wrong type.
    """
    config = load_config()
    sampling = SamplingParams(
        temperature=_require_float(config, "temperature"),
        topk=_require_int(config, "topk"),
        repetition_penalty=_require_float(config, "repetition_penalty"),
        top_p=_require_float(config, "top_p"),
    )
    runtime = KokoroRuntimeConfig(
        tts_cli=Path(_require_str(config, "tts_cli")),
        model_path=Path(_require_str(config, "model")),
        n_threads=_require_int(config, "n_threads"),
        timeout_seconds=_require_int(config, "timeout_seconds"),
        sampling=sampling,
    )
    return ChatConfig(
        runtime=runtime,
        output_dir=Path(_require_str(config, "output_dir")),
        sample_rate=_require_int(config, "sample_rate"),
        lead_silence_ms=_require_int(config, "lead_silence_ms"),
        save_wav=_require_bool(config, "save_wav"),
        simplify_punctuation_enabled=_require_bool(config, "simplify_punctuation"),
    )


def select_voice(voices: list[str]) -> str:
    """Display available voices and let the user select one.

    Args:
        voices: List of available voice names.

    Returns:
        Selected voice name.
    """
    print("\nAvailable voices:")
    for i, voice in enumerate(voices, 1):
        print(f"  {i}. {voice}")

    while True:
        choice = input(f"\nSelect voice [1-{len(voices)}]: ").strip()
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(voices):
                return voices[idx]
        print(f"Invalid choice. Enter a number between 1 and {len(voices)}.")


def read_input(prompt: str) -> str | None:
    """Read a line of input, character by character. Returns None on exit.

    Enter once inserts a newline; Enter twice submits the buffer. Pressing
    Enter twice with an empty buffer exits, as does pressing ESC twice.

    Args:
        prompt: The prompt to display.

    Returns:
        The entered text, or None if the user chose to exit.
    """
    sys.stdout.write(prompt)
    sys.stdout.flush()

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    buf: list[str] = []
    last_was_esc = False
    last_was_enter = False

    try:
        tty.setraw(fd)
        while True:
            ch = sys.stdin.read(1)

            if ch == "\x1b":  # ESC
                if last_was_esc:
                    sys.stdout.write("\r\n")
                    sys.stdout.flush()
                    return None
                last_was_esc = True
                last_was_enter = False
                continue

            if ch in ("\r", "\n"):  # Enter
                if last_was_enter:
                    sys.stdout.write("\r\n")
                    sys.stdout.flush()
                    return "".join(buf) or None
                last_was_enter = True
                last_was_esc = False
                sys.stdout.write("\r\n")
                sys.stdout.flush()
                continue

            if ch in ("\x7f", "\x08"):  # Backspace
                last_was_esc = False
                last_was_enter = False
                if buf:
                    buf.pop()
                    sys.stdout.write("\b \b")
                    sys.stdout.flush()
                continue

            if ch == "\x03":  # Ctrl+C
                sys.stdout.write("\r\n")
                sys.stdout.flush()
                return None

            if last_was_enter:
                buf.append("\n")
            last_was_esc = False
            last_was_enter = False
            buf.append(ch)
            sys.stdout.write(ch)
            sys.stdout.flush()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def list_outputs(output_dir: Path) -> None:
    """List previously generated audio files in the output directory.

    Args:
        output_dir: Directory containing generated WAV files.
    """
    if not output_dir.exists():
        print("No output directory found. No audio has been generated yet.")
        return

    wav_files = sorted(output_dir.glob("*.wav"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not wav_files:
        print("No audio files found in output directory.")
        return

    print(f"\nGenerated audio files in {output_dir}:")
    for wav in wav_files:
        size_kb = wav.stat().st_size // 1024
        mtime = wav.stat().st_mtime
        ts = datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
        print(f"  {ts}  {wav.name}  ({size_kb} KB)")


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and return the CLI argument parser.

    Returns:
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(description="Interactive chat with Kokoro TTS via the patched TTS.cpp CLI")
    parser.add_argument("text", nargs="?", help="Text to convert to speech (or enter interactively)")
    parser.add_argument("--voice", help="Voice to use for synthesis (or select interactively)")
    parser.add_argument(
        "--list-outputs",
        action="store_true",
        help="List previously generated audio files and exit",
    )
    return parser


def prepare_text(text: str, simplify_punct: bool) -> str:
    """Clean text and optionally simplify punctuation.

    Args:
        text: Raw input text.
        simplify_punct: Whether punctuation simplification is enabled.

    Returns:
        Cleaned text, optionally with simplified punctuation.
    """
    prepared = clean_text(text)
    if prepared and simplify_punct:
        prepared = simplify_punctuation(prepared)
    return prepared


def _generate_chat_item(
    runtime: KokoroRuntimeConfig,
    text: str,
    voice: str,
    output_dir: Path,
    save_wav: bool,
) -> tuple[Path, Path | None, bool] | None:
    """Synthesize one utterance to a WAV file and return playback metadata.

    Args:
        runtime: Validated Kokoro runtime configuration.
        text: Cleaned text to synthesize.
        voice: Voice to use for synthesis.
        output_dir: Directory for generated WAV files.
        save_wav: Whether generated audio should be retained after playback.

    Returns:
        Tuple of (wav_path, reported_path, delete_after_playback), or None when
        generation failed.
    """
    output_path = make_output_path(output_dir)
    reported_path = output_path if save_wav else None
    delete_after_playback = not save_wav

    try:
        generate_wav(runtime, text, voice, output_path)
    except (FileExistsError, RuntimeError, TimeoutError, ValueError) as error:
        sys.stderr.write(f"\r\n  Error: {error}\r\n")
        sys.stderr.flush()
        if delete_after_playback and output_path.exists():
            output_path.unlink()
        return None

    return output_path, reported_path, delete_after_playback


def _submit_playback(player: AudioPlayer, pending: tuple[Path, Path | None, bool]) -> threading.Event:
    """Queue one generated WAV for playback and return a completion event.

    Args:
        player: The persistent audio player.
        pending: Tuple of (wav_path, reported_path, delete_after_playback).

    Returns:
        Event that is set once the WAV finished playing or failed.
    """
    wav_path, reported_path, delete_after_playback = pending
    done = threading.Event()

    def on_complete(_output_path: Path | None) -> None:
        done.set()

    def on_error(error: Exception) -> None:
        sys.stderr.write(f"\r\n  Error: {error}\r\n")
        sys.stderr.flush()
        done.set()

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


def chat_audio_worker(
    work_queue: queue.Queue[str | None],
    runtime: KokoroRuntimeConfig,
    voice: str,
    output_dir: Path,
    sample_rate: int,
    lead_silence_ms: int,
    save_wav: bool,
) -> None:
    """Background worker that generates Kokoro WAVs and plays them in order.

    The next utterance is generated while the current one is still playing, so
    there is no gap between submissions.

    Args:
        work_queue: Queue of cleaned text strings. None signals shutdown.
        runtime: Validated Kokoro runtime configuration.
        voice: Voice to use for synthesis.
        output_dir: Directory for generated WAV files.
        sample_rate: Playback sample rate in Hz.
        lead_silence_ms: Silence written after each audio stream open/reopen.
        save_wav: Whether generated audio should be retained after playback.
    """
    player = AudioPlayer(sample_rate, lead_silence_ms)
    playback_done: threading.Event | None = None

    try:
        while True:
            text = work_queue.get()
            if text is None:
                break

            generated = _generate_chat_item(runtime, text, voice, output_dir, save_wav)
            work_queue.task_done()
            if generated is None:
                continue

            if playback_done is not None:
                playback_done.wait()
            playback_done = _submit_playback(player, generated)

        if playback_done is not None:
            playback_done.wait()
    finally:
        player.close()


def shutdown_worker(work_queue: queue.Queue[str | None], worker: threading.Thread) -> None:
    """Wait for queued work to finish, then stop and join the worker thread."""
    work_queue.join()
    work_queue.put(None)
    worker.join()


def resolve_voice(cli_voice: str | None, voices: list[str]) -> str:
    """Resolve the voice from the CLI argument or interactive selection.

    Args:
        cli_voice: Voice from the --voice argument, or None.
        voices: Available voice names.

    Returns:
        A voice guaranteed to be present in ``voices``.

    Raises:
        ValueError: If a CLI-provided voice is not available.
    """
    if cli_voice:
        validate_voice(cli_voice, voices)
        return cli_voice
    return select_voice(voices)


def run_repl(work_queue: queue.Queue[str | None], simplify_punct: bool) -> None:
    """Run the interactive read/submit loop until the user exits.

    Args:
        work_queue: Queue consumed by the audio worker.
        simplify_punct: Whether punctuation simplification is enabled.
    """
    print("Type text. Enter twice submits (single Enter = newline). Enter twice on empty input or ESC twice quits.\n")
    while True:
        result = read_input("Text: ")
        if result is None:
            break
        text = prepare_text(result, simplify_punct)
        if not text:
            continue
        work_queue.put(text)
        print()


def main() -> None:
    """Entry point for the interactive Kokoro chat REPL."""
    parser = create_argument_parser()
    args = parser.parse_args()

    if args.list_outputs:
        list_outputs(OUTPUT_DIR)
        return

    config = load_chat_config()
    validate_runtime_config(config.runtime)

    voices = kokoro_voices()
    try:
        voice = resolve_voice(args.voice, voices)
    except ValueError as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)

    oneshot_text: str | None = None
    if args.text:
        oneshot_text = prepare_text(args.text, config.simplify_punctuation_enabled)
        if not oneshot_text:
            print("Error: text is empty after cleaning", file=sys.stderr)
            sys.exit(1)

    print(f"\nModel: {config.runtime.model_path}")
    print(f"Voice: {voice}\n")

    work_queue: queue.Queue[str | None] = queue.Queue()
    worker = threading.Thread(
        target=chat_audio_worker,
        args=(
            work_queue,
            config.runtime,
            voice,
            config.output_dir,
            config.sample_rate,
            config.lead_silence_ms,
            config.save_wav,
        ),
        daemon=True,
    )
    worker.start()

    if oneshot_text is not None:
        work_queue.put(oneshot_text)
        shutdown_worker(work_queue, worker)
        return

    run_repl(work_queue, config.simplify_punctuation_enabled)
    shutdown_worker(work_queue, worker)


if __name__ == "__main__":
    main()
