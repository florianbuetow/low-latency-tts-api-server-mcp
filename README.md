# low-latency-tts-service-mcp

![Made with AI](https://img.shields.io/badge/Made%20with-AI-333333?labelColor=f00) ![Verified by Humans](https://img.shields.io/badge/Verified%20by-Humans-333333?labelColor=brightgreen)

Low-latency local text-to-speech powered by Kokoro through [TTS.cpp](https://github.com/mmwillet/TTS.cpp). The service shells out to a local `tts-cli` binary for fast Kokoro GGUF inference, then handles queued playback, status tracking, and MCP integration for AI agents. It offers a FastAPI server for HTTP clients, a TypeScript MCP relay for Claude Code, Claude Desktop, or any MCP-compatible client, and an interactive terminal chat REPL (`just chat`) for typing text and hearing it spoken right away.

### Features

| Feature | Description |
|---------|-------------|
| Queued Playback | Sequential speech playback through a background worker, with request status tracking |
| REST API Server | FastAPI server with `/say`, `/voices`, `/status/{message_id}`, and `/health` endpoints |
| MCP Server | Ready-to-use MCP bridge that exposes `say`, `get_voices`, and `get_status` tools |
| Interactive Chat REPL | Terminal REPL (`just chat`) that synthesizes and plays each line as you type, generating the next line while the current one is still playing |
| TTS.cpp Runtime | Uses the local TTS.cpp `tts-cli` for low-latency Kokoro GGUF speech generation |
| Kokoro Voices | 27 English no-espeak Kokoro voices, including `af_heart`, `af_sky`, `am_adam`, and `bm_george` |
| WAV Output | Generated audio is saved as timestamped WAV files under `data/output/` when enabled |
| Explicit Runtime Config | TTS.cpp binary, GGUF model path, thread count, host, port, and playback settings are read from `config.yaml` |

Under the hood, the project converts text to phonemes with [misaki](https://github.com/hexgrad/misaki), the grapheme-to-phoneme library Kokoro was trained with, then shells out to a local [TTS.cpp](https://github.com/mmwillet/TTS.cpp) `tts-cli` binary for Kokoro generation. Words missing from misaki's lexicons fall back to the phonemizer built into the Kokoro GGUF. It uses [sounddevice](https://python-sounddevice.readthedocs.io/) for audio output, and uses [FastAPI](https://fastapi.tiangolo.com/) for the HTTP server. The MCP server is a lightweight TypeScript stdio-to-HTTP relay using the [Model Context Protocol SDK](https://modelcontextprotocol.io/).

## Design Principles

All runtime configuration is explicit. If a required value is missing from `config.yaml`, or if the configured `tts-cli` executable or Kokoro GGUF model is not present, the service fails immediately with a clear error. The service does not silently fall back to another model, voice, port, binary, or output directory.

Audio files are written to `data/output/` as WAV files with timestamps when `save_wav: true` in `config.yaml`. The server serializes requests through a queue and a background audio worker. The worker generates audio for a request, starts playback, and can generate the next queued request while the current one is playing.

## Architecture

There are three entry paths into the system. HTTP clients call the FastAPI server directly. AI agents call the MCP relay (`mcp/tts-mcp.ts`), which reads the server host and port from `config.yaml`, checks `/health`, and forwards tool calls to the FastAPI API. Terminal users run the interactive chat REPL (`src/main.py`), which drives the shared TTS runtime directly without going through the HTTP server. In every case, TTS inference runs out-of-process through the configured TTS.cpp `tts-cli`, using the Kokoro no-espeak GGUF model.

```text
┌─────────────────────────┐    ┌────────────────────┐      ┌─────────────────────────┐
│        AI Agent         │    │    HTTP Client     │      │      Terminal User      │
│  Claude Code / Desktop  │    │   curl / scripts   │      │        just chat        │
└────────────┬────────────┘    └─────────┬──────────┘      └────────────┬────────────┘
             │ MCP stdio                 │ HTTP                         │ keystrokes
             ▼                           │                              ▼
┌─────────────────────────┐              │                 ┌─────────────────────────┐
│  MCP Server (Node.js)   │              │                 │       src/main.py       │
│     mcp/tts-mcp.ts      │              │                 │  interactive chat REPL  │
│ tools: say, get_voices, │              │                 │ type text -> synth+play │
│       get_status        │              │                 └────────────┬────────────┘
└────────────┬────────────┘              │                              │
             │ HTTP                      │                              │
             ▼                           ▼                              │
┌───────────────────────────────────────────────────────┐               │
│                    FastAPI Server                     │               │
│       src/low_latency_tts_service_mcp/server.py       │               │
│                                                       │               │
│     POST /say  GET /voices  /status/{id}  /health     │               │
│       work queue -> audio worker -> status map        │               │
└───────────────────┬───────────────────────────────────┘               │
                    │                                                   │
                    ▼                                                   ▼
┌────────────────────────────────────────────────────────────────────────────────────┐
│            Shared TTS Runtime - src/low_latency_tts_service_mcp/tts.py             │
│                                                                                    │
│       text cleanup -> TTS.cpp command -> WAV reader -> sounddevice playback        │
└───────────────────┬───────────────────────────────────┬────────────────────────────┘
                    │                                   │
                    ▼                                   ▼
        ┌───────────────────────┐           ┌───────────────────────┐
        │    TTS.cpp tts-cli    │           │   data/output/*.wav   │
        │ Kokoro_no_espeak.gguf │           │   timestamped audio   │
        └───────────┬───────────┘           └───────────────────────┘
                    │
                    ▼
              ┌───────────┐
              │ Speakers  │
              └───────────┘
```

## Prerequisites

- **Python 3.12+**
- **uv** - Python package manager ([install](https://docs.astral.sh/uv/getting-started/installation/))
- **just** - Command runner ([install](https://github.com/casey/just#installation))
- **Node.js 18+** - For the MCP server
- **git and cmake** - Used by `just build-tts` to fetch, patch and compile TTS.cpp
- **Local audio output device** - Required for playback through `sounddevice`

## Project Structure

```text
.
├── src/
│   ├── main.py                         # Interactive chat REPL (just chat / just run)
│   ├── server.py                       # Module wrapper for uv run -m src.server
│   └── low_latency_tts_service_mcp/
│       ├── server.py                   # FastAPI server, queue, statuses, worker
│       └── tts.py                      # Config, Kokoro command, WAV playback
├── tests/
│   ├── test_main.py
│   ├── test_server.py
│   ├── test_tts.py
│   ├── test_integration.py
│   └── architecture/                   # Architecture import rule tests
├── scripts/
│   └── download-model.sh               # Interactive Kokoro model downloader
├── mcp/
│   ├── tts-mcp.ts                      # MCP relay to FastAPI server
│   ├── package.json
│   └── tsconfig.json
├── config/
│   ├── semgrep/                        # Static analysis rules
│   └── codespell/                      # Spell-check configuration
├── data/
│   ├── models/                         # Downloaded Kokoro GGUF model
│   └── output/                         # Generated WAV files
├── stubs/                              # Local type stubs for strict checking
├── vendor/
│   └── TTS.cpp/                        # Local TTS.cpp checkout/build location
├── config.yaml                         # Runtime configuration
├── justfile                            # Command recipes
└── pyproject.toml                      # Project metadata and dependencies
```

## Setup

```bash
just init
```

Builds the TTS.cpp binaries, creates report directories, and installs Python dependencies via `uv sync --all-extras`. If no Kokoro model is present and the command is running interactively, `just init` prompts for a model download. In non-interactive contexts it tells you to run `just download`.

### Build the TTS.cpp Binaries

```bash
just build-tts
```

Run automatically by `just init`, and a no-op once the binaries exist — delete `vendor/` to force a rebuild. `vendor/` is gitignored, so the binaries are a local build artifact; what this repository tracks is everything needed to reproduce them:

| Item | Purpose |
|------|---------|
| `TTS_CPP_COMMIT` in `scripts/build-tts.sh` | The pinned upstream TTS.cpp commit |
| `patches/tts-cpp.patch` | Local changes applied on top of that commit |

The patch adds a `--phonemes` flag so `tts-cli` accepts misaki's phoneme string directly instead of re-phonemizing it, keeps `.`, `!` and `?` in the phonemized prompt so Kokoro produces sentence-boundary pauses instead of running sentences together, builds the `phonemize` helper used as the fallback for unknown words, trims unused example targets, and adds load and generation timing output. The ggml Accelerate, BLAS and Metal backends are pinned off so every machine builds the same CPU binary and generates identical audio.

To move to a newer upstream TTS.cpp, bump `TTS_CPP_COMMIT`, delete `vendor/`, and re-run. If the patch no longer applies, re-create it with `git -C vendor/TTS.cpp diff > patches/tts-cpp.patch`.

### Download a Model

```bash
just download
```

Downloads the Kokoro no-espeak GGUF model used by the default configuration:

| Model | Size | Notes |
|-------|------|-------|
| `Kokoro_no_espeak.gguf` | ~354 MB | English no-espeak Kokoro model with 27 built-in voices |

The default path is:

```text
data/models/Kokoro_no_espeak.gguf
```

### Getting Started

1. Run `just init` - builds the TTS.cpp binaries and installs Python dependencies
2. Run `just download` - downloads `Kokoro_no_espeak.gguf` if it is not already present
3. Confirm `config.yaml` points at the local `tts-cli` executable and downloaded model
4. Run `just start` - starts the FastAPI TTS server
5. Send requests via HTTP or through the MCP bridge

To try synthesis without the server, run `just chat` for an interactive terminal REPL: type a line, press Enter twice, and hear it spoken.

## Configuration

TTS and server runtime settings live in `config.yaml` at the project root. Some operational constants, such as status retention and MCP request timeouts, are defined in code. Example:

```yaml
tts_cli: ./vendor/TTS.cpp/build/bin/tts-cli
phonemize_cli: ./vendor/TTS.cpp/build/bin/phonemize
model: ./data/models/Kokoro_no_espeak.gguf
output_dir: ./data/output
sample_rate: 24000
lead_silence_ms: 200
default_voice: af_heart
save_wav: true
simplify_punctuation: false
n_threads: 8
timeout_seconds: 120
host: 0.0.0.0
port: 12000
```

| Key | Description |
|-----|-------------|
| `tts_cli` | Path to the local TTS.cpp `tts-cli` executable |
| `phonemize_cli` | Path to the TTS.cpp `phonemize` executable, used for words misaki does not know |
| `model` | Path to `Kokoro_no_espeak.gguf` |
| `output_dir` | Directory for generated WAV files |
| `sample_rate` | Expected WAV sample rate in Hz |
| `lead_silence_ms` | Silence written before playback starts on a new audio stream |
| `default_voice` | Voice used when `/say` omits a voice |
| `save_wav` | Save generated audio to WAV files in `output_dir` (`true` or `false`) |
| `simplify_punctuation` | Simplify punctuation before synthesis (`true` or `false`) |
| `n_threads` | Number of threads passed to TTS.cpp |
| `timeout_seconds` | Maximum duration for one TTS.cpp generation command |
| `host` | Server listen address |
| `port` | Server listen port |

### Voices

The no-espeak Kokoro model exposes these voice identifiers:

```text
af_alloy, af_aoede, af_bella, af_heart, af_jessica, af_kore, af_nicole,
af_nova, af_river, af_sarah, af_sky, am_adam, am_echo, am_eric,
am_fenrir, am_liam, am_michael, am_onyx, am_puck, am_santa, bf_alice,
bf_emma, bf_isabella, bf_lily, bm_daniel, bm_fable, bm_george
```

## Usage

| Command | Description |
|---------|-------------|
| `just chat` | Start the interactive chat REPL (type text, hear speech) |
| `just run` | Alias for `just chat` |
| `just start` | Start the FastAPI TTS server in the foreground |
| `just stop` | Stop the running server |
| `just status` | Check if the server is running |
| `just mcp-install` | Install Node dependencies for the MCP relay |
| `just mcp-start` | Start the MCP stdio relay from the terminal |
| `just mcp-typecheck` | Type-check the MCP TypeScript relay |

### Interactive Chat (REPL)

```bash
just chat
```

Starts an interactive terminal REPL that synthesizes and plays each submission with the local TTS.cpp `tts-cli`. Generation for the next line overlaps playback of the current one, so there is no gap between utterances. The REPL drives the shared TTS runtime directly and does not require the FastAPI server to be running.

If `--voice` is not supplied, the REPL prompts you to pick a voice; otherwise it uses the one you pass. It reads settings (model, sample rate, `save_wav`, and more) from `config.yaml` and fails immediately if the configured `tts-cli` or GGUF model is missing.

Input controls:

| Key | Action |
|-----|--------|
| `Enter` once | Insert a newline into the current line |
| `Enter` twice | Submit the buffered text for synthesis |
| `Enter` twice on empty input | Quit |
| `ESC` twice | Quit |
| `Backspace` | Delete the previous character |

Run it directly with `uv run` for more options:

```bash
# Pick a voice interactively, then type lines to speak
uv run -m src.main

# Skip voice selection
uv run -m src.main --voice af_heart

# One-shot: synthesize a single string and exit
uv run -m src.main --voice am_adam "Hello from Kokoro."

# List previously generated WAV files in data/output/ and exit
uv run -m src.main --list-outputs
```

When `save_wav: true`, each utterance is written to a timestamped WAV under `output_dir`; when `false`, audio is played and the temporary file is removed.

### Server

```bash
just start
```

Starts a FastAPI server with queued playback. The server validates `config.yaml` at startup and processes requests sequentially through a background worker.

## API

FastAPI auto-generates interactive docs at `/docs` (Swagger) and `/redoc` (ReDoc) when the server is running.

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Liveness check |
| GET | `/voices` | List available voices and default voice |
| POST | `/say` | Queue text for synthesis and playback |
| GET | `/status/{message_id}` | Check status of a queued, generating, playing, completed, or failed message |

### POST /say

```json
{
  "text": "Hello, this is a Kokoro TTS request.",
  "voice": "af_heart"
}
```

Returns `202 Accepted` with a message ID and queue position:

```json
{
  "message_id": "msg_20260627_130430_001",
  "status": "queued",
  "queue_position": 0
}
```

Audio plays through the server machine's speakers.

### Message Lifecycle

```text
queued -> generating -> playing -> completed
```

Failures are reported as:

```text
error
```

Completed and failed statuses are evicted lazily after 1 hour when later `/say` or `/status` requests trigger status cleanup.

## MCP Server

The MCP server (`mcp/tts-mcp.ts`) is a transparent relay between MCP clients and the FastAPI server. It exposes three tools:

| Tool | Description |
|------|-------------|
| `say` | Queue text for speech synthesis with a specified voice |
| `get_voices` | List all available voices |
| `get_status` | Check status of a speech request by message ID |

### Setup

```bash
just mcp-install
```

### Usage with Claude Code / Claude Desktop

Start the FastAPI server first:

```bash
just start
```

Then configure the MCP client to run the TypeScript relay directly. For Claude Code, from the project directory:

```bash
claude mcp add --scope local kokoro-tts-project \
  -e KOKORO_TTS_CONFIG_PATH=/path/to/low-latency-tts-service-mcp/config.yaml \
  -- /path/to/low-latency-tts-service-mcp/mcp/node_modules/.bin/tsx \
  /path/to/low-latency-tts-service-mcp/mcp/tts-mcp.ts
```

For JSON-based MCP configuration:

```json
{
  "mcpServers": {
    "kokoro-tts-project": {
      "command": "/path/to/low-latency-tts-service-mcp/mcp/node_modules/.bin/tsx",
      "args": ["/path/to/low-latency-tts-service-mcp/mcp/tts-mcp.ts"],
      "env": {
        "KOKORO_TTS_CONFIG_PATH": "/path/to/low-latency-tts-service-mcp/config.yaml"
      }
    }
  }
}
```

The MCP relay reads `host` and `port` from `config.yaml` and calls `/health` before tool requests. Successful FastAPI JSON responses are returned as MCP text content; health check failures and non-OK HTTP responses are wrapped as structured MCP error results.

## Development

### Code Quality

| Command | Description |
|---------|-------------|
| `just code-format` | Auto-fix code style and formatting |
| `just code-style` | Check code style and formatting (read-only) |
| `just code-typecheck` | Run static type checking with mypy |
| `just code-lspchecks` | Run strict type checking with Pyright (LSP-based) |
| `just code-security` | Run security checks with bandit |
| `just code-deptry` | Check dependency hygiene with deptry |
| `just code-spell` | Check spelling in code and documentation |
| `just code-semgrep` | Run Semgrep static analysis |
| `just code-audit` | Scan dependencies for known vulnerabilities |
| `just code-architecture` | Run architecture import rule tests |
| `just code-stats` | Generate code statistics with pygount |

### Testing

| Command | Description |
|---------|-------------|
| `just test` | Run unit and integration tests |
| `just test-coverage` | Run tests with coverage report |

### CI

- `just ci` - Run all validation checks (verbose)
- `just ci-quiet` - Run all checks (silent, fail-fast)

The CI pipeline runs in order: init, code-format, code-style, code-typecheck, code-security, code-deptry, code-spell, code-semgrep, code-audit, test, code-architecture, code-lspchecks, and mcp-typecheck.

## AI-Assisted Development

This project includes an [AGENTS.md](AGENTS.md) file with development rules for AI coding assistants.
