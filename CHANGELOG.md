# Changelog

All notable changes to this project are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## 2026-10-03

### Added

- Added text file to MP3 conversion via the `--input-file` and `--output` options.
- Added `just convert` for batch speech synthesis of text files.

### Changed

- MP3 encoding now uses ffmpeg instead of soundfile dependency.

### Removed

- Removed the `CLAUDE.md` symlink.

### Fixed

- Fixed pronunciation of Microsoft and preserved apostrophes in simplified punctuation.

## 2026-09-23

### Added

- Added misaki phonemization, with fallback to the TTS.cpp phonemizer for unknown words.
- Added a reproducible TTS.cpp build via `just init`, using a pinned upstream commit and patch.

### Changed

- Improved speech quality by keeping sentence-ending punctuation, restoring pauses between sentences.

### Removed

- Removed the unused `temperature`, `topk`, `repetition_penalty` and `top_p` settings.

### Security

- Raised minimum versions of anyio, click, cryptography, gitpython, mcp and pip to clear known vulnerabilities.

## 2026-06-27

### Added

- Added a low-latency local text-to-speech server using Kokoro through TTS.cpp.
- Added queued playback, a REST API and an MCP relay exposing `say`, `get_voices`, `get_status`.
- Added an interactive terminal chat REPL via `just chat`.
