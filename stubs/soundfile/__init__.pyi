"""Minimal soundfile stubs used by this project."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt

class _SoundFileInfo:
    format: str
    samplerate: int
    channels: int
    duration: float

def write(file: str | Path, data: npt.NDArray[np.float32], samplerate: int, *, format: str) -> None: ...
def info(file: str | Path) -> _SoundFileInfo: ...
