from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class KeyEvent:
    kind: Literal['down', 'up']
    key: str


@dataclass(frozen=True)
class AudioChunk:
    seq: int
    data: np.ndarray
    sample_rate: int


@dataclass(frozen=True)
class TranscribedText:
    seq: int
    raw: str


@dataclass(frozen=True)
class CleanText:
    seq: int
    text: str
