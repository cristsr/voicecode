import numpy as np


def concatenate_chunks(chunks: list[np.ndarray]) -> np.ndarray:
    if not chunks:
        return np.array([], dtype=np.float32)
    return np.concatenate(chunks).astype(np.float32)


def duration_ms(samples: np.ndarray, sample_rate: int) -> float:
    return (len(samples) / sample_rate) * 1000
