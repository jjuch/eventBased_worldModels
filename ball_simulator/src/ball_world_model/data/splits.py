from __future__ import annotations

import hashlib

from ball_world_model.config import SplitConfig

def deterministic_fraction(trajectory_id: str, seed: int) -> float:
    payload = f"{seed}:{trajectory_id}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    value = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return value / 2**64

def assign_split(trajectory_id: str, config: SplitConfig) -> str:
    value = deterministic_fraction(trajectory_id, config.seed)
    if value < config.train_fraction:
        return "train"
    if value < config.train_fraction + config.validation_fraction:
        return "validation"
    return "test"