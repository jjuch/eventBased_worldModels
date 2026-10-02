"""Shared evaluation I/O and batch utilities."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Any

import numpy as np
import torch


def as_numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy()


def limited_batches(loader: Iterable, maximum: int) -> Iterator[dict[str, Any]]:
    """Yield at most ``maximum`` windows without mutating the source loader."""
    used = 0
    for batch in loader:
        if used >= maximum:
            break
        batch_size = int(batch["context_rgb"].shape[0])
        keep = min(batch_size, maximum - used)
        if keep != batch_size:
            batch = {
                key: value[:keep] if isinstance(value, (torch.Tensor, list, tuple)) else value
                for key, value in batch.items()
            }
        used += keep
        yield batch


def write_csv(path: str | Path, rows: list[Mapping[str, object]]) -> None:
    """Write heterogeneous dictionaries using first-seen column order."""
    if not rows:
        return
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)

def write_json(path: str | Path, value: object) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(value, indent=2), encoding="utf-8")


def read_json(path: str | Path) -> object:
    return json.loads(Path(path).read_text(encoding="utf-8"))
