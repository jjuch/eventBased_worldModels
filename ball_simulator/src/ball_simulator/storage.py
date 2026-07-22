from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np
import json

from .trajectory import Trajectory


class HDF5TrajectoryWriter:
    def __init__(
            self, 
            path: str | Path, 
            compression: str = "gzip", 
            level: int = 4,
            metadata: dict[str, object] | None = None,
    ) -> None:
        self.path = Path(path)
        self.compression = None if compression == "none" else compression
        self.level = level
        self.metadata = metadata or {}
        self._file: h5py.File | None = None

    def __enter__(self) -> "HDF5TrajectoryWriter":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = h5py.File(self.path, "w")

        self._file.attrs["format_version"] = "1.1"
        self._file.attrs["quaternion_order"] = "xyzw"
        self._file.attrs["contact_mode"] = "0=free,1=sticking,2=sliding"
        self._file.attrs["metadata_json"] = json.dumps(self.metadata)
        self._file.attrs["environment_kind"] = self.metadata["selected_environment"]
        return self

    def __exit__(self, *args: Any) -> None:
        if self._file is not None:
            self._file.close()

    def _write_arrays(self, group: h5py.Group, arrays: dict[str, np.ndarray]) -> None:
        for name, data in arrays.items():
            kwargs: dict[str, Any] = {}
            if self.compression is not None and data.size > 16:
                kwargs["compression"] = self.compression
                if self.compression == "gzip":
                    kwargs["compression_opts"] = self.level
                kwargs["shuffle"] = True
            group.create_dataset(name, data=data, **kwargs)

    def write(
        self, 
        index: int, 
        trajectory: Trajectory,
    ) -> None:
        if self._file is None:
            raise RuntimeError("Writer must be used as a context manager")
        
        group = self._file.create_group(f"trajectories/{index:08d}")

        group.attrs["environment_kind"] = trajectory.environment_kind
        group.attrs["surface_ids_json"] = json.dumps(trajectory.surface_ids)

        self._write_arrays(
            group.create_group("observations"), trajectory.observations,
        )

        if trajectory.high_rate is not None:
            self._write_arrays(
                group.create_group("high_rate"), trajectory.high_rate,
            )

        parameters = group.create_group("parameters")
        for name in trajectory.parameters.__dataclass_fields__:
            parameters.attrs[name] = getattr(
                trajectory.parameters, 
                name,
            )
