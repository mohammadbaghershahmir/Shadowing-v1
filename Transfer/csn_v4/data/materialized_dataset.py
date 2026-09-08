"""Fast CSN-V4 dataset loading pre-materialized NPZ tiles."""
from __future__ import annotations

import json
from pathlib import Path

from torch.utils.data import Dataset

from csn_v4.data.materialize import load_materialized_npz
from csn_v4.data.paths import resolve_dataset_path


class CSNV4MaterializedDataset(Dataset):
    def __init__(
        self,
        index_path: str | Path,
        dataset_root: str | Path,
        *,
        split_filter: str | None = "train",
    ):
        self.dataset_root = Path(dataset_root)
        self.index_path = resolve_dataset_path(index_path, self.dataset_root)
        if not self.index_path.is_file():
            raise FileNotFoundError(f"Materialized index not found: {self.index_path}")

        self.entries: list[dict] = []
        with self.index_path.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                rec = json.loads(line)
                if split_filter is not None and rec.get("split") != split_filter:
                    continue
                self.entries.append(rec)

        if not self.entries:
            raise ValueError(f"No materialized tiles with split_filter={split_filter!r} in {self.index_path}")

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> dict:
        rec = self.entries[idx]
        npz_path = self.dataset_root / rec["path"]
        if not npz_path.is_file():
            raise FileNotFoundError(f"Missing materialized tile: {npz_path}")
        return load_materialized_npz(npz_path)
