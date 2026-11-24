from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def canonical_id(value: str | Path) -> str:
    name = Path(str(value)).name
    return name[:-4] if name.endswith(".npy") else Path(name).stem


def resolve_npy(directory: Path, bag_name: str) -> Path:
    target = canonical_id(bag_name)
    candidates = sorted(directory.rglob(f"{target}.npy"))
    if not candidates:
        raise FileNotFoundError(f"No .npy file for {bag_name!r} under {directory}")
    if len(candidates) > 1:
        exact = [p for p in candidates if p.parent.name == directory.name]
        if len(exact) == 1:
            return exact[0]
    return candidates[0]


@dataclass(frozen=True)
class BagRecord:
    bag_name: str
    label: int
    split: str


class Camelyon16Bags(Dataset):
    """Lightweight on-demand adapter for the official CAMELYON16 files.

    It intentionally exposes coordinates and instance labels without using them
    in the training collate function. This avoids adjacency construction and
    keeps the same interface available to future models.
    """

    def __init__(
        self,
        root: str | Path,
        partition: str | None = None,
        bag_names: Sequence[str] | None = None,
        fields: Sequence[str] = ("X", "Y"),
        load_at_init: bool = False,
    ) -> None:
        self.root = Path(root)
        self.fields = tuple(fields)
        self.load_at_init = load_at_init
        self.patch_root = self.root / "patches_512"
        self.feature_dir = self.patch_root / "features" / "features_UNI"
        self.label_dir = self.patch_root / "labels"
        self.coords_dir = self.patch_root / "coords"
        self.instance_dir = self.patch_root / "patch_labels"
        self._records = self._read_records(partition=partition, bag_names=bag_names)
        self._cache: dict[str, dict[str, torch.Tensor]] = {}

    def _read_records(
        self, partition: str | None, bag_names: Sequence[str] | None
    ) -> list[BagRecord]:
        split_path = self.root / "splits.csv"
        if not split_path.exists():
            raise FileNotFoundError(f"Missing official split file: {split_path}")
        frame = pd.read_csv(split_path)
        required = {"bag_name", "split"}
        if not required.issubset(frame.columns):
            raise ValueError(f"splits.csv must contain {sorted(required)}; got {list(frame.columns)}")
        selected = frame if partition is None else frame[frame["split"].astype(str) == partition]
        allowed = None if bag_names is None else {canonical_id(x) for x in bag_names}
        records: list[BagRecord] = []
        for row in selected.itertuples(index=False):
            bag = str(getattr(row, "bag_name"))
            if allowed is not None and canonical_id(bag) not in allowed:
                continue
            label = int(np.asarray(np.load(resolve_npy(self.label_dir, bag))).reshape(-1)[0])
            records.append(BagRecord(bag_name=canonical_id(bag), label=label, split=str(getattr(row, "split"))))
        if not records:
            raise ValueError(f"No bags selected for partition={partition!r}")
        return records

    @property
    def records(self) -> list[BagRecord]:
        return list(self._records)

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        record = self._records[index]
        if record.bag_name in self._cache:
            return dict(self._cache[record.bag_name])
        paths = {
            "X": resolve_npy(self.feature_dir, record.bag_name),
            "Y": resolve_npy(self.label_dir, record.bag_name),
            "coords": resolve_npy(self.coords_dir, record.bag_name),
            "y_inst": resolve_npy(self.instance_dir, record.bag_name),
        }
        output: dict[str, torch.Tensor | str] = {"bag_name": record.bag_name}
        for field in self.fields:
            if field not in paths:
                raise KeyError(f"Unknown bag field {field}")
            array = np.load(paths[field], mmap_mode=None if self.load_at_init else "r")
            tensor = torch.from_numpy(np.asarray(array).copy())
            if field == "X":
                tensor = tensor.float()
            elif field in {"Y", "y_inst"}:
                tensor = tensor.long()
            else:
                tensor = tensor.float()
            output[field] = tensor
        if self.load_at_init:
            self._cache[record.bag_name] = dict(output)  # type: ignore[arg-type]
        return output

    def get_bag_labels(self) -> np.ndarray:
        return np.asarray([record.label for record in self._records], dtype=np.int64)

    def subset(self, indices: Iterable[int]) -> "Camelyon16Bags":
        chosen = [self._records[i] for i in indices]
        child = object.__new__(Camelyon16Bags)
        child.root = self.root
        child.fields = self.fields
        child.load_at_init = self.load_at_init
        child.patch_root = self.patch_root
        child.feature_dir = self.feature_dir
        child.label_dir = self.label_dir
        child.coords_dir = self.coords_dir
        child.instance_dir = self.instance_dir
        child._records = chosen
        child._cache = {}
        return child


def collate_bags(batch: list[dict[str, torch.Tensor | str]]) -> dict[str, torch.Tensor | list[str]]:
    """Pad variable-length bags, returning only tensors present in the batch.

    The ABMIL path consumes only X, Y and mask. No adjacency matrix or patch
    labels are constructed here.
    """
    if not batch:
        raise ValueError("Cannot collate an empty batch")
    x_values = [item["X"] for item in batch]
    if not all(isinstance(value, torch.Tensor) for value in x_values):
        raise TypeError("Every bag must contain tensor X")
    tensors = [value for value in x_values if isinstance(value, torch.Tensor)]
    max_n = max(int(value.shape[0]) for value in tensors)
    dim = int(tensors[0].shape[-1])
    padded = torch.zeros(len(tensors), max_n, dim, dtype=tensors[0].dtype)
    mask = torch.zeros(len(tensors), max_n, dtype=torch.bool)
    for index, value in enumerate(tensors):
        if value.ndim != 2 or value.shape[-1] != dim:
            raise ValueError(f"Expected X shape [N, D], got {tuple(value.shape)}")
        n = value.shape[0]
        padded[index, :n] = value
        mask[index, :n] = True
    output: dict[str, torch.Tensor | list[str]] = {
        "X": padded,
        "mask": mask,
        "Y": torch.stack([item["Y"].reshape(-1)[0].long() for item in batch if isinstance(item["Y"], torch.Tensor)]),
        "bag_name": [str(item["bag_name"]) for item in batch],
    }
    return output

