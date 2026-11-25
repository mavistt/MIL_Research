from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.datasets.camelyon16 import canonical_id, resolve_npy


def size_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path("."))
    args = parser.parse_args()
    root = args.project_root / "data" / "camelyon16"
    frame = pd.read_csv(root / "splits.csv")
    if not {"bag_name", "split"}.issubset(frame.columns):
        raise ValueError(f"Unexpected splits.csv columns: {list(frame.columns)}")
    frame["bag_id"] = frame["bag_name"].map(canonical_id)
    duplicate_ids = frame.loc[frame["bag_id"].duplicated(keep=False), "bag_id"].unique().tolist()
    train = set(frame.loc[frame["split"].astype(str) == "train", "bag_id"])
    test = set(frame.loc[frame["split"].astype(str) == "test", "bag_id"])
    overlap = sorted(train & test)
    patches = root / "patches_512"
    dirs = {
        "features": patches / "features" / "features_UNI",
        "labels": patches / "labels",
        "coords": patches / "coords",
        "patch_labels": patches / "patch_labels",
    }
    missing: dict[str, list[str]] = {key: [] for key in dirs}
    invalid: list[str] = []
    patch_counts: list[int] = []
    labels: list[int] = []
    feature_shapes: set[tuple[int, ...]] = set()
    for bag in frame["bag_id"]:
        try:
            feature_path = resolve_npy(dirs["features"], bag)
            label_path = resolve_npy(dirs["labels"], bag)
            coords_path = resolve_npy(dirs["coords"], bag)
            inst_path = resolve_npy(dirs["patch_labels"], bag)
        except FileNotFoundError as exc:
            missing_key = next((key for key, directory in dirs.items() if not list(directory.rglob(f"{bag}.npy"))), "unknown")
            missing[missing_key].append(bag)
            invalid.append(f"{bag}: {exc}")
            continue
        features = np.load(feature_path, mmap_mode="r")
        label = np.asarray(np.load(label_path)).reshape(-1)
        coords = np.asarray(np.load(coords_path))
        inst = np.asarray(np.load(inst_path)).reshape(-1)
        feature_shapes.add(tuple(features.shape))
        if features.ndim != 2 or features.shape[1] != 1024 or features.shape[0] == 0:
            invalid.append(f"{bag}: feature_shape={features.shape}")
        if not np.isfinite(np.asarray(features)).all():
            invalid.append(f"{bag}: non-finite feature values")
        if coords.shape != (features.shape[0], 2):
            invalid.append(f"{bag}: coords_shape={coords.shape} expected={(features.shape[0], 2)}")
        if inst.shape[0] != features.shape[0]:
            invalid.append(f"{bag}: patch_labels_len={inst.shape[0]} expected={features.shape[0]}")
        if label.size != 1 or int(label[0]) not in (0, 1):
            invalid.append(f"{bag}: slide_label={label.tolist()}")
        else:
            labels.append(int(label[0]))
        patch_counts.append(int(features.shape[0]))
    report = args.project_root / "reports" / "project" / "data_audit.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# CAMELYON16 data audit",
        "",
        f"- Total WSIs: {len(frame)}",
        f"- Train WSIs: {len(train)}",
        f"- Test WSIs: {len(test)}",
        f"- Positive WSIs: {sum(label == 1 for label in labels)}",
        f"- Negative WSIs: {sum(label == 0 for label in labels)}",
        f"- Train/test overlap: {len(overlap)}",
        f"- Duplicate WSI IDs: {len(duplicate_ids)}",
        f"- Feature shapes observed: {sorted(feature_shapes)}",
        f"- Patch count min/max/mean/median: {min(patch_counts) if patch_counts else 'n/a'} / {max(patch_counts) if patch_counts else 'n/a'} / {np.mean(patch_counts) if patch_counts else 'n/a'} / {np.median(patch_counts) if patch_counts else 'n/a'}",
        f"- Data directory size: {size_bytes(root) / (1024**3):.2f} GiB",
        "",
        "## Missing files",
        "",
    ]
    for key, values in missing.items():
        lines.append(f"- {key}: {len(values)}")
        if values:
            lines.append(f"  - {', '.join(values[:20])}")
    lines += ["", "## Invalid values or shapes", ""]
    lines += [f"- {item}" for item in invalid[:100]] or ["- none"]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    if overlap or duplicate_ids or any(missing.values()) or invalid:
        raise SystemExit("Data audit failed; see reports/project/data_audit.md")


if __name__ == "__main__":
    main()

