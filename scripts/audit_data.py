from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.datasets.camelyon16 import canonical_id


EXPECTED_CHALLENGE_TOTAL = 400
KNOWN_UPSTREAM_MISSING = "test_049"


def size_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def index_npy(directory: Path) -> dict[str, list[Path]]:
    indexed: dict[str, list[Path]] = {}
    for path in sorted(directory.rglob("*.npy")):
        indexed.setdefault(canonical_id(path), []).append(path)
    return indexed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path("."))
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    root = project_root / "data" / "camelyon16"
    frame = pd.read_csv(root / "splits.csv")
    if not {"bag_name", "split"}.issubset(frame.columns):
        raise ValueError(f"Unexpected splits.csv columns: {list(frame.columns)}")
    frame["bag_id"] = frame["bag_name"].map(canonical_id)
    manifest_ids = set(frame["bag_id"])
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
    indexes = {key: index_npy(directory) for key, directory in dirs.items()}
    missing: dict[str, list[str]] = {key: [] for key in dirs}
    duplicate_candidates: dict[str, list[str]] = {}
    invalid: list[str] = []
    patch_label_value_violations: list[str] = []
    label_patch_mismatches: list[str] = []
    patch_counts: list[int] = []
    labels: list[int] = []
    feature_shapes: set[tuple[int, ...]] = set()

    for bag in frame["bag_id"]:
        candidates = {key: indexes[key].get(bag, []) for key in dirs}
        for key, paths in candidates.items():
            if not paths:
                missing[key].append(bag)
            elif len(paths) > 1:
                duplicate_candidates[f"{key}:{bag}"] = [str(path.relative_to(root)) for path in paths]
        if any(not paths for paths in candidates.values()):
            invalid.append(f"{bag}: one or more required files are missing")
            continue

        feature_path = candidates["features"][0]
        label_path = candidates["labels"][0]
        coords_path = candidates["coords"][0]
        inst_path = candidates["patch_labels"][0]
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
        unique_inst = set(np.unique(inst).tolist())
        if not unique_inst.issubset({0, 1}):
            patch_label_value_violations.append(f"{bag}: values={sorted(unique_inst)}")
        if label.size != 1 or int(label[0]) not in (0, 1):
            invalid.append(f"{bag}: slide_label={label.tolist()}")
        else:
            slide_label = int(label[0])
            labels.append(slide_label)
            patch_positive = int(np.any(inst == 1))
            if slide_label != patch_positive:
                label_patch_mismatches.append(f"{bag}: slide_label={slide_label}, any_patch_positive={patch_positive}")
        patch_counts.append(int(features.shape[0]))

    extra_feature_ids = sorted(set(indexes["features"]) - manifest_ids)
    unexpected_extra_features = [bag for bag in extra_feature_ids if bag != KNOWN_UPSTREAM_MISSING]
    known_missing_present_as_extra = KNOWN_UPSTREAM_MISSING in extra_feature_ids
    challenge_gap_ok = len(frame) == EXPECTED_CHALLENGE_TOTAL - 1 and KNOWN_UPSTREAM_MISSING not in manifest_ids

    report = project_root / "reports" / "project" / "data_audit.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# CAMELYON16 data audit",
        "",
        "## Manifest and upstream gap",
        "",
        f"- TorchMIL/project manifest rows: {len(frame)}",
        f"- Original CAMELYON16 Challenge expected WSIs: {EXPECTED_CHALLENGE_TOTAL}",
        f"- Known upstream missing WSI: `{KNOWN_UPSTREAM_MISSING}`",
        f"- Known 399-vs-400 gap matches expected missing WSI: {'yes' if challenge_gap_ok else 'no'}",
        f"- Train WSIs: {len(train)}",
        f"- Test WSIs: {len(test)}",
        f"- Train/test overlap: {len(overlap)}",
        f"- Duplicate WSI IDs in manifest: {len(duplicate_ids)}",
        "",
        "## Labels and shapes",
        "",
        f"- Positive WSIs: {sum(label == 1 for label in labels)}",
        f"- Negative WSIs: {sum(label == 0 for label in labels)}",
        f"- Feature shapes observed: {sorted(feature_shapes)}",
        f"- Patch count min/max/mean/median: {min(patch_counts) if patch_counts else 'n/a'} / {max(patch_counts) if patch_counts else 'n/a'} / {np.mean(patch_counts) if patch_counts else 'n/a'} / {np.median(patch_counts) if patch_counts else 'n/a'}",
        f"- WSI label vs any-positive patch label mismatches: {len(label_patch_mismatches)}",
        f"- Illegal patch-label value bags: {len(patch_label_value_violations)}",
        "",
        "## Feature manifest coverage",
        "",
        f"- Feature WSI IDs not referenced by manifest: {len(extra_feature_ids)}",
        f"- Known `test_049` feature present outside manifest: {'yes' if known_missing_present_as_extra else 'no'}",
        f"- Unexpected extra feature WSI IDs: {len(unexpected_extra_features)}",
        f"- WSI/modality IDs with multiple candidate file paths: {len(duplicate_candidates)}",
        f"- Data directory size: {size_bytes(root) / (1024**3):.2f} GiB",
        "",
        "## Missing files",
        "",
    ]
    for key, values in missing.items():
        lines.append(f"- {key}: {len(values)}")
        if values:
            lines.append(f"  - {', '.join(values[:20])}")
    lines += ["", "## Label consistency mismatches", ""]
    lines += [f"- {item}" for item in label_patch_mismatches[:100]] or ["- none"]
    lines += ["", "## Illegal patch-label values", ""]
    lines += [f"- {item}" for item in patch_label_value_violations[:100]] or ["- none"]
    lines += ["", "## Multiple candidate paths", ""]
    lines += [f"- {key}: {', '.join(paths)}" for key, paths in list(duplicate_candidates.items())[:100]] or ["- none"]
    lines += ["", "## Invalid values or shapes", ""]
    lines += [f"- {item}" for item in invalid[:100]] or ["- none"]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))

    failed = (
        not challenge_gap_ok
        or overlap
        or duplicate_ids
        or any(missing.values())
        or invalid
        or patch_label_value_violations
        or label_patch_mismatches
        or unexpected_extra_features
        or duplicate_candidates
    )
    if failed:
        raise SystemExit("Data audit failed; see reports/project/data_audit.md")


if __name__ == "__main__":
    main()
