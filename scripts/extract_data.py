from __future__ import annotations

import argparse
import shutil
import tarfile
from pathlib import Path


ARCHIVES = {
    "features_UNI.tar.gz": "features/features_UNI",
    "labels.tar.gz": "labels",
    "coords.tar.gz": "coords",
    "patch_labels.tar.gz": "patch_labels",
}


def safe_members(archive: tarfile.TarFile, destination: Path):
    root = destination.resolve()
    for member in archive.getmembers():
        target = (destination / member.name).resolve()
        if root not in target.parents and target != root:
            raise RuntimeError(f"Unsafe archive member: {member.name}")
        yield member


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path("."))
    args = parser.parse_args()
    downloads = args.project_root / "data" / "downloads" / "camelyon16_hf"
    data_root = args.project_root / "data" / "camelyon16"
    patches = data_root / "patches_512"
    patches.mkdir(parents=True, exist_ok=True)
    for archive_name, expected_subdir in ARCHIVES.items():
        candidates = list(downloads.rglob(archive_name))
        if not candidates:
            raise FileNotFoundError(f"Missing downloaded archive {archive_name} under {downloads}")
        archive_path = candidates[0]
        destination = patches / expected_subdir
        destination.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive_path, "r:gz") as archive:
            members = list(archive.getmembers())
            print(f"archive={archive_path} members={len(members)} first={members[:5]}")
            archive.extractall(destination, members=safe_members(archive, destination))
    split_candidates = list(downloads.rglob("splits.csv"))
    if not split_candidates:
        raise FileNotFoundError(f"Missing splits.csv under {downloads}")
    shutil.copy2(split_candidates[0], data_root / "splits.csv")
    print(f"data_root={data_root.resolve()}")


if __name__ == "__main__":
    main()

