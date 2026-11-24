from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path("."))
    args = parser.parse_args()
    root = args.project_root / "data" / "camelyon16"
    frame = pd.read_csv(root / "splits.csv")
    labels = pd.DataFrame({"bag_name": frame["bag_name"].astype(str), "official_split": frame["split"].astype(str)})
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.datasets.camelyon16 import canonical_id, resolve_npy

    labels["bag_name"] = labels["bag_name"].map(canonical_id)
    label_dir = root / "patches_512" / "labels"
    labels["label"] = labels["bag_name"].map(lambda name: int(pd.Series(__import__("numpy").load(resolve_npy(label_dir, name)).reshape(-1)).iloc[0]))
    train_frame = labels[labels["official_split"] == "train"].copy()
    train_ids, val_ids = train_test_split(
        train_frame["bag_name"],
        test_size=0.2,
        random_state=1234,
        stratify=train_frame["label"],
    )
    labels["partition"] = "test"
    labels.loc[labels["bag_name"].isin(train_ids), "partition"] = "train"
    labels.loc[labels["bag_name"].isin(val_ids), "partition"] = "val"
    output = args.project_root / "data" / "splits" / "seed1234.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    labels.to_csv(output, index=False)
    print(labels.groupby(["partition", "label"]).size())
    print(f"saved={output}")


if __name__ == "__main__":
    main()

