from __future__ import annotations

import argparse
import copy
from pathlib import Path
import sys

import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.datasets import Camelyon16Bags, collate_bags
from src.metrics.classification import compute_metrics, dump_metrics, save_curves
from src.models import build_model
from src.utils.config import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--config", type=Path, default=Path("configs/experiments/abmil.yaml"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    root = args.project_root.resolve()
    config = load_yaml(root / args.config)
    config = copy.deepcopy(config)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = build_model(config).to(device)
    checkpoint = torch.load(root / args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    split = pd.read_csv(root / "data/splits/seed1234.csv")
    names = split.loc[split["partition"] == "test", "bag_name"].tolist()
    dataset = Camelyon16Bags(root / "data/camelyon16", bag_names=names, fields=("X", "Y"))
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_bags, num_workers=0)
    model.eval()
    ys, ps, ids = [], [], []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["X"].to(device), batch["mask"].to(device))
            ys.extend(batch["Y"].numpy().astype(int).tolist())
            ps.extend(torch.sigmoid(logits).cpu().numpy().tolist())
            ids.extend(batch["bag_name"])
    if args.output_dir is not None:
        output = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    else:
        output = root / "outputs" / "abmil_evaluated"
    output.mkdir(parents=True, exist_ok=True)
    dump_metrics(compute_metrics(ys, ps), output / "metrics_test.json")
    pd.DataFrame({"wsi_id": ids, "label": ys, "probability": ps, "predicted_label": (pd.Series(ps) >= 0.5).astype(int), "checkpoint": str(args.checkpoint)}).to_csv(output / "predictions_test.csv", index=False)
    save_curves(ys, ps, output)
    print(f"output={output}")


if __name__ == "__main__":
    main()

