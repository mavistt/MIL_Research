from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.datasets import Camelyon16Bags, collate_bags
from src.metrics.classification import compute_metrics, dump_metrics, save_curves
from src.models import build_model
from src.utils.config import load_yaml, save_yaml
from src.utils.seed import seed_everything


def run_epoch(model, loader, optimizer, device, training: bool):
    model.train(training)
    losses, ys, ps, names = [], [], [], []
    for batch in loader:
        x = batch["X"].to(device)
        mask = batch["mask"].to(device)
        y = batch["Y"].float().to(device)
        with torch.set_grad_enabled(training):
            logits = model(x, mask)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        losses.append(float(loss.detach().cpu()))
        ys.extend(y.detach().cpu().numpy().astype(int).tolist())
        ps.extend(torch.sigmoid(logits).detach().cpu().numpy().tolist())
        names.extend(batch["bag_name"])
    return float(np.mean(losses)), ys, ps, names


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--config", type=Path, default=Path("configs/experiments/abmil.yaml"))
    args = parser.parse_args()
    root = args.project_root.resolve()
    config = load_yaml(root / args.config)
    seed = int(config["reproducibility"]["training_seed"])
    seed_everything(seed)
    output = root / "outputs" / f"abmil_seed{seed}"
    output.mkdir(parents=True, exist_ok=True)
    save_yaml(config, output / "config.yaml")
    shutil.copy2(root / "data/splits/seed1234.csv", output / "split_used.csv")
    split = pd.read_csv(root / "data/splits/seed1234.csv")
    device_name = config["training"].get("device", "auto")
    device = torch.device("cuda:0" if device_name == "auto" and torch.cuda.is_available() else "cpu") if device_name == "auto" else torch.device(device_name)
    model = build_model(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["training"]["learning_rate"]), weight_decay=float(config["training"]["weight_decay"]))
    loaders = {}
    for partition in ("train", "val", "test"):
        names = split.loc[split["partition"] == partition, "bag_name"].tolist()
        dataset = Camelyon16Bags(root / "data/camelyon16", bag_names=names, fields=("X", "Y"))
        loaders[partition] = DataLoader(dataset, batch_size=1, shuffle=partition == "train", collate_fn=collate_bags, num_workers=0)
    best = -float("inf")
    patience = 0
    history = []
    for epoch in range(1, int(config["training"]["epochs"]) + 1):
        train_loss, _, _, _ = run_epoch(model, loaders["train"], optimizer, device, True)
        val_loss, val_y, val_p, _ = run_epoch(model, loaders["val"], optimizer, device, False)
        val_metrics = compute_metrics(val_y, val_p)
        score = val_metrics["auroc"] if np.isfinite(val_metrics["auroc"]) else val_metrics["auprc"]
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_auroc": val_metrics["auroc"], "val_auprc": val_metrics["auprc"]})
        print(history[-1], flush=True)
        if score > best:
            best = score
            patience = 0
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_metrics": val_metrics}, output / "checkpoint_best.pt")
        else:
            patience += 1
            if patience >= int(config["training"]["early_stopping_patience"]):
                break
    pd.DataFrame(history).to_csv(output / "training_history.csv", index=False)
    checkpoint = torch.load(output / "checkpoint_best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    for partition in ("val", "test"):
        _, y, p, names = run_epoch(model, loaders[partition], optimizer, device, False)
        metrics = compute_metrics(y, p, threshold=float(config["evaluation"]["threshold"]))
        dump_metrics(metrics, output / f"metrics_{partition}.json")
        if partition == "test":
            pd.DataFrame({"wsi_id": names, "label": y, "probability": p, "predicted_label": (np.asarray(p) >= 0.5).astype(int), "checkpoint": "checkpoint_best.pt"}).to_csv(output / "predictions_test.csv", index=False)
            save_curves(y, p, output)
    print(f"output={output}")


if __name__ == "__main__":
    main()

