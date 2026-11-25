from __future__ import annotations

import argparse
import copy
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


def _forward_logits(model, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    result = model(x, mask)
    if isinstance(result, (tuple, list)):
        result = result[0]
    if not isinstance(result, torch.Tensor):
        raise TypeError(f"Model returned {type(result)!r}, expected a tensor")
    return result.reshape(-1)


def _loss_and_logits(model, y: torch.Tensor, x: torch.Tensor, mask: torch.Tensor):
    compute_loss = getattr(model, "compute_loss", None)
    if compute_loss is None:
        logits = _forward_logits(model, x, mask)
        losses = {"bag_loss": torch.nn.functional.binary_cross_entropy_with_logits(logits, y.float())}
        return logits, losses
    logits, loss_terms = compute_loss(y=y, x=x, mask=mask)
    if isinstance(loss_terms, torch.Tensor):
        loss = loss_terms
    elif isinstance(loss_terms, dict):
        values = [value.reshape(()) for value in loss_terms.values() if isinstance(value, torch.Tensor)]
        if not values:
            raise TypeError("Model returned an empty loss dictionary")
        loss = torch.stack(values).sum()
    else:
        raise TypeError(f"Model returned unsupported loss terms: {type(loss_terms)!r}")
    return logits.reshape(-1), {"total_loss": loss}


def run_epoch(model, loader, optimizer, device, training: bool):
    model.train(training)
    losses, ys, ps, names = [], [], [], []
    for batch in loader:
        x = batch["X"].to(device)
        mask = batch["mask"].to(device)
        y = batch["Y"].float().to(device)
        with torch.set_grad_enabled(training):
            if training:
                logits, loss_terms = _loss_and_logits(model, y, x, mask)
                loss = loss_terms["total_loss"]
            else:
                logits = _forward_logits(model, x, mask)
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
    parser.add_argument("--seed", type=int, default=None, help="Override training_seed only; split seed remains fixed")
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    root = args.project_root.resolve()
    config = load_yaml(root / args.config)
    config = copy.deepcopy(config)
    if args.seed is not None:
        config.setdefault("reproducibility", {})["training_seed"] = int(args.seed)
    seed = int(config["reproducibility"]["training_seed"])
    seed_everything(seed)
    if args.output_dir is not None:
        output = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    elif args.experiment_name:
        output = root / "outputs" / "camelyon16_uni" / args.experiment_name / f"seed_{seed}"
    else:
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
        loaders[partition] = DataLoader(
            dataset,
            batch_size=int(config["training"].get("batch_size", 1)),
            shuffle=partition == "train",
            collate_fn=collate_bags,
            num_workers=int(config["training"].get("num_workers", 0)),
        )
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
            threshold = float(config["evaluation"]["threshold"])
            pd.DataFrame({"wsi_id": names, "label": y, "probability": p, "predicted_label": (np.asarray(p) >= threshold).astype(int), "checkpoint": "checkpoint_best.pt"}).to_csv(output / "predictions_test.csv", index=False)
            save_curves(y, p, output)
    print(f"output={output}")


if __name__ == "__main__":
    main()

