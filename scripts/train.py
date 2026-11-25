from __future__ import annotations

import argparse
import copy
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
    compute_training_loss = getattr(model, "compute_training_loss", None)
    if compute_training_loss is not None:
        logits, loss, details = compute_training_loss(y=y, x=x, mask=mask)
        return logits.reshape(-1), loss.reshape(()), details
    logits = _forward_logits(model, x, mask)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y.float())
    return logits, loss, {"bag_loss": loss}


def _validation_loss_and_logits(model, y: torch.Tensor, x: torch.Tensor, mask: torch.Tensor):
    compute_validation_loss = getattr(model, "compute_validation_loss", None)
    if compute_validation_loss is not None:
        logits, loss, details = compute_validation_loss(y=y, x=x, mask=mask)
        return logits.reshape(-1), loss.reshape(()), details
    logits = _forward_logits(model, x, mask)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y.float())
    return logits, loss, {"bag_loss": loss}


def run_epoch(model, loader, optimizer, device, training: bool):
    model.train(training)
    losses, ys, ps, names = [], [], [], []
    for batch in loader:
        x = batch["X"].to(device)
        mask = batch["mask"].to(device)
        y = batch["Y"].float().to(device)
        with torch.set_grad_enabled(training):
            loss_fn = _loss_and_logits if training else _validation_loss_and_logits
            logits, loss, _ = loss_fn(model, y, x, mask)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        losses.append(float(loss.detach().cpu()))
        ys.extend(y.detach().cpu().numpy().astype(int).tolist())
        # For two-class CE adapters the scalar is logit_1-logit_0, whose
        # sigmoid is exactly the softmax probability of class 1.
        ps.extend(torch.sigmoid(logits).detach().cpu().numpy().tolist())
        names.extend(batch["bag_name"])
    return float(np.mean(losses)), ys, ps, names


def build_optimizer(model: torch.nn.Module, training_cfg: dict) -> torch.optim.Optimizer:
    name = str(training_cfg.get("optimizer", "adam")).lower()
    lr = float(training_cfg["learning_rate"])
    weight_decay = float(training_cfg.get("weight_decay", 0.0))
    betas = tuple(float(v) for v in training_cfg.get("betas", [0.9, 0.999]))
    if name in {"adam", "adamw"}:
        optimizer_cls = torch.optim.AdamW if name == "adamw" else torch.optim.Adam
        return optimizer_cls(model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
    if name == "radam":
        from src.models.transmil.official_radam import RAdam

        return RAdam(model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
    if name in {"lookahead_radam", "lookahead-radam"}:
        from src.models.transmil.official_lookahead import Lookahead
        from src.models.transmil.official_radam import RAdam

        base = RAdam(model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        return Lookahead(base, alpha=float(training_cfg.get("lookahead_alpha", 0.5)), k=int(training_cfg.get("lookahead_k", 6)))
    raise ValueError(f"Unsupported optimizer={name!r}")


def build_scheduler(optimizer, training_cfg: dict, epochs: int):
    name = str(training_cfg.get("scheduler", "none")).lower()
    if name in {"none", "", "null"}:
        return None
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(training_cfg.get("scheduler_t_max", epochs)),
            eta_min=float(training_cfg.get("scheduler_eta_min", 0.0)),
        )
    raise ValueError(f"Unsupported scheduler={name!r}")


def selection_score(val_loss: float, val_metrics: dict, metric: str) -> tuple[float, bool]:
    metric = metric.lower()
    if metric in {"val_loss", "loss"}:
        return -val_loss, True
    value = float(val_metrics.get(metric.replace("val_", ""), float("nan")))
    if not np.isfinite(value):
        value = float(val_metrics.get("auprc", float("nan")))
    return value, False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--config", type=Path, default=Path("configs/experiments/abmil.yaml"))
    parser.add_argument("--seed", type=int, default=None, help="Override training_seed only; split seed remains fixed")
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    root = args.project_root.resolve()
    config = copy.deepcopy(load_yaml(root / args.config))
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
    training_cfg = config["training"]
    device_name = training_cfg.get("device", "auto")
    device = torch.device("cuda:0" if device_name == "auto" and torch.cuda.is_available() else "cpu") if device_name == "auto" else torch.device(device_name)
    model = build_model(config).to(device)
    epochs = int(training_cfg["epochs"])
    optimizer = build_optimizer(model, training_cfg)
    scheduler = build_scheduler(optimizer, training_cfg, epochs)
    loaders = {}
    for partition in ("train", "val", "test"):
        names = split.loc[split["partition"] == partition, "bag_name"].tolist()
        dataset = Camelyon16Bags(root / "data/camelyon16", bag_names=names, fields=("X", "Y"))
        loaders[partition] = DataLoader(
            dataset,
            batch_size=int(training_cfg.get("batch_size", 1)),
            shuffle=partition == "train" and bool(training_cfg.get("data_shuffle", True)),
            collate_fn=collate_bags,
            num_workers=int(training_cfg.get("num_workers", 0)),
        )
    best = -float("inf")
    best_epoch = 0
    patience = 0
    history = []
    metric_name = str(training_cfg.get("selection_metric", "auroc"))
    use_early_stopping = bool(training_cfg.get("early_stopping", True))
    min_epoch = int(training_cfg.get("early_stopping_min_epoch", 1))
    patience_limit = int(training_cfg.get("early_stopping_patience", 10))
    for epoch in range(1, epochs + 1):
        train_loss, _, _, _ = run_epoch(model, loaders["train"], optimizer, device, True)
        val_loss, val_y, val_p, _ = run_epoch(model, loaders["val"], optimizer, device, False)
        val_metrics = compute_metrics(val_y, val_p)
        score, lower_is_better = selection_score(val_loss, val_metrics, metric_name)
        improved = score > best
        history.append({"epoch": epoch, "train_loss": train_loss, "val_bag_loss": val_loss, "val_loss": val_loss, "val_auroc": val_metrics["auroc"], "val_auprc": val_metrics["auprc"], "selection_score": score})
        print(history[-1], flush=True)
        if improved:
            best = score
            best_epoch = epoch
            patience = 0
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_metrics": val_metrics, "selection_metric": metric_name, "selection_score": score}, output / "checkpoint_best.pt")
        elif use_early_stopping and epoch >= min_epoch:
            patience += 1
            if patience >= patience_limit:
                break
        if scheduler is not None:
            scheduler.step()
    pd.DataFrame(history).to_csv(output / "training_history.csv", index=False)
    (output / "run_metadata.json").write_text(json.dumps({"device": str(device), "best_epoch": best_epoch, "checkpoint_selection_metric": metric_name, "selection_direction": "minimize" if lower_is_better else "maximize", "training_loss_definition": "model-specific official training objective", "validation_loss_definition": "model-specific bag-level validation loss", "data_shuffle": bool(training_cfg.get("data_shuffle", True)), "fp16": bool(training_cfg.get("fp16", False)), "gradient_accumulation_steps": int(training_cfg.get("gradient_accumulation_steps", 1)), "protocol_note": "Project-level validation/checkpoint protocol is configured independently from author-specific training loops."}, indent=2), encoding="utf-8")
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
