from __future__ import annotations

import argparse
import json
import math
import tempfile
import time
import traceback
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.datasets import Camelyon16Bags, collate_bags
from src.models import build_model
from src.utils.config import load_yaml
from scripts.train import build_optimizer


def bag_size(dataset: Camelyon16Bags, index: int) -> int:
    return int(dataset[index]["X"].shape[0])


def choose_bags(dataset: Camelyon16Bags) -> dict[str, int]:
    sizes = [(bag_size(dataset, i), i) for i in range(len(dataset))]
    sizes.sort()
    median_size = sizes[len(sizes) // 2][0]
    medium = min(sizes, key=lambda pair: abs(pair[0] - median_size))
    return {"small": sizes[0][1], "medium": medium[1], "large": sizes[-1][1]}


def finite(value: torch.Tensor) -> bool:
    return bool(torch.isfinite(value).all().item())


def smoke_one(model_name: str, config: dict, item: dict, device: torch.device, checkpoint_dir: Path) -> dict:
    batch = collate_bags([item])
    x = batch["X"].to(device)
    mask = batch["mask"].to(device)
    y = batch["Y"].float().to(device)
    model = build_model(config).to(device)
    optimizer = build_optimizer(model, config["training"])
    model.train()
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters() if parameter.requires_grad}
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    _, loss, loss_details = model.compute_training_loss(y=y, x=x, mask=mask)
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    gradients_finite = bool(gradients) and all(finite(gradient) for gradient in gradients)
    optimizer.step()
    elapsed = time.perf_counter() - started
    parameter_changed = any(not torch.equal(before[name], parameter.detach()) for name, parameter in model.named_parameters() if parameter.requires_grad)
    checkpoint_path = checkpoint_dir / f"{model_name}.pt"
    torch.save({"model": model.state_dict()}, checkpoint_path)

    model.eval()
    with torch.no_grad():
        eval_score, eval_loss, _ = model.compute_validation_loss(y=y, x=x, mask=mask)
    reloaded = build_model(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    reloaded.load_state_dict(checkpoint["model"])
    reloaded.eval()
    with torch.no_grad():
        reload_score = reloaded(x, mask)
    evaluation_finite = finite(eval_score) and finite(eval_loss) and finite(reload_score)
    probability = torch.sigmoid(reload_score)
    output_valid = bool(reload_score.shape == (1,) and finite(probability) and ((probability >= 0) & (probability <= 1)).all().item())
    peak_mb = None if device.type != "cuda" else round(torch.cuda.max_memory_allocated(device) / (1024 * 1024), 2)
    result = {
        "status": "passed" if all((finite(loss), gradients_finite, parameter_changed, evaluation_finite, output_valid)) else "failed",
        "bag_name": str(item["bag_name"]),
        "n_instances": int(mask.sum().item()),
        "training_loss": float(loss.detach().cpu()),
        "training_loss_terms": {key: float(value.detach().cpu()) for key, value in loss_details.items() if isinstance(value, torch.Tensor)},
        "validation_bag_loss": float(eval_loss.detach().cpu()),
        "elapsed_seconds": round(elapsed, 3),
        "peak_memory_mb": peak_mb,
        "gradient_count": len(gradients),
        "gradient_finite": gradients_finite,
        "optimizer_step_changed_parameter": parameter_changed,
        "checkpoint_saved_and_reloaded": True,
        "evaluation_output_finite_and_valid": output_valid,
        "transmil_internal_padding": int(math.ceil(math.sqrt(int(mask.sum().item()))) ** 2 - int(mask.sum().item())) if model_name == "transmil" else 0,
    }
    del reloaded, model, optimizer, x, mask, y
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=Path("reports/project/official_smoke_test.json"))
    args = parser.parse_args()
    root = args.project_root.resolve()
    device = torch.device(args.device)
    split = pd.read_csv(root / "data/splits/seed1234.csv")
    train_names = split.loc[split["partition"].astype(str) == "train", "bag_name"].tolist()
    dataset = Camelyon16Bags(root / "data/camelyon16", bag_names=train_names, fields=("X", "Y"))
    chosen = choose_bags(dataset)
    config_paths = {
        "mean_pooling": root / "configs/experiments/mean_pooling.yaml",
        "abmil": root / "configs/experiments/abmil.yaml",
        "gated_abmil": root / "configs/experiments/abmil_gated.yaml",
        "clam_sb": root / "configs/experiments/clam_sb.yaml",
        "dsmil": root / "configs/experiments/dsmil.yaml",
        "transmil": root / "configs/experiments/transmil.yaml",
    }
    results: dict = {"device": str(device), "bags": {}, "models": {}, "passed_cases": 0, "failed_cases": 0}
    for size_name, index in chosen.items():
        record = dataset.records[index]
        results["bags"][size_name] = {"bag_name": record.bag_name, "label": record.label, "n_instances": bag_size(dataset, index)}
    with tempfile.TemporaryDirectory(prefix="mil_smoke_checkpoints_") as checkpoint_dir_name:
        checkpoint_dir = Path(checkpoint_dir_name)
        for model_name, config_path in config_paths.items():
            config = load_yaml(config_path)
            results["models"][model_name] = {}
            for size_name, index in chosen.items():
                item = dataset[index]
                try:
                    case = smoke_one(model_name, config, item, device, checkpoint_dir)
                except Exception as exc:  # keep all model/bag failures visible in one report
                    case = {"status": "failed", "bag_name": str(item["bag_name"]), "n_instances": int(item["X"].shape[0]), "error": repr(exc), "traceback": traceback.format_exc()}
                results["models"][model_name][size_name] = case
                results["passed_cases"] += int(case["status"] == "passed")
                results["failed_cases"] += int(case["status"] != "passed")
    output = args.output if args.output.is_absolute() else root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    if results["failed_cases"]:
        raise SystemExit("Official real-data smoke test failed")


if __name__ == "__main__":
    main()
