from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.datasets import Camelyon16Bags, collate_bags
from src.models import build_model
from src.utils.config import load_yaml


def bag_size(dataset: Camelyon16Bags, index: int) -> int:
    return int(dataset[index]["X"].shape[0])


def choose_bags(dataset: Camelyon16Bags) -> dict[str, int]:
    sizes = [(bag_size(dataset, i), i) for i in range(len(dataset))]
    sizes.sort()
    median_size = sizes[len(sizes) // 2][0]
    medium = min(sizes, key=lambda pair: abs(pair[0] - median_size))
    return {"small": sizes[0][1], "medium": medium[1], "large": sizes[-1][1]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=Path("reports/project/official_smoke_test.json"))
    args = parser.parse_args()
    root = args.project_root.resolve()
    device = torch.device(args.device)
    dataset = Camelyon16Bags(root / "data/camelyon16", fields=("X", "Y"))
    chosen = choose_bags(dataset)
    config_paths = {
        "mean_pooling": root / "configs/experiments/mean_pooling.yaml",
        "abmil": root / "configs/experiments/abmil.yaml",
        "gated_abmil": root / "configs/experiments/abmil_gated.yaml",
        "clam_sb": root / "configs/experiments/clam_sb.yaml",
        "dsmil": root / "configs/experiments/dsmil.yaml",
        "transmil": root / "configs/experiments/transmil.yaml",
    }
    results = {"device": str(device), "bags": {}, "models": {}}
    for size_name, index in chosen.items():
        record = dataset.records[index]
        results["bags"][size_name] = {"bag_name": record.bag_name, "label": record.label, "n_instances": bag_size(dataset, index)}
    for model_name, config_path in config_paths.items():
        model = build_model(load_yaml(config_path)).to(device).eval()
        results["models"][model_name] = {}
        for size_name, index in chosen.items():
            item = dataset[index]
            batch = collate_bags([item])
            x = batch["X"].to(device)
            mask = batch["mask"].to(device)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            started = time.perf_counter()
            with torch.no_grad():
                score = model(x, mask)
            elapsed = time.perf_counter() - started
            peak_mb = None if device.type != "cuda" else round(torch.cuda.max_memory_allocated(device) / (1024 * 1024), 2)
            n = int(mask.sum().item())
            results["models"][model_name][size_name] = {
                "bag_name": item["bag_name"],
                "n_instances": n,
                "score": float(score.reshape(-1)[0].cpu()),
                "elapsed_seconds": round(elapsed, 3),
                "peak_memory_mb": peak_mb,
                "transmil_internal_padding": int(math.ceil(math.sqrt(n)) ** 2 - n) if model_name == "transmil" else 0,
            }
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    output = args.output if args.output.is_absolute() else root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
