from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.datasets import Camelyon16Bags, collate_bags
from src.models import build_model
from src.utils.config import load_yaml
from src.utils.seed import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path("."))
    args = parser.parse_args()
    config = load_yaml(args.project_root / "configs/experiments/abmil.yaml")
    seed_everything(1234)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset = Camelyon16Bags(args.project_root / "data/camelyon16", partition="train", fields=("X", "Y"))
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_bags, num_workers=0)
    batch = next(iter(loader))
    model = build_model(config).to(device)
    x = batch["X"].to(device)
    mask = batch["mask"].to(device)
    y = batch["Y"].float().to(device)
    logits = model(x, mask)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    print(f"device={device}")
    print(f"X_shape={tuple(x.shape)} mask_shape={tuple(mask.shape)} Y_shape={tuple(y.shape)}")
    print(f"logits_shape={tuple(logits.shape)} loss={float(loss):.6f}")
    print(f"gradient_tensors={len(gradients)}")
    if "y_inst" in batch or "coords" in batch:
        raise AssertionError("Smoke test unexpectedly passed evaluation-only fields")


if __name__ == "__main__":
    main()

