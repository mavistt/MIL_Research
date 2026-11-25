from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


METRICS = (
    "auroc",
    "auprc",
    "accuracy",
    "precision",
    "recall_sensitivity",
    "specificity",
    "f1",
    "balanced_accuracy",
)
EXPERIMENTS = (
    "abmil_ungated",
    "gated_abmil",
    "mean_pooling",
    "clam_sb",
    "dsmil",
    "transmil",
)
COMMON_SEEDS = (0, 1, 2)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare all baseline models on common seeds 0, 1, 2.")
    parser.add_argument("--project-root", type=Path, default=Path("."))
    args = parser.parse_args()
    root = args.project_root.resolve()
    output_root = root / "outputs" / "camelyon16_uni"
    report_root = root / "reports" / "project"
    report_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    reference_test: tuple[list[str], list[int]] | None = None
    reference_split_hash: str | None = None
    provenance: list[dict[str, object]] = []
    consistency: list[str] = []

    for experiment in EXPERIMENTS:
        experiment_root = output_root / experiment
        per_seed: list[dict[str, object]] = []
        for seed in COMMON_SEEDS:
            seed_root = experiment_root / f"seed_{seed}"
            metrics_path = seed_root / "metrics_test.json"
            predictions_path = seed_root / "predictions_test.csv"
            split_path = seed_root / "split_used.csv"
            metadata_path = seed_root / "run_metadata.json"
            if not all(path.exists() for path in (metrics_path, predictions_path, split_path, metadata_path)):
                raise SystemExit(f"Incomplete result for {experiment} seed {seed}: {seed_root}")

            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            predictions = pd.read_csv(predictions_path)
            test_signature = (predictions["wsi_id"].astype(str).tolist(), predictions["label"].astype(int).tolist())
            split_hash = sha256(split_path)
            if reference_test is None:
                reference_test = test_signature
                reference_split_hash = split_hash
            else:
                if test_signature != reference_test:
                    consistency.append(f"test WSI IDs/labels differ: {experiment} seed {seed}")
                if split_hash != reference_split_hash:
                    consistency.append(f"split_used.csv SHA256 differs: {experiment} seed {seed}")

            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            provenance.append(
                {
                    "baseline": experiment,
                    "seed": seed,
                    "code_commit": metadata.get("code_commit", "unknown"),
                    "parameter_count": metadata.get("parameter_count"),
                    "trainable_parameter_count": metadata.get("trainable_parameter_count"),
                    "actual_epochs": metadata.get("actual_epochs"),
                    "best_epoch": metadata.get("best_epoch"),
                    "total_training_seconds": metadata.get("total_training_seconds"),
                    "peak_gpu_memory_mb": metadata.get("peak_gpu_memory_mb"),
                    "mean_test_model_seconds_per_wsi": metadata.get("evaluation_timing", {}).get("test", {}).get("mean_model_compute_seconds_per_wsi"),
                }
            )
            per_seed.append({"seed": seed, **{metric: metrics.get(metric, np.nan) for metric in METRICS}})

        frame = pd.DataFrame(per_seed)
        row: dict[str, object] = {"baseline": experiment, "common_seed_count": len(COMMON_SEEDS)}
        for metric in METRICS:
            values = pd.to_numeric(frame[metric], errors="coerce").dropna().to_numpy(dtype=float)
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_sample_std"] = float(np.std(values, ddof=1))
            row[f"{metric}_valid_n"] = int(len(values))
        rows.append(row)

    if consistency:
        raise SystemExit("Consistency check failed:\n" + "\n".join(consistency))

    comparison = pd.DataFrame(rows)
    comparison.to_csv(report_root / "all_baselines_comparison.csv", index=False)
    pd.DataFrame(provenance).to_csv(report_root / "all_baselines_compute.csv", index=False)

    def formatted(row: pd.Series, metric: str) -> str:
        return f"{row[f'{metric}_mean']:.6f} ± {row[f'{metric}_sample_std']:.6f} (n={int(row[f'{metric}_valid_n'])})"

    lines = [
        "# All Baselines Comparison",
        "",
        "主表只使用所有模型共有的 training seed 0、1、2；std 为样本标准差（ddof=1）。",
        f"固定 split SHA256: `{reference_split_hash}`。测试 WSI ID 和 label 在全部模型/seed 间完全一致，共 {len(reference_test[0]) if reference_test else 0} 张 test WSI。",
        "",
        "## Common seeds 0/1/2",
        "",
        "| Baseline | AUROC | AUPRC | Accuracy | Precision | Sensitivity | Specificity | F1 | Balanced Accuracy |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in comparison.iterrows():
        lines.append("| " + " | ".join([str(row["baseline"]), *(formatted(row, metric) for metric in METRICS)]) + " |")
    lines.extend(
        [
            "",
            "## Provenance and compute records",
            "",
            "每个 seed 的 code commit、参数量、实际 epoch、最佳 checkpoint epoch、总训练时间、峰值显存和单 WSI 测试模型计算时间见 `reports/project/all_baselines_compute.csv`。",
            "",
            "标准 ABMIL 的历史 seed1234 与新 seed 0–3 的五 Seed 结果继续保留在 `reports/abmil_ungated/summary.*`，不混入本表的 common-seed 主比较。",
        ]
    )
    (report_root / "all_baselines_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"comparison={report_root / 'all_baselines_comparison.csv'}")
    print(f"markdown={report_root / 'all_baselines_comparison.md'}")
    print(f"compute={report_root / 'all_baselines_compute.csv'}")


if __name__ == "__main__":
    main()
