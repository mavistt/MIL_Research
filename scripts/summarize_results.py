from __future__ import annotations

import argparse
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize test metrics across independent seeds.")
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--exclude-historical", action="store_true")
    args = parser.parse_args()
    root = args.project_root.resolve()

    metric_rows: list[dict[str, object]] = []
    experiment_root = root / "outputs" / "camelyon16_uni" / args.experiment_name
    include_historical = args.experiment_name == "abmil_ungated" and not args.exclude_historical
    for output in sorted(experiment_root.glob("seed_*")):
        metrics_path = output / "metrics_test.json"
        if not metrics_path.exists():
            continue
        try:
            seed = int(output.name.split("_", 1)[1])
        except ValueError:
            continue
        if args.experiment_name == "abmil_ungated" and seed == 1234 and args.exclude_historical:
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        metric_rows.append({"seed": seed, **{name: metrics.get(name, np.nan) for name in METRICS}})

    if not metric_rows:
        raise SystemExit(f"No completed metrics found under {experiment_root}")
    per_seed = pd.DataFrame(metric_rows).sort_values("seed")
    summary_rows = []
    for metric in METRICS:
        values = pd.to_numeric(per_seed[metric], errors="coerce").dropna().to_numpy(dtype=float)
        summary_rows.append(
            {
                "metric": metric,
                "mean": float(np.mean(values)) if len(values) else np.nan,
                "sample_std": float(np.std(values, ddof=1)) if len(values) > 1 else np.nan,
                "valid_n": int(len(values)),
            }
        )
    summary = pd.DataFrame(summary_rows)
    reports = root / "reports" / args.experiment_name
    reports.mkdir(parents=True, exist_ok=True)
    per_seed_path = reports / "per_seed.csv"
    summary_path = reports / "summary.csv"
    markdown_path = reports / "summary.md"
    per_seed.to_csv(per_seed_path, index=False)
    summary.to_csv(summary_path, index=False)

    def fmt(value: object) -> str:
        return "nan" if pd.isna(value) else f"{float(value):.6f}"

    per_seed_headers = ["seed", *METRICS]
    historical_note = (
        "历史 ABMIL seed1234 仅在 abmil_ungated 汇总中自动纳入。"
        if include_historical
        else "本汇总只包含当前实验目录中的 seed，不纳入历史 ABMIL seed1234。"
    )
    lines = [
        f"# {args.experiment_name} 多 Seed 测试集汇总",
        "",
        f"固定 split_seed=1234；表中 std 为样本标准差（ddof=1）。{historical_note}",
        "",
        "## 每个 seed",
        "",
        "| " + " | ".join(per_seed_headers) + " |",
        "|" + "|".join("---:" for _ in per_seed_headers) + "|",
    ]
    for _, row in per_seed.iterrows():
        lines.append("| " + " | ".join([str(int(row["seed"])), *(fmt(row[metric]) for metric in METRICS)]) + " |")
    lines.extend(
        [
            "",
            "## 均值 ± 样本标准差",
            "",
            "| 指标 | mean | sample std | valid n | mean ± std |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in summary.itertuples(index=False):
        mean = fmt(row.mean)
        std = fmt(row.sample_std)
        lines.append(f"| {row.metric} | {mean} | {std} | {row.valid_n} | {mean} ± {std} |")
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"per_seed={per_seed_path}")
    print(f"summary={summary_path}")
    print(f"markdown={markdown_path}")


if __name__ == "__main__":
    main()
