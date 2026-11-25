from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def _complete(output: Path) -> bool:
    required = ("checkpoint_best.pt", "training_history.csv", "metrics_test.json", "predictions_test.csv")
    return all((output / name).exists() for name in required)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run independent MIL seeds sequentially.")
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--skip-complete", action="store_true")
    args = parser.parse_args()

    root = args.project_root.resolve()
    logs = root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    status_path = root / "reports" / args.experiment_name / "seeds_status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status: list[dict[str, object]] = []
    train_script = root / "scripts" / "train.py"
    config = args.config if args.config.is_absolute() else root / args.config

    for seed in args.seeds:
        output = root / "outputs" / "camelyon16_uni" / args.experiment_name / f"seed_{seed}"
        log_path = logs / f"{args.experiment_name}_seed_{seed}.log"
        if args.skip_complete and _complete(output):
            record = {"seed": seed, "status": "skipped_complete", "output": str(output)}
            status.append(record)
            print(record, flush=True)
            continue

        command = [
            sys.executable,
            str(train_script),
            "--project-root",
            str(root),
            "--config",
            str(config),
            "--seed",
            str(seed),
            "--experiment-name",
            args.experiment_name,
        ]
        started = datetime.now(timezone.utc).isoformat()
        print({"seed": seed, "status": "started", "log": str(log_path)}, flush=True)
        with log_path.open("w", encoding="utf-8") as handle:
            result = subprocess.run(command, cwd=root, stdout=handle, stderr=subprocess.STDOUT, env=os.environ.copy())
        record = {
            "seed": seed,
            "status": "completed" if result.returncode == 0 else "failed",
            "returncode": result.returncode,
            "output": str(output),
            "log": str(log_path),
            "started_at": started,
            "finished_at": datetime.now(timezone.utc).isoformat(),
        }
        status.append(record)
        print(record, flush=True)
        status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
        if result.returncode != 0:
            raise SystemExit(result.returncode)

    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(f"status={status_path}", flush=True)


if __name__ == "__main__":
    main()
