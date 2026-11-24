# Foundation Model-based WSI MIL Research Bench

This project is an isolated, reproducible baseline for CAMELYON16 WSI-level binary classification using fixed UNI patch embeddings and standard ABMIL. All large files, caches, checkpoints, and the virtual environment belong under this project directory.

The first experiment deliberately uses slide-level labels only. `coords` and `y_inst` remain available for later localization/evaluation, but are never passed to the ABMIL training step.

## Layout

```text
WSI-MIL-Research/
├── configs/                 YAML experiment and dataset configuration
├── data/downloads/          only the selected Hugging Face archives/files
├── data/camelyon16/         extracted dataset
├── src/                     reusable dataset, model, training and metrics code
├── scripts/                 download, extract, audit, smoke test, train, evaluate
├── outputs/                 checkpoints, metrics, predictions and plots
├── reports/                 audit and environment reports
└── .venv/                   project-local Python environment
```

## Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/print_environment.py
```

The scripts verify the installed TorchMIL API rather than assuming that an older `Camelyon16MIL`/2048-dimensional example is still current. The intended first model is `torchmil.models.ABMIL`, not `SmABMIL`.

## Download and prepare data

Only these five items are downloaded from `torchmil/Camelyon16_MIL`:

- `dataset/patches_512/features/features_UNI.tar.gz`
- `dataset/patches_512/labels.tar.gz`
- `dataset/patches_512/coords.tar.gz`
- `dataset/patches_512/patch_labels.tar.gz`
- `dataset/splits.csv`

Run from the project root:

```bash
python scripts/download_data.py --project-root .
python scripts/extract_data.py --project-root .
python scripts/audit_data.py --project-root .
python scripts/make_splits.py --project-root .
```

If the HPC cannot reach Hugging Face directly, expose the local proxy through an SSH reverse tunnel. For example, keep this command running in a local CMD window, using remote port `6066` and local proxy port `6064`:

```cmd
ssh -i "D:\ssh_key\shenzhen" -o IdentitiesOnly=yes -N -T -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -R 127.0.0.1:6066:127.0.0.1:6064 yutong@100.121.37.21
```

Then run the download process on the HPC with the remote proxy port:

```bash
HTTP_PROXY=http://127.0.0.1:6066 HTTPS_PROXY=http://127.0.0.1:6066 \
  python scripts/download_data.py --project-root . --proxy http://127.0.0.1:6066
```

The downloader uses fixed byte-range requests, writes each file to a `.part` path, and atomically renames it after completion. Restarting the command resumes from the existing `.part` size; it never calls a full-repository snapshot. Keep archives and generated data outside Git.

## Smoke test and baseline

Do not train until the audit passes:

```bash
python scripts/smoke_test.py --project-root .
python scripts/train.py --project-root . --config configs/experiments/abmil.yaml
python scripts/evaluate.py --project-root . --config configs/experiments/abmil.yaml \
  --checkpoint outputs/abmil_seed1234/checkpoint_best.pt
```

The official test partition is never used for early stopping, model selection, or threshold optimization. The output directory contains the configuration, checkpoint, history, validation/test JSON, test predictions, ROC/PR plots, and confusion matrix.

