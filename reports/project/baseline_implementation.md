# Baseline implementation notes

## Scope

All baselines use the same CAMELYON16 fixed split `data/splits/seed1234.csv`, the existing 1024-dimensional UNI features, and batch size 1. The strict official settings are model-specific; they are recorded in `reports/project/official_implementation_audit.md` and the experiment YAML files rather than forcing every model into a shared 30-epoch recipe. No instance labels, coordinates, or test labels are used during training.

## Implemented models

| Model | Implementation | Training objective |
|---|---|---|
| Mean Pooling | Project-local masked mean pooling + linear classifier | Bag-level BCE |
| ABMIL | TorchMIL 1.0.2 `ABMIL(gated=False)` | Bag-level BCE |
| Gated ABMIL | TorchMIL 1.0.2 `ABMIL(gated=True)` | Bag-level BCE |
| CLAM-SB | Vendored official `CLAM_SB` at the pinned author commit | `0.7 * bag CE + 0.3 * instance CE` |
| DSMIL | Vendored official `FCLayer/BClassifier/MILNet` at the pinned author commit | `0.5 * bag BCE + 0.5 * max-instance BCE` |
| TransMIL | Vendored official `TransMIL` at the pinned author commit | Two-class CrossEntropy; scalar evaluation score is `logit_1-logit_0` |

The project wrappers are adapters only: they normalize output shapes and expose one project-wide `forward(x, mask)` / `compute_training_loss(y, x, mask)` contract. CLAM, DSMIL and TransMIL use the author implementations rather than simplified substitutes. ABMIL/Gated ABMIL remain TorchMIL 1.0.2 adapters and are explicitly labeled as such.

## Important compatibility note

The official TransMIL forward signature has no external mask argument and internally squares a bag by repeating instances. The project wrapper therefore removes only collator padding before calling it; it never truncates a bag. The experiment configurations keep `batch_size: 1`, so the collator does not add cross-bag padding.

## Experiment isolation

`scripts/run_seeds.py` writes each run to `outputs/camelyon16_uni/<experiment>/seed_<seed>/` and each seed has an independent log in `logs/`. The historical ABMIL seed 1234 is also stored under `outputs/camelyon16_uni/abmil_ungated/seed_1234/`; `scripts/summarize_results.py` can include or exclude it with `--exclude-historical`.
