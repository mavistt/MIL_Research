# Baseline implementation notes

## Scope

All baselines use the same CAMELYON16 fixed split `data/splits/seed1234.csv`, the existing 1024-dimensional UNI features, batch size 1, a maximum of 30 epochs, and validation-based early stopping. No instance labels, coordinates, or test labels are used during training.

## Implemented models

| Model | Implementation | Training objective |
|---|---|---|
| Mean Pooling | Project-local masked mean pooling + linear classifier | Bag-level BCE |
| ABMIL | TorchMIL 1.0.2 `ABMIL(gated=False)` | Bag-level BCE |
| Gated ABMIL | TorchMIL 1.0.2 `ABMIL(gated=True)` | Bag-level BCE |
| CLAM-SB | TorchMIL 1.0.2 `CLAM_SB` | Official bag loss plus official instance clustering loss |
| DSMIL | TorchMIL 1.0.2 `DSMIL` | Official critical-instance and bag losses returned by `compute_loss` |
| TransMIL | TorchMIL 1.0.2 `TransMIL` | Bag-level BCE through the official forward path |

The TorchMIL wrappers are adapters only: they normalize output shapes and expose one project-wide `forward(x, mask)` / `compute_loss(y, x, mask)` contract. They do not replace the official attention, instance-selection, or transformer modules with simplified substitutes.

## Important compatibility note

TorchMIL 1.0.2's TransMIL forward signature has no external mask argument and internally squares a bag by repeating instances. The project wrapper therefore rejects externally padded bags instead of silently treating padding as data. The experiment configurations keep `batch_size: 1`, so the collator does not add cross-bag padding; the official TransMIL sequence-squaring behavior remains explicit and documented.

## Experiment isolation

`scripts/run_seeds.py` writes each run to `outputs/camelyon16_uni/<experiment>/seed_<seed>/` and each seed has an independent log in `logs/`. The historical `outputs/abmil_seed1234/` directory is not overwritten. `scripts/summarize_results.py` reports per-seed metrics, mean, sample standard deviation, and valid count; it ignores `outputs/abmil_evaluated/`.
