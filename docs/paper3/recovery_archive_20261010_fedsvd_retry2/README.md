# Paper 3 — FedSVD Recovery Run

## Verification status
- All 10 round checkpoints (rounds 0–9) were verified.
- The `latest` checkpoint records round 9.
- The final checkpoint's model weights match round 9.
- History JSON and metrics CSV cover rounds 0–9.
- Full training log and the exact saved runner/checkpoint-manager source are included.

## Model and training configuration
- Base model: `xlm-roberta-base`
- Method: FedSVD + LoRA
- LoRA rank: 32
- LoRA alpha: 64
- Languages: English, Hindi, German, Spanish, Chinese
- Clients: 5
- Rounds: 10
- Local epochs: 2
- Batch size: 8
- Max sequence length: 64
- Learning rate: 0.0002
- Seed / split seed / partition seed: 42
- Differential Privacy: OFF

## Compact checkpoint format
These are compact trainable-parameter checkpoints, not full 1+ GiB base-model snapshots.
Restore by reconstructing the base model and LoRA architecture with the included source/configuration, then loading the saved trainable state using the matching compact-checkpoint restore path.

The files for rounds 0–8 are kept separately. The final `latest` file represents round 9, so a duplicate round-9 checkpoint file is omitted from this archive.

## Contents
- `checkpoints/`: round checkpoints 0–8, final `latest`, and history JSON
- `evidence/`: results, metrics and training log
- `source_snapshot/`: runner and checkpoint manager used for this recovery
- `experiment_metadata.json`: run configuration
- `environment.json`: Python/package/GPU environment
- `SHA256SUMS.json`: checksums for archive files

This recovery run is not a Differential Privacy-protected result.
