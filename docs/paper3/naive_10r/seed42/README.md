# Paper 3 — FedAvg 10-Round Seed 42 Evidence

This folder contains the reproducibility/evidence artifacts for the
Paper 3 XLM-R + LoRA FedAvg 10-round run with seed 42.

## Configuration

- Model: `xlm-roberta-base`
- LoRA rank: 32
- LoRA alpha: 64
- Languages: `eng, hin, deu, esp, chn`
- Clients: 5
- Rounds: 10
- Local epochs: 2
- Batch size: 8
- Max length: 64
- Learning rate: 2e-4
- Dirichlet alpha: 0.5
- Seed: 42
- Max grad norm: 1.0
- Aggregation: FedAvg

## Final reported metrics from the run

- Test Macro-F1: 0.8209
- Test Micro-F1: 0.8222
- Validation Macro-F1: 0.8132
- Validation Micro-F1: 0.8160

## Artifact policy

Large model checkpoints are intentionally NOT stored in GitHub.
They remain archived on Google Drive.

## Source run

Google Drive:
`/content/drive/MyDrive/paper3_fairdp_xlm/production_10r/fedavg/seed42/`

Generated on: 2026-10-08T19:20:22.053137
