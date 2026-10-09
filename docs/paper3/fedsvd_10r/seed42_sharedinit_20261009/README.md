# Paper 3 — Matched FedSVD 10-Round Evidence (Seed 42)

Repository: https://github.com/girijashankardewangan/federated-multilingual-emotion-analytics
Branch: `paper3-fairdp-xlm`

## Experiment

- Model: XLM-RoBERTa (`xlm-roberta-base`) + LoRA
- Aggregation: FedSVD
- Rounds: 10; final stored round index is zero-based (`9`)
- Seed: 42
- Languages: English, Hindi, German, Spanish, and Mandarin
- Clients: 5
- Train / validation / test: 24,912 / 5,338 / 5,339 records

## Final metrics

- Validation Macro-F1: 0.7888
- Validation Micro-F1: 0.7916
- Test Macro-F1: 0.7933
- Test Micro-F1: 0.7954
- Runtime: 89.02 minutes

## Matched seed-42 comparison

The previously archived FedAvg run reported test Macro-F1 0.8209 and
Micro-F1 0.8222. The fresh FedSVD run reported 0.7933 and
0.7954, respectively.

This is a **single-seed descriptive comparison**, not a five-seed
significance test or a claim of general superiority.

## Privacy

This experiment did **not** enable differential privacy. The blank
epsilon field is not a privacy certification; no DP guarantee is claimed.

## Reproducibility artifacts

- `fedsvd_10r_seed42_summary.json`: configuration and final results
- `fedsvd_10r_seed42_per_language.csv`: language-wise and label-wise results
- `matched_comparison_seed42.csv`: descriptive matched comparison
- `table_matched_seed42.tex`: LaTeX table
- `seed42/fedsvd_10r_seed42_metrics.csv`: original round-level metrics
- `seed42/fedsvd_10r_seed42_log.txt`: full training log
- `seed42/*history*.json`: round history, if produced
- `source_snapshot/`: exact runner and XLM-R model source snapshots
- `SHA256SUMS.json`: integrity hashes for archived files

Large model checkpoints remain on Google Drive and are intentionally
not included in this GitHub bundle.

## Google Drive

Metrics: `/content/drive/MyDrive/paper3_fairdp_xlm/production_10r/fedsvd/seed42/results/fedsvd_10r_seed42_metrics.csv`
Checkpoints: `/content/drive/MyDrive/paper3_fairdp_xlm/production_10r/fedsvd/seed42/checkpoints`
Log: `/content/drive/MyDrive/paper3_fairdp_xlm/production_10r/logs/fedsvd_10r_seed42.log`
