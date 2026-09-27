# Paper 3 — Full 5-Seed Results

## Setup
- XLM-R base + LoRA (r=32, alpha=64)
- Languages: eng, hin, deu, esp, chn
- Clients: 5, Dirichlet alpha=0.5
- Rounds: 10, local_epochs: 2, batch_size: 8, max_length: 64, LR: 2e-4
- Seeds: 42, 43, 44, 45, 46
- Methods: FedSVD baseline, FedSVD + FairBatch alpha=0.2

## Key Findings

### Aggregate Utility (Macro-F1)
- Baseline: 0.8006 +/- 0.0112
- FairBatch: 0.7931 +/- 0.0105
- Delta: -0.0075 (paired t-test p=0.0398, Cohen's d=-1.34)

FairBatch significantly reduces aggregate utility.

### Cross-Lingual Gap
- Baseline: 0.1600 +/- 0.0176
- FairBatch: 0.1581 +/- 0.0106
- Delta: -0.0019 (p=0.7550, d=-0.15, not significant)

FairBatch has no significant effect on gap.

### German (Worst Language)
- Baseline: 0.6959 +/- 0.0171
- FairBatch: 0.6930 +/- 0.0101
- Delta: -0.0029 (p=0.6969, not significant)

## Conclusion
FairBatch alpha=0.2 does NOT reduce cross-lingual disparity in federated multilingual LoRA.

## Files
- pilot_fedsvd_seed{42..46}_metrics.csv
- pilot_fairbatch_a02_seed{42..46}_metrics.csv
- summary.csv

## Date
2026-09-27
