# Paper 3 — `naive_10r` Evidence

Repository: `girijashankardewangan/federated-multilingual-emotion-analytics`  
Branch: `paper3-fairdp-xlm`

This bundle archives the completed XLM-R–LoRA FedAvg 10-round
seed-42 evidence.

## Configuration

- Model: `xlm-roberta-base`
- LoRA rank: 32
- LoRA alpha: 64
- Languages: `eng, hin, deu, esp, chn`
- Clients: 5
- Communication rounds: 10
- Local epochs: 2
- Batch size: 8
- Max length: 64
- Learning rate: 2e-4
- Dirichlet alpha: 0.5
- Seed: 42
- Max grad norm: 1.0
- Aggregation: FedAvg

## Final seed-42 metrics

- Validation Macro-F1: 0.8132
- Validation Micro-F1: 0.8160
- Test Macro-F1: 0.8209
- Test Micro-F1: 0.8222

## Per-language test Macro-F1

| Language | Macro-F1 |
|---|---:|
| CHN | 0.7872 |
| DEU | 0.7232 |
| ENG | 0.7902 |
| ESP | 0.8562 |
| HIN | 0.8802 |

## Files

- `fedavg_10r_per_language.csv`
- `fedavg_10r_summary.json`
- `fedavg_10r_aggregate.json`
- `table_matched_10r.tex`
- `seed42/fedavg_10r_seed42_history.json`
- `seed42/fedavg_10r_seed42_metrics.csv`
- `seed42/paper3-fedavg-seed42-log.txt`

Large model checkpoints are **not** stored in GitHub.
They remain archived on Google Drive.

Note: this archive currently represents seed 42 only.
Five-seed aggregate claims should be made only after seeds 43–46
are completed and independently verified.
