# 5. Experimental Setup

## 5.1 Dataset

We use the English subset of the BRIGHTER dataset [5], accessed via the
HuggingFace Hub under the identifier
`brighter-dataset/BRIGHTER-emotion-categories`. BRIGHTER provides
multi-label emotion annotations across 28 languages. We use English (config
`eng`) to keep the federated comparison controlled; extension to multiple
languages is left for future work.

**Labels.** BRIGHTER annotates six emotions: joy, anger, fear, sadness,
surprise, and disgust. The disgust label has very few positive annotations
in English and is heavily imbalanced. To avoid the label sparsity problem
documented in prior federated experiments, we retain the five well-populated
labels: joy, anger, fear, sadness, and surprise. This reduces the number of
output classes from 6 to 5.

**Splits.** We concatenate the official train, dev, and test splits of
BRIGHTER English and re-split them into 70% train, 15% validation, and 15%
test using a fixed seed of 42. This gives approximately:

- Training set: 5,965 examples
- Validation set: 1,278 examples
- Test set: 1,279 examples

**Class distribution.** The label distribution is roughly:

- joy: ~20% positive
- anger: ~24% positive
- fear: ~18% positive
- sadness: ~18% positive
- surprise: ~13% positive

Multi-label examples (more than one positive label) account for roughly
8% of the dataset.

## 5.2 Federated Client Partitioning

We simulate K = 5 federated clients using Dirichlet partitioning with
concentration parameter alpha = 0.5. The partitioning is applied
independently to each label's positive examples and then unioned, following
the standard non-IID federated benchmark procedure [22].

For seed 42, the resulting client sizes are approximately:

| Client | Size |
|---|---|
| 0 | 1,014 |
| 1 | 2,463 |
| 2 | 544 |
| 3 | 915 |
| 4 | 1,029 |

Client sizes vary by a factor of roughly 4.5, mirroring realistic
federated deployments where clients have unequal data volumes.

## 5.3 Model

We use XLM-RoBERTa-base [9] as the shared encoder, accessed via HuggingFace
under the identifier `xlm-roberta-base`. The base model has approximately
278 million parameters, with a hidden size of 768.

We configure LoRA with rank r = 32, alpha = 64, and dropout = 0.1. LoRA is
applied to the query, key, value, and dense projections of all 12 attention
layers. The classification head is a single linear layer from the encoder's
pooled output to 5 labels.

Trainable parameter count after LoRA:
- LoRA A matrices: ~590K
- LoRA B matrices: ~590K
- Classifier: ~3.8K
- Total trainable: ~5.36M (~1.89% of 283M)

## 5.4 Training Hyperparameters

Unless otherwise noted, we use the following configuration:

| Parameter | Value |
|---|---|
| Federated rounds | 5 |
| Clients | 5 |
| Client fraction | 1.0 (all clients per round) |
| Local epochs | 2 |
| Batch size | 8 |
| Max sequence length | 64 |
| Optimizer | AdamW |
| Learning rate | 2e-4 |
| LoRA rank | 32 |
| LoRA alpha | 64 |
| LoRA dropout | 0.1 |
| Dirichlet alpha | 0.5 |
| Random seeds | 42, 43, 44, 45, 46 |

For DP-SGD experiments, additional parameters:

| Parameter | Value |
|---|---|
| Target epsilon | 4.0 |
| Delta | 1e-5 |
| Max gradient norm C | 1.0 |
| Accounting | RDP |
| Noise multiplier | auto-computed by Opacus |

## 5.5 Evaluation Metrics

We report macro-F1 and micro-F1 on the held-out test set. Macro-F1 is the
primary metric because it treats all five emotion labels equally, which is
important for multi-label classification with imbalanced labels. Micro-F1
is reported for completeness.

For cross-lingual fairness analysis, we additionally compute per-language
macro-F1 and the cross-lingual gap, defined as:

    gap = max_L F1_L - min_L F1_L

where L ranges over the five language groups present in the BRIGHTER
English test set (the dataset provides language subtags even within the
English config).

## 5.6 Statistical Tests

For the DP-SGD comparison against non-DP FedSVD, we use the Wilcoxon
signed-rank test (paired, two-sided, n = 5 seeds). We report both the W
statistic and the two-sided p-value. Given the small sample size (n = 5),
we treat p-values as descriptive rather than conclusive, and we interpret
them alongside the observed per-seed effect sizes.

## 5.7 Hardware and Reproducibility

All experiments were run on Google Colab with NVIDIA Tesla T4 and NVIDIA
L4 GPUs, using PyTorch 2.x, Transformers 5.x, PEFT 0.21, and Opacus 1.6.
Peak GPU memory usage was approximately 10 GB for 5-client experiments
and 3.8 GB for 1-client experiments.

To support reproducibility, all code, training logs, and metrics are
archived at:

- Repository: `girijashankardewangan/federated-multilingual-emotion-analytics`
- Branch: `paper3-fairdp-xlm`
- Key commits:
  - `028a123`: pooler freeze fix + VRAM instrumentation
  - `8bed10b`: five-seed FedSVD+DP results
  - `21a023b`: pre-fix DP failure logs
  - `a306444`: manuscript figures

The exact command to reproduce the flagship FedSVD+DP run is:

    python run_paper3.py
        --model_name xlm-roberta-base
        --languages eng
        --clients 5 --rounds 5 --local_epochs 2
        --batch_size 8 --max_length 64 --lr 2e-4
        --use_lora --lora_r 32 --lora_alpha 64
        --use_fedsvd --use_dp --target_epsilon 4.0
        --seed 42
        --checkpoint_dir /content/paper3_checkpoints
        --experiment_name paper3_fedsvd_dp_fixed_s42

DP accounting can be independently verified by running the RDP accountant
on the input tuple (sigma = 1.1, q = 8/5965, T = 3750, delta = 1e-5),
which yields epsilon = 0.6283 (see `docs/paper3/rdp_calculation_log.json`).
