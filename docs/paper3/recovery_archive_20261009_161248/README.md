# Paper 3 — Recovery archive

Repository: https://github.com/girijashankardewangan/federated-multilingual-emotion-analytics
GitHub username inferred from repository URL: girijashankardewangan
Canonical paper branch: paper3-fairdp-xlm
Archive branch: paper3-archive-20261009_161248

## Contents
- Patched source files and a source snapshot
- Ten-round validation metrics, final test CSV/JSON and full training log
- Compact LoRA adapter and classifier weights
- Research identity/contact metadata
- Environment versions and SHA-256 checksums

## Verified experiment
FedAvg, seed 42, fixed split/partition seed 42, XLM-R, five languages,
five clients, ten rounds, two local epochs, LoRA r=32/alpha=64.
Test Macro-F1: 0.823
Test Micro-F1: 0.8238
DP and FairBatch were disabled.

Compact adapter restoration was tested against the loaded final model.
Maximum absolute logit difference: 9.5367431640625e-07

The frozen base encoder and raw BRIGHTER records are not included.
To restore the model, load `xlm-roberta-base` and the adapter/configuration
in this folder, then load `classifier.pt`. The original full checkpoint
remains at its existing Colab path and was not uploaded or deleted.

File integrity hashes are in `SHA256SUMS.json`.
