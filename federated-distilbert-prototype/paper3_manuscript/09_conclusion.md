# 9. Conclusion

We presented a systematic empirical study of LoRA-based federated
learning for multilingual emotion classification on the BRIGHTER English
dataset. We compared five configurations across five random seeds:
centralized training, naive FedAvg, FedSVD, FedSVD + FairBatch, and
FedSVD + DP-SGD.

Our contributions are threefold.

**First**, we demonstrated that LoRA-based federated learning closely
approaches centralized training for multilingual emotion classification.
Naive FedAvg achieves 0.7749 ± 0.0130 macro-F1 compared with
0.7997 ± 0.0058 for centralized training — a 3.1% relative gap — while
training only 1.89% of model parameters. This makes LoRA-based federated
learning a practical option for communication-constrained multilingual
deployments.

**Second**, we showed that FairBatch reweighting, designed for centralized
group fairness, does not transfer cleanly to server-side federated
aggregation. Despite using a validation-based update rule that avoids
test leakage, FairBatch did not meaningfully reduce the cross-lingual
gap in our setting and slightly reduced mean macro-F1. We attribute this
to the fact that language-group membership is nearly client-aligned in
our Dirichlet partition, which undermines the mechanism by which
FairBatch typically operates.

**Third**, we documented a negative result for DP-SGD + LoRA. At
epsilon = 4 with delta = 1e-5, the combination fails catastrophically:
macro-F1 drops from 0.7642 to 0.0387 (20x reduction), and two of five
seeds fail to converge entirely. We identified two causes. The first was
a specific integration bug (PEFT attaches LoRA to the unused pooler
layer, causing Opacus to raise "per sample gradient is not initialized"),
which we fixed by freezing pooler parameters before attaching the
PrivacyEngine. The second is a fundamental signal-to-noise mismatch:
LoRA gradients have norms on the order of 10^-3, whereas DP-SGD noise at
epsilon = 4 has magnitude ~0.5. The resulting SNR is two to three orders
of magnitude worse than full fine-tuning.

Our findings suggest that the current combination of parameter-efficient
training and differential privacy is not viable for federated
multilingual classification at practical privacy budgets. Practitioners
must choose between communication efficiency (LoRA without DP) and
formal privacy (full fine-tuning with DP), or wait for algorithmic
innovation that closes this gap.

## Future Work

We see four directions for future work.

**Cross-lingual federated experiments.** Our study used a single language
(English) to isolate the federated comparison. The natural extension is
to train with clients drawn from different languages, where the
cross-lingual gap and group-vs-client fairness dynamics would differ
substantially.

**Alternative privacy mechanisms.** Secure aggregation, local
differential privacy, and hybrid mechanisms may offer better
utility-privacy trade-offs than DP-SGD for LoRA-based federated learning.
We leave their empirical evaluation as future work.

**Higher-rank LoRA under DP.** Increasing LoRA rank from 32 to 64 or 128
would increase gradient magnitude and might partially mitigate the
signal-to-noise mismatch. This would need to be balanced against the
increased communication cost.

**Federated FairBatch with cross-cutting groups.** Our negative result
on FairBatch is specific to language-as-client-level-attribute. Future
work could explore FairBatch in settings where group membership
cross-cuts clients (e.g., demographic groups within each client).

## Reproducibility Statement

All code, training logs, and metrics for this paper are archived in the
public repository `federated-multilingual-emotion-analytics` on GitHub,
branch `paper3-fairdp-xlm`. Key commits are documented in Section 5.7.
The manuscript and all figures are regenerable from the repository. We
welcome replication and extension of this work.
