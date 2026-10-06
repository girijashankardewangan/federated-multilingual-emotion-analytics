# 2. Introduction

Emotion classification from text is a foundational capability in applications
ranging from mental-health monitoring [1] and content moderation [2] to
customer-service analytics [3] and social-media analysis [4]. As these systems
deploy across global user bases, focus has shifted from English-only
classification toward multilingual emotion analysis. The recently released
BRIGHTER dataset [5] has made multilingual emotion research practical by
providing human-annotated multi-label emotion data across 28 languages, with
categories including joy, anger, fear, sadness, surprise, and disgust.
Training such multilingual emotion models, however, requires collecting
user-generated text, which raises significant privacy concerns.

Federated Learning (FL) [6] offers a promising alternative: clients train
local models on their own data and share only model updates, not the underlying
text. This preserves privacy in principle and aligns with emerging
data-protection regulations such as the GDPR. However, standard FL carries
substantial communication cost, particularly for large multilingual transformer
models where each round requires exchanging hundreds of millions of parameters.
Parameter-efficient fine-tuning methods such as Low-Rank Adaptation (LoRA) [7]
reduce this cost by training only a small subset of parameters, typically
under 2% of the total.

Three open questions motivate our study:

1. **Classification quality.** Does LoRA-based federated learning preserve
   emotion-classification quality on multilingual data, and how does it
   compare with centralized training and other aggregation strategies?

2. **Cross-lingual fairness.** How does cross-lingual disparity behave under
   federated settings, and can existing fairness methods mitigate it?

3. **Privacy integration.** Can differentially private stochastic gradient
   descent (DP-SGD) [8] integrate cleanly with LoRA in a federated setting,
   given LoRA's very small parameter footprint?

We conduct a systematic empirical study of federated multilingual emotion
classification on the BRIGHTER English subset, using XLM-RoBERTa-base [9] as
the shared encoder. We compare five training configurations across five random
seeds: centralized training, naive FedAvg [6] with LoRA, FedSVD aggregation,
FedSVD with FairBatch reweighting [10], and FedSVD with DP-SGD.

## 2.1 Key Findings

**Finding 1 — Aggregation comparison.** LoRA-based federated learning
closely approaches centralized training: naive FedAvg achieves
macro-F1 = 0.7749 ± 0.0130, compared with 0.7997 ± 0.0058 for centralized
training (a 3.1% relative gap). FedSVD achieves 0.7642 ± 0.0121, marginally
below naive FedAvg.

**Finding 2 — Fairness.** Cross-lingual disparity is substantial: in naive
FedAvg the difference between the best- and worst-performing language groups
is 0.1663 macro-F1. Applying FairBatch reweighting at the aggregation level
(with validation-based weight updates, avoiding test leakage) yields
macro-F1 = 0.7557 ± 0.0168 — a 1.1% relative decrease with no meaningful
reduction in cross-lingual gap. This suggests that FairBatch-style
reweighting, designed for centralized group fairness, does not transfer
cleanly to federated aggregation.

**Finding 3 — DP-SGD failure.** Naively combining DP-SGD with LoRA-based
federated learning causes catastrophic degradation. Macro-F1 drops from
0.7642 (FedSVD, no DP) to 0.0387 ± 0.0477 (FedSVD + DP at epsilon ≈ 4,
delta = 1e-5) — a twentyfold reduction. Two of five seeds fail to converge
entirely (F1 ≤ 0.0077), and one seed produces F1 = 0.0000. The Wilcoxon
signed-rank test yields W = 0.0 and p = 0.0625 — a strong but not
statistically significant effect at n = 5, consistent with the seed-dependent
variability observed.

**Finding 4 — Root cause.** We identify and document two distinct causes.
First, a specific integration bug: PEFT attaches LoRA adapters to the
encoder's pooler layer, but our forward pass uses the first token's hidden
state and never invokes the pooler. The unused pooler parameter receives no
gradient during backward, causing Opacus to raise "Per sample gradient is
not initialized. Not updated in backward pass?". We fix this by freezing
pooler parameters before attaching the PrivacyEngine. Second, once the bug
is fixed, the residual failure is a fundamental signal-to-noise mismatch:
LoRA gradients have norms on the order of 10^-3, whereas DP-SGD noise at
epsilon ≈ 4 has magnitude approximately 0.5 — a signal-to-noise ratio
roughly two orders of magnitude worse than full fine-tuning.

**Finding 5 — Privacy accounting.** We verify DP accounting using the
Opacus RDP accountant. For the configuration sigma = 1.1, q = 8/5965,
T = 3750, delta = 1e-5, the accountant yields epsilon = 0.6283, numerically
consistent with the value epsilon = 0.63 reported in the earlier version of
this work [11]. This confirms our accounting is correct even though the
training utility is poor.

## 2.2 Contributions

1. We present the first systematic empirical evaluation of LoRA-based
   federated learning on the BRIGHTER multilingual emotion benchmark,
   comparing five aggregation and privacy configurations across five seeds.

2. We demonstrate that FairBatch-style reweighting, designed for centralized
   group fairness, does not improve federated aggregation metrics in our
   setting.

3. We identify, isolate, and document a concrete integration failure
   between Opacus and PEFT-wrapped models, and provide a minimal fix
   (pooler freeze) that makes DP-SGD compatible with LoRA-adapted
   transformers.

4. We report a documented negative result: DP-SGD and LoRA do not combine
   cleanly on multilingual emotion classification at practical privacy
   budgets, and we quantify the signal-to-noise mismatch that explains why.

5. We provide complete reproducibility artifacts, including raw training
   logs, accounting calculations, and failure evidence.

## 2.3 Organisation

Section 3 surveys related work. Section 4 describes our methodology,
including the federated setup, LoRA fine-tuning, FedSVD aggregation,
FairBatch, and DP-SGD. Section 5 details the experimental setup and dataset.
Section 6 reports results. Section 7 discusses the signal-to-noise mismatch
and its implications. Section 8 covers limitations, and Section 9 concludes.
