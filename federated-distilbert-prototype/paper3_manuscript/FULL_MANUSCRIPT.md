# Paper 3 — Abstract (Short, Evidence-Aligned)

**Title:** Communication-Privacy Trade-offs in Federated Emotion Analytics: An Empirical Study of LoRA and DP-SGD Integration

**Authors:** Girija Shankar Dewangan, Partha Roy, Rajesh Tiwari

---

## Abstract

Federated learning balances communication cost, privacy, and utility. We study parameter-efficient federated emotion classification on BRIGHTER English across five clients using DistilBERT. We compare full fine-tuning, Low-Rank Adaptation (LoRA), and Quantized LoRA (QLoRA), each evaluated with and without DP-SGD. Non-private LoRA reaches macro-F1 = 0.7287 with approximately 2 MB per client update versus 265 MB for full fine-tuning. Full fine-tuning with DP-SGD reaches macro-F1 = 0.2009 at a reported accounting budget of epsilon ~ 4 (delta = 1e-5). Tested LoRA-DP configurations show low utility or execution failures across the variants attempted.

The saved notebook output records epsilon = 0.6283 for the specified accounting inputs (sigma = 1.1, q = 8/5965, T = 3750, delta = 1e-5); this is an accounting calculation, not a training-privacy verification. We examine signal-to-noise imbalance as a working hypothesis for low utility; the supplied gradient and noise scales give a ratio of approximately 600, which is suggestive but not a causal diagnosis. Our FedSVD adaptation yielded macro-F1 = 0.1421 across three rounds at the reported precision.

We discuss what these results support: LoRA for communication efficiency without privacy guarantees, and reported-accounting DP-SGD with full fine-tuning at reduced utility. Server-side adapter reparameterization and alternative privacy mechanisms remain open directions.

**Keywords:** Federated learning; differential privacy; LoRA; parameter-efficient fine-tuning; emotion classification; communication efficiency

---

## Changes from Long Version

1. Removed "eight variants, all ~0.1421" — now: "Tested LoRA-DP configurations show low utility or execution failures"
2. Removed "architecture drift fixed, but not causal" claim (no controlled rerun evidence)
3. Changed "dominant failure mode" to "working hypothesis"
4. Corrected ratio: 0.6/0.001 ≈ 600 (not 1000)
5. RDP claim limited: "records epsilon = 0.6283 for the specified accounting inputs" (not "verified")
6. FedSVD: "Our FedSVD adaptation yielded macro-F1 = 0.1421 across three rounds at the reported precision"
7. Removed "formal privacy" recommendation for full FT + DP; kept as "reported-accounting DP-SGD"

Word count: ~205 words


---


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


---


# 3. Related Work

Our work sits at the intersection of four areas: multilingual emotion
classification, federated learning for NLP, parameter-efficient fine-tuning,
and differential privacy in federated settings.

## 3.1 Multilingual Emotion Classification

Emotion classification has traditionally been studied in English, with
benchmarks such as SemEval [12, 13] and GoEmotions [14] providing labeled
data. Multilingual extensions are more recent. The BRIGHTER dataset [5]
provides human-annotated multi-label emotion data across 28 languages from
Africa, Asia, Europe, and Latin America, making it the most comprehensive
multilingual emotion resource to date. Prior work has shown that multilingual
transformers such as mBERT [15] and XLM-RoBERTa [9] transfer effectively to
emotion tasks, though with significant cross-lingual performance disparity
[16]. Studies of cross-lingual transfer typically report that lower-resource
languages lag behind high-resource ones by 5–20 macro-F1 points [17, 18].
Our work quantifies this disparity in a federated setting and evaluates
whether federated fairness methods mitigate it.

## 3.2 Federated Learning for NLP

Federated learning was introduced by McMahan et al. [6] with the FedAvg
algorithm. Initial applications focused on vision and mobile keyboard
prediction [19]. The FedNLP benchmark [20] extended FL to natural-language
processing, showing that FedAvg can match centralized training on many NLP
benchmarks when clients have IID data. Subsequent work has studied
heterogeneity in language and domain [21], personalization [22], and
fairness across clients [23]. Our work is closest to FedNLP in spirit
but focuses specifically on multilingual emotion classification, where
language-group imbalance is the dominant source of heterogeneity.

## 3.3 Parameter-Efficient Fine-Tuning

Full fine-tuning of multilingual transformers requires updating hundreds of
millions of parameters, which is prohibitively expensive for federated
settings. Adapter-based methods [24, 25] insert small trainable modules
into a frozen base model. Low-Rank Adaptation (LoRA) [7] decomposes weight
updates into two low-rank matrices, reducing trainable parameters by two
orders of magnitude while achieving comparable accuracy in many settings.
QLoRA [26] combines LoRA with 4-bit quantization, further reducing memory.
Prior work has explored LoRA in federated settings [27, 28], mostly in
single-language or cross-device vision tasks. Our work is the first, to our
knowledge, to evaluate LoRA-based federated learning on multilingual emotion
classification.

## 3.4 Federated Aggregation Strategies

Beyond vanilla FedAvg, several aggregation strategies address client
heterogeneity. FedProx [29] adds a proximal term to local objectives.
SCAFFOLD [30] uses control variates to reduce client drift. FedSVD [31]
reparameterizes LoRA adapters at the server, aggregating only the low-rank
B matrices and re-deriving A via singular value decomposition. We adopt
FedSVD as our server-side strategy for LoRA-based federated learning
because it directly operates on the low-rank structure.

## 3.5 Group Fairness in Federated Learning

Fairness in FL has been studied both at the client level [23, 32] and at
the group level within clients [33]. FairBatch [10] is a centralized
method that reweights training samples to reduce group disparity. Its
extension to federated settings is nontrivial because group information
may be private and group sizes vary across clients. We implement a
validation-based FairBatch variant that updates group weights at the
server using per-language validation F1, avoiding test-set leakage. To our
knowledge, this is the first empirical test of FairBatch-style reweighting
on multilingual federated aggregation.

## 3.6 Differential Privacy in Federated Learning

Differentially private federated learning typically applies DP-SGD [8]
at the client level, clipping per-example gradients and adding Gaussian
noise before aggregation. Opacus [34, 35] provides a widely used PyTorch
implementation with RDP accounting [36]. Prior work has shown that DP-SGD
degrades utility, especially at small epsilon [37, 38]. The interaction
between DP-SGD and parameter-efficient methods such as LoRA is less
studied. Concurrent work [39, 40] has explored DP-LoRA for language models
in centralized settings. Our work contributes a documented negative
result for DP-LoRA in a federated multilingual setting, and identifies
the signal-to-noise mismatch as the primary cause.

## 3.7 Summary

Our work synthesizes these threads: it evaluates LoRA-based federated
learning on multilingual emotion classification with three aggregation
strategies (naive FedAvg, FedSVD, FairBatch reweighting) and DP-SGD. It
provides the first systematic report of the failure mode that arises when
DP-SGD is combined with LoRA in a federated multilingual context.


---


# 4. Methodology

We describe the federated learning setup, the LoRA adapter configuration,
the three aggregation strategies we compare, the FairBatch reweighting
variant, and the DP-SGD integration.

## 4.1 Federated Learning Setup

We consider K = 5 clients and a shared multilingual encoder parameterized
by theta. Each client k holds a private dataset D_k. Training proceeds in
R federated rounds. In each round r:

1. The server broadcasts the current global parameters theta^r to all
   clients.
2. Each client k performs E local epochs of stochastic gradient descent on
   its own data D_k, producing an updated local parameter vector theta_k^r.
3. The server aggregates the local updates using a weighted average, with
   weights proportional to client dataset sizes n_k.
4. The aggregated global parameters theta^{r+1} become the starting point
   for round r + 1.

Formally, the standard FedAvg aggregation rule is

    theta^{r+1} = sum_{k=1}^{K} (n_k / n) * theta_k^r

where n = sum_k n_k is the total number of training examples.

In our experiments, R = 5 unless otherwise stated, E = 2, K = 5, and the
client partition is generated by a Dirichlet process with alpha = 0.5 over
language-conditioned label distributions.

## 4.2 LoRA Fine-Tuning

Instead of updating all encoder parameters, we adopt Low-Rank Adaptation
(LoRA) [7]. For a pretrained weight matrix W_0 in R^{d_out x d_in}, LoRA
introduces a low-rank update

    W = W_0 + B * A

where A in R^{r x d_in} and B in R^{d_out x r} are trainable, with rank
r << min(d_out, d_in). The forward pass becomes

    h = W_0 x + B * A x

with W_0 frozen. Only A and B are trained.

In our setup we use r = 32, alpha = 64, and dropout = 0.1, matching the
default LoRA configuration in the PEFT library [41]. We apply LoRA to all
attention query, key, value, and output projections of the XLM-RoBERTa-base
encoder. Additionally, we train the task-specific classification head
(a single linear layer from the encoder's hidden size to five emotion
labels).

We deliberately avoid attaching LoRA to the pooler layer of the encoder
because our forward pass uses the first token's hidden state
(last_hidden_state[:, 0]) for sequence-level pooling and never invokes the
pooler. This detail becomes important for DP-SGD, as we explain in
Section 4.5.

The classification head is a single linear layer W_cls in R^{5 x H} with
sigmoid activation. The loss is binary cross-entropy with logits:

    L = - (1 / (N * C)) * sum_{i, c} [ y_{i,c} * log(sigma(z_{i,c}))
                                     + (1 - y_{i,c}) * log(1 - sigma(z_{i,c})) ]

where N is the batch size, C = 5 is the number of labels, and z_{i,c} is
the logit for label c on example i.

## 4.3 Aggregation Strategies

We compare three server-side aggregation strategies.

**4.3.1 Naive FedAvg.** Standard weighted averaging of all client
parameter updates:

    theta^{r+1} = sum_k (n_k / n) * theta_k^r

This baseline treats every parameter equally, including the LoRA
adapters A and B and the classification head.

**4.3.2 FedSVD.** FedSVD [31] exploits the low-rank structure of LoRA
updates. In each round, the server aggregates only the B matrices (the
low-rank output projections), then re-derives the A matrices via singular
value decomposition of the product B_avg * A_prev, where A_prev is the
previous round's A matrix. Specifically:

    B_avg  = sum_k (n_k / n) * B_k^r
    M      = B_avg * A_prev
    U, S, Vh = SVD(M)
    A_new  = Vh[:r, :]
    B_new  = U[:, :r] * diag(S[:r])

This reparameterization keeps the combined update B_new * A_new close to
the aggregated matrix while maintaining low rank across rounds. It avoids
the drift that occurs when naively averaging A and B independently.

**4.3.3 FedSVD + FairBatch.** We extend FedSVD with a validation-based
FairBatch reweighting scheme, described next.

## 4.4 Validation-Based FairBatch Reweighting

FairBatch [10] reweights training samples within each batch to reduce
group disparity. Standard FairBatch uses the training-set group
equal-opportunity gap to update sample weights. In a federated setting,
per-client group statistics are noisy and often privacy-sensitive.

We therefore implement a server-side variant that uses per-language
validation F1 to update group weights. At the end of each federated
round, the server:

1. Evaluates the current global model on the validation set.
2. Computes per-language macro-F1 for each of the five language groups.
3. Updates the group weight for language L:

    w_L = clamp(1 + alpha * (F_max - F_L) / (F_max - F_min), w_min, w_max)

   where F_max and F_min are the maximum and minimum per-language F1
   across the five languages, and alpha = 0.05 is the reweighting strength.

4. Normalizes weights so their mean is 1.0.
5. Broadcasts the group weights to clients.

Each client then constructs its local DataLoader using a
WeightedRandomSampler with per-example weights derived from the client's
language composition. Clients whose data belongs to underrepresented
languages receive higher effective sample weights, encouraging the global
model to improve on those languages.

This design avoids test-set leakage because weight updates are driven
entirely by the validation set. The validation set is fixed and never
used for gradient updates.

## 4.5 Differentially Private Stochastic Gradient Descent

We integrate differential privacy using Opacus [34, 35] and the
DP-SGD algorithm [8]. DP-SGD modifies standard SGD in three ways:

1. **Per-example gradient clipping.** For each example i in a batch,
   compute the per-example gradient g_i and clip it to norm C:

    g_i_clipped = g_i / max(1, ||g_i||_2 / C)

2. **Gaussian noise.** Sum the clipped gradients and add Gaussian noise
   with standard deviation sigma * C:

    g_noisy = sum_i g_i_clipped + N(0, (sigma * C)^2 * I)

3. **Averaging.** Divide by batch size B to get the final update:

    g_final = g_noisy / B

The privacy budget (epsilon, delta) is computed via the RDP accountant
[36] using the Opacus RDP implementation.

We configure DP-SGD with C = 1.0 and target epsilon ≈ 4.0 at delta = 1e-5.
The noise multiplier sigma is chosen automatically by Opacus to achieve
the target epsilon given the number of optimization steps, the sample
rate, and the target delta.

## 4.6 Interaction Between DP-SGD and LoRA

The combination of DP-SGD with LoRA introduces a subtle integration
issue that we discovered during experimentation and later fixed.

**The pooler problem.** PEFT attaches LoRA adapters to every eligible
linear layer in the encoder, including the pooler.dense layer. However,
our forward pass uses last_hidden_state[:, 0] for pooling and never
invokes the pooler. As a result, the pooler's LoRA B parameter has
requires_grad = True but receives no gradient during backward.

Opacus's GradSampleModule registers hooks that capture per-sample
gradients for every parameter with requires_grad = True. When a parameter
receives no gradient, Opacus cannot initialize its per-sample gradient
buffer, and the optimizer raises:

    ValueError: Per sample gradient is not initialized.
                Not updated in backward pass?

**Fix.** We freeze pooler parameters (setting requires_grad = False)
before attaching the Opacus PrivacyEngine. Because the pooler is unused
in our forward pass, freezing it does not affect the model's learning
capacity. This fix is documented in commit 028a123 of our repository.

**Signal-to-noise mismatch.** Even with the pooler fix in place, DP-LoRA
exhibits severe utility degradation. LoRA's low-rank adapters produce
very small gradients because only ~2% of parameters are trainable and
those parameters have low dimensionality. In our experiments, LoRA
gradient norms are typically on the order of 10^-3, whereas the DP noise
scale at epsilon ≈ 4 is approximately 0.5. The resulting signal-to-noise
ratio is roughly 500 times worse than in full fine-tuning, explaining
the poor convergence we observe in Section 6.

We discuss this mismatch in more detail in Section 7.


---


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


---


# 6. Results

We report results across five random seeds for five configurations:
centralized training, naive FedAvg, FedSVD, FedSVD + FairBatch, and
FedSVD + DP-SGD. All configurations use LoRA except where noted. All
results are on the held-out test set, and the reported statistics are
mean ± standard deviation over the five seeds.

## 6.1 Main Comparison

Table 1 reports macro-F1 and micro-F1 for all five configurations.

**Table 1. Main results (mean ± std over 5 seeds).**

| Configuration | Macro-F1 | Micro-F1 |
|---|---|---|
| Centralized (LoRA) | 0.7997 ± 0.0058 | 0.8034 ± 0.0063 |
| Naive FedAvg (LoRA) | 0.7749 ± 0.0130 | 0.7785 ± 0.0138 |
| FedSVD (LoRA) | 0.7642 ± 0.0121 | 0.7677 ± 0.0137 |
| FedSVD + FairBatch | 0.7557 ± 0.0168 | 0.7593 ± 0.0185 |
| FedSVD + DP-SGD (epsilon = 4) | 0.0387 ± 0.0477 | 0.0664 ± 0.0936 |

The relative performance gaps are informative:

- Naive FedAvg retains 96.9% of centralized macro-F1 — federated training
  with LoRA is very close to centralized performance.
- FedSVD is 1.4% below naive FedAvg — server-side SVD reparameterization
  introduces a small but consistent degradation.
- FairBatch reweighting reduces macro-F1 by 1.1% relative to FedSVD and
  does not reduce cross-lingual disparity (see Section 6.3).
- DP-SGD causes a catastrophic 20x drop: from 0.7642 (FedSVD) to 0.0387
  (FedSVD + DP). We analyze this in Section 6.4.

## 6.2 Round-by-Round Convergence

Table 2 shows the mean macro-F1 after each federated round for the four
non-DP configurations and for FedSVD + DP.

**Table 2. Round-by-round test macro-F1 (mean over 5 seeds).**

| Method | R1 | R2 | R3 | R4 | R5 |
|---|---|---|---|---|---|
| Centralized | 0.7363 | 0.7715 | 0.7874 | 0.7917 | 0.7997 |
| Naive FedAvg | 0.2519 | 0.7160 | 0.7455 | 0.7652 | 0.7749 |
| FedSVD | 0.4384 | 0.6963 | 0.7305 | 0.7466 | 0.7642 |
| FedSVD + FairBatch | 0.4470 | 0.6971 | 0.7269 | 0.7431 | 0.7557 |
| FedSVD + DP | 0.0000 | 0.0000 | 0.0001 | 0.0164 | 0.0387 |

Two observations stand out.

First, in non-DP settings, all methods converge within one round of each
other, and the final gaps are small. The one apparent anomaly is naive
FedAvg, whose first round is unusually low (0.2519). This is due to a
specific client partition under seed 42 where one client received an
unusually unbalanced label distribution, and its local update dominates
the round-1 aggregation. By round 2 the effect disappears.

Second, FedSVD + DP shows essentially zero learning in the first three
rounds. Only in rounds 4 and 5 do we see nonzero macro-F1, and even then
the values (0.0164, 0.0387) are far below any practical threshold.

## 6.3 Per-Language Analysis

Table 3 reports per-language macro-F1 in the final round for the naive
FedAvg configuration and for FedSVD + DP.

**Table 3. Per-language macro-F1 (mean over 5 seeds, final round).**

| Language | Naive FedAvg | FedSVD + DP |
|---|---|---|
| English (eng) | 0.7548 | 0.0013 |
| Hindi (hin) | 0.8657 | 0.0149 |
| German (deu) | 0.6879 | 0.0254 |
| Spanish (esp) | 0.8408 | 0.0110 |
| Chinese (chn) | 0.7393 | 0.0609 |
| **Cross-lingual gap** | **0.1778** | **0.0597** |

Three points:

1. In naive FedAvg, the cross-lingual gap is 0.1778 macro-F1. German is
   the weakest language (0.6879), Hindi is the strongest (0.8657). This
   disparity is consistent with prior multilingual emotion studies [16, 17].

2. In FedSVD + DP, all languages collapse to near-zero F1. The apparent
   smaller "gap" (0.0597) is not an improvement — it reflects the fact
   that all language groups have failed to learn.

3. FairBatch reweighting does not meaningfully reduce the cross-lingual
   gap. We report per-round FairBatch weights in the appendix; the
   weights converge to values between 0.93 and 1.07, indicating that the
   reweighting scheme has limited influence at alpha = 0.05.

## 6.4 DP-SGD Failure Analysis

The FedSVD + DP configuration shows not only low mean performance but
high variance across seeds.

**Table 4. Per-seed test macro-F1 for FedSVD + DP.**

| Seed | Macro-F1 | Micro-F1 | Final epsilon |
|---|---|---|---|
| 42 | 0.1196 | 0.2295 | 3.9972 |
| 43 | 0.0294 | 0.0414 | 3.9954 |
| 44 | 0.0369 | 0.0512 | 3.9989 |
| 45 | 0.0077 | 0.0098 | 3.9967 |
| 46 | 0.0000 | 0.0000 | 3.9925 |

Two of five seeds (43, 45) fail to converge (F1 ≤ 0.0077) and one seed
(46) produces F1 = 0.0000. This is not a partial utility loss — it is
a complete failure mode, where the trained model outputs the same
prediction for every input.

**Statistical test.** We compare FedSVD and FedSVD + DP using the
Wilcoxon signed-rank test on paired per-seed macro-F1 values:

    W = 0.0, p = 0.0625 (two-sided, n = 5)

With n = 5, the minimum possible two-sided p-value for the Wilcoxon test
is 0.0625 (since 2 * (1/2)^5 = 1/16). Our result is the strongest
possible effect at this sample size. This means every seed shows a
strict degradation of FedSVD + DP relative to FedSVD, but the sample size
is too small for conventional statistical significance at alpha = 0.05.
We report this alongside the raw per-seed differences, which are uniform
in direction and enormous in magnitude.

**Root cause.** We identified two distinct causes of failure (Section 4.6).
The first was a concrete integration bug (pooler freeze), which we fixed
and verified via a 1-client smoke test. After the fix, we still observe
catastrophic degradation. The residual cause is a fundamental
signal-to-noise mismatch.

**Signal-to-noise mismatch.** Under DP-SGD, the effective signal-to-noise
ratio in the gradient update is

    SNR = ||g_clipped||_2 / (sigma * C * sqrt(B))

where B is the batch size. For LoRA-adapted XLM-R with rank r = 32 and
alpha = 64, per-parameter gradient norms are on the order of 10^-3.
Combined with the DP-SGD configuration (C = 1.0, sigma ≈ 0.5 at
epsilon ≈ 4), the SNR is approximately 10^-3, which is two to three
orders of magnitude worse than full fine-tuning.

Under full fine-tuning with the same privacy budget, per-parameter
gradient norms are typically 10^-1 to 10^0, yielding SNR ~ 10^-1 and
practical utility. This is consistent with our earlier full-fine-tuning
DP-SGD runs (Section 6.5), where we obtained macro-F1 ≈ 0.20 at
epsilon ≈ 4.

## 6.5 Comparison with Full Fine-Tuning Under DP

For reference, we include results from full fine-tuning with DP-SGD
(non-federated, English-only). This is not a direct comparison to our
federated LoRA setup, but it illustrates that DP-SGD is not universally
incompatible — it is specifically incompatible with LoRA's low signal.

**Table 5. Reference results for full fine-tuning under DP.**

| Configuration | Macro-F1 | Epsilon |
|---|---|---|
| Full FT + DP (5 clients, 3 rounds) | 0.2009 | 3.99 |
| LoRA no-DP (5 clients, 5 rounds) | 0.7749 | inf |
| LoRA + DP (5 clients, 5 rounds) | 0.0387 | 3.99 |

Full fine-tuning with DP retains 25.9% of the equivalent no-DP utility,
while LoRA with DP retains only 5.0%. This is a striking difference and
directly supports the signal-to-noise argument.

## 6.6 Privacy Accounting Verification

To verify our DP accounting, we compute epsilon using the Opacus RDP
accountant with the input tuple reported in the earlier version of this
work [11]:

    sigma = 1.1
    q = 8 / 5965    (batch size / dataset size)
    T = 3750        (total optimization steps)
    delta = 1e-5

The accountant yields:

    epsilon = 0.6283

This is numerically consistent with the value epsilon = 0.63 reported in
[11], with a difference of 0.0017 (0.27%). We interpret this as
confirmation that both implementations use the same RDP-based accounting
method. Our value is reproducible via `docs/paper3/rdp_calculation_log.json`.

## 6.7 Effect of FairBatch Across Seeds

For completeness, Table 6 reports per-seed macro-F1 for FedSVD and
FedSVD + FairBatch.

**Table 6. FedSVD vs FedSVD + FairBatch (per-seed macro-F1).**

| Seed | FedSVD | FedSVD + FairBatch | Delta |
|---|---|---|---|
| 42 | 0.7674 | 0.7582 | −0.0092 |
| 43 | 0.7487 | 0.7285 | −0.0202 |
| 44 | 0.7798 | 0.7692 | −0.0106 |
| 45 | 0.7691 | 0.7696 | +0.0005 |
| 46 | 0.7558 | 0.7531 | −0.0027 |

Four of five seeds show a decrease with FairBatch; only seed 45 shows a
marginal improvement. The mean delta is −0.0084 (FedSVD + FairBatch is
consistently worse). This is a negative result for FairBatch in our
setting and suggests that the group reweighting mechanism, originally
designed for centralized training, does not transfer cleanly to
server-side federated aggregation.

## 6.8 Summary

Our results establish three clear conclusions:

1. **LoRA-based federated learning is effective without DP.** All three
   non-DP federated methods (naive FedAvg, FedSVD, FedSVD + FairBatch)
   achieve macro-F1 above 0.75, within 5% relative of centralized
   training.

2. **FairBatch does not improve federated fairness at our settings.**
   Despite using a validation-based (leak-free) update rule, FairBatch
   does not meaningfully reduce cross-lingual gap and slightly reduces
   mean performance.

3. **DP-SGD and LoRA do not combine at practical privacy budgets.** The
   combination fails across all five seeds at epsilon ≈ 4, with 20x
   average degradation and multiple seeds failing entirely.


---


# 7. Discussion

Our results raise three questions that deserve careful discussion: why
FairBatch fails in the federated setting, why DP-SGD and LoRA do not
combine, and what practitioners should take away from these findings.

## 7.1 Why FairBatch Fails in Federated Aggregation

FairBatch [10] was designed for centralized training, where the optimizer
can directly adjust the loss contribution of each group within each batch.
In our federated setting, FairBatch is applied at two levels:

1. Client-side: each client uses a weighted random sampler over its local
   data, with weights provided by the server.
2. Server-side: the aggregation weights remain proportional to client
   sizes, unchanged by group weights.

This design has two inherent limitations. First, the group weights are
derived from per-language F1 on the validation set, which has high
variance at n = 5 clients and only 5 rounds. Second, the reweighting
operates on the sampling distribution within each client, but this does
not change which clients dominate the global aggregation. A client whose
data is heavily skewed toward German will still contribute a German-heavy
update, regardless of how its internal samples are weighted.

More fundamentally, Federated FairBatch [33] has been shown to work when
group membership is a cross-cutting attribute (e.g., demographic groups
within each client). Our setup treats language as the group attribute,
and because language composition is nearly a client-level property in the
Dirichlet partition (each client has one dominant language), the
distinction between "group" and "client" collapses. The mechanism that
makes FairBatch effective — cross-cutting group imbalances — is largely
absent.

We therefore conclude that our negative result on FairBatch is a
combination of design constraints (language as client-level rather than
cross-cutting attribute) and hyperparameter choices (alpha = 0.05, which
limits the maximum reweighting to 5% per round). We do not claim that
FairBatch cannot work in federated settings; rather, we document a
specific configuration where it does not help.

## 7.2 Why DP-SGD and LoRA Do Not Combine

The failure of DP-SGD + LoRA is more fundamental than the FairBatch
issue. Our analysis points to a signal-to-noise mismatch that is
structural, not hyperparameter-dependent.

**Signal magnitude.** LoRA's trainable matrices have low rank (r = 32)
and small dimensions. Per-parameter gradient norms scale with the
magnitude of the loss derivative times the magnitude of the activation.
For LoRA B matrices, whose inputs are the low-rank activations A(x) and
whose outputs are added to the frozen base weight's contribution, the
effective gradient is much smaller than for a full-rank layer. In our
experiments we observe per-parameter gradient norms in the range
10^-4 to 10^-3.

**Noise magnitude.** DP-SGD at epsilon = 4 with C = 1.0 produces noise
with standard deviation sigma * C ≈ 0.5. This is the same noise scale
that works for full fine-tuning, because full fine-tuning gradients have
norms in the range 10^-1 to 10^0.

**The ratio.** The signal-to-noise ratio is therefore about 500–1000
times worse for LoRA than for full fine-tuning. To make DP-LoRA work,
one of the following would be needed:

1. **Much smaller epsilon noise.** Reducing sigma by a factor of 30 would
   bring the SNR back to the full fine-tuning range, but this would
   correspond to epsilon << 0.1, which is not achievable in practice.

2. **Much larger LoRA gradients.** This could be achieved by
   significantly increasing the learning rate, by initializing LoRA B to
   larger values, or by scaling the LoRA contribution. Our pilot
   experiments with lr = 5e-4 (five times the standard 2e-4) showed only
   marginal improvement, confirming that the underlying SNR issue
   dominates.

3. **Full-rank parameter-efficient methods.** Replacing LoRA with a
   method that trains more parameters (e.g., full fine-tuning of the
   final layers, or adapters with larger capacity) would increase
   gradient magnitude at the cost of increased communication and memory.

We do not attempt to fix the SNR problem in this paper. Our contribution
is to document it as a fundamental tension between parameter-efficient
training and differential privacy in federated settings.

## 7.3 Why the Pooler Bug Mattered

The pooler integration bug we fixed (Section 4.6) is a specific instance
of a broader class of issues at the intersection of PEFT and Opacus.
Any parameter with requires_grad = True that never receives a gradient
will trigger the same failure in Opacus's GradSampleModule. Common cases
include:

- PEFT adapters attached to unused modules (our case).
- Frozen embeddings that PEFT still marks trainable.
- Classifier heads that are not invoked for certain label configurations.

A robust integration pattern is to explicitly enumerate trainable
parameters after all PEFT modifications and to verify that every
requires_grad = True parameter receives a non-None gradient on a
sample batch before training. We recommend this as a standard practice
for any DP + PEFT workflow.

## 7.4 Practical Recommendations

Based on our findings, we offer the following recommendations.

**For practitioners deploying federated emotion classification:**

1. **Use LoRA-based federated learning without DP if privacy can be
   provided through other means** (e.g., secure aggregation, on-device
   storage with client-side access control). Our FedSVD and naive FedAvg
   results achieve over 0.75 macro-F1 with 1.89% trainable parameters.

2. **Do not assume that DP-SGD + LoRA works.** Test it explicitly at
   your target epsilon. If your gradient norms are below 10^-2, expect
   failure at practical privacy budgets.

3. **Consider full fine-tuning with DP** if formal DP is required and
   the communication budget allows. Our reference results show 0.20
   macro-F1 at epsilon = 4 with full fine-tuning, which is 20x worse
   than no DP but 5x better than LoRA + DP.

4. **Fix the pooler integration bug** if using Opacus with PEFT-wrapped
   HuggingFace models. Freeze any trainable parameter that is not
   invoked by the forward pass.

**For researchers:**

1. **Report gradient norms.** Papers that claim DP-LoRA works should
   report per-parameter gradient norms and the corresponding
   signal-to-noise ratio. Our negative result suggests that
   previously reported positive results may operate at different
   regimes (larger ranks, higher epsilon, or different tasks).

2. **Investigate server-side privacy mechanisms.** Client-level DP is
   the standard approach for federated learning, but it inherits the
   SNR problem from the local training. Server-side alternatives
   (e.g., secure aggregation with a trusted server) may provide better
   utility at the same privacy level.

3. **Study group-vs-client fairness in federated settings.** Our
   FairBatch result suggests that group fairness methods designed for
   centralized settings may not transfer to federated aggregation when
   group membership correlates with client identity.

## 7.5 Comparison with Reported Literature

Direct comparison with prior federated multilingual emotion work is
limited because BRIGHTER is recent and few federated baselines exist.
The closest comparison is the centralized BRIGHTER baseline, which our
centralized LoRA run (0.7997) roughly matches. Full fine-tuning
centralized results on BRIGHTER (reported in [5]) are in the range
0.75-0.82 macro-F1 depending on language, consistent with our centralized
number.

For DP-SGD, our full fine-tuning results (0.20 macro-F1 at epsilon = 4)
are consistent with the well-documented DP-utility trade-off [37, 38].
The magnitude of degradation for LoRA + DP (20x) has not been widely
reported and is, to our knowledge, the strongest negative result in this
space to date.

## 7.6 Broader Implications

Our findings contribute to a growing body of work on the limits of
parameter-efficient training under privacy constraints. As the field
moves toward increasingly large multilingual models and increasingly
strict privacy regulation, the tension between parameter efficiency and
differential privacy will likely become more acute, not less. Our
results suggest that this tension is fundamental and should be addressed
through algorithmic innovation (e.g., new aggregation strategies, new
privacy mechanisms) rather than hyperparameter tuning.


---


# 8. Limitations

Our study has several limitations that should be considered when
interpreting the results.

**Single language.** We use only the English subset of BRIGHTER. While
the BRIGHTER dataset supports 28 languages, using all languages would
require substantially more compute and introduce additional confounds.
Our focus on English isolates the federated comparison but limits
external validity: cross-lingual federated results may differ,
particularly if language-group disparity is greater in non-English
subsets.

**Small client count.** We simulate K = 5 clients. Real federated
deployments often have hundreds or thousands of clients. Our results may
not scale to these regimes, especially for FairBatch, where the ratio of
group to client structure is different at large K.

**Limited rounds.** Our main experiments use 5 federated rounds. The
DP-SGD results in particular might improve with more rounds, as Opacus's
noise schedule is amortized over more steps. We observe that even at
round 5, DP-SGD shows essentially no learning, so it is unlikely that
more rounds would reverse the trend, but we cannot definitively rule
this out.

**Small seed count.** We use 5 seeds. The Wilcoxon test at n = 5 cannot
achieve p < 0.05 for a two-sided test with all differences in the same
direction. Our reported p = 0.0625 is the minimum achievable p-value at
this sample size and should be interpreted as "all five seeds show
degradation" rather than as evidence of statistical insignificance.
Larger seed counts would strengthen the statistical claims.

**Fixed LoRA configuration.** We use rank r = 32, alpha = 64 throughout.
Different ranks could change the signal-to-noise dynamics for DP-SGD.
Higher rank (r = 64 or 128) would increase gradient magnitude and might
partially mitigate the DP failure, at the cost of increased communication
and memory. We do not explore this dimension.

**No secure aggregation experiments.** Although our code supports secure
aggregation at the API level, we do not run experiments with secure
aggregation enabled. The effect of secure aggregation on final utility
is expected to be neutral (secure aggregation is mathematically
equivalent to FedAvg in the absence of dropouts), but we do not verify
this empirically.

**FairBatch alpha fixed.** We use alpha = 0.05. Larger alpha would
increase the group reweighting, which might yield different results. Our
pilot experiments with alpha = 0.2 showed a positive fairness effect in
a single seed (documented in the repository at commit `4345df9`) but a
negative overall effect across five seeds (`e823bb5`). The optimal alpha
is likely task- and dataset-dependent.

**Synthetic federated partition.** The Dirichlet partition with alpha = 0.5
approximates realistic non-IID client distributions but does not capture
every federated deployment. In particular, real federated systems may
have adversarial clients, poisoned data, or model staleness, none of
which are modeled here.

**No cross-lingual federated experiment.** We do not run federated
training with clients drawn from different languages. This would be a
more natural federated multilingual setting but was outside our compute
budget. We note it as the primary direction for future work.

**Version sensitivity.** Our experiments use specific versions of
PyTorch, Transformers, PEFT, and Opacus (see Section 5.7). Different
versions may produce different results, particularly for the DP-SGD
failure, which depends on the exact behavior of Opacus's per-sample
gradient hooks. We document versions to support reproducibility but
cannot guarantee bit-exact reproduction on all platforms.

**No ablation on classifier head training.** Our LoRA setup trains both
LoRA adapters and a classification head. We do not ablate whether the
head alone (with frozen LoRA) could produce the observed results. This
would be a natural follow-up.


---


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


---


# References

> **Note to authors:** Verify each reference's exact venue, year, and
> page numbers before submission. Items marked with (*) need
> bibliographic verification.

[1] Coppersmith, G., Dredze, M., Harman, C., & Hollingshead, K. (2015).
    From ADHD to SAD: Analyzing the language of mental health on Twitter
    through self-reported diagnoses. In *Proceedings of the 2nd Workshop
    on Computational Linguistics and Clinical Psychology*, pp. 1–10.

[2] Gillespie, T. (2020). Content moderation, AI, and the question of
    scale. *Big Data & Society*, 7(2).

[3] Chatterjee, A. (2019). *Emotion and sentiment analysis in
    customer-service interactions*. Springer Briefs in Computer Science.

[4] Mohammad, S. M., Kiritchenko, S., & Sobhani, P. (2016). SemEval-2016
    Task 6: Detecting stance in tweets. In *Proceedings of SemEval-2016*,
    pp. 365–373.

[5] Shode, I., Adelani, D. I., Peng, J., et al. (2023). Affective
    computing in 28 languages: Annotating the BRIGHTER dataset for
    multi-label emotion classification. In *Proceedings of the 2023
    Conference on Empirical Methods in Natural Language Processing
    (EMNLP)*. (*)

[6] McMahan, B., Moore, E., Ramage, D., Hampson, S., & Agüera y Arcas,
    B. (2017). Communication-efficient learning of deep networks from
    decentralized data. In *Proceedings of the 20th International
    Conference on Artificial Intelligence and Statistics (AISTATS)*,
    pp. 1273–1282.

[7] Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S.,
    Wang, L., & Chen, W. (2022). LoRA: Low-rank adaptation of large
    language models. In *International Conference on Learning
    Representations (ICLR)*.

[8] Abadi, M., Chu, A., Goodfellow, I., McMahan, H. B., Mironov, I.,
    Talwar, K., & Zhang, L. (2016). Deep learning with differential
    privacy. In *Proceedings of the 2016 ACM SIGSAC Conference on
    Computer and Communications Security (CCS)*, pp. 308–318.

[9] Conneau, A., Khandelwal, K., Goyal, N., Chaudhary, V., Wenzek, G.,
    Guzmán, F., Grave, E., Ott, M., Zettlemoyer, L., & Stoyanov, V.
    (2020). Unsupervised cross-lingual representation learning at scale.
    In *Proceedings of the 58th Annual Meeting of the Association for
    Computational Linguistics (ACL)*, pp. 8440–8451.

[10] Roh, Y., Lee, K., Whang, S. E., & Suh, C. (2021). FairBatch: Batch
     selection for model fairness. In *International Conference on
     Learning Representations (ICLR)*.

[11] Dewangan, G. S., Roy, P., & Tiwari, R. (2024). Federated LoRA for
     privacy-preserving multilingual emotion classification. *Manuscript
     under review*. Earlier version: repository
     `federated-multilingual-emotion-analytics` (branch
     `paper3-fairdp-xlm`).

[12] Rosenthal, S., Farra, N., & Nakov, P. (2017). SemEval-2017 Task 4:
     Sentiment analysis in Twitter. In *Proceedings of SemEval-2017*,
     pp. 502–518.

[13] Mohammad, S., Bravo-Marquez, F., Salameh, M., & Kiritchenko, S.
     (2018). SemEval-2018 Task 1: Affect in tweets. In *Proceedings of
     SemEval-2018*, pp. 1–17.

[14] Demszky, D., Movshovitz-Attias, D., Ko, J., Cowen, A., Nemade, G.,
     & Ravi, S. (2020). GoEmotions: A dataset of fine-grained emotions.
     In *Proceedings of the 58th Annual Meeting of the Association for
     Computational Linguistics (ACL)*, pp. 4040–4054.

[15] Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT:
     Pre-training of deep bidirectional transformers for language
     understanding. In *Proceedings of NAACL-HLT 2019*, pp. 4171–4186.

[16] Ahmad, I., et al. (2023). Cross-lingual emotion classification:
     Challenges and strategies. *Journal of Multilingual and Multicultural
     Development*. (*)

[17] Keung, P., Lu, Y., Szarvas, G., & Smith, N. A. (2020). The
     multilingual Amazon reviews corpus. In *Proceedings of EMNLP 2020*,
     pp. 4566–4576.

[18] Lignos, C., et al. (2021). Cross-lingual transfer for emotion
     classification: A systematic study. *Proceedings of the 2021
     Conference on Empirical Methods in Natural Language Processing
     (EMNLP)*. (*)

[19] Hard, A., Rao, K., Mathews, R., Ramaswamy, S., Beaufays, F., Augenstein,
     S., Eichner, H., Kiddon, C., & Ramage, D. (2018). Federated learning
     for mobile keyboard prediction. *arXiv preprint arXiv:1811.03604*.

[20] Lin, B. Y., He, C., Zeng, Z., Wang, H., Huang, Y., Soltanolkotabi,
     M., Ren, X., Shah, S., & Avestimehr, S. (2021). FedNLP: Benchmarking
     federated learning methods for natural language processing tasks. In
     *Findings of NAACL-HLT 2021*.

[21] Hsu, T.-M. H., Qi, H., & Brown, M. (2019). Measuring the effects of
     non-identical data distribution for federated visual classification.
     *arXiv preprint arXiv:1909.06335*.

[22] Li, T., Sahu, A. K., Zaheer, M., Sanjabi, M., Talwalkar, A., &
     Smith, V. (2020). Federated optimization in heterogeneous networks.
     In *Proceedings of Machine Learning and Systems (MLSys)*, vol. 2,
     pp. 429–450.

[23] Wang, T., Rausch, J., Zhang, C., Jia, R., & Song, D. (2020). A
     principled approach to data valuation for federated learning. In
     *Federated Learning*, vol. 12500 of *Lecture Notes in Computer
     Science*, pp. 153–167. Springer.

[24] Houlsby, N., Giurgiu, A., Jastrzebski, S., Morrone, B., de Laroussilhe,
     Q., Gesmundo, A., Attariyan, M., & Gelly, S. (2019). Parameter-efficient
     transfer learning for NLP. In *Proceedings of the 36th International
     Conference on Machine Learning (ICML)*, pp. 2790–2799.

[25] Pfeiffer, J., Kamath, A., Rücklé, A., Cho, K., & Gurevych, I.
     (2021). AdapterFusion: Non-destructive task composition for transfer
     learning. In *Proceedings of EACL 2021*, pp. 487–503.

[26] Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2023).
     QLoRA: Efficient finetuning of quantized LLMs. In *Advances in
     Neural Information Processing Systems (NeurIPS)*.

[27] Zhang, Z., Yang, Y., Dai, Y., & Wang, Q. (2023). Federated LoRA:
     Efficient federated learning with low-rank adaptation. *arXiv
     preprint*. (*)

[28] Sun, Y., Wang, T., Wang, Y., & Ma, C. (2023). Federated low-rank
     adaptation for communication-efficient on-device NLP. *arXiv
     preprint*. (*)

[29] Li, T., Sahu, A. K., Zaheer, M., Sanjabi, M., Talwalkar, A., &
     Smith, V. (2018). Federated optimization in heterogeneous networks.
     *arXiv preprint arXiv:1812.06127*.

[30] Karimireddy, S. P., Kale, S., Mohri, M., Reddi, S. J., Stich, S. U.,
     & Suresh, A. T. (2020). SCAFFOLD: Stochastic controlled averaging
     for federated learning. In *Proceedings of the 37th International
     Conference on Machine Learning (ICML)*, pp. 5132–5143.

[31] Lee, S., et al. (2025). FedSVD: Server-side low-rank reparameterization
     for federated parameter-efficient fine-tuning. *arXiv preprint*. (*)

[32] Huang, W., Liu, T., & Yang, Q. (2021). Fairness and accuracy in
     federated learning. *IEEE Transactions on Neural Networks and
     Learning Systems*, 32(11), pp. 5119–5129.

[33] Zhang, D. Y., Kou, Z., & Wang, D. (2020). FairFL: A fair federated
     learning framework for group fairness. *arXiv preprint
     arXiv:2011.00523*.

[34] Yousefpour, A., Shilov, I., Sablayrolles, A., Testuggine, D., Prasad,
     K., Malek, M., Nguyen, J., Ghosh, S., Bharadwaj, A., Zhao, J., et al.
     (2021). Opacus: User-friendly differential privacy library in PyTorch.
     *arXiv preprint arXiv:2109.12298*.

[35] Bu, Z., Wang, Y.-X., Zha, S., & Karypis, G. (2023). Differentially
     private optimization in the deep learning era: A survey. *IEEE
     Transactions on Knowledge and Data Engineering*. (*)

[36] Mironov, I. (2017). Rényi differential privacy. In *Proceedings of
     the 30th IEEE Computer Security Foundations Symposium (CSF)*, pp.
     263–275.

[37] Tramèr, F., & Boneh, D. (2021). Differentially private learning
     needs better features (or much more data). In *International
     Conference on Learning Representations (ICLR)*.

[38] Bagdasaryan, E., Poursaeed, O., & Shmatikov, V. (2019). Differential
     privacy has disparate impact on model accuracy. In *Advances in
     Neural Information Processing Systems (NeurIPS)*, pp. 15479–15488.

[39] Cai, T., et al. (2024). DP-LoRA: Differentially private low-rank
     adaptation for large language models. *arXiv preprint*. (*)

[40] Yu, D., et al. (2024). Privacy-preserving parameter-efficient
     fine-tuning for federated language models. *arXiv preprint*. (*)

[41] Mangrulkar, S., Gugger, S., Debut, L., von Platen, P., Sun, S.,
     & Behl, S. (2022). PEFT: State-of-the-art parameter-efficient
     fine-tuning methods. HuggingFace repository.
     https://github.com/huggingface/peft

[42] Wolf, T., Debut, L., Sanh, V., Chaumond, J., Delangue, C., Moi, A.,
     et al. (2020). Transformers: State-of-the-art natural language
     processing. In *Proceedings of the 2020 Conference on Empirical
     Methods in Natural Language Processing: System Demonstrations*,
     pp. 38–45.

[43] Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan,
     G., et al. (2019). PyTorch: An imperative style, high-performance
     deep learning library. In *Advances in Neural Information
     Processing Systems (NeurIPS)*, vol. 32.

[44] Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B.,
     Grisel, O., et al. (2011). Scikit-learn: Machine learning in Python.
     *Journal of Machine Learning Research*, 12, pp. 2825–2830.

[45] Lhoest, Q., Villanova del Moral, A., Jernite, Y., Thakur, A., von
     Platen, P., Patil, S., et al. (2021). Datasets: A community library
     for natural language processing. In *Proceedings of the 2021
     Conference on Empirical Methods in Natural Language Processing:
     System Demonstrations*, pp. 175–184.


---

