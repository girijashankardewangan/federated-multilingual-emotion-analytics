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
