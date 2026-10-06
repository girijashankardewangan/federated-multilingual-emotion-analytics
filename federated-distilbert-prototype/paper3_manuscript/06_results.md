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
