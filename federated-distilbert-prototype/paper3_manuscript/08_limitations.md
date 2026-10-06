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
