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
