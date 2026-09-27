"""Paper 3: Generate figures from full_5seeds CSVs.

Produces 4 publication-ready figures:
  1. fig1_round_trajectory.png   - FedSVD vs FairBatch over 10 rounds
  2. fig2_macro_f1_bars.png      - Final macro-F1 comparison
  3. fig3_per_language_bars.png  - Per-language macro-F1
  4. fig4_gap_trajectory.png     - Cross-lingual gap over rounds
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.size"] = 10

OUT = "results/full_5seeds/figures"
os.makedirs(OUT, exist_ok=True)
SEEDS = [42, 43, 44, 45, 46]
LANGS = ["eng", "hin", "deu", "esp", "chn"]


def load_round(name):
    return pd.read_csv(f"results/full_5seeds/{name}_metrics.csv")


# ---------- Load data ----------
fedsvd_curves, fb_curves = [], []
fedsvd_gap_curves, fb_gap_curves = [], []
fedsvd_final, fb_final = [], []
fedsvd_lang, fb_lang = [], []

for s in SEEDS:
    try:
        df = load_round(f"pilot_fedsvd_seed{s}")
        fedsvd_curves.append(df["test_macro_f1"].values)
        fedsvd_gap_curves.append(df["cross_lingual_gap"].values)
        fedsvd_final.append(df["test_macro_f1"].iloc[-1])
        fedsvd_lang.append([df[f"{l}_macro_f1"].iloc[-1] for l in LANGS])
    except FileNotFoundError:
        pass
    try:
        df = load_round(f"pilot_fairbatch_a02_seed{s}")
        fb_curves.append(df["test_macro_f1"].values)
        fb_gap_curves.append(df["cross_lingual_gap"].values)
        fb_final.append(df["test_macro_f1"].iloc[-1])
        fb_lang.append([df[f"{l}_macro_f1"].iloc[-1] for l in LANGS])
    except FileNotFoundError:
        pass

print(f"Loaded FedSVD: {len(fedsvd_curves)} seeds")
print(f"Loaded FairBatch: {len(fb_curves)} seeds")

fedsvd_curves = np.array(fedsvd_curves)
fb_curves = np.array(fb_curves)
fedsvd_gap = np.array(fedsvd_gap_curves)
fb_gap = np.array(fb_gap_curves)
fedsvd_lang = np.array(fedsvd_lang)
fb_lang = np.array(fb_lang)


# ---------- FIGURE 1: Round trajectory ----------
fig, ax = plt.subplots(figsize=(7, 4.2))
rounds = np.arange(1, fedsvd_curves.shape[1] + 1)

m1 = fedsvd_curves.mean(axis=0)
s1 = fedsvd_curves.std(axis=0, ddof=1)
m2 = fb_curves.mean(axis=0)
s2 = fb_curves.std(axis=0, ddof=1)

ax.plot(rounds, m1, "-o", color="#1f77b4", label="FedSVD (10r)", lw=2, ms=6)
ax.fill_between(rounds, m1 - s1, m1 + s1, alpha=0.18, color="#1f77b4")
ax.plot(rounds, m2, "-s", color="#d62728", label="FedSVD + FairBatch α=0.2 (10r)", lw=2, ms=6)
ax.fill_between(rounds, m2 - s2, m2 + s2, alpha=0.18, color="#d62728")

ax.axhline(0.7997, ls="--", color="gray", lw=1, label="Centralized (5r) = 0.7997")
ax.axhline(0.7749, ls=":", color="black", lw=1, label="Naive FedAvg (5r) = 0.7749")

ax.set_xlabel("Federated round")
ax.set_ylabel("Test macro-F1")
ax.set_title("Test macro-F1 over 10 federated rounds (mean ± SD, n=5)")
ax.legend(fontsize=8, loc="lower right")
ax.grid(alpha=0.3)
ax.set_ylim(0, 1)
fig.tight_layout()
fig.savefig(f"{OUT}/fig1_round_trajectory.png", bbox_inches="tight")
plt.close(fig)
print("Saved fig1_round_trajectory.png")


# ---------- FIGURE 2: Macro-F1 bars ----------
fig, ax = plt.subplots(figsize=(6, 4))
labels = ["Centralized\n(5r)", "Naive FedAvg\n(5r)", "FedSVD\n(10r)", "FairBatch α=0.2\n(10r)"]
means = [0.7997, 0.7749, np.mean(fedsvd_final), np.mean(fb_final)]
stds = [0.0058, 0.0130, np.std(fedsvd_final, ddof=1), np.std(fb_final, ddof=1)]
colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]

bars = ax.bar(labels, means, yerr=stds, capsize=5, color=colors, edgecolor="black")
for i, (m, s) in enumerate(zip(means, stds)):
    ax.text(i, m + s + 0.005, f"{m:.4f}", ha="center", fontsize=9, fontweight="bold")

ax.set_ylabel("Test macro-F1")
ax.set_title("Final-round macro-F1 comparison")
ax.set_ylim(0, 1)
ax.grid(alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(f"{OUT}/fig2_macro_f1_bars.png", bbox_inches="tight")
plt.close(fig)
print("Saved fig2_macro_f1_bars.png")


# ---------- FIGURE 3: Per-language bars ----------
fig, ax = plt.subplots(figsize=(9, 4))
x = np.arange(len(LANGS))
width = 0.28

fedsvd_lang_mean = fedsvd_lang.mean(axis=0)
fedsvd_lang_std = fedsvd_lang.std(axis=0, ddof=1)
fb_lang_mean = fb_lang.mean(axis=0)
fb_lang_std = fb_lang.std(axis=0, ddof=1)

ax.bar(x - width, fedsvd_lang_mean, width, yerr=fedsvd_lang_std, capsize=3,
       label="FedSVD (10r)", color="#ff7f0e")
ax.bar(x, fb_lang_mean, width, yerr=fb_lang_std, capsize=3,
       label="FairBatch α=0.2 (10r)", color="#d62728")
ax.bar(x + width, [0.7548, 0.8657, 0.6879, 0.8408, 0.7393], width,
       label="Centralized (5r)", color="#1f77b4", alpha=0.7)

ax.set_xticks(x)
ax.set_xticklabels(["English", "Hindi", "German", "Spanish", "Chinese"])
ax.set_ylabel("Test macro-F1")
ax.set_title("Language-wise macro-F1 at final round (mean ± SD, n=5)")
ax.legend(fontsize=9)
ax.grid(alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(f"{OUT}/fig3_per_language_bars.png", bbox_inches="tight")
plt.close(fig)
print("Saved fig3_per_language_bars.png")


# ---------- FIGURE 4: Gap trajectory ----------
fig, ax = plt.subplots(figsize=(7, 4.2))

g1 = fedsvd_gap.mean(axis=0)
g1s = fedsvd_gap.std(axis=0, ddof=1)
g2 = fb_gap.mean(axis=0)
g2s = fb_gap.std(axis=0, ddof=1)

ax.plot(rounds, g1, "-o", color="#ff7f0e", label="FedSVD (10r)", lw=2, ms=6)
ax.fill_between(rounds, g1 - g1s, g1 + g1s, alpha=0.18, color="#ff7f0e")
ax.plot(rounds, g2, "-s", color="#d62728", label="FairBatch α=0.2 (10r)", lw=2, ms=6)
ax.fill_between(rounds, g2 - g2s, g2 + g2s, alpha=0.18, color="#d62728")

ax.set_xlabel("Federated round")
ax.set_ylabel("Cross-lingual gap (max − min macro-F1)")
ax.set_title("Cross-lingual gap over 10 federated rounds (mean ± SD, n=5)")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{OUT}/fig4_gap_trajectory.png", bbox_inches="tight")
plt.close(fig)
print("Saved fig4_gap_trajectory.png")

print("\nAll 4 figures saved to:", OUT)
