"""Paper 3: 5-Seed Analysis"""
import pandas as pd, numpy as np, glob, os
from scipy.stats import wilcoxon, ttest_rel

results = {}
for f in sorted(glob.glob("results/full_5seeds/*.csv")):
    name = os.path.basename(f).replace("pilot_", "").replace("_metrics.csv", "")
    results[name] = pd.read_csv(f).iloc[-1]

seeds = [42, 43, 44, 45, 46]
b, fa = [], []
for s in seeds:
    br = results[f"fedsvd_seed{s}"]
    fr = results[f"fairbatch_a02_seed{s}"]
    b.append({"seed": s, "macro_f1": br["test_macro_f1"], "gap": br["cross_lingual_gap"], "deu": br["deu_macro_f1"]})
    fa.append({"seed": s, "macro_f1": fr["test_macro_f1"], "gap": fr["cross_lingual_gap"], "deu": fr["deu_macro_f1"]})

bdf, fdf = pd.DataFrame(b), pd.DataFrame(fa)
for m in ["macro_f1", "gap", "deu"]:
    W, p = wilcoxon(bdf[m], fdf[m])
    t, pt = ttest_rel(bdf[m], fdf[m])
    d = (fdf[m] - bdf[m]).mean() / (fdf[m] - bdf[m]).std(ddof=1)
    print(f"{m:10s} | Wilcoxon W={W:.1f} p={p:.4f} | t-test t={t:.4f} p={pt:.4f} | d={d:+.4f}")
