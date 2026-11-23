"""Анализ для пересмотренной версии статьи (ответ рецензенту).

python analyze2.py full.csv abl.csv curve.csv > report2.txt
"""
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

warnings.filterwarnings("ignore")
F = ["func", "m", "rho", "marg", "wconc", "noise", "ntrain"]
full = pd.read_csv(sys.argv[1])
abl = pd.read_csv(sys.argv[2])
curve = pd.read_csv(sys.argv[3])

# контроль: общие методы совпадают
chk = full.merge(abl, on=F + ["rep", "method"], suffixes=("_o", "_n"))
print("== check identical (max |diff|):", chk.groupby("method").apply(lambda g: (g.kendall_o - g.kendall_n).abs().max()).to_dict())
abl = abl[~abl.method.isin(["WSM-entropy", "FIS-entropy", "Ridge"])]
d = pd.concat([full, abl], ignore_index=True)
d["kendall"] = d["kendall"].astype(float)

rng = np.random.default_rng(1)
B = 1000


def boot_mean(df, col="kendall", B=B):
    """CI по повторам: внутри каждого сценария повторы ресэмплируются, сценарии фиксированы."""
    piv = df.pivot_table(index=F, columns="rep", values=col).to_numpy()  # (S, R)
    S, R = piv.shape
    idx = rng.integers(0, R, size=(B, R))
    bs = np.array([piv[:, i].mean() for i in idx])
    return piv.mean(), np.percentile(bs, [2.5, 97.5])


def boot_diff(a, b, col="kendall", B=B):
    pa = a.pivot_table(index=F, columns="rep", values=col)
    pb = b.pivot_table(index=F, columns="rep", values=col).loc[pa.index, pa.columns]
    D = (pa - pb).to_numpy()
    S, R = D.shape
    idx = rng.integers(0, R, size=(B, R))
    bs = np.array([D[:, i].mean() for i in idx])
    sc = D.mean(1)
    return D.mean(), np.percentile(bs, [2.5, 97.5]), (sc > 0).mean()


M = lambda name: d[d.method == name]

print("\n== Table 1: mean tau [95% CI], spearman, ndcg")
methods = d.method.unique()
rows = []
for mt in methods:
    sub = M(mt)
    mu, ci = boot_mean(sub)
    rows.append((mt, mu, ci[0], ci[1], sub.spearman.mean(), sub.ndcg10.mean()))
T1 = pd.DataFrame(rows, columns=["method", "tau", "lo", "hi", "rho", "ndcg"]).sort_values("tau", ascending=False)
print(T1.round(3).to_string(index=False))
print("max CI half-width:", ((T1.hi - T1.lo) / 2).max().round(4))

print("\n== Ablation: fuzzy variant minus crisp counterpart (same weights and logic)")
PAIRS = [("FIS-entropy", "WSM-entropy", "Mamdani L=5 FCM"), ("FIS-L9-c", "WSM-entropy", "Mamdani L=9"),
         ("FIS-L17-c", "WSM-entropy", "Mamdani L=17"), ("FIS-unif-c", "WSM-entropy", "Mamdani uniform centres"),
         ("TSK-index-c", "WSM-entropy", "TSK index"), ("TSK-centre-c", "WSM-entropy", "TSK centre FCM"),
         ("TSK-centre-unif-c", "WSM-entropy", "TSK centre uniform"),
         ("FIS-andness0.5", "OWA-andness0.5", "Mamdani L=5 FCM"), ("FIS-L9-n", "OWA-andness0.5", "Mamdani L=9"),
         ("FIS-L17-n", "OWA-andness0.5", "Mamdani L=17"), ("FIS-unif-n", "OWA-andness0.5", "Mamdani uniform centres"),
         ("TSK-index-n", "OWA-andness0.5", "TSK index"), ("TSK-centre-n", "OWA-andness0.5", "TSK centre FCM"),
         ("TSK-centre-unif-n", "OWA-andness0.5", "TSK centre uniform"),
         ("FIS-entropy-K5", "WSM-entropy", "Mamdani K=5"), ("FIS-andness0.5-K5", "OWA-andness0.5", "Mamdani K=5")]
for a, b, lab in PAIRS:
    mu, ci, sh = boot_diff(M(a), M(b))
    vmu, vci, vsh = boot_diff(M(a)[M(a).func == "veto"], M(b)[M(b).func == "veto"])
    print(f"{lab:26s} {a:20s} vs {b:16s} d={mu:+.4f} [{ci[0]:+.4f},{ci[1]:+.4f}] better={sh:.3f} | veto d={vmu:+.4f} better={vsh:.3f}")

print("\n== Ablation by marginal distribution (d tau vs crisp)")
for a, b, lab in PAIRS[:7] + PAIRS[7:14]:
    out = []
    for mg in ("uniform", "skewed", "sparse"):
        A, Bm = M(a), M(b)
        out.append(f"{mg}={(A[A.marg == mg].kendall.mean() - Bm[Bm.marg == mg].kendall.mean()):+.3f}")
    print(f"{lab:26s} {a:20s} " + " ".join(out))

print("\n== Scenario-level best method (scenario means, oracle excluded)")
s = d[d.method != "WSM-oracle"].groupby(F + ["method"]).kendall.mean().unstack()
mainset = ["WSM-entropy", "WSM-equal", "TOPSIS-entropy", "FIS-entropy", "FIS-equal", "FIS-andness0.5", "OWA-andness0.5",
           "FIS-entropy-K5", "FIS-andness0.5-K5", "FIS-WangMendel", "Ridge", "RandomForest", "GradBoost",
           "TSK-learned", "Choquet-learned", "MonoGBM"]
ss = s[mainset]
best = ss.idxmax(axis=1)
print((best.value_counts() / len(ss) * 100).round(1).to_string())
print("ties:", (ss.eq(ss.max(axis=1), axis=0).sum(axis=1) > 1).sum())
fis = [c for c in mainset if c.startswith("FIS")]
print("fuzzy best:", best.isin(fis).mean(), "within .01:", (ss[fis].max(axis=1) >= ss.max(axis=1) - 0.01).mean())
sup = ["Ridge", "RandomForest", "GradBoost", "FIS-WangMendel", "TSK-learned", "Choquet-learned", "MonoGBM"]
uns = [c for c in mainset if c not in sup]
print("best sup > best unsup:", (ss[sup].max(axis=1) > ss[uns].max(axis=1)).mean())

print("\n== By form and ntrain (scenario means)")
show = ["WSM-entropy", "WSM-expert0.3", "WSM-expert0.6", "OWA-andness0.5", "MIX-expert0.3", "Ridge", "Choquet-learned",
        "MonoGBM", "GradBoost", "RandomForest", "TSK-learned", "FIS-WangMendel"]
print(d[d.method.isin(show)].pivot_table(index="method", columns=["func", "ntrain"], values="kendall").loc[show].round(3).to_string())
v = s.reset_index()
for n in (50, 200):
    vv = v[(v.func == "veto") & (v.ntrain == n)]
    print("veto", n, "OWA beats best supervised (incl Choquet/MonoGBM):", (vv["OWA-andness0.5"] > vv[sup].max(axis=1)).mean().round(3),
          "best sup mean", vv[sup].mean().round(3).to_dict())
print("\n== overall by ntrain")
print(d[d.method.isin(show)].pivot_table(index="method", columns="ntrain", values="kendall").loc[show].round(3).to_string())
print("\n== compensatory forms, ridge by ntrain and form")
print(d[(d.method == "Ridge")].pivot_table(index="func", columns="ntrain", values="kendall").round(3))

print("\n== Expert weights vs entropy, by wconc")
print(d[d.method.isin(["WSM-entropy", "WSM-equal", "WSM-expert0.3", "WSM-expert0.6", "OWA-andness0.5", "MIX-expert0.3", "MIX-expert0.6", "Ridge"])]
      .pivot_table(index="method", columns="wconc", values="kendall").round(3))
for e in ("WSM-expert0.3", "WSM-expert0.6"):
    for n in (50, 200):
        x = s.reset_index()
        x = x[(x.ntrain == n) & (x.func != "veto")]
        print(e, n, "ridge beats expert WSM (compensatory):", (x["Ridge"] > x[e]).mean().round(3))

print("\n== ANOVA (instance level, main effects + two-way interactions): partial eta^2")
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
for mt in ["WSM-entropy", "FIS-entropy", "OWA-andness0.5", "Ridge", "RandomForest"]:
    sub = M(mt).copy()
    for c in F:
        sub[c] = sub[c].astype(str)
    fac = " + ".join(f"C({c})" for c in F)
    model = smf.ols(f"kendall ~ ({fac})**2", data=sub).fit()
    a = anova_lm(model, typ=2)
    ssr = a.loc["Residual", "sum_sq"]
    eta = (a.sum_sq / (a.sum_sq + ssr)).drop("Residual").sort_values(ascending=False)
    print(mt, "R2=%.3f" % model.rsquared, "; ".join(f"{k.replace('C(', '').replace(')', '')}:{v:.3f}" for k, v in eta.head(6).items()))

print("\n== Learning curve: power vs exponential (ridge, mean by n)")
pw = lambda n, a, b, c: a - b * n ** (-c)
ex = lambda n, a, b, c: a - b * np.exp(-c * n)
for func, g in curve[curve.method == "Ridge"].groupby("func"):
    o = g.groupby("n").kendall.mean()
    x, y = o.index.values.astype(float), o.values
    res = {}
    for nm, fn, p0 in (("power", pw, [0.95, 1, 0.6]), ("exp", ex, [0.95, 0.2, 0.02])):
        try:
            p, _ = curve_fit(fn, x, y, p0=p0, maxfev=50000)
            rss = ((fn(x, *p) - y) ** 2).sum()
            r2 = 1 - rss / ((y - y.mean()) ** 2).sum()
            aic = len(x) * np.log(rss / len(x)) + 2 * 3
            res[nm] = (round(r2, 5), round(aic, 1))
        except Exception as e:
            res[nm] = str(e)[:40]
    print(func, res)
