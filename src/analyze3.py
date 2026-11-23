"""Анализ для второй рецензии: все числа статьи и онлайн-приложения.

python analyze3.py all3.csv r3.csv curve.csv curve_wmin.csv shape.csv bench3.csv > report3.txt
"""
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

warnings.filterwarnings("ignore")
F = ["func", "m", "rho", "marg", "wconc", "noise", "ntrain"]
d = pd.read_csv(sys.argv[1]); r3 = pd.read_csv(sys.argv[2])
curve = pd.read_csv(sys.argv[3]); curve_w = pd.read_csv(sys.argv[4])
shape = pd.read_csv(sys.argv[5]); bench = pd.read_csv(sys.argv[6])
rng = np.random.default_rng(1)
B = 1000
M = lambda name: d[d.method == name]

UNS = ["WSM-entropy", "WSM-equal", "TOPSIS-entropy", "OWA-andness0.5", "FIS-entropy", "FIS-andness0.5",
       "FIS-entropy-K5", "FIS-andness0.5-K5"]
SUP = ["Ridge", "RandomForest", "GradBoost", "MonoGBM", "Choquet-learned", "WMin-learned", "TSK-learned", "FIS-WangMendel"]
EXP = ["WSM-expert0.3", "MIX-expert0.3", "WSM-expert0.6", "MIX-expert0.6"]


def boot_mean(df, col="kendall"):
    piv = df.pivot_table(index=F, columns="rep", values=col).to_numpy()
    idx = rng.integers(0, piv.shape[1], size=(B, piv.shape[1]))
    bs = np.array([piv[:, i].mean() for i in idx])
    return piv.mean(), np.percentile(bs, [2.5, 97.5])


print("== Table 1 (20 methods + reference): tau [95% CI], tau n=50, n=200, spearman, ndcg")
rows = []
for mt in ["WSM-oracle"] + UNS + SUP + EXP:
    sub = M(mt)
    mu, ci = boot_mean(sub)
    rows.append((mt, mu, (ci[1] - ci[0]) / 2, sub[sub.ntrain == 50].kendall.mean(), sub[sub.ntrain == 200].kendall.mean(),
                 sub.spearman.mean(), sub.ndcg10.mean(), sub[sub.func != "veto"].kendall.mean(), sub[sub.func == "veto"].kendall.mean()))
T1 = pd.DataFrame(rows, columns=["method", "tau", "ci_half", "tau50", "tau200", "rho", "ndcg", "comp", "veto"]).sort_values("tau", ascending=False)
print(T1.round(4).to_string(index=False))
print("max CI half-width:", T1.ci_half.max().round(4), "| NDCG range:", T1.ndcg.min().round(3), T1.ndcg.max().round(3))

print("\n== tau by form x ntrain (selected)")
show = ["WMin-learned", "Choquet-learned", "Ridge", "MonoGBM", "GradBoost", "RandomForest", "OWA-andness0.5", "WSM-entropy",
        "WSM-expert0.3", "WSM-expert0.6", "MIX-expert0.3", "MIX-expert0.6"]
print(d[d.method.isin(show)].pivot_table(index="method", columns=["func", "ntrain"], values="kendall").loc[show].round(3).to_string())
d["grp"] = np.where(d.func == "veto", "veto", "comp")
print(d[d.method.isin(show)].pivot_table(index="method", columns=["grp", "ntrain"], values="kendall").loc[show].round(3).to_string())

print("\n== Best method per scenario; candidates = 8 unsupervised + 8 learned (no ablations, expert, reference)")
s = d.groupby(F + ["method"]).kendall.mean().unstack()
ss = s[UNS + SUP]
best = ss.idxmax(axis=1)
print((best.value_counts() / len(ss) * 100).round(1).to_string())
print("ties:", int((ss.eq(ss.max(axis=1), axis=0).sum(axis=1) > 1).sum()), "| learned best:", best.isin(SUP).mean().round(3),
      "| best learned > best unsup:", (ss[SUP].max(axis=1) > ss[UNS].max(axis=1)).mean().round(3))
print("WMin-learned > Ridge:", (s["WMin-learned"] > s["Ridge"]).mean().round(3), "> Choquet:", (s["WMin-learned"] > s["Choquet-learned"]).mean().round(3))
v = s.reset_index()
for n in (50, 200):
    vv = v[(v.func == "veto") & (v.ntrain == n)]
    print(f"veto n={n}: WMin0.5 > best learned: {(vv['OWA-andness0.5'] > vv[SUP].max(axis=1)).mean():.3f};"
          f" WMin0.5 > best learned excl. WMin-learned: {(vv['OWA-andness0.5'] > vv[[x for x in SUP if x != 'WMin-learned']].max(axis=1)).mean():.3f}")
    cc = v[(v.func != "veto") & (v.ntrain == n)]
    for e in ("WSM-expert0.3", "WSM-expert0.6"):
        print(f"   comp n={n}: Ridge > {e}: {(cc['Ridge'] > cc[e]).mean():.3f}; WMin-learned > {e}: {(cc['WMin-learned'] > cc[e]).mean():.3f}")
    print(f"   veto n={n}: MIX-expert0.3 > best learned: {(vv['MIX-expert0.3'] > vv[SUP].max(axis=1)).mean():.3f}")
lam = r3[r3.method == "WMin-learned"]
print("learned lambda (median, IQR) by form:", lam.groupby("func").lam_learned.apply(lambda x: (round(x.median(), 2), round(x.quantile(.25), 2), round(x.quantile(.75), 2))).to_dict())

print("\n== Ties in the true utility (share of tied test pairs)")
t = r3[r3.method == "WSM-entropy"]
print("by marg:", t.groupby("marg").ties_true.mean().round(3).to_dict())
print(t[t.marg == "sparse"].pivot_table(index="m", columns="rho", values="ties_true").round(3).to_string())

print("\n== Veto x marginal distribution x ntrain")
vm = ["WSM-entropy", "OWA-andness0.5", "MIX-expert0.3", "WMin-learned", "Choquet-learned", "MonoGBM", "Ridge", "FIS-entropy", "FIS-andness0.5"]
print(d[(d.func == "veto") & d.method.isin(vm)].pivot_table(index="method", columns=["marg", "ntrain"], values="kendall").loc[vm].round(3).to_string())
g = d[(d.func == "veto")].pivot_table(index="method", columns="marg", values="kendall")
print("WMin0.5 - WSM by marg:", (g.loc["OWA-andness0.5"] - g.loc["WSM-entropy"]).round(3).to_dict())
for n in (50, 200):
    vv = v[(v.func == "veto") & (v.ntrain == n)]
    print(f"   n={n} WMin0.5 > best learned by marg:", vv.assign(w=vv["OWA-andness0.5"] > vv[SUP].max(axis=1)).groupby("marg").w.mean().round(3).to_dict())

print("\n== lambda sweep (crisp WMin, TSK equal centres, Mamdani K=3)")
SW = {"WMin": ["WSM-entropy", "WMin-0.25", "OWA-andness0.5", "WMin-0.75", "WMin-1.0"],
      "TSK": ["TSK-centre-unif-c", "TSKu-0.25", "TSK-centre-unif-n", "TSKu-0.75", "TSKu-1.0"],
      "Mamdani": ["FIS-entropy", "FIS-0.25", "FIS-andness0.5", "FIS-0.75", "FIS-1.0"]}
LAMS = [0, 0.25, 0.5, 0.75, 1.0]
tab = []
for fam, ms in SW.items():
    for l, mt in zip(LAMS, ms):
        sub = M(mt)
        tab.append(dict(family=fam, lam=l, comp=sub[sub.func != "veto"].kendall.mean(), veto=sub[sub.func == "veto"].kendall.mean(), all=sub.kendall.mean()))
SWT = pd.DataFrame(tab)
print(SWT.pivot(index="lam", columns="family", values=["comp", "veto"]).round(3).to_string())
SWT.to_csv("../results/lambda_sweep.csv", index=False)

print("\n== Monotonicity of Mamdani output (reps 0-4, 1920 problems): z_j + 0.05")
mo = t.dropna(subset=["mono_viol_0.0"])
for lam_ in ("0.0", "0.5"):
    print(f"lambda={lam_}: share of (alternative, criterion) increments with lower output: {mo['mono_viol_' + lam_].mean():.4f};"
          f" problems with any violation: {(mo['mono_viol_' + lam_] > 0).mean():.3f};"
          f" max drop (0-100 scale): median {mo['mono_maxdrop_' + lam_].median():.2f}, max {mo['mono_maxdrop_' + lam_].max():.2f}")

print("\n== Sensitivity: change in tau when one factor changes level (mean over other factors)")
FACT = [("rho", 0.0, 0.6), ("marg", "uniform", "sparse"), ("marg", "uniform", "skewed"), ("m", 3, 5), ("wconc", "balanced", "concentrated"),
        ("noise", 0.0, 0.2), ("ntrain", 50, 200)]
SM = ["WSM-entropy", "FIS-entropy", "OWA-andness0.5", "Ridge", "WMin-learned", "Choquet-learned", "RandomForest"]
sens = []
for f, a, b in FACT:
    g = d[d.method.isin(SM)].groupby(["method", f]).kendall.mean()
    sens.append({"factor": f"{f}: {a}->{b}", **{m: g[(m, b)] - g[(m, a)] for m in SM}})
SENS = pd.DataFrame(sens).set_index("factor")
print(SENS.round(3).to_string())
SENS.to_csv("../results/sensitivity.csv")

print("\n== ANOVA (unit = 11,520 problem instances; 7 factors + all two-way interactions): partial eta^2, top 3")
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
for mt in ["WSM-entropy", "FIS-entropy", "Ridge", "WMin-learned", "RandomForest"]:
    sub = M(mt).copy()
    for c in F:
        sub[c] = sub[c].astype(str)
    fac = " + ".join(f"C({c})" for c in F)
    a = anova_lm(smf.ols(f"kendall ~ ({fac})**2", data=sub).fit(), typ=2)
    ssr = a.loc["Residual", "sum_sq"]
    eta = (a.sum_sq / (a.sum_sq + ssr)).drop("Residual").sort_values(ascending=False)
    print(mt, "; ".join(f"{k.replace('C(', '').replace(')', '')}:{v:.2f}" for k, v in eta.head(3).items()))

print("\n== Learning curves: n* (mean curve) with bootstrap CIs; per-scenario break-even; risk percentiles")
f = lambda n, a, b, c: a - b * n ** (-c)
Fc = ["func", "m", "rho", "marg", "wconc", "noise"]


def nstar_of(y, base):
    try:
        a, b, c = curve_fit(f, y.index.values.astype(float), y.values, p0=[0.95, 1, 0.6], maxfev=20000)[0]
    except Exception:
        return np.nan
    return (b / (a - base)) ** (1 / c) if a > base else np.inf


rule = curve[curve.method.isin(["WSM", "OWA"])]
for lname, cdf in (("Ridge", curve[curve.method == "Ridge"]), ("WMin-learned", curve_w)):
    print(f"-- {lname}")
    for func in ["linear", "concave", "interaction", "veto"]:
        rn = "OWA" if func == "veto" else "WSM"
        R = cdf[cdf.func == func].pivot_table(index=Fc + ["rep"], columns="n", values="kendall")
        B0 = rule[(rule.func == func) & (rule.method == rn)].set_index(Fc + ["rep"]).kendall.loc[R.index]
        ns = nstar_of(R.mean(), B0.mean())
        S_ = len(R) // 30
        Rv = R.to_numpy().reshape(S_, 30, -1); Bv = B0.to_numpy().reshape(S_, 30)
        bc = []
        for _ in range(400):
            j = rng.integers(0, S_, S_)
            bc.append(nstar_of(pd.Series(Rv[j].mean((0, 1)), index=R.columns), Bv[j].mean()))
        ci = np.nanpercentile(np.where(np.isinf(bc), np.nan, bc), [2.5, 97.5]).round(1) if np.isfinite(ns) else "-"
        Rc = R.groupby(Fc).mean(); Bc = B0.groupby(Fc).mean()
        D = Rc.sub(Bc, axis=0)
        first = D.gt(0).idxmax(axis=1).where(D.gt(0).any(axis=1))
        print(f"{func:12s} n*={ns:.1f} scenario-bootstrap CI={ci}; first grid n with gain: median={first.median()}, "
              f"never={first.isna().mean():.2f}; share gain>0 at 50: {(D[50] > 0).mean():.3f}; p10 diff at 10/20/50: "
              f"{D[10].quantile(.1):.3f}/{D[20].quantile(.1):.3f}/{D[50].quantile(.1):.3f}; plateau(500)={Rc[500].mean():.3f}")
    comp = cdf[cdf.func != "veto"].groupby(Fc + ["n"]).kendall.mean().unstack()
    Bc = rule[(rule.func != "veto") & (rule.method == "WSM")].groupby(Fc).kendall.mean()
    D = comp.sub(Bc, axis=0)
    first = D.gt(0).idxmax(axis=1).where(D.gt(0).any(axis=1))
    nz = D.index.get_level_values("noise")
    print(f"compensatory pooled (N={len(D)}): first n median={first.median()}, IQR={first.quantile(.25)}-{first.quantile(.75)}, p90={first.quantile(.9)};"
          f" gain>0 at 50: {(D[50] > 0).mean():.3f} ({int((D[50] > 0).sum())}/{len(D)}); sigma=0: {(D[50] > 0)[nz == 0].mean():.3f}"
          f" ({int((D[50] > 0)[nz == 0].sum())}/{int((nz == 0).sum())}); sigma=0.2: {(D[50] > 0)[nz == 0.2].mean():.3f}"
          f" ({int((D[50] > 0)[nz == 0.2].sum())}/{int((nz == 0.2).sum())}); p10 diff at 10/20/50: {D[10].quantile(.1):.3f}/{D[20].quantile(.1):.3f}/{D[50].quantile(.1):.3f}")

print("\n== Shape sensitivity (veto: theta, beta; interaction: gamma); identity check with base")
base = shape[shape.variant == "base"].merge(d, on=F + ["rep", "method"], suffixes=("_s", "_d"))
print("identity max |diff| (base vs main run):", (base.kendall_s - base.kendall_d).abs().max())
SH = ["WSM-entropy", "OWA-andness0.5", "FIS-entropy", "FIS-andness0.5", "WMin-learned", "Choquet-learned", "MonoGBM", "Ridge", "RandomForest"]
for func in ("veto", "interaction"):
    x = shape[(shape.func == func) & shape.method.isin(SH)]
    print(x.pivot_table(index="method", columns=["variant", "ntrain"], values="kendall").loc[SH].round(3).to_string())
    sv = x.groupby(F + ["variant", "method"]).kendall.mean().unstack().reset_index()
    sup_ = [c for c in SH if c in SUP]
    for (var, n), g in sv.groupby(["variant", "ntrain"]):
        print(f"   {func} {var} n={n}: WMin0.5-WSM={g['OWA-andness0.5'].mean() - g['WSM-entropy'].mean():+.3f};"
              f" Mamdani(0.5)-WMin0.5={g['FIS-andness0.5'].mean() - g['OWA-andness0.5'].mean():+.3f};"
              f" WMin0.5 > best learned: {(g['OWA-andness0.5'] > g[sup_].max(axis=1)).mean():.3f}; best mean: {g[SH].mean().idxmax()}")

print("\n== Real data (20 splits): tau_b, Somers' D on pairs with different grades")
bm = ["WMin-learned", "Ridge", "Choquet-learned", "MonoGBM", "RandomForest", "GradBoost", "WSM-equal", "WSM-entropy", "TOPSIS-entropy",
      "OWA-andness0.5", "FIS-entropy", "FIS-andness0.5", "FIS-WangMendel", "TSK-learned", "FIS-entropy-K5", "FIS-andness0.5-K5"]
for n in (200, 50):
    b = bench[(bench.ntrain == n) & bench.method.isin(bm)]
    for met in ("kendall", "somers_d"):
        print(f"-- n={n} {met} mean")
        print(b.pivot_table(index="method", columns="dataset", values=met).reindex(bm).round(3).to_string())
    print(f"-- n={n} SD of tau over splits (max over datasets):", b.groupby(["method", "dataset"]).kendall.std().groupby("method").max().reindex(bm).round(3).to_dict())
    print(f"-- n={n} tau_max:", b.groupby("dataset").tau_max.mean().round(3).to_dict())
    lrn = ["WMin-learned", "Ridge", "Choquet-learned", "MonoGBM", "RandomForest", "GradBoost", "FIS-WangMendel", "TSK-learned"]
    uns = ["WSM-equal", "WSM-entropy", "TOPSIS-entropy", "OWA-andness0.5", "FIS-entropy", "FIS-andness0.5", "FIS-entropy-K5", "FIS-andness0.5-K5"]
    for met in ("kendall", "somers_d"):
        p = b.pivot_table(index=["dataset", "rep"], columns="method", values=met)
        win = (p[[c for c in lrn if c in p]].max(axis=1) > p[[c for c in uns if c in p]].max(axis=1)).groupby("dataset").sum()
        gain = (p[[c for c in lrn if c in p]].mean().max())
        means = p.groupby("dataset").mean()
        bl = means[[c for c in lrn if c in means]].idxmax(axis=1); bu = means[[c for c in uns if c in means]].idxmax(axis=1)
        g = pd.Series({ds: means.loc[ds, bl[ds]] - means.loc[ds, bu[ds]] for ds in means.index})
        print(f"   {met}: best learned {bl.to_dict()}; best unsup {bu.to_dict()}; gain {g.round(3).to_dict()};"
              f" splits where best learned (per split) > best unsup: {win.to_dict()}")
        # splits where the best learned method (by mean) beats the best unsupervised (by mean)
        w2 = {ds: int((p.loc[ds][bl[ds]] > p.loc[ds][bu[ds]]).sum()) for ds in means.index}
        print(f"   {met}: splits where mean-best learned > mean-best unsup: {w2}")
