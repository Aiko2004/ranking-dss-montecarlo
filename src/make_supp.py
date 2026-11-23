"""Онлайн-приложение (Supplementary Material) из сырых результатов.

python make_supp.py results_dir paper_dir
"""
import sys

import numpy as np
import pandas as pd

R, P = sys.argv[1], sys.argv[2]
d = pd.read_csv(f"{R}/all3_R30.csv.gz")
r4 = pd.read_csv(f"{R}/r4_R30.csv.gz")
d = pd.concat([d, r4[r4.method.isin(["Ridge-pos", "Ridge-0.1"])][[c for c in d.columns if c in r4.columns]]], ignore_index=True)
r3 = pd.read_csv(f"{R}/r3_R30.csv.gz")
shape = pd.read_csv(f"{R}/shape_R10.csv.gz")
bench = pd.read_csv(f"{R}/bench3_R20.csv.gz")
F = ["func", "m", "rho", "marg", "wconc", "noise", "ntrain"]
NAME = {"WSM-oracle": "WSM, true weights (reference)", "WMin-learned": "WMin~*λ*~, learned", "Choquet-learned": "Choquet-type, learned",
        "Ridge": "Ridge regression", "WSM-expert0.3": "WSM, expert, *σ*~w~ = 0.3", "MIX-expert0.3": "WMin~0.5~, expert, *σ*~w~ = 0.3",
        "MonoGBM": "Monotone GBM", "WSM-expert0.6": "WSM, expert, *σ*~w~ = 0.6", "MIX-expert0.6": "WMin~0.5~, expert, *σ*~w~ = 0.6",
        "GradBoost": "Gradient boosting", "RandomForest": "Random forest", "WSM-equal": "WSM, equal weights", "WSM-entropy": "WSM",
        "TSK-learned": "TSK, learned", "OWA-andness0.5": "WMin~0.5~", "TOPSIS-entropy": "TOPSIS", "FIS-entropy-K5": "Mamdani, *K* = 5",
        "FIS-entropy": "Mamdani, *K* = 3", "FIS-andness0.5": "Mamdani, *K* = 3, *λ* = 0.5", "FIS-andness0.5-K5": "Mamdani, *K* = 5, *λ* = 0.5",
        "FIS-WangMendel": "Wang–Mendel", "Ridge-pos": "Ridge, non-negative, α = 0.1 (control)", "Ridge-0.1": "Ridge, α = 0.1 (control)"}
f3 = lambda x: "–" if pd.isna(x) else f"{x:.3f}"
out = []
w = out.append


def table(df, header, align=None):
    w("| " + " | ".join(header) + " |")
    w("|" + "|".join(align or [":---"] + ["---:"] * (len(header) - 1)) + "|")
    for row in df:
        w("| " + " | ".join(row) + " |")
    w("")


# S2 ties
w("## S2 Ties in the True Utility\n")
t = r3[r3.method == "WSM-entropy"]
w("Share of the 44,850 test pairs tied in the true utility, sparse marginals (mean over forms, weights, noise and *n*~train~; uniform and skewed marginals have no ties). Kendall's *τ*~b~ accounts for these ties through *N*~1~ in (6).\n")
pt = t[t.marg == "sparse"].pivot_table(index="m", columns="rho", values="ties_true")
table([[f"*m* = {m}"] + [f"{pt.loc[m, r]:.3f}" for r in pt.columns] for m in pt.index], ["", "*ρ* = 0", "*ρ* = 0.6"])

# penalised share and FCM fallback
t4 = r4[r4.method == "Ridge"]
pp = t4.pivot_table(index="marg", columns="m", values="penalised")
w("### Frequency of the veto-type penalty\n")
w("Share of test alternatives with at least one criterion below *θ* = 0.25 (mean over all scenarios; the share does not depend on the utility form). Overall: "
  f"{t4.penalised.mean():.3f}.\n")
table([[mg] + [f"{pp.loc[mg, m]:.3f}" for m in pp.columns] for mg in ["uniform", "skewed", "sparse"]], ["Marginal", "*m* = 3", "*m* = 5"])
fb = t4.dropna(subset=["fcm_fb_K3"])
w(f"*FCM fallback.* In replications 0–4 ({len(fb)} problems, {int(fb.m.sum())} criteria), FCM fell back to equally spaced terms for "
  f"{int(fb.fcm_fb_K3.sum())} criteria with *K* = 3 and {int(fb.fcm_fb_K5.sum())} with *K* = 5.\n")

# S3 full method table and by form x n
w("## S3 All Methods: Kendall's *τ*~b~, Spearman's *ρ*~s~ and NDCG@10%\n")
w("Mean over 384 scenarios with half-width of the 95% bootstrap interval of *τ*~b~, and *τ*~b~ by utility form and number of past decisions.\n")
rows = []
order = d[d.method.isin(NAME)].groupby("method").kendall.mean().sort_values(ascending=False).index
piv = d.pivot_table(index="method", columns=["func", "ntrain"], values="kendall")
rng = np.random.default_rng(1)
for mt in order:
    sub = d[d.method == mt]
    pv = sub.pivot_table(index=F, columns="rep", values="kendall").to_numpy()
    idx = rng.integers(0, 30, size=(1000, 30))
    bs = np.array([pv[:, i].mean() for i in idx])
    ci = (np.percentile(bs, 97.5) - np.percentile(bs, 2.5)) / 2
    rows.append([NAME[mt], f3(sub.kendall.mean()), f"{ci:.4f}", f3(sub.spearman.mean()), f3(sub.ndcg10.mean()),
                 [f3(piv.loc[mt, (fn, n)]) for fn in ["linear", "concave", "interaction", "veto"] for n in (50, 200)]])
table([r[:5] for r in rows], ["Method", "*τ*~b~", "±CI", "*ρ*~s~", "NDCG@10%"])
w("*τ*~b~ by utility form and number of past decisions.\n")
table([[r[0]] + r[5] for r in rows], ["Method", "Lin. 50", "Lin. 200", "Conc. 50", "Conc. 200", "Int. 50", "Int. 200", "Veto 50", "Veto 200"])
w("NDCG@10% varied between 0.93 and 0.99 and ranked the top methods in the same order as *τ*~b~; it is therefore not reported in the paper.\n")

w("### Learned WMin~*λ*~ against the regularisation control\n")
w("Mean *τ*~b~ by utility form. Non-negative ridge uses the same penalty (0.1) and sign constraint as the learned WMin~*λ*~ but no minimum feature; the difference isolates the contribution of the minimum. Paired 95% bootstrap interval of the overall difference: 0.0146–0.0150.\n")
CM = ["WMin-learned", "Ridge-pos", "Ridge-0.1", "Ridge", "Choquet-learned"]
pc = d[d.method.isin(CM)].pivot_table(index="method", columns="func", values="kendall")
rows = [[NAME[m]] + [f3(pc.loc[m, f]) for f in ["linear", "concave", "interaction", "veto"]] + [f3(d[d.method == m].kendall.mean())] for m in CM]
rows.append(["WMin~*λ*~ minus non-negative ridge"] + [f"{pc.loc['WMin-learned', f] - pc.loc['Ridge-pos', f]:+.3f}" for f in ["linear", "concave", "interaction", "veto"]]
            + [f"{d[d.method == 'WMin-learned'].kendall.mean() - d[d.method == 'Ridge-pos'].kendall.mean():+.3f}"])
table(rows, ["Method", "Linear", "Concave", "Interaction", "Veto", "All"])

w("### Best method with ablation variants as candidates\n")
UNS = ["WSM-entropy", "WSM-equal", "TOPSIS-entropy", "OWA-andness0.5", "FIS-entropy", "FIS-andness0.5", "FIS-entropy-K5", "FIS-andness0.5-K5"]
SUP = ["Ridge", "RandomForest", "GradBoost", "MonoGBM", "Choquet-learned", "WMin-learned", "TSK-learned", "FIS-WangMendel"]
sc = d.groupby(F + ["method"]).kendall.mean().unstack()
ABL = [c for c in sc.columns if c.startswith(("TSK-", "TSKu", "FIS-L", "FIS-unif", "FIS-0", "FIS-1")) and c != "TSK-learned"]
CR = ["WMin-0.25", "WMin-0.75", "WMin-1.0"]
cand = sc[UNS + SUP + ABL + CR]
bb = cand.idxmax(axis=1).value_counts()
LAB2 = {**NAME, "WMin-0.25": "WMin~0.25~", "WMin-0.75": "WMin~0.75~", "WMin-1.0": "WMin~1~", "TSKu-1.0": "TSK, equal terms, *λ* = 1",
        "TSKu-0.25": "TSK, equal terms, *λ* = 0.25", "TSKu-0.75": "TSK, equal terms, *λ* = 0.75", "FIS-L17-n": "Mamdani, *L* = 17, *λ* = 0.5",
        "FIS-L17-c": "Mamdani, *L* = 17, *λ* = 0"}
w("Candidates: the 16 methods of the paper, all 18 ablation and *λ*-sweep fuzzy variants and crisp WMin~*λ*~ for *λ* = 0.25, 0.75 and 1. Number of scenarios (of 384) in which each method had the highest mean *τ*~b~.\n")
table([[LAB2.get(k, k), str(v)] for k, v in bb.items()], ["Method", "Scenarios"])

# Veto x marginal
w("### Veto form by marginal distribution and number of past decisions\n")
vm = ["WSM-entropy", "OWA-andness0.5", "MIX-expert0.3", "WMin-learned", "Choquet-learned", "MonoGBM", "Ridge", "FIS-entropy", "FIS-andness0.5"]
pv = d[(d.func == "veto") & d.method.isin(vm)].pivot_table(index="method", columns=["marg", "ntrain"], values="kendall").loc[vm]
cols = [(mg, n) for mg in ("uniform", "skewed", "sparse") for n in (50, 200)]
table([[NAME[m]] + [f3(pv.loc[m, c]) for c in cols] for m in vm], ["Method"] + [f"{mg[:4]}. {n}" for mg, n in cols])
w("With sparse criteria the minimum is zero for most alternatives, so WMin~0.5~ and WSM nearly coincide.\n")

# S4 lambda sweep
w("## S4 Sensitivity to the Weight of the Minimum *λ*\n")
SW = {"WMin (crisp)": ["WSM-entropy", "WMin-0.25", "OWA-andness0.5", "WMin-0.75", "WMin-1.0"],
      "TSK, equal centres": ["TSK-centre-unif-c", "TSKu-0.25", "TSK-centre-unif-n", "TSKu-0.75", "TSKu-1.0"],
      "Mamdani, *K* = 3": ["FIS-entropy", "FIS-0.25", "FIS-andness0.5", "FIS-0.75", "FIS-1.0"]}
LAMS = [0, 0.25, 0.5, 0.75, 1.0]
rows = []
for grp, sel in (("Without veto", d.func != "veto"), ("Veto", d.func == "veto")):
    for fam, ms in SW.items():
        rows.append([f"{grp}: {fam}"] + [f3(d[sel & (d.method == m)].kendall.mean()) for m in ms])
table(rows, ["Form and system"] + [f"*λ* = {l}" for l in LAMS])
w("For the veto form the best crisp result (0.862 at *λ* = 0.5) is not exceeded by any fuzzy variant; at *λ* ≥ 0.75 the TSK system is better than the crisp operator at the same *λ*, because smoothing weakens an overly strong minimum.\n")

# monotonicity
w("### Monotonicity of the defuzzified Mamdani output\n")
mo = t.dropna(subset=["mono_viol_0.0"])
w(f"Replications 0–4 of every scenario ({len(mo)} problems). Each criterion of each test alternative was raised by 0.05 (capped at 1) with the partitions and rules fixed. "
  f"Share of increments that lowered the output: {mo['mono_viol_0.0'].mean():.3f} (*λ* = 0) and {mo['mono_viol_0.5'].mean():.3f} (*λ* = 0.5); "
  f"every problem had at least one violation; the largest drop per problem had median {mo['mono_maxdrop_0.0'].median():.2f} and maximum {mo['mono_maxdrop_0.0'].max():.2f} "
  f"(*λ* = 0), {mo['mono_maxdrop_0.5'].median():.2f} and {mo['mono_maxdrop_0.5'].max():.2f} (*λ* = 0.5) points on the 0–100 output scale.\n")

# learned lambda
lam = r3[r3.method == "WMin-learned"]
w("### Learned share of the minimum\n")
g = lam.groupby(["func", "ntrain"]).lam_learned
table([[fn, str(n), f"{g.median()[(fn, n)]:.2f}", f"{g.quantile(.25)[(fn, n)]:.2f}–{g.quantile(.75)[(fn, n)]:.2f}"]
       for fn in ["linear", "concave", "interaction", "veto"] for n in (50, 200)], ["Form", "*n*~train~", "Median *λ*", "IQR"])

# S5 sensitivity to factors
w("## S5 Effects of the Problem Conditions\n")
sens = pd.read_csv(f"{R}/sensitivity.csv").set_index("factor")
SM = ["WSM-entropy", "FIS-entropy", "OWA-andness0.5", "Ridge", "WMin-learned", "Choquet-learned", "RandomForest"]
w("Change in mean *τ*~b~ when one factor changes level (mean over all other factors).\n")
table([[i] + [f"{sens.loc[i, m]:+.3f}" for m in SM] for i in sens.index], ["Factor"] + [NAME[m] for m in SM])
w("Partial *η*^2^ (ANOVA of 11,520 problem-level values on seven factors and all two-way interactions), three largest effects: WSM: *ρ* 0.63, weight concentration 0.61, marginal 0.30; Mamdani: *ρ* 0.74, weight concentration 0.61, marginal 0.43; ridge: form 0.60, *ρ* 0.40, form × *ρ* 0.38; learned WMin~*λ*~: form 0.41, noise 0.24, *n*~train~ 0.23; random forest: noise 0.46, *ρ* 0.42, marginal 0.29.\n")


# S6 shape sensitivity
w("## S6 Sensitivity to the Parameters of the Utility Forms\n")
w("Veto form (2) with *θ* ∈ {0.15, 0.25, 0.40} and *β* ∈ {0, 0.2, 0.5} (one parameter changed at a time) and interaction form (3) with *γ* ∈ {2, 4, 8}; the same 96 scenarios and seeds as in the main experiment, replications 0–9 (the base variant reproduces the main experiment exactly). Mean *τ*~b~.\n")
SH = ["WSM-entropy", "OWA-andness0.5", "FIS-andness0.5", "WMin-learned", "Choquet-learned", "MonoGBM", "Ridge"]
VAR = {"veto": ["theta=0.15", "base", "theta=0.4", "beta=0.0", "beta=0.5"], "interaction": ["gamma=2.0", "base", "gamma=8.0"]}
LAB = {"theta=0.15": "*θ* = 0.15", "theta=0.4": "*θ* = 0.40", "beta=0.0": "*β* = 0", "beta=0.5": "*β* = 0.5", "gamma=2.0": "*γ* = 2", "gamma=8.0": "*γ* = 8"}
for fn in ("veto", "interaction"):
    x = shape[(shape.func == fn) & shape.method.isin(SH)].pivot_table(index="method", columns=["variant", "ntrain"], values="kendall")
    for n in (50, 200):
        cols = [(v, n) for v in VAR[fn]]
        w(f"*{fn.capitalize()} form, {n} past decisions*\n")
        table([[NAME[m]] + [f3(x.loc[m, c]) for c in cols] for m in SH],
              ["Method"] + [LAB.get(v, "base") for v, _ in cols])

# S7 learning curves
w("## S7 Learning Curves\n")
w("Break-even points of the mean curves with 95% intervals from 400 resamples of the 48 scenarios of a form; first simulated size at which the learned model beat the rule in a scenario; 10th percentile of the scenario-level gain over the rule.\n")
w(open(f"{R}/curve_stats.md").read())

# S8 real data
w("## S8 Real Datasets\n")
bm = ["WMin-learned", "Ridge", "Choquet-learned", "MonoGBM", "RandomForest", "GradBoost", "TSK-learned", "FIS-WangMendel",
      "WSM-equal", "WSM-entropy", "TOPSIS-entropy", "OWA-andness0.5", "FIS-entropy", "FIS-andness0.5", "FIS-entropy-K5", "FIS-andness0.5-K5"]
DS = ["Car", "ERA", "ESL", "LEV", "SWD"]
for n in (200, 50):
    b = bench[bench.ntrain == n]
    for met, lab in (("kendall", "Kendall's *τ*~b~"), ("somers_d", "Somers' *D* on pairs with different grades")):
        mu = b.pivot_table(index="method", columns="dataset", values=met)
        sd = b.pivot_table(index="method", columns="dataset", values=met, aggfunc="std")
        w(f"*{lab}, {n} past decisions (mean ± SD over 20 splits)*\n")
        rows = []
        if met == "kendall":
            tm = b.groupby("dataset").tau_max.mean()
            rows.append(["*τ*~max~"] + [f3(tm[ds]) for ds in DS])
        for m in bm:
            if m in mu.index:
                rows.append([NAME[m]] + [("–" if pd.isna(mu.loc[m, ds]) else f"{mu.loc[m, ds]:.3f} ± {sd.loc[m, ds]:.3f}") for ds in DS])
        table(rows, ["Method", "Car", "ERA", "ESL", "LEV", "SWD"])

open(f"{P}/supp_tables.md", "w").write("\n".join(out))
print("ok")
