"""n*: точка безубыточности средней кривой с бутстреп-интервалом, распределение посценарных n*,
перцентили посценарной разности (ridge - правило). python curve_stats.py curve.csv"""
import sys, numpy as np, pandas as pd
from scipy.optimize import curve_fit
d = pd.read_csv(sys.argv[1])
F = ["func", "m", "rho", "marg", "wconc", "noise"]
f = lambda n, a, b, c: a - b * n ** (-c)
rng = np.random.default_rng(1)

def nstar_of(curve_by_n, base):
    x = curve_by_n.index.values.astype(float); y = curve_by_n.values
    try:
        a, b, c = curve_fit(f, x, y, p0=[0.95, 1, 0.6], maxfev=20000)[0]
    except Exception:
        return np.nan
    return (b / (a - base)) ** (1 / c) if a > base else np.inf

for func in ["linear", "concave", "interaction", "veto"]:
    s = d[d.func == func]
    rule = "OWA" if func == "veto" else "WSM"
    R = s[s.method == "Ridge"].pivot_table(index=F + ["rep"], columns="n", values="kendall")
    B0 = s[s.method == rule].set_index(F + ["rep"]).kendall.loc[R.index]
    # точечная оценка
    ns = nstar_of(R.mean(), B0.mean())
    # бутстреп: повторы внутри сценариев (сетка фиксирована) и, отдельно, сценарии
    cells = R.index.droplevel("rep").unique()
    Rv = R.to_numpy().reshape(len(cells), 30, -1); Bv = B0.to_numpy().reshape(len(cells), 30)
    bs_rep, bs_cell = [], []
    for _ in range(500):
        i = rng.integers(0, 30, 30)
        bs_rep.append(nstar_of(pd.Series(Rv[:, i].mean((0, 1)), index=R.columns), Bv[:, i].mean()))
        j = rng.integers(0, len(cells), len(cells))
        bs_cell.append(nstar_of(pd.Series(Rv[j].mean((0, 1)), index=R.columns), Bv[j].mean()))
    q = lambda v: np.nanpercentile(np.array(v, float), [2.5, 97.5]).round(1)
    # посценарные кривые
    Rc = R.groupby(F).mean(); Bc = B0.groupby(F).mean()
    per = np.array([nstar_of(Rc.loc[k], Bc.loc[k]) for k in Rc.index])
    diff = Rc.sub(Bc, axis=0)
    print(f"{func}: n*={ns:.1f} CI_rep={q(bs_rep)} CI_scen={q(bs_cell)} | per-scenario n*: median={np.median(per):.1f} "
          f"q25={np.percentile(per,25):.1f} q75={np.percentile(per,75):.1f} inf={np.isinf(per).mean():.2f} N={len(per)}")
    print("   diff ridge-rule by n: mean / p10 / share>0:",
          {n: (round(diff[n].mean(), 3), round(diff[n].quantile(0.1), 3), round((diff[n] > 0).mean(), 3)) for n in [10, 20, 50, 100, 200, 500]})
comp = d[d.func != "veto"]
R = comp[comp.method == "Ridge"].groupby(F + ["n"]).kendall.mean().unstack()
B = comp[comp.method == "WSM"].groupby(F).kendall.mean()
dd = R.sub(B, axis=0)
per = []
for k in R.index:
    per.append(nstar_of(R.loc[k], B.loc[k]))
per = np.array(per)
print("compensatory pooled: N scen", len(dd), "share ridge>rule at 50:", (dd[50] > 0).mean().round(3), (dd[50] > 0).sum(),
      "noise0:", (dd[50] > 0)[dd.index.get_level_values('noise') == 0].mean().round(3),
      "noise.2:", (dd[50] > 0)[dd.index.get_level_values('noise') == 0.2].mean().round(3))
print("pooled per-scenario n*: median", np.median(per).round(1), "q10/q90", np.percentile(per, [10, 25, 75, 90]).round(1))
print("p10 of diff at n=10, 20, 50:", dd[10].quantile(.1).round(3), dd[20].quantile(.1).round(3), dd[50].quantile(.1).round(3))
print("lower-curve gap check: mean of q10 curve at 10 minus rule:", (R[10].groupby(level='func').quantile(.1) - B.groupby(level='func').mean()).round(3).to_dict())
