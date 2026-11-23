"""Кривая обучения: точность гребневой регрессии в зависимости от числа прошлых решений n.
Для каждого сценария (без фактора ntrain) и каждого n — R повторов.
Стартовое правило: WSM (энтропийные веса) и OWA_0.5 на тех же тестовых альтернативах.
"""
import itertools, sys, warnings
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from scipy.stats import kendalltau
sys.path.insert(0, sys.path[0])
import methods as M
from simulate import gen_criteria, utility, FACTORS
from sklearn.linear_model import Ridge
warnings.filterwarnings("ignore")

NGRID = [10, 20, 30, 50, 75, 100, 150, 200, 300, 500]
F = {k: v for k, v in FACTORS.items() if k != "ntrain"}

def run(cell, cid, rep):
    rng = np.random.default_rng(777000 + cid * 100 + rep)
    m = cell["m"]; w = rng.dirichlet(np.full(m, 10.0 if cell["wconc"] == "balanced" else 0.5))
    nmax = max(NGRID); X = gen_criteria(nmax + 300, m, cell["rho"], cell["marg"], rng)
    U = utility(X, w, cell["func"])
    pct = pd.Series(U).rank(pct=True).to_numpy()
    yn = pct + (rng.normal(0, cell["noise"], len(U)) if cell["noise"] > 0 else 0)
    Xtr_all, Xte, ute = X[:nmax], X[nmax:], U[nmax:]
    Zte, _, _ = M.minmax(Xte); we = M.entropy_weights(Zte)
    out = [{**cell, "rep": rep, "n": 0, "method": "WSM", "kendall": kendalltau(ute, M.wsm(Zte, we)).statistic},
           {**cell, "rep": rep, "n": 0, "method": "OWA", "kendall": kendalltau(ute, M.owa_crisp(Zte, we, 0.5)).statistic}]
    for n in NGRID:
        cuts = np.quantile(yn[:n], [0.2, 0.4, 0.6, 0.8]); y = np.searchsorted(cuts, yn[:n], side="right").astype(float)
        Ztr, lo, hi = M.minmax(Xtr_all[:n]); Zt, _, _ = M.minmax(Xte, lo, hi)
        s = Ridge(alpha=1.0).fit(Ztr, y).predict(Zt)
        out.append({**cell, "rep": rep, "n": n, "method": "Ridge", "kendall": kendalltau(ute, s).statistic})
    return out

cells = [dict(zip(F, v)) for v in itertools.product(*F.values())]
R = int(sys.argv[2])
rows = Parallel(n_jobs=2, batch_size=16)(delayed(run)(c, i, r) for i, c in enumerate(cells) for r in range(R))
pd.DataFrame([x for r in rows for x in r]).to_csv(sys.argv[1], index=False)
print(len(cells), "cells x", R)
