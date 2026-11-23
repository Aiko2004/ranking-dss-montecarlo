"""Четвёртый прогон (третья рецензия): контроль регуляризации и описательные величины.

* Ridge-pos: гребневая регрессия alpha = 0.1, positive = True БЕЗ признака min (контроль к обучаемому WMin);
* Ridge-0.1: гребневая регрессия alpha = 0.1 без ограничений знака;
* доля тестовых альтернатив хотя бы с одним критерием ниже порога вето theta = 0.25 (по исходным x);
* частота отката FCM к равномерным термам (по критериям тестового множества, K = 3 и 5; повторы 0-4).
Те же зёрна и данные, что в simulate.py.

Запуск: python simulate4.py OUT.csv REPS [N_JOBS]
"""
from __future__ import annotations

import itertools
import sys
import time
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import Ridge

sys.path.insert(0, sys.path[0])
import methods as M  # noqa: E402
from simulate import FACTORS, N_TEST, gen_criteria, utility, metrics  # noqa: E402
from fuzzy_priority.variables import fcm_partition  # noqa: E402

warnings.filterwarnings("ignore")


def fcm_fallbacks(Z, K):
    names = [f"t{i}" for i in range(K)]
    n = 0
    for j in range(Z.shape[1]):
        try:
            fcm_partition("x", Z[:, j], names, M.U_IN, seed=0)
        except ValueError:
            n += 1
    return n


def run_one(cell, rep, cell_id, base_seed=2026):
    seed = base_seed + cell_id * 1000 + rep
    rng = np.random.default_rng(seed)
    m, ntr = cell["m"], cell["ntrain"]
    alpha = 10.0 if cell["wconc"] == "balanced" else 0.5
    w = rng.dirichlet(np.full(m, alpha))
    X = gen_criteria(ntr + N_TEST, m, cell["rho"], cell["marg"], rng)
    Ut = utility(X, w, cell["func"])
    pct = pd.Series(Ut).rank(pct=True).to_numpy()
    y_noisy = pct + rng.normal(0, cell["noise"], size=len(Ut)) if cell["noise"] > 0 else pct
    cuts = np.quantile(y_noisy[:ntr], [0.2, 0.4, 0.6, 0.8])
    y_ord = np.searchsorted(cuts, y_noisy, side="right").astype(float)
    Xtr, Xte, ytr, ute = X[:ntr], X[ntr:], y_ord[:ntr], Ut[ntr:]
    Zte, _, _ = M.minmax(Xte)
    Ztr, lo, hi = M.minmax(Xtr)
    Zte_tr, _, _ = M.minmax(Xte, lo, hi)
    extra = {"penalised": float((Xte < 0.25).any(axis=1).mean())}
    if rep < 5:  # откат FCM считается на повторах 0-4 (1920 задач)
        extra.update(fcm_fb_K3=fcm_fallbacks(Zte, 3), fcm_fb_K5=fcm_fallbacks(Zte, 5))
    res = {"Ridge-pos": Ridge(alpha=0.1, positive=True).fit(Ztr, ytr).predict(Zte_tr),
           "Ridge-0.1": Ridge(alpha=0.1).fit(Ztr, ytr).predict(Zte_tr),
           "Ridge": Ridge(alpha=1.0).fit(Ztr, ytr).predict(Zte_tr)}  # контроль совпадения
    return [{**cell, "rep": rep, "method": k, **metrics(ute, v), **extra} for k, v in res.items()]


def main(out, reps, n_jobs=2):
    cells = [dict(zip(FACTORS, v)) for v in itertools.product(*FACTORS.values())]
    jobs = [(c, r, i) for i, c in enumerate(cells) for r in range(reps)]
    t = time.time()
    rows = Parallel(n_jobs=n_jobs, batch_size=16)(delayed(run_one)(c, r, i) for c, r, i in jobs)
    pd.DataFrame([x for rr in rows for x in rr]).to_csv(out, index=False)
    print(f"{len(jobs)} instances, {time.time() - t:.0f} s")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 2)
