"""Чувствительность выводов к параметрам форм полезности (вторая рецензия).

veto:        U = (sum w_j x_j) * prod_j (beta + (1 - beta) * min(1, x_j / theta)),
             базовые theta = 0.25, beta = 0.2; варианты theta = 0.15, 0.40; beta = 0, 0.5.
interaction: U = sum w_j x_j + gamma * sum_{j<k} w_j w_k x_j x_k, базовое gamma = 4; варианты 2, 8.
Те же сценарии и зёрна, что в simulate.py (cell_id из полной сетки).

Запуск: python simulate_shape.py OUT.csv REPS [N_JOBS]
"""
from __future__ import annotations

import itertools
import sys
import time
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, sys.path[0])
import methods as M  # noqa: E402
import methods2 as M2  # noqa: E402
from simulate import FACTORS, N_TEST, gen_criteria, metrics  # noqa: E402
from simulate3 import wmin_learned  # noqa: E402

warnings.filterwarnings("ignore")

VARIANTS = {
    "veto": [("theta", 0.15), ("theta", 0.40), ("beta", 0.0), ("beta", 0.5), ("base", None)],
    "interaction": [("gamma", 2.0), ("gamma", 8.0), ("base", None)],
}


def utility_v(X, w, func, par, val):
    if func == "veto":
        theta = val if par == "theta" else 0.25
        beta = val if par == "beta" else 0.2
        return (X @ w) * np.prod(beta + (1 - beta) * np.minimum(1.0, X / theta), axis=1)
    gamma = val if par == "gamma" else 4.0
    m = X.shape[1]
    syn = sum(w[j] * w[k] * X[:, j] * X[:, k] for j in range(m) for k in range(j + 1, m))
    return X @ w + gamma * syn


def run_one(cell, rep, cell_id, par, val, base_seed=2026):
    seed = base_seed + cell_id * 1000 + rep
    rng = np.random.default_rng(seed)
    m, ntr = cell["m"], cell["ntrain"]
    alpha = 10.0 if cell["wconc"] == "balanced" else 0.5
    w = rng.dirichlet(np.full(m, alpha))
    X = gen_criteria(ntr + N_TEST, m, cell["rho"], cell["marg"], rng)
    Ut = utility_v(X, w, cell["func"], par, val)
    pct = pd.Series(Ut).rank(pct=True).to_numpy()
    y_noisy = pct + rng.normal(0, cell["noise"], size=len(Ut)) if cell["noise"] > 0 else pct
    cuts = np.quantile(y_noisy[:ntr], [0.2, 0.4, 0.6, 0.8])
    y_ord = np.searchsorted(cuts, y_noisy, side="right").astype(float)
    Xtr, Xte, ytr, ute = X[:ntr], X[ntr:], y_ord[:ntr], Ut[ntr:]
    Zte, _, _ = M.minmax(Xte)
    Ztr, lo, hi = M.minmax(Xtr)
    Zte_tr, _, _ = M.minmax(Xte, lo, hi)
    w_ent = M.entropy_weights(Zte)
    w_ent = w_ent / w_ent.sum()
    res = {
        "WSM-entropy": M.wsm(Zte, w_ent),
        "WSM-equal": M.wsm(Zte, np.full(m, 1 / m)),
        "OWA-andness0.5": M.owa_crisp(Zte, w_ent, 0.5),
        "FIS-entropy": M.fis_unsupervised(Zte, "entropy", "fcm"),
        "FIS-andness0.5": M.fis_unsupervised(Zte, "entropy", "fcm", lam=0.5),
        "WSM-oracle": M.wsm(Zte, w),
        "Choquet-learned": M2.choquet_learned(Ztr, ytr, Zte_tr),
        "MonoGBM": M2.mono_gbm(Ztr, ytr, Zte_tr, seed % 10_000),
        "WMin-learned": wmin_learned(Ztr, ytr, Zte_tr)[0],
    }
    for name, mdl in M.ml_models(seed % 10_000).items():
        res[name] = mdl.fit(Ztr, ytr).predict(Zte_tr)
    tag = "base" if par == "base" else f"{par}={val}"
    return [{**cell, "rep": rep, "variant": tag, "method": k, **metrics(ute, v)} for k, v in res.items()]


def main(out, reps, n_jobs=2):
    cells = [dict(zip(FACTORS, v)) for v in itertools.product(*FACTORS.values())]
    jobs = [(c, r, i, p, v) for i, c in enumerate(cells) if c["func"] in VARIANTS
            for p, v in VARIANTS[c["func"]] for r in range(reps)]
    t = time.time()
    out_rows = Parallel(n_jobs=n_jobs, batch_size=8)(delayed(run_one)(*j) for j in jobs)
    pd.DataFrame([row for rows in out_rows for row in rows]).to_csv(out, index=False)
    print(f"{len(jobs)} instances, {time.time() - t:.0f} s")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 2)
