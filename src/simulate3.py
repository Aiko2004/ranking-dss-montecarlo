"""Третий прогон на тех же 11 520 задачах (те же зёрна): ответ на вторую рецензию.

* доля связей истинной полезности на тестовом множестве (N1 в tau-b);
* обучаемый WMin_lambda: неотрицательная гребневая регрессия на [z_1..z_m, min_j z_j];
* lambda-sweep (0.25, 0.75, 1) для чёткого WMin, TSK на равномерных термах и Мамдани K=3;
* эмпирическая проверка монотонности выхода Мамдани (повторы 0-4):
  z_j -> min(1, z_j + 0.05), доля пар (альтернатива, критерий), где оценка уменьшилась.

Запуск: python simulate3.py OUT.csv REPS [N_JOBS]
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
import methods2 as M2  # noqa: E402
from simulate import FACTORS, N_TEST, gen_criteria, utility, metrics  # noqa: E402

warnings.filterwarnings("ignore")


def wmin_features(Z):
    return np.hstack([Z, Z.min(1, keepdims=True)])


def wmin_learned(Ztr, ytr, Zte, alpha=0.1):
    """Обучаемый WMin_lambda: y ~ a + sum b_j z_j + g min_j z_j, b, g >= 0.
    Тогда w = b / sum(b), lambda = g / (sum(b) + g)."""
    mdl = Ridge(alpha=alpha, positive=True).fit(wmin_features(Ztr), ytr)
    b, g = mdl.coef_[:-1], mdl.coef_[-1]
    lam = g / max(b.sum() + g, 1e-12)
    return mdl.predict(wmin_features(Zte)), lam


def tied_share(u):
    _, cnt = np.unique(u, return_counts=True)
    n = len(u)
    return float((cnt * (cnt - 1) / 2).sum() / (n * (n - 1) / 2))


def mamdani_fixed(Z, parts, w, lam, K=3):
    ant, cons = M.andness_rules(Z.shape[1], K, w, lam)
    return M.mamdani(M.memberships(Z, parts), ant, cons)


def mono_check(Z, w, lam, delta=0.05, K=3):
    parts = M.build_partitions(Z, K, "fcm")
    base = mamdani_fixed(Z, parts, w, lam, K)
    n, m = Z.shape
    P = np.repeat(Z[None], m, 0)  # (m, n, m)
    for j in range(m):
        P[j, :, j] = np.minimum(1.0, P[j, :, j] + delta)
    up = mamdani_fixed(P.reshape(-1, m), parts, w, lam, K).reshape(m, n)
    moved = P[np.arange(m), :, np.arange(m)] > Z.T + 1e-12  # где критерий действительно вырос
    drop = (up < base[None] - 1e-9) & moved
    return drop.sum() / max(moved.sum(), 1), float(np.max(np.where(drop, base[None] - up, 0.0)))


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
    w_ent = M.entropy_weights(Zte)
    w_ent = w_ent / w_ent.sum()

    res = {}
    extra = {"ties_true": tied_share(ute)}
    res["WSM-entropy"] = M.wsm(Zte, w_ent)  # контроль совпадения
    res["WMin-learned"], extra["lam_learned"] = wmin_learned(Ztr, ytr, Zte_tr)
    for lam in (0.25, 0.75, 1.0):
        res[f"WMin-{lam}"] = M.owa_crisp(Zte, w_ent, lam)
        res[f"TSKu-{lam}"] = M2.tsk0(Zte, w_ent, 3, lam, "centre", "uniform")
        res[f"FIS-{lam}"] = M.fis_unsupervised(Zte, "entropy", "fcm", lam=lam)
    if rep < 5:
        for lam in (0.0, 0.5):
            extra[f"mono_viol_{lam}"], extra[f"mono_maxdrop_{lam}"] = mono_check(Zte, w_ent, lam)
    rows = [{**cell, "rep": rep, "method": k, **metrics(ute, v), **extra} for k, v in res.items()]
    return rows


def main(out, reps, n_jobs=2):
    cells = [dict(zip(FACTORS, v)) for v in itertools.product(*FACTORS.values())]
    jobs = [(c, r, i) for i, c in enumerate(cells) for r in range(reps)]
    t = time.time()
    out_rows = Parallel(n_jobs=n_jobs, batch_size=8)(delayed(run_one)(c, r, i) for c, r, i in jobs)
    pd.DataFrame([row for rows in out_rows for row in rows]).to_csv(out, index=False)
    print(f"{len(jobs)} instances, {time.time() - t:.0f} s")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 2)
