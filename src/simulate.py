"""Исследование Монте-Карло: когда нечёткий вывод лучше линейных и ML-моделей
для многокритериального ранжирования.

Запуск: python simulate.py OUT.csv REPS [N_JOBS]
"""
from __future__ import annotations

import itertools
import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm, beta, spearmanr, kendalltau
from joblib import Parallel, delayed

sys.path.insert(0, sys.path[0])
import methods as M  # noqa: E402

warnings.filterwarnings("ignore")

N_TEST = 300
FACTORS = {
    "func": ["linear", "concave", "veto", "interaction"],
    "m": [3, 5],
    "rho": [0.0, 0.6],
    "marg": ["uniform", "skewed", "sparse"],
    "wconc": ["balanced", "concentrated"],
    "noise": [0.0, 0.2],
    "ntrain": [50, 200],
}


# ------------------------------------------------------------------ генератор
def gen_criteria(n, m, rho, marg, rng):
    C = np.full((m, m), rho) + (1 - rho) * np.eye(m)
    Zn = rng.multivariate_normal(np.zeros(m), C, size=n)
    U = norm.cdf(Zn)
    if marg == "uniform":
        return U
    if marg == "skewed":
        return beta.ppf(U, 2, 5)
    # разреженный признак: 70% нулей, далее равномерно
    return np.where(U < 0.7, 0.0, (U - 0.7) / 0.3)


def utility(X, w, func):
    if func == "linear":
        return X @ w
    if func == "concave":  # убывающая отдача
        return np.sqrt(X) @ w
    if func == "veto":  # некомпенсаторность: слабый критерий «обнуляет» вклад остальных
        pen = np.prod(0.2 + 0.8 * np.minimum(1.0, X / 0.25), axis=1)
        return (X @ w) * pen
    # interaction: синергия пар критериев
    m = X.shape[1]
    syn = sum(w[j] * w[k] * X[:, j] * X[:, k] for j in range(m) for k in range(j + 1, m))
    return X @ w + 4.0 * syn


def ndcg_top(true, score, frac=0.1):
    n = len(true)
    k = max(1, int(round(frac * n)))
    gain = pd.Series(true).rank(pct=True).to_numpy()  # релевантность = процентиль истинной полезности
    disc = 1 / np.log2(np.arange(2, k + 2))
    order = np.lexsort((np.random.default_rng(0).random(n), -score))[:k]
    ideal = np.sort(gain)[::-1][:k]
    return float((gain[order] * disc).sum() / (ideal * disc).sum())


def metrics(true, score):
    r = spearmanr(true, score).statistic
    t = kendalltau(true, score).statistic
    return {"spearman": float(np.nan_to_num(r)), "kendall": float(np.nan_to_num(t)), "ndcg10": ndcg_top(true, score)}


# ------------------------------------------------------------------ один прогон
def run_one(cell, rep, cell_id, base_seed=2026):
    # воспроизводимое зерно: не зависит от PYTHONHASHSEED
    seed = base_seed + cell_id * 1000 + rep
    rng = np.random.default_rng(seed)
    m, ntr = cell["m"], cell["ntrain"]
    alpha = 10.0 if cell["wconc"] == "balanced" else 0.5
    w = rng.dirichlet(np.full(m, alpha))
    X = gen_criteria(ntr + N_TEST, m, cell["rho"], cell["marg"], rng)
    Ut = utility(X, w, cell["func"])
    # прошлые решения ЛПР: порядковая оценка (5 уровней) зашумлённой полезности
    # шум задаётся на шкале процентилей полезности: σ = 0.2 означает, что ЛПР
    # в среднем ошибается на ~20 процентилей независимо от вида функции
    pct = pd.Series(Ut).rank(pct=True).to_numpy()
    y_noisy = pct + rng.normal(0, cell["noise"], size=len(Ut)) if cell["noise"] > 0 else pct
    cuts = np.quantile(y_noisy[:ntr], [0.2, 0.4, 0.6, 0.8])
    y_ord = np.searchsorted(cuts, y_noisy, side="right").astype(float)

    Xtr, Xte, ytr, ute = X[:ntr], X[ntr:], y_ord[:ntr], Ut[ntr:]
    Zte, _, _ = M.minmax(Xte)
    Ztr, lo, hi = M.minmax(Xtr)
    Zte_tr, _, _ = M.minmax(Xte, lo, hi)

    res = {}
    t0 = time.perf_counter()
    w_ent = M.entropy_weights(Zte)
    res["WSM-entropy"] = M.wsm(Zte, w_ent)
    res["WSM-equal"] = M.wsm(Zte, np.full(m, 1 / m))
    res["TOPSIS-entropy"] = M.topsis(Zte, w_ent)
    res["FIS-entropy"] = M.fis_unsupervised(Zte, "entropy", "fcm")
    res["FIS-equal"] = M.fis_unsupervised(Zte, "equal", "fcm")
    res["FIS-andness0.5"] = M.fis_unsupervised(Zte, "entropy", "fcm", lam=0.5)
    res["OWA-andness0.5"] = M.owa_crisp(Zte, w_ent, 0.5)
    res["FIS-entropy-K5"] = M.fis_unsupervised(Zte, "entropy", "fcm", K=5)
    res["FIS-andness0.5-K5"] = M.fis_unsupervised(Zte, "entropy", "fcm", K=5, lam=0.5)
    res["WSM-oracle"] = M.wsm(Zte, w)  # эталон: истинные веса известны
    res["FIS-WangMendel"] = M.fis_wang_mendel(Ztr, ytr, Zte_tr)
    for name, mdl in M.ml_models(seed % 10_000).items():
        mdl.fit(Ztr, ytr)
        res[name] = mdl.predict(Zte_tr)
    dt = time.perf_counter() - t0

    rows = []
    for meth, sc in res.items():
        rows.append({**cell, "rep": rep, "method": meth, **metrics(ute, sc)})
    rows[0]["_sec"] = dt
    return rows


def main(out, reps, n_jobs=2):
    cells = [dict(zip(FACTORS, v)) for v in itertools.product(*FACTORS.values())]
    jobs = [(c, r, i) for i, c in enumerate(cells) for r in range(reps)]
    t = time.time()
    out_rows = Parallel(n_jobs=n_jobs, batch_size=8)(delayed(run_one)(c, r, i) for c, r, i in jobs)
    df = pd.DataFrame([row for rows in out_rows for row in rows])
    df.to_csv(out, index=False)
    print(f"{len(cells)} cells x {reps} reps = {len(jobs)} instances, {time.time() - t:.0f} s")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 2)
