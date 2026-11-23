"""Дополнительный прогон на тех же 11 520 задачах (те же зёрна, что в simulate.py).

Запуск: python simulate2.py OUT.csv REPS [N_JOBS]
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
from simulate import FACTORS, N_TEST, gen_criteria, utility, metrics  # noqa: E402

warnings.filterwarnings("ignore")


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
    t0 = time.perf_counter()
    # контроль совпадения с основным прогоном
    res["WSM-entropy"] = M.wsm(Zte, w_ent)
    res["FIS-entropy"] = M.fis_unsupervised(Zte, "entropy", "fcm")
    res["Ridge"] = M.ml_models(seed % 10_000)["Ridge"].fit(Ztr, ytr).predict(Zte_tr)
    # абляции нечёткого вывода, lam = 0 и 0.5
    for lam, tag in ((0.0, "c"), (0.5, "n")):
        for L in (9, 17):
            res[f"FIS-L{L}-{tag}"] = M2.fis_mamdani(Zte, w_ent, 3, lam, L)
        res[f"FIS-unif-{tag}"] = M2.fis_mamdani(Zte, w_ent, 3, lam, 5, "uniform")
        res[f"TSK-index-{tag}"] = M2.tsk0(Zte, w_ent, 3, lam, "index")
        res[f"TSK-centre-{tag}"] = M2.tsk0(Zte, w_ent, 3, lam, "centre")
        res[f"TSK-centre-unif-{tag}"] = M2.tsk0(Zte, w_ent, 3, lam, "centre", "uniform")
    # обучаемые
    res["TSK-learned"] = M2.tsk_learned(Ztr, ytr, Zte_tr)
    res["Choquet-learned"] = M2.choquet_learned(Ztr, ytr, Zte_tr)
    res["MonoGBM"] = M2.mono_gbm(Ztr, ytr, Zte_tr, seed % 10_000)
    # зашумлённые экспертные веса (отдельный поток случайных чисел)
    rng_e = np.random.default_rng(seed + 7_000_000)
    for s in (0.3, 0.6):
        we = M2.expert_weights(w, s, rng_e)
        res[f"WSM-expert{s}"] = M.wsm(Zte, we)
        res[f"MIX-expert{s}"] = M.owa_crisp(Zte, we, 0.5)
    dt = time.perf_counter() - t0
    rows = [{**cell, "rep": rep, "method": k, **metrics(ute, v)} for k, v in res.items()]
    rows[0]["_sec"] = dt
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
