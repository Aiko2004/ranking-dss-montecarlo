"""Проверка на реальных наборах данных многокритериальных решений (PMLB).

Протокол повторяет симуляцию: случайная обучающая выборка из ntrain «прошлых
решений», остальное — тест; методы без учителя ранжируют тест без меток.
Качество — τ_b Кендалла и Spearman с реальной порядковой оценкой, NDCG@10%.
Запуск: python bench.py BENCH_DIR OUT.csv REPS
"""
from __future__ import annotations

import sys
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import kendalltau, spearmanr

sys.path.insert(0, sys.path[0])
import methods as M  # noqa: E402
from simulate import ndcg_top  # noqa: E402

warnings.filterwarnings("ignore")

DATASETS = {
    "ESL": ("1027_ESL", []),
    "LEV": ("1029_LEV", []),
    "SWD": ("1028_SWD", []),
    "ERA": ("1030_ERA", []),
    "Car": ("car_evaluation", ["buying_price", "maintenance_costs"]),  # критерии типа cost
}


def load(bench_dir, key):
    fname, cost = DATASETS[key]
    d = pd.read_csv(f"{bench_dir}/{fname}.tsv.gz", sep="\t")
    X = d.drop(columns="target").astype(float)
    for c in cost:
        X[c] = X[c].max() - X[c]  # приводим к направлению «больше = лучше»
    return X.to_numpy(), d["target"].to_numpy(float)


def run(key, X, y, ntrain, rep):
    rng = np.random.default_rng(1000 * rep + ntrain)
    idx = rng.permutation(len(y))
    tr, te = idx[:ntrain], idx[ntrain:]
    m = X.shape[1]
    Ztr, lo, hi = M.minmax(X[tr])
    Zte_tr, _, _ = M.minmax(X[te], lo, hi)
    Zte, _, _ = M.minmax(X[te])
    w = M.entropy_weights(Zte)
    res = {
        "WSM-entropy": M.wsm(Zte, w),
        "WSM-equal": M.wsm(Zte, np.full(m, 1 / m)),
        "TOPSIS-entropy": M.topsis(Zte, w),
        "FIS-entropy": M.fis_unsupervised(Zte, "entropy", "fcm"),
        "FIS-andness0.5": M.fis_unsupervised(Zte, "entropy", "fcm", lam=0.5),
        "OWA-andness0.5": M.owa_crisp(Zte, w, 0.5),
        "FIS-WangMendel": M.fis_wang_mendel(Ztr, y[tr], Zte_tr),
    }
    if m <= 6:  # 5 термов: 5^m правил — только для малого числа критериев
        res["FIS-entropy-K5"] = M.fis_unsupervised(Zte, "entropy", "fcm", K=5)
        res["FIS-andness0.5-K5"] = M.fis_unsupervised(Zte, "entropy", "fcm", K=5, lam=0.5)
    for name, mdl in M.ml_models(rep).items():
        mdl.fit(Ztr, y[tr])
        res[name] = mdl.predict(Zte_tr)
    out = []
    for meth, sc in res.items():
        out.append({"dataset": key, "ntrain": ntrain, "rep": rep, "method": meth,
                    "kendall": kendalltau(y[te], sc).statistic,
                    "spearman": spearmanr(y[te], sc).statistic,
                    "ndcg10": ndcg_top(y[te], sc)})
    return out


def main(bench_dir, out_csv, reps):
    jobs = []
    for key in DATASETS:
        X, y = load(bench_dir, key)
        for ntr in (50, 200):
            for r in range(reps):
                jobs.append((key, X, y, ntr, r))
    rows = Parallel(n_jobs=2)(delayed(run)(*j) for j in jobs)
    pd.DataFrame([x for r in rows for x in r]).to_csv(out_csv, index=False)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]))
