"""Верификация результата: сопоставление с эталонными методами.

Нечёткий вывод сравнивается с TOPSIS и VIKOR на одной и той же
нормализованной матрице решений и одном и том же векторе объективных весов.
Единственное различие между методами — способ агрегирования, поэтому
согласованность ранжирований является свидетельством того, что результат
не является артефактом выбранного метода.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

from .reference_methods import topsis, vikor

__all__ = ["cross_method_validation"]


def _pairwise(a: np.ndarray, b: np.ndarray, top_n: int) -> dict:
    rho, rho_p = spearmanr(a, b)
    tau, tau_p = kendalltau(a, b)
    shifts = np.abs(a - b)
    top_a = set(np.argsort(a)[:top_n].tolist())
    top_b = set(np.argsort(b)[:top_n].tolist())
    return {
        "spearman_rho": round(float(rho), 4),
        "spearman_p": round(float(rho_p), 6),
        "kendall_tau": round(float(tau), 4),
        "kendall_p": round(float(tau_p), 6),
        "max_rank_shift": int(shifts.max()),
        "mean_rank_shift": round(float(shifts.mean()), 3),
        f"top{top_n}_overlap": len(top_a & top_b),
        "top1_same": bool(int(np.argmin(a)) == int(np.argmin(b))),
    }


def cross_method_validation(
    df: pd.DataFrame,
    criteria=None,
    weighting: str = "entropy",
    partition: str = "quantile",
    defuzz: str = "centroid",
    inference: str = "mamdani",
    vikor_v: float = 0.5,
    top_n: int = 5,
) -> dict:
    """Сравнение ранжирований FIS / TOPSIS / VIKOR.

    Возвращает ранги по каждому методу, попарные меры согласованности
    и сводную таблицу по альтернативам.
    """
    from .pipeline import PriorityModel  # локальный импорт: избегаем цикла

    kwargs = {
        "weighting": weighting,
        "partition": partition,
        "defuzz": defuzz,
        "inference": inference,
    }
    if criteria is not None:
        kwargs["criteria"] = criteria

    model = PriorityModel(**kwargs).fit(df)
    fis = model.rank()

    # Те же входные данные, что у нечёткого вывода: нормализованная матрица
    # (все критерии приведены к типу benefit) и объективные веса.
    matrix = model.norm_.to_numpy()
    weights = model.weights_

    top_n = max(1, min(int(top_n), len(df)))

    t = topsis(matrix, weights)
    v = vikor(matrix, weights, v=vikor_v)

    index = model.norm_.index
    fis_ranks = fis["rank"].reindex(index).to_numpy()

    comparison = pd.DataFrame(
        {
            "fis_priority": fis["priority"].reindex(index).round(3),
            "fis_rank": fis_ranks,
            "topsis_closeness": np.round(t.closeness, 4),
            "topsis_rank": t.ranks,
            "vikor_q": np.round(v.q, 4),
            "vikor_rank": v.ranks,
        },
        index=index,
    )
    comparison["max_disagreement"] = comparison[
        ["fis_rank", "topsis_rank", "vikor_rank"]
    ].max(axis=1) - comparison[["fis_rank", "topsis_rank", "vikor_rank"]].min(axis=1)

    pairs = {
        "fis_vs_topsis": _pairwise(fis_ranks, t.ranks, top_n),
        "fis_vs_vikor": _pairwise(fis_ranks, v.ranks, top_n),
        "topsis_vs_vikor": _pairwise(t.ranks, v.ranks, top_n),
    }

    return {
        "settings": {
            "weighting": weighting,
            "partition": partition,
            "defuzz": defuzz,
            "inference": inference,
            "vikor_v": vikor_v,
            "top_n": top_n,
        },
        "weights": {
            c.key: round(float(w), 8) for c, w in zip(model.criteria, weights)
        },
        "alternatives": len(index),
        "agreement": pairs,
        "vikor_conditions": {
            "acceptable_advantage": v.acceptable_advantage,
            "acceptable_stability": v.acceptable_stability,
            "compromise_set": [str(index[i]) for i in v.compromise_set],
        },
        "comparison": comparison.sort_values("fis_rank"),
    }
