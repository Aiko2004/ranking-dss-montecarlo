"""Анализ устойчивости ранжирования.

Проверяет, насколько итоговый порядок альтернатив зависит от выбора метода
взвешивания, типа разбиения, оператора дефаззификации и способа вывода.
Это ключевой аргумент в пользу объективности модели: если ранги устойчивы,
результат не является артефактом настроек.

Аддитивный вывод имеет фиксированную дефаззификацию (центр по вершинам
термов, в настройках — defuzz="centroid"), поэтому перебирается один раз на
пару (взвешивание, разбиение): 3 × 2 × 3 сценария Мамдани + 3 × 2 аддитивных.

Для каждой конфигурации сообщается число различных оценок: дефаззификация
mom на этой модели схлопывает оценки до нескольких значений (вершины выходных
термов), и тогда корреляция рангов отражает шум, а не различие методов.
"""

from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .config import PRIORITY_DECIMALS

__all__ = [
    "sensitivity_analysis",
    "WEIGHTINGS",
    "PARTITIONS",
    "DEFUZZ_METHODS",
    "INFERENCES",
]

WEIGHTINGS = ["entropy", "critic", "equal"]
PARTITIONS = ["quantile", "fcm"]
DEFUZZ_METHODS = ["centroid", "bisector", "mom"]
INFERENCES = ["mamdani", "additive"]

_SCORE_DECIMALS = PRIORITY_DECIMALS


def sensitivity_analysis(
    df: pd.DataFrame,
    criteria=None,
    baseline: tuple[str, ...] = ("entropy", "quantile", "centroid", "mamdani"),
    weightings=None,
    partitions=None,
    defuzz_methods=None,
    inferences=None,
) -> dict:
    """Перебор допустимых конфигураций; сравнение рангов с базовой.

    baseline — (weighting, partition, defuzz[, inference]); без inference
    базовой считается "mamdani".
    """
    from .pipeline import PriorityModel  # локальный импорт: избегаем цикла

    weightings = weightings or WEIGHTINGS
    partitions = partitions or PARTITIONS
    defuzz_methods = defuzz_methods or DEFUZZ_METHODS
    inferences = inferences or INFERENCES
    if len(baseline) == 3:
        baseline = (*baseline, "mamdani")

    def run(w, p, d, i):
        kwargs = {"weighting": w, "partition": p, "defuzz": d, "inference": i}
        if criteria is not None:
            kwargs["criteria"] = criteria
        return PriorityModel(**kwargs).fit(df).rank()

    base = run(*baseline)
    base_ranks = base["rank"].reindex(df.index)

    scenarios = []
    rank_matrix = {}
    configs = [
        (w, p, d, i)
        for w, p, d, i in product(weightings, partitions, defuzz_methods, inferences)
        if i != "additive" or d == "centroid"
    ]
    for w, p, d, i in configs:
        res = run(w, p, d, i)
        ranks = res["rank"].reindex(df.index)
        rho, pval = spearmanr(base_ranks.to_numpy(), ranks.to_numpy())
        shifts = (ranks - base_ranks).abs()
        scenarios.append(
            {
                "weighting": w,
                "partition": p,
                "defuzz": d,
                "inference": i,
                "spearman_rho": round(float(rho), 4),
                "p_value": round(float(pval), 6),
                "max_rank_shift": int(shifts.max()),
                "mean_rank_shift": round(float(shifts.mean()), 3),
                "top1_preserved": bool(
                    ranks.idxmin() == base_ranks.idxmin()
                ),
                "distinct_scores": int(res["priority"].round(_SCORE_DECIMALS).nunique()),
            }
        )
        rank_matrix[f"{w}|{p}|{d}|{i}"] = ranks

    rm = pd.DataFrame(rank_matrix)
    stability = pd.DataFrame(
        {
            "baseline_rank": base_ranks,
            "rank_min": rm.min(axis=1).astype(int),
            "rank_max": rm.max(axis=1).astype(int),
            "rank_std": rm.std(axis=1).round(3),
        }
    ).sort_values("baseline_rank")

    rhos = np.array([s["spearman_rho"] for s in scenarios])
    return {
        "baseline": dict(zip(("weighting", "partition", "defuzz", "inference"), baseline)),
        "scenario_count": len(scenarios),
        "scenarios": scenarios,
        "spearman_min": round(float(rhos.min()), 4),
        "spearman_mean": round(float(rhos.mean()), 4),
        "stability": stability,
    }
