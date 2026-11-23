"""Эталонные методы многокритериального выбора для верификации результата.

TOPSIS и VIKOR выбраны намеренно: оба работают на объективных весах
(энтропия Шеннона, CRITIC) и не требуют экспертных парных сравнений.
Метод анализа иерархий (AHP) для сопоставления не применяется — он
строится на экспертных суждениях, что противоречит проектному ограничению
работы.

Все три метода (нечёткий вывод, TOPSIS, VIKOR) получают ОДНУ И ТУ ЖЕ
нормализованную матрицу решений и ОДИН И ТОТ ЖЕ вектор весов. Благодаря
этому единственным различием между ними остаётся способ агрегирования,
и расхождение ранжирований можно приписать именно ему, а не разнице
в подготовке данных.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["TopsisResult", "VikorResult", "topsis", "vikor"]


def _check(matrix: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    m = np.asarray(matrix, dtype=float)
    w = np.asarray(weights, dtype=float)
    if m.ndim != 2:
        raise ValueError("ожидается матрица (альтернативы x критерии)")
    if m.shape[0] < 2:
        raise ValueError("нужно не менее двух альтернатив")
    if w.shape != (m.shape[1],):
        raise ValueError("длина вектора весов не совпадает с числом критериев")
    if not np.isfinite(m).all() or not np.isfinite(w).all():
        raise ValueError("матрица или веса содержат NaN/inf")
    if w.sum() <= 0:
        raise ValueError("сумма весов должна быть положительной")
    return m, w / w.sum()


@dataclass
class TopsisResult:
    closeness: np.ndarray  # коэффициент близости C, больше = лучше
    ranks: np.ndarray  # 1 = высший приоритет
    d_positive: np.ndarray  # расстояние до идеального решения
    d_negative: np.ndarray  # расстояние до антиидеального решения


@dataclass
class VikorResult:
    q: np.ndarray  # компромиссная мера Q, меньше = лучше
    s: np.ndarray  # мера полезности группы
    r: np.ndarray  # мера индивидуального сожаления
    ranks: np.ndarray
    acceptable_advantage: bool  # условие C1
    acceptable_stability: bool  # условие C2
    compromise_set: list[int]  # индексы альтернатив компромиссного множества


def _ranks_desc(values: np.ndarray) -> np.ndarray:
    """Ранги по убыванию значения: наибольшему значению — ранг 1."""
    order = np.argsort(-values, kind="stable")
    ranks = np.empty(len(values), dtype=int)
    ranks[order] = np.arange(1, len(values) + 1)
    return ranks


def _ranks_asc(values: np.ndarray) -> np.ndarray:
    """Ранги по возрастанию значения: наименьшему значению — ранг 1."""
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=int)
    ranks[order] = np.arange(1, len(values) + 1)
    return ranks


def topsis(matrix: np.ndarray, weights: np.ndarray) -> TopsisResult:
    """TOPSIS (Hwang & Yoon, 1981).

    Матрица должна быть приведена к типу benefit: больше значение = лучше.
    В проекте это обеспечивает нормализация в PriorityModel, где критерии
    типа cost инвертируются.
    """
    m, w = _check(matrix, weights)

    # Векторная нормализация
    norms = np.sqrt((m**2).sum(axis=0))
    norms = np.where(norms < 1e-12, 1.0, norms)
    r = m / norms

    v = r * w

    ideal_best = v.max(axis=0)
    ideal_worst = v.min(axis=0)

    d_pos = np.sqrt(((v - ideal_best) ** 2).sum(axis=1))
    d_neg = np.sqrt(((v - ideal_worst) ** 2).sum(axis=1))

    denom = d_pos + d_neg
    closeness = np.where(denom < 1e-12, 0.0, d_neg / np.where(denom < 1e-12, 1.0, denom))

    return TopsisResult(
        closeness=closeness,
        ranks=_ranks_desc(closeness),
        d_positive=d_pos,
        d_negative=d_neg,
    )


def vikor(matrix: np.ndarray, weights: np.ndarray, v: float = 0.5) -> VikorResult:
    """VIKOR (Opricovic & Tzeng, 2004).

    Parameters
    ----------
    v : вес стратегии «большинства критериев». 0.5 — консенсус,
        принятое значение по умолчанию.
    """
    if not 0.0 <= v <= 1.0:
        raise ValueError("параметр v должен лежать в [0, 1]")
    m, w = _check(matrix, weights)
    n = m.shape[0]

    f_best = m.max(axis=0)
    f_worst = m.min(axis=0)
    spread = f_best - f_worst
    spread = np.where(spread < 1e-12, 1.0, spread)

    normalized = w * (f_best - m) / spread

    s = normalized.sum(axis=1)
    r = normalized.max(axis=1)

    def _scale(x: np.ndarray) -> np.ndarray:
        lo, hi = x.min(), x.max()
        return np.zeros_like(x) if hi - lo < 1e-12 else (x - lo) / (hi - lo)

    q = v * _scale(s) + (1.0 - v) * _scale(r)
    ranks = _ranks_asc(q)

    # Условия компромиссного решения
    order = np.argsort(q, kind="stable")
    a1, a2 = int(order[0]), int(order[1])

    # C1 — приемлемое преимущество
    dq = 1.0 / (n - 1) if n > 1 else 0.0
    acceptable_advantage = bool((q[a2] - q[a1]) >= dq)

    # C2 — приемлемая устойчивость: лидер по Q лучший хотя бы по S или по R
    acceptable_stability = bool(a1 == int(np.argmin(s)) or a1 == int(np.argmin(r)))

    if acceptable_advantage and acceptable_stability:
        compromise = [a1]
    elif not acceptable_stability:
        compromise = [a1, a2]
    else:
        # C1 не выполнено: в компромиссное множество входят все альтернативы,
        # для которых Q[i] - Q[a1] < DQ
        compromise = [int(i) for i in order if (q[i] - q[a1]) < dq]

    return VikorResult(
        q=q,
        s=s,
        r=r,
        ranks=ranks,
        acceptable_advantage=acceptable_advantage,
        acceptable_stability=acceptable_stability,
        compromise_set=compromise,
    )
