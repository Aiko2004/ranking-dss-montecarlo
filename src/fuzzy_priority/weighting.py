"""Объективное взвешивание критериев.

Используются только методы, извлекающие веса из самой матрицы решений:
без анкет, экспертных парных сравнений и AHP.
"""

from __future__ import annotations

import numpy as np

__all__ = ["entropy_weights", "critic_weights", "equal_weights"]


def _check(matrix: np.ndarray) -> np.ndarray:
    m = np.asarray(matrix, dtype=float)
    if m.ndim != 2:
        raise ValueError("ожидается матрица (альтернативы x критерии)")
    if not np.isfinite(m).all():
        raise ValueError("матрица содержит NaN/inf")
    return m


def _nonnegative_weights(raw: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Обрезает сырые веса снизу нулём и нормирует; при нулевой сумме — равные веса.

    Теоретически сырые веса неотрицательны (1 - энтропия >= 0, СКО x конфликтность
    >= 0), но в арифметике с плавающей точкой возможны значения порядка -1e-16.
    Неотрицательность весов нужна для монотонности базы правил (docs/19-monotonicity.md,
    предположение A5).
    """
    clipped = np.clip(raw, 0.0, None)
    total = clipped.sum()
    if total <= 0:
        return equal_weights(matrix)
    return clipped / total


def equal_weights(matrix: np.ndarray) -> np.ndarray:
    """Равные веса — базовый вариант для сравнения."""
    m = _check(matrix)
    return np.full(m.shape[1], 1.0 / m.shape[1])


def entropy_weights(matrix: np.ndarray) -> np.ndarray:
    """Метод энтропии Шеннона.

    Ожидает нормализованную матрицу в [0, 1], где больше = лучше.
    Чем ниже энтропия распределения критерия по альтернативам,
    тем выше его различающая способность и вес.
    """
    m = _check(matrix)
    n = m.shape[0]
    shifted = m + 1e-12
    p = shifted / shifted.sum(axis=0, keepdims=True)
    e = -(p * np.log(p)).sum(axis=0) / np.log(n)
    return _nonnegative_weights(1.0 - e, m)


def critic_weights(matrix: np.ndarray) -> np.ndarray:
    """Метод CRITIC (Diakoulaki et al., 1995).

    Вес = контраст (СКО) x конфликтность (сумма 1 - корреляция с другими).
    """
    m = _check(matrix)
    if m.shape[1] < 2:
        return equal_weights(m)
    sd = m.std(axis=0, ddof=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.corrcoef(m, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)
    conflict = (1.0 - corr).sum(axis=1)
    return _nonnegative_weights(sd * conflict, m)
