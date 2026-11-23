"""Функции принадлежности (векторизованные, на numpy).

Все функции принимают скаляр или np.ndarray и возвращают массив степеней
принадлежности в [0, 1].
"""

from __future__ import annotations

import numpy as np

__all__ = ["trapmf", "trimf", "gaussmf"]


def trapmf(x, params) -> np.ndarray:
    """Трапециевидная ФП с параметрами (a, b, c, d), a <= b <= c <= d.

    Вырожденные случаи a == b (левое плечо) и c == d (правое плечо)
    обрабатываются корректно.
    """
    a, b, c, d = (float(p) for p in params)
    if not (a <= b <= c <= d):
        raise ValueError(f"Требуется a <= b <= c <= d, получено {params}")

    x = np.atleast_1d(np.asarray(x, dtype=float))

    left = (x - a) / (b - a) if b > a else np.where(x >= b, 1.0, 0.0)
    right = (d - x) / (d - c) if d > c else np.where(x <= c, 1.0, 0.0)

    return np.clip(np.minimum(left, right), 0.0, 1.0)


def trimf(x, params) -> np.ndarray:
    """Треугольная ФП с параметрами (a, b, c)."""
    a, b, c = (float(p) for p in params)
    return trapmf(x, (a, b, b, c))


def gaussmf(x, mean: float, sigma: float) -> np.ndarray:
    """Гауссова ФП."""
    if sigma <= 0:
        raise ValueError("sigma должна быть положительной")
    x = np.atleast_1d(np.asarray(x, dtype=float))
    return np.exp(-((x - float(mean)) ** 2) / (2.0 * float(sigma) ** 2))
