"""Лингвистические переменные и построение разбиений по данным.

Ключевой принцип: параметры функций принадлежности НЕ задаются экспертом.
Они выводятся либо из эмпирического распределения показателя (квантили),
либо из центров нечёткой кластеризации (FCM), либо из нормативных границ.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from .membership import trapmf

__all__ = [
    "LinguisticVariable",
    "quantile_partition",
    "fcm_partition",
    "uniform_partition",
]

MF = Callable[[np.ndarray], np.ndarray]


@dataclass
class LinguisticVariable:
    """Лингвистическая переменная: имя, универсум, набор термов."""

    name: str
    universe: np.ndarray
    terms: dict[str, MF]
    _grid: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.universe = np.asarray(self.universe, dtype=float)
        if self.universe.ndim != 1 or self.universe.size < 2:
            raise ValueError("universe должен быть одномерным массивом длины >= 2")
        if not self.terms:
            raise ValueError("нужен хотя бы один терм")
        self._grid = np.vstack([mf(self.universe) for mf in self.terms.values()])

    @property
    def term_names(self) -> list[str]:
        return list(self.terms)

    @property
    def n_terms(self) -> int:
        return len(self.terms)

    @property
    def grid(self) -> np.ndarray:
        """Матрица (n_terms, n_points) — ФП всех термов на универсуме."""
        return self._grid

    def fuzzify(self, x: float) -> np.ndarray:
        """Вектор степеней принадлежности (n_terms,) для чёткого значения."""
        arr = np.asarray([float(x)])
        return np.array([float(mf(arr)[0]) for mf in self.terms.values()])

    def fuzzify_named(self, x: float) -> dict[str, float]:
        return dict(zip(self.term_names, self.fuzzify(x)))


def _partition_from_centers(
    name: str,
    centers: Sequence[float],
    term_names: Sequence[str],
    universe: np.ndarray,
) -> LinguisticVariable:
    """Строит треугольное разбиение по заданным центрам термов.

    Крайние термы — плечевые (трапеции), внутренние — треугольные.
    Соседние термы пересекаются на уровне 0.5, разбиение является
    разбиением единицы (сумма ФП в любой точке = 1).
    """
    centers = np.sort(np.asarray(centers, dtype=float))
    if len(centers) != len(term_names):
        raise ValueError("число центров должно совпадать с числом термов")
    if len(np.unique(centers)) != len(centers):
        raise ValueError(f"центры термов переменной '{name}' совпадают: {centers}")

    lo, hi = float(universe[0]), float(universe[-1])
    n = len(centers)
    terms: dict[str, MF] = {}

    for i, term in enumerate(term_names):
        left = centers[i - 1] if i > 0 else lo
        right = centers[i + 1] if i < n - 1 else hi
        if i == 0:
            params = (lo, lo, centers[i], right)
        elif i == n - 1:
            params = (left, centers[i], hi, hi)
        else:
            params = (left, centers[i], centers[i], right)
        terms[term] = (lambda p: (lambda x: trapmf(x, p)))(params)

    return LinguisticVariable(name=name, universe=universe, terms=terms)


def quantile_partition(
    name: str,
    values: np.ndarray,
    term_names: Sequence[str],
    universe: np.ndarray,
) -> LinguisticVariable:
    """Разбиение по эмпирическим квантилям выборки.

    Центры термов = квантили уровня 0, 1/(n-1), ..., 1. Это делает разбиение
    воспроизводимым и полностью определяемым данными.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError(f"нет валидных значений для переменной '{name}'")
    n = len(term_names)
    centers = np.quantile(values, np.linspace(0.0, 1.0, n))
    # если распределение вырождено (много одинаковых значений) — раздвигаем
    if len(np.unique(centers)) != n:
        centers = np.linspace(float(universe[0]), float(universe[-1]), n)
    return _partition_from_centers(name, centers, term_names, universe)


def fcm_partition(
    name: str,
    values: np.ndarray,
    term_names: Sequence[str],
    universe: np.ndarray,
    m: float = 2.0,
    max_iter: int = 300,
    tol: float = 1e-6,
    seed: int = 0,
) -> LinguisticVariable:
    """Разбиение по центрам нечёткой кластеризации c-means (одномерной)."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)].reshape(-1, 1)
    c = len(term_names)
    rng = np.random.default_rng(seed)
    u = rng.random((values.shape[0], c))
    u /= u.sum(axis=1, keepdims=True)

    centers = np.zeros(c)
    for _ in range(max_iter):
        um = u**m
        new_centers = (um * values).sum(axis=0) / um.sum(axis=0)
        dist = np.abs(values - new_centers.reshape(1, -1))
        dist = np.fmax(dist, 1e-12)
        inv = dist ** (-2.0 / (m - 1.0))
        new_u = inv / inv.sum(axis=1, keepdims=True)
        shift = np.abs(new_u - u).max()
        u, centers = new_u, new_centers
        if shift < tol:
            break

    return _partition_from_centers(name, centers, term_names, universe)


def uniform_partition(
    name: str,
    term_names: Sequence[str],
    universe: np.ndarray,
) -> LinguisticVariable:
    """Равномерное разбиение универсума — используется для выходной переменной,
    шкала которой определена математически, а не по данным."""
    centers = np.linspace(float(universe[0]), float(universe[-1]), len(term_names))
    return _partition_from_centers(name, centers, term_names, universe)
