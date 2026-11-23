"""Система нечёткого вывода Мамдани (векторизованная реализация).

Собственная реализация вместо scikit-fuzzy: полный контроль над операторами
t-нормы, импликации, агрегирования и дефаззификации — это нужно для анализа
чувствительности и для воспроизводимости результатов в работе.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np

from .rules import Rule
from .variables import LinguisticVariable

__all__ = ["MamdaniFIS", "InferenceResult", "centroid_defuzz", "weighted_peak_defuzz", "term_peaks"]

TNorm = Literal["min", "prod"]
Aggregation = Literal["max", "sum"]
Defuzz = Literal["centroid", "bisector", "mom", "som", "lom", "peak"]

_TNORMS = ("min", "prod")
_AGGREGATIONS = ("max", "sum")


def term_peaks(output: LinguisticVariable) -> np.ndarray:
    """Вершины термов выхода: середина множества точек, где ФП максимальна."""
    u = output.universe
    peaks = []
    for g in output.grid:
        idx = np.flatnonzero(g >= g.max() - 1e-12)
        peaks.append(float(u[idx].mean()))
    return np.array(peaks)


def centroid_defuzz(agg: np.ndarray, universe: np.ndarray) -> float:
    """Центр тяжести выходного множества."""
    return float(np.trapezoid(agg * universe, universe) / np.trapezoid(agg, universe))


def weighted_peak_defuzz(firing: np.ndarray, peaks: np.ndarray) -> float:
    """Σ f_r·c(k_r) / Σ f_r — среднее вершин заключений, взвешенное срабатываниями.

    В отличие от центроида суммарного множества, здесь у каждого правила вес
    равен его срабатыванию, а не срабатыванию × площади терма заключения.
    Площади крайних (плечевых) термов вдвое меньше средних, и с весами-площадями
    сдвиг срабатываний к более высоким заключениям может понизить центроид.
    """
    total = float(firing.sum())
    if total <= 1e-12:
        return float("nan")
    return float(np.dot(firing, peaks) / total)


@dataclass
class InferenceResult:
    """Результат вывода для одной альтернативы."""

    value: float
    firing: np.ndarray  # степени активации правил (n_rules,)
    aggregated: np.ndarray  # агрегированное выходное множество (n_points,)


class MamdaniFIS:
    """Нечёткий вывод с настраиваемыми операторами.

    and_op="min", implication="min", aggregation="max" — классический
    min-max вывод Мамдани. and_op="prod", implication="prod",
    aggregation="sum", defuzz="peak" — аддитивный вывод с дефаззификацией
    по вершинам термов: при разбиениях единицы на входах и монотонной базе
    правил его выход не убывает по каждому входу (docs/19-monotonicity.md).
    С defuzz="centroid" сумма (SAM) монотонна не в общем случае: веса
    правил тогда пропорциональны площадям термов заключения.
    """

    def __init__(
        self,
        inputs: Sequence[LinguisticVariable],
        output: LinguisticVariable,
        rules: Sequence[Rule],
        and_op: TNorm = "min",
        implication: TNorm = "min",
        aggregation: Aggregation = "max",
        defuzz: Defuzz = "centroid",
    ) -> None:
        if and_op not in _TNORMS or implication not in _TNORMS:
            raise ValueError(f"t-норма должна быть одной из {_TNORMS}")
        if aggregation not in _AGGREGATIONS:
            raise ValueError(f"агрегирование должно быть одним из {_AGGREGATIONS}")
        if not inputs:
            raise ValueError("нужен хотя бы один вход")
        if not rules:
            raise ValueError("база правил пуста")

        n_in = len(inputs)
        for r in rules:
            if len(r.antecedent) != n_in:
                raise ValueError("длина посылки не совпадает с числом входов")
            for j, t in enumerate(r.antecedent):
                if not (0 <= t < inputs[j].n_terms):
                    raise ValueError(f"недопустимый индекс терма {t} для входа {j}")
            if not (0 <= r.consequent < output.n_terms):
                raise ValueError(f"недопустимый индекс терма заключения {r.consequent}")

        self.inputs = list(inputs)
        self.output = output
        self.rules = list(rules)
        self.and_op = and_op
        self.implication = implication
        self.aggregation = aggregation
        self.defuzz = defuzz

        self._ant = np.array([r.antecedent for r in self.rules], dtype=int)
        self._rule_w = np.array([r.weight for r in self.rules], dtype=float)
        # ФП термов заключения каждого правила: (n_rules, n_points)
        self._cons = output.grid[[r.consequent for r in self.rules], :]
        # вершины термов заключения каждого правила: (n_rules,)
        self._peaks = term_peaks(output)[[r.consequent for r in self.rules]]

    # ------------------------------------------------------------------ #

    @property
    def n_rules(self) -> int:
        return len(self.rules)

    def _firing(self, x: Sequence[float]) -> np.ndarray:
        if len(x) != len(self.inputs):
            raise ValueError("размерность входного вектора не совпадает с числом входов")
        # матрица степеней принадлежности: (n_inputs, max_terms)
        max_terms = max(v.n_terms for v in self.inputs)
        mu = np.zeros((len(self.inputs), max_terms))
        for j, var in enumerate(self.inputs):
            mu[j, : var.n_terms] = var.fuzzify(float(x[j]))

        cols = np.arange(len(self.inputs))[None, :]
        degrees = mu[cols, self._ant]  # (n_rules, n_inputs)
        if self.and_op == "min":
            firing = degrees.min(axis=1)
        else:
            firing = degrees.prod(axis=1)
        return firing * self._rule_w

    def _aggregate(self, firing: np.ndarray) -> np.ndarray:
        if self.implication == "min":
            clipped = np.minimum(firing[:, None], self._cons)
        else:
            clipped = firing[:, None] * self._cons
        if self.aggregation == "sum":
            # Значения могут превышать 1: это не нечёткое множество в строгом
            # смысле, а взвешенная сумма заключений — для центроида это и нужно.
            return clipped.sum(axis=0)
        return clipped.max(axis=0)

    def _defuzzify(self, agg: np.ndarray, firing: np.ndarray) -> float:
        if self.defuzz == "peak":
            return weighted_peak_defuzz(firing, self._peaks)
        u = self.output.universe
        total = agg.sum()
        if total <= 1e-12:
            return float("nan")

        if self.defuzz == "centroid":
            return centroid_defuzz(agg, u)
        if self.defuzz == "bisector":
            cum = np.cumsum(agg)
            return float(u[np.searchsorted(cum, cum[-1] / 2.0)])
        peak = agg.max()
        idx = np.flatnonzero(agg >= peak - 1e-12)
        if self.defuzz == "mom":
            return float(u[idx].mean())
        if self.defuzz == "som":
            return float(u[idx].min())
        return float(u[idx].max())

    # ------------------------------------------------------------------ #

    def infer(self, x: Sequence[float]) -> InferenceResult:
        firing = self._firing(x)
        agg = self._aggregate(firing)
        return InferenceResult(value=self._defuzzify(agg, firing), firing=firing, aggregated=agg)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Вывод для матрицы альтернатив (n_alternatives, n_inputs)."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        return np.array([self.infer(row).value for row in X])
