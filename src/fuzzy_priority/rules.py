"""База правил, порождаемая математически.

Вместо экспертных продукций правила строятся детерминированной функцией
агрегирования: индекс терма заключения определяется взвешенной суммой
порядковых индексов термов посылки. Это корректно при условии, что все
критерии приведены к единому направлению (больше = выше приоритет),
и делает базу правил полной, непротиворечивой и монотонной по построению.

Терм заключения: k(a) = floor(s + 1/2), где s = (K-1) * <w, a> / (m-1),
округлённое до 9 знаков. Округление s снимает шум весов порядка 1e-15,
иначе s на границе x.5 переключало бы заключения от сдвига шкалы данных;
округляется именно s, а не уровень <w, a> / (m-1): ничья s = k + 1/2 точна
при любом K, тогда как уровень (2k+1) / (2(K-1)) двоичной дробью бывает не всегда.
Ничья округляется вверх (а не к чётному, как round()). База правил остаётся
ступенчатой функцией весов — это свойство метода, исправлен только численный шум.

Монотонна именно база правил: терм заключения не убывает при повышении
терма любого входа. Монотонность выхода из этого не следует: при min-max
выводе Мамдани с центроидом оценка может локально убывать по критерию.
Неубывающий выход даёт аддитивный вывод (PriorityModel(inference="additive")).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import product

import numpy as np

__all__ = ["Rule", "generate_monotone_rules", "rules_to_table"]

# Знаков округления (K-1)·уровня: шум весов ~1e-15 исчезает, различимые значения сохраняются
_LEVEL_DECIMALS = 9


@dataclass(frozen=True)
class Rule:
    """Правило вида ЕСЛИ x1 = T[a1] И ... И xn = T[an] ТО y = U[c]."""

    antecedent: tuple[int, ...]  # индексы термов по каждому входу
    consequent: int  # индекс терма выхода
    weight: float = 1.0


def generate_monotone_rules(
    n_inputs: int,
    n_in_terms: int,
    n_out_terms: int,
    weights: np.ndarray | None = None,
) -> list[Rule]:
    """Порождает полную базу из n_in_terms ** n_inputs правил.

    Parameters
    ----------
    weights : веса критериев (объективные). None = равные веса.
    """
    if n_inputs < 1 or n_in_terms < 2 or n_out_terms < 2:
        raise ValueError("некорректная размерность базы правил")

    if weights is None:
        w = np.full(n_inputs, 1.0 / n_inputs)
    else:
        w = np.asarray(weights, dtype=float)
        if w.shape != (n_inputs,):
            raise ValueError("длина вектора весов не совпадает с числом входов")
        if w.sum() <= 0:
            raise ValueError("сумма весов должна быть положительной")
        w = w / w.sum()

    max_idx = n_in_terms - 1
    rules: list[Rule] = []
    for antecedent in product(range(n_in_terms), repeat=n_inputs):
        level = float(np.dot(w, antecedent)) / max_idx  # в [0, 1]
        scaled = round(level * (n_out_terms - 1), _LEVEL_DECIMALS)  # в [0, K-1]
        consequent = math.floor(scaled + 0.5)  # ничья — вверх
        rules.append(Rule(antecedent=tuple(antecedent), consequent=consequent))
    return rules


def rules_to_table(
    rules: list[Rule],
    input_names: list[str],
    in_term_names: list[str],
    out_name: str,
    out_term_names: list[str],
) -> list[dict]:
    """Человекочитаемое представление базы правил (для приложения к работе)."""
    table = []
    for i, r in enumerate(rules, start=1):
        row = {"№": i}
        for name, idx in zip(input_names, r.antecedent):
            row[name] = in_term_names[idx]
        row[out_name] = out_term_names[r.consequent]
        table.append(row)
    return table
