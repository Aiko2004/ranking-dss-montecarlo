"""Сквозная модель: данные -> нормализация -> веса -> ФП -> правила -> ранжирование."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Sequence

import numpy as np
import pandas as pd

from .config import DEFAULT_CRITERIA, IN_TERMS, OUT_TERMS, OUTPUT_RANGE, PRIORITY_DECIMALS, Criterion
from .mamdani import MamdaniFIS
from .rules import generate_monotone_rules
from .variables import (
    LinguisticVariable,
    fcm_partition,
    quantile_partition,
    uniform_partition,
)
from .weighting import critic_weights, entropy_weights, equal_weights

__all__ = ["PriorityModel"]

WeightMethod = Literal["entropy", "critic", "equal"]
PartitionMethod = Literal["quantile", "fcm"]
InferenceMethod = Literal["mamdani", "additive"]

_WEIGHTERS = {
    "entropy": entropy_weights,
    "critic": critic_weights,
    "equal": equal_weights,
}

# Операторы вывода. "mamdani" — классический min-max вывод; база правил
# монотонна, но выход с центроидом может локально убывать по критерию.
# "additive" — prod-И, prod-импликация, sum-агрегирование и дефаззификация
# Σ f_r·c(k_r) / Σ f_r по вершинам термов заключения: при разбиениях единицы
# на входах выход не убывает по каждому критерию (docs/19-monotonicity.md).
# Дефаззификация аддитивного вывода фиксирована: в настройках ей соответствует
# только defuzz="centroid", прочие методы для него не определены.
INFERENCE_OPERATORS: dict[str, dict[str, str]] = {
    "mamdani": {"and_op": "min", "implication": "min", "aggregation": "max"},
    "additive": {"and_op": "prod", "implication": "prod", "aggregation": "sum", "defuzz": "peak"},
}


@dataclass
class PriorityModel:
    criteria: Sequence[Criterion] = field(default_factory=lambda: list(DEFAULT_CRITERIA))
    in_terms: Sequence[str] = field(default_factory=lambda: list(IN_TERMS))
    out_terms: Sequence[str] = field(default_factory=lambda: list(OUT_TERMS))
    weighting: WeightMethod = "entropy"
    partition: PartitionMethod = "quantile"
    resolution: int = 501
    defuzz: str = "centroid"
    # По умолчанию "mamdani": так воспроизводятся расчёты, сделанные до
    # появления параметра. Смена значения по умолчанию — отдельное решение.
    inference: InferenceMethod = "mamdani"

    fitted_: bool = field(default=False, init=False)

    # ------------------------------------------------------------------ #

    @property
    def keys(self) -> list[str]:
        return [c.key for c in self.criteria]

    def _normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Min-max нормализация с приведением всех критериев к типу benefit."""
        out = pd.DataFrame(index=df.index, columns=self.keys, dtype=float)
        self.bounds_ = {}
        for c in self.criteria:
            col = df[c.key].astype(float)
            lo, hi = float(col.min()), float(col.max())
            self.bounds_[c.key] = (lo, hi)
            if hi - lo < 1e-12:
                out[c.key] = 0.5
                continue
            z = (col - lo) / (hi - lo)
            out[c.key] = z if c.direction == "benefit" else 1.0 - z
        return out

    def fit(self, df: pd.DataFrame) -> "PriorityModel":
        """df: индекс — альтернативы, колонки — ключи критериев (сырые значения)."""
        missing = [k for k in self.keys if k not in df.columns]
        if missing:
            raise ValueError(f"в данных отсутствуют критерии: {missing}")
        if self.inference not in INFERENCE_OPERATORS:
            raise ValueError(
                f"неизвестный способ вывода '{self.inference}'; "
                f"допустимы: {list(INFERENCE_OPERATORS)}"
            )
        if self.inference == "additive" and self.defuzz != "centroid":
            raise ValueError(
                "в аддитивном выводе дефаззификация фиксирована (центр по вершинам "
                f"термов), допустим только defuzz='centroid', получено '{self.defuzz}'"
            )

        self.raw_ = df[self.keys].copy()
        self.norm_ = self._normalize(self.raw_)

        matrix = self.norm_.to_numpy()
        self.weights_ = _WEIGHTERS[self.weighting](matrix)

        universe_in = np.linspace(0.0, 1.0, self.resolution)
        builder = quantile_partition if self.partition == "quantile" else fcm_partition
        self.input_vars_: list[LinguisticVariable] = [
            builder(c.key, matrix[:, j], list(self.in_terms), universe_in)
            for j, c in enumerate(self.criteria)
        ]

        universe_out = np.linspace(*OUTPUT_RANGE, self.resolution)
        self.output_var_ = uniform_partition("priority", list(self.out_terms), universe_out)

        self.rules_ = generate_monotone_rules(
            n_inputs=len(self.criteria),
            n_in_terms=len(self.in_terms),
            n_out_terms=len(self.out_terms),
            weights=self.weights_,
        )

        self.fis_ = MamdaniFIS(
            inputs=self.input_vars_,
            output=self.output_var_,
            rules=self.rules_,
            **self.operators(),
        )
        self.fitted_ = True
        return self

    def operators(self) -> dict[str, str]:
        """Операторы вывода MamdaniFIS для выбранных inference и defuzz."""
        return {"defuzz": self.defuzz, **INFERENCE_OPERATORS[self.inference]}

    def rank(self, df: pd.DataFrame | None = None) -> pd.DataFrame:
        if not self.fitted_:
            raise RuntimeError("модель не обучена: сначала вызовите fit()")
        norm = self.norm_ if df is None else self._normalize(df[self.keys])
        scores = self.fis_.predict(norm.to_numpy())
        result = pd.DataFrame(
            {"priority": scores},
            index=norm.index,
        )
        # ранг — по отдаваемой оценке: равные в ответе priority получают равный ранг
        shown = result["priority"].round(PRIORITY_DECIMALS)
        result["rank"] = shown.rank(ascending=False, method="min").astype(int)
        result["class"] = [self._classify(s) for s in scores]
        return result.sort_values("rank")

    def _classify(self, score: float) -> str:
        mu = self.output_var_.fuzzify(score)
        return self.output_var_.term_names[int(np.argmax(mu))]

    def weights_table(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "критерий": [c.name_ru for c in self.criteria],
                "направление": [c.direction for c in self.criteria],
                "источник": [c.origin for c in self.criteria],
                "вес": self.weights_,
            },
            index=self.keys,
        )

    # ---------------------- интроспекция для API ---------------------- #

    def partition_centers(self) -> dict[str, list[float]]:
        """Центры термов каждого входа на нормализованной шкале."""
        return {
            var.name: [float(var.universe[int(np.argmax(g))]) for g in var.grid]
            for var in self.input_vars_
        }

    def membership_curves(self, n_points: int = 101) -> dict[str, dict]:
        """Кривые ФП всех переменных — для построения графиков на фронтенде."""
        step = max(1, self.resolution // n_points)
        out: dict[str, dict] = {}
        for var in list(self.input_vars_) + [self.output_var_]:
            out[var.name] = {
                "universe": var.universe[::step].round(4).tolist(),
                "terms": {
                    name: var.grid[i][::step].round(4).tolist()
                    for i, name in enumerate(var.term_names)
                },
            }
        return out

    def explain(self, alternative: str, top_rules: int = 10, n_points: int = 101) -> dict:
        """Трассировка вывода для одной альтернативы.

        Возвращает степени принадлежности по каждому критерию, наиболее
        активные правила и агрегированное выходное множество.
        """
        if not self.fitted_:
            raise RuntimeError("модель не обучена: сначала вызовите fit()")
        if alternative not in self.norm_.index:
            raise KeyError(f"альтернатива '{alternative}' не найдена")

        x = self.norm_.loc[alternative].to_numpy(dtype=float)
        res = self.fis_.infer(x)

        fuzzified = {
            var.name: dict(zip(var.term_names, var.fuzzify(x[j]).round(4).tolist()))
            for j, var in enumerate(self.input_vars_)
        }

        order = np.argsort(-res.firing)[:top_rules]
        fired = [
            {
                "rule_no": int(i) + 1,
                "firing": round(float(res.firing[i]), 4),
                "if": {
                    var.name: var.term_names[self.rules_[i].antecedent[j]]
                    for j, var in enumerate(self.input_vars_)
                },
                "then": self.output_var_.term_names[self.rules_[i].consequent],
            }
            for i in order
            if res.firing[i] > 0
        ]

        step = max(1, self.resolution // n_points)
        return {
            "alternative": alternative,
            "inference": self.inference,
            "operators": self.operators(),
            "raw": self.raw_.loc[alternative].to_dict(),
            "normalized": self.norm_.loc[alternative].round(4).to_dict(),
            "fuzzified": fuzzified,
            "active_rules": fired,
            "active_rule_count": int((res.firing > 0).sum()),
            "aggregated": {
                "universe": self.output_var_.universe[::step].round(3).tolist(),
                "membership": res.aggregated[::step].round(4).tolist(),
            },
            "priority": round(float(res.value), PRIORITY_DECIMALS),
            "class": self._classify(float(res.value)),
        }
