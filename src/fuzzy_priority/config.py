"""Конфигурация критериев.

Каждый критерий обязан иметь прослеживаемый источник: официальная статистика,
нормативный документ или геопространственный расчёт. Поле `origin` принимает
значения "data" (из данных), "normative" (из СН/СП/СНиП) или "geospatial".
Экспертных оценок в системе нет.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

__all__ = ["Criterion", "DEFAULT_CRITERIA", "IN_TERMS", "OUT_TERMS", "OUTPUT_RANGE"]

Direction = Literal["benefit", "cost"]
Origin = Literal["data", "normative", "geospatial"]


@dataclass(frozen=True)
class Criterion:
    key: str
    name_ru: str
    unit: str
    direction: Direction  # benefit = больше -> выше приоритет
    origin: Origin
    source: str


DEFAULT_CRITERIA: list[Criterion] = [
    Criterion(
        key="deficit",
        name_ru="Нормативный дефицит мощности объекта",
        unit="доля от нормативной потребности",
        direction="benefit",
        origin="normative",
        source="СН РК 3.01-01 / отраслевые нормативы обеспеченности + БНС АСПиР РК",
    ),
    Criterion(
        key="demand_growth",
        name_ru="Темп прироста обслуживаемого населения",
        unit="% в год",
        direction="benefit",
        origin="data",
        source="БНС АСПиР РК, численность населения по районам г. Астана",
    ),
    Criterion(
        key="density",
        name_ru="Плотность населения территории обслуживания",
        unit="чел./га",
        direction="benefit",
        origin="data",
        source="БНС АСПиР РК + границы районов",
    ),
    Criterion(
        key="accessibility",
        name_ru="Транспортно-пешеходная недоступность существующих объектов",
        unit="средневзвешенное время в пути, мин",
        direction="benefit",
        origin="geospatial",
        source="изохроны по дорожному графу OpenStreetMap",
    ),
    Criterion(
        key="unit_cost",
        name_ru="Удельная стоимость строительства",
        unit="тыс. тг/м²",
        direction="cost",
        origin="normative",
        source="сметно-нормативная база РК + кадастровая стоимость земли",
    ),
]

IN_TERMS: list[str] = ["низкий", "средний", "высокий"]

OUT_TERMS: list[str] = [
    "очень низкий",
    "низкий",
    "средний",
    "высокий",
    "очень высокий",
]

OUTPUT_RANGE: tuple[float, float] = (0.0, 100.0)

# Знаков оценки приоритетности в ответе API и в core-service
# (calc_result.priority_score NUMERIC(8,3)); ранг считается по той же оценке
PRIORITY_DECIMALS: int = 3
