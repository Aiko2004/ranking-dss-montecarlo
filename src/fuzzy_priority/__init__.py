"""Система определения приоритетности строительства альтернативных зданий
на основе нечёткой логики (модель Мамдани)."""

from .config import DEFAULT_CRITERIA, Criterion
from .mamdani import MamdaniFIS
from .pipeline import PriorityModel
from .rules import Rule, generate_monotone_rules
from .variables import LinguisticVariable

__version__ = "0.1.0"
__all__ = [
    "Criterion",
    "DEFAULT_CRITERIA",
    "LinguisticVariable",
    "MamdaniFIS",
    "PriorityModel",
    "Rule",
    "generate_monotone_rules",
]
