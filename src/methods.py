"""Методы ранжирования для исследования Монте-Карло.

Без учителя (типичный модуль СППР): WSM, TOPSIS, нечёткий вывод Мамдани
с параметрами из данных (как в fis-engine).
С учителем (обучение на прошлых решениях ЛПР): Ridge, случайный лес,
градиентный бустинг, нечёткий вывод с правилами Ванга–Менделя.
"""
from __future__ import annotations

import sys
import numpy as np

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from fuzzy_priority.variables import fcm_partition, quantile_partition, uniform_partition  # noqa: E402
from fuzzy_priority.rules import generate_monotone_rules  # noqa: E402
from fuzzy_priority.weighting import entropy_weights, critic_weights, equal_weights  # noqa: E402

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

WEIGHT = {"entropy": entropy_weights, "critic": critic_weights, "equal": equal_weights}
RES = 201
U_IN = np.linspace(0, 1, RES)
OUT = uniform_partition("y", [f"o{i}" for i in range(5)], np.linspace(0, 100, RES))


# ------------------------------------------------------------------ utils
def minmax(X, lo=None, hi=None):
    X = np.asarray(X, float)
    lo = X.min(0) if lo is None else lo
    hi = X.max(0) if hi is None else hi
    return np.clip((X - lo) / np.where(hi - lo > 0, hi - lo, 1), 0, 1), lo, hi


def wsm(Z, w):
    return Z @ w


def topsis(Z, w):
    V = Z / np.sqrt((Z ** 2).sum(0) + 1e-12) * w
    dp = np.sqrt(((V - V.max(0)) ** 2).sum(1))
    dn = np.sqrt(((V - V.min(0)) ** 2).sum(1))
    return dn / (dp + dn + 1e-12)


# ------------------------------------------------------------------ fuzzy core
def build_partitions(Z, K=3, partition="fcm", seed=0):
    names = [f"t{i}" for i in range(K)]
    out = []
    for j in range(Z.shape[1]):
        col = Z[:, j]
        if partition == "fcm":
            try:
                out.append(fcm_partition("x", col, names, U_IN, seed=seed))
            except ValueError:  # совпавшие центры — равномерное разбиение
                out.append(uniform_partition("x", names, U_IN))
        elif partition == "quantile":
            out.append(quantile_partition("x", col, names, U_IN))
        else:
            out.append(uniform_partition("x", names, U_IN))
    return out


def memberships(Z, parts):
    # (m, n, K)
    return np.stack([np.stack([np.interp(Z[:, j], U_IN, p.grid[t]) for t in range(p.n_terms)], 1)
                     for j, p in enumerate(parts)])


def mamdani(mu, ant, cons, n_out=5):
    """Векторизованный min–max вывод с дефаззификацией по центру тяжести."""
    m, n, _ = mu.shape
    B, yu = OUT.grid, OUT.universe
    scores = np.empty(n)
    step = max(20, int(4e6 // (ant.shape[0] * m)))
    for s in range(0, n, step):
        sl = slice(s, s + step)
        fire = np.min(np.stack([mu[j, sl][:, ant[:, j]] for j in range(m)]), 0)  # (b, R)
        alpha = np.stack([fire[:, cons == t].max(1) if (cons == t).any() else np.zeros(fire.shape[0])
                          for t in range(n_out)], 1)
        agg = np.max(np.minimum(alpha[:, :, None], B[None]), 1)
        num = np.trapezoid(agg * yu, yu, axis=1)
        den = np.trapezoid(agg, yu, axis=1)
        scores[sl] = np.where(den > 1e-12, num / np.maximum(den, 1e-12), 50.0)
    return scores


def andness_rules(m, K, w, lam, n_out=5):
    """База правил с регулируемой некомпенсаторностью.

    Индекс заключения = (1-λ)·взвешенное среднее + λ·минимум индексов термов посылки.
    λ = 0 — компенсаторная (монотонная генерация из fis-engine),
    λ = 1 — «слабое звено»: приоритет определяется худшим критерием.
    Правила монотонны при любом λ ∈ [0, 1].
    """
    import itertools
    ant = np.array(list(itertools.product(range(K), repeat=m)))
    level = (1 - lam) * (ant @ w) + lam * ant.min(1)
    cons = np.rint(level / (K - 1) * (n_out - 1)).astype(int)
    return ant, cons


def fis_unsupervised(Z, weighting="entropy", partition="fcm", K=3, lam=0.0):
    """Нечёткий вывод с параметрами из данных: ФП по FCM, веса по энтропии, правила с andness λ."""
    w = WEIGHT[weighting](Z)
    parts = build_partitions(Z, K, partition)
    ant, cons = andness_rules(Z.shape[1], K, w / w.sum(), lam)
    return mamdani(memberships(Z, parts), ant, cons)


def owa_crisp(Z, w, lam):
    """Чёткий аналог: смесь взвешенного среднего и минимума (оператор с andness λ)."""
    return (1 - lam) * (Z @ w) + lam * Z.min(1)


def fis_wang_mendel(Ztr, ytr, Zte, K=3, partition="fcm"):
    """Нечёткий вывод с правилами Ванга–Менделя (1992), обученными на прошлых решениях.

    Каждый обучающий пример порождает правило: термы с наибольшей принадлежностью
    по входам -> терм выхода по метке. Конфликты решаются по степени правила.
    Непокрытые посылки заполняются монотонным генератором (равные веса).
    """
    m = Ztr.shape[1]
    parts = build_partitions(Ztr, K, partition)
    mu_tr = memberships(Ztr, parts)  # (m, n, K)
    idx = mu_tr.argmax(2).T  # (n, m)
    deg = mu_tr.max(2).prod(0)  # (n,)
    # метка -> терм выхода: квантили обучающих меток на 5 уровней
    q = np.quantile(ytr, [0.2, 0.4, 0.6, 0.8])
    yterm = np.searchsorted(q, ytr, side="right")
    # заключение правила = среднее термов меток, взвешенное степенью примера
    # (вместо «победителя» по максимальной степени: на дискретных данных
    # степени часто равны, и выбор победителя становится произвольным)
    acc = {}
    for a, d, t in zip(map(tuple, idx), deg, yterm):
        sw, swt = acc.get(a, (0.0, 0.0))
        acc[a] = (sw + d, swt + d * t)
    base = generate_monotone_rules(m, K, 5, None)
    ant = np.array([r.antecedent for r in base])
    cons = np.array([int(round(acc[tuple(a)][1] / acc[tuple(a)][0])) if tuple(a) in acc and acc[tuple(a)][0] > 0
                     else r.consequent for a, r in zip(ant, base)])
    return mamdani(memberships(Zte, parts), ant, cons)


# ------------------------------------------------------------------ supervised ML
def ml_models(seed=0):
    return {
        "Ridge": Ridge(alpha=1.0),
        "RandomForest": RandomForestRegressor(n_estimators=200, min_samples_leaf=2, n_jobs=1, random_state=seed),
        # классический бустинг с настройками для малых выборок (≥50 примеров)
        "GradBoost": GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=2,
                                               min_samples_leaf=3, subsample=0.8, random_state=seed),
    }
