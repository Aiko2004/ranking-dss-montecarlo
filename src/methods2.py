"""Дополнительные методы для ответа рецензенту.

Абляции нечёткого вывода (К1):
  * Mamdani с L = 5, 9, 17 выходными термами (уменьшение квантования заключений);
  * Mamdani с равномерными центрами термов вместо FCM;
  * TSK нулевого порядка (произведение, взвешенное среднее) с непрерывными заключениями:
      - по номерам термов (то же правило (4), но без округления),
      - по центрам термов (заключение = смесь взвешенного среднего и минимума центров).
Обучаемые модели (К2, замечание 3):
  * TSK нулевого порядка с заключениями, найденными гребневой регрессией (шаг МНК ANFIS);
  * 2-аддитивный интеграл Шоке, обученный неотрицательной гребневой регрессией;
  * монотонный градиентный бустинг (HistGradientBoosting с monotonic_cst).
Холодный старт (замечание 2):
  * взвешенная сумма и смесь с минимумом при зашумлённых «экспертных» весах.
"""
from __future__ import annotations

import itertools

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor

import methods as M
from fuzzy_priority.variables import uniform_partition


# ------------------------------------------------------------------ Mamdani с L термами
def _out_partition(L):
    return uniform_partition("y", [f"o{i}" for i in range(L)], np.linspace(0, 100, M.RES))


OUTS = {L: _out_partition(L) for L in (5, 9, 17)}


def mamdani_L(mu, ant, cons, L):
    m, n, _ = mu.shape
    B, yu = OUTS[L].grid, OUTS[L].universe
    scores = np.empty(n)
    step = max(20, int(4e6 // (ant.shape[0] * m)))
    for s in range(0, n, step):
        sl = slice(s, s + step)
        fire = np.min(np.stack([mu[j, sl][:, ant[:, j]] for j in range(m)]), 0)
        alpha = np.stack([fire[:, cons == t].max(1) if (cons == t).any() else np.zeros(fire.shape[0])
                          for t in range(L)], 1)
        agg = np.max(np.minimum(alpha[:, :, None], B[None]), 1)
        num = np.trapezoid(agg * yu, yu, axis=1)
        den = np.trapezoid(agg, yu, axis=1)
        scores[sl] = np.where(den > 1e-12, num / np.maximum(den, 1e-12), 50.0)
    return scores


def rules(m, K, w, lam):
    ant = np.array(list(itertools.product(range(K), repeat=m)))
    level = ((1 - lam) * (ant @ w) + lam * ant.min(1)) / (K - 1)  # непрерывное, в [0, 1]
    return ant, level


def fis_mamdani(Z, w, K=3, lam=0.0, L=5, partition="fcm"):
    parts = M.build_partitions(Z, K, partition)
    ant, level = rules(Z.shape[1], K, w, lam)
    cons = np.rint(level * (L - 1)).astype(int)
    return mamdani_L(M.memberships(Z, parts), ant, cons, L)


# ------------------------------------------------------------------ TSK нулевого порядка
def centres(parts):
    """Центр терма = середина ядра (точек с принадлежностью 1)."""
    C = []
    for p in parts:
        g = p.grid
        C.append(np.array([M.U_IN[g[t] >= g[t].max() - 1e-9].mean() for t in range(p.n_terms)]))
    return np.array(C)  # (m, K)


def firing(mu, ant):
    """Произведение принадлежностей: (n, R). При разбиении единицы сумма по правилам = 1."""
    m = mu.shape[0]
    f = np.ones((mu.shape[1], ant.shape[0]))
    for j in range(m):
        f *= mu[j][:, ant[:, j]]
    return f


def tsk0(Z, w, K=3, lam=0.0, conseq="index", partition="fcm"):
    parts = M.build_partitions(Z, K, partition)
    mu = M.memberships(Z, parts)
    ant, level = rules(Z.shape[1], K, w, lam)
    if conseq == "centre":
        C = centres(parts)
        cv = C[np.arange(Z.shape[1])[None, :], ant]  # (R, m) центры термов посылки
        level = (1 - lam) * (cv @ w) + lam * cv.min(1)
    f = firing(mu, ant)
    return (f @ level) / np.maximum(f.sum(1), 1e-12)


# ------------------------------------------------------------------ обучаемые модели
def tsk_learned(Ztr, ytr, Zte, K=3, alpha=1.0):
    """TSK-0: посылки по FCM на обучающих данных, заключения — гребневая регрессия
    по нормированным степеням срабатывания (шаг наименьших квадратов ANFIS)."""
    parts = M.build_partitions(Ztr, K, "fcm")
    ant = np.array(list(itertools.product(range(K), repeat=Ztr.shape[1])))
    ftr = firing(M.memberships(Ztr, parts), ant)
    fte = firing(M.memberships(Zte, parts), ant)
    ftr /= np.maximum(ftr.sum(1, keepdims=True), 1e-12)
    fte /= np.maximum(fte.sum(1, keepdims=True), 1e-12)
    return Ridge(alpha=alpha).fit(ftr, ytr).predict(fte)


def choquet_features(Z):
    m = Z.shape[1]
    cols = [Z]
    for j, k in itertools.combinations(range(m), 2):
        cols.append(np.minimum(Z[:, j], Z[:, k])[:, None])
        cols.append(np.maximum(Z[:, j], Z[:, k])[:, None])
    return np.hstack(cols)


def choquet_learned(Ztr, ytr, Zte, alpha=0.1):
    """2-аддитивный интеграл Шоке в представлении через x_j, min и max пар
    с неотрицательными коэффициентами (монотонная модель)."""
    mdl = Ridge(alpha=alpha, positive=True).fit(choquet_features(Ztr), ytr)
    return mdl.predict(choquet_features(Zte))


def mono_gbm(Ztr, ytr, Zte, seed=0):
    mdl = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_leaf_nodes=4,
                                        min_samples_leaf=5, monotonic_cst=[1] * Ztr.shape[1],
                                        early_stopping=False, random_state=seed)
    return mdl.fit(Ztr, ytr).predict(Zte)


def expert_weights(w, sigma, rng):
    v = w * np.exp(rng.normal(0, sigma, size=len(w)))
    return v / v.sum()
