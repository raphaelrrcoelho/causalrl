"""Strong causal-inference contenders for the real-data demos: IPW, doubly-robust AIPW, and
propensity-score stratification.

These are the methods a competent practitioner actually uses (and what DoWhy / EconML compute under
the hood), so the causal agent is benchmarked against *real* contenders rather than a naive
difference-in-means strawman. Needs scikit-learn. This is example-only comparison code, not a
causalrl library API -- causalrl's value-add is the decision + certificate layer, not the ATE
estimators themselves (pair it with DoWhy/EconML for those).
"""

from __future__ import annotations

import itertools

import numpy as np


def _propensity(x: np.ndarray, a: np.ndarray) -> np.ndarray:
    """P(A=1 | X) via standardized logistic regression, clipped away from 0/1 for stability."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    xs = StandardScaler().fit_transform(x)
    e = LogisticRegression(max_iter=1000).fit(xs, a).predict_proba(xs)[:, 1]
    return np.clip(e, 0.02, 0.98)


def _outcome(x: np.ndarray, a: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-action ridge outcome models, each predicted on ALL rows: returns (mu0, mu1)."""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    xs = StandardScaler().fit_transform(x)
    predictions = []
    for action in (0, 1):
        mask = a == action
        predictions.append(Ridge().fit(xs[mask], y[mask]).predict(xs))
    return predictions[0], predictions[1]


def ipw_ate(x: np.ndarray, a: np.ndarray, y: np.ndarray) -> float:
    """Inverse-propensity-weighted ATE (Hajek / self-normalized)."""
    e = _propensity(x, a)
    w1, w0 = a / e, (1 - a) / (1 - e)
    return float((w1 * y).sum() / w1.sum() - (w0 * y).sum() / w0.sum())


def aipw_ate(x: np.ndarray, a: np.ndarray, y: np.ndarray) -> float:
    """Doubly-robust (augmented IPW) ATE — the efficient, misspecification-robust estimator."""
    e = _propensity(x, a)
    mu0, mu1 = _outcome(x, a, y)
    dr1 = mu1 + a * (y - mu1) / e
    dr0 = mu0 + (1 - a) * (y - mu0) / (1 - e)
    return float((dr1 - dr0).mean())


def propensity_strata_ate(x: np.ndarray, a: np.ndarray, y: np.ndarray, n_strata: int = 5) -> float:
    """Propensity-score stratification ATE (the Dehejia-Wahba subclassification approach)."""
    e = _propensity(x, a)
    edges = np.quantile(e, np.linspace(0.0, 1.0, n_strata + 1))
    edges[0] -= 1e-9
    edges[-1] += 1e-9
    total, total_weight = 0.0, 0.0
    for lo, hi in itertools.pairwise(edges):
        mask = (e > lo) & (e <= hi)
        a_s, y_s = a[mask], y[mask]
        if (a_s == 1).sum() == 0 or (a_s == 0).sum() == 0:
            continue
        weight = float(mask.sum())
        total += weight * (y_s[a_s == 1].mean() - y_s[a_s == 0].mean())
        total_weight += weight
    return float(total / total_weight) if total_weight else float("nan")


def slearner_cate(x: np.ndarray, a: np.ndarray, y: np.ndarray) -> np.ndarray:
    """S-learner CATE: one model with treatment as a feature, differenced at a=1 vs a=0."""
    from sklearn.ensemble import GradientBoostingRegressor

    model = GradientBoostingRegressor(random_state=0).fit(np.column_stack([x, a]), y)
    ones, zeros = np.ones(len(x)), np.zeros(len(x))
    return model.predict(np.column_stack([x, ones])) - model.predict(np.column_stack([x, zeros]))


def xlearner_cate(x: np.ndarray, a: np.ndarray, y: np.ndarray) -> np.ndarray:
    """X-learner CATE (Kunzel et al.): cross-fitted imputed effects, propensity-weighted."""
    from sklearn.ensemble import GradientBoostingRegressor as GBR

    mu0 = GBR(random_state=0).fit(x[a == 0], y[a == 0])
    mu1 = GBR(random_state=0).fit(x[a == 1], y[a == 1])
    tau1 = GBR(random_state=0).fit(x[a == 1], y[a == 1] - mu0.predict(x[a == 1]))
    tau0 = GBR(random_state=0).fit(x[a == 0], mu1.predict(x[a == 0]) - y[a == 0])
    e = _propensity(x, a)
    return e * tau0.predict(x) + (1.0 - e) * tau1.predict(x)
