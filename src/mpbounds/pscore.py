"""Propensity score for D in {-1, 0, +1}: an ordered probit whose probabilities are renormalised over the
Bluebook menu, so P(D=d | X=x, Z=z) = 0 exactly when d is not in x:

    pi_d(z) = ordered-probit cell probability,   p(d | x, z) = pi_d(z) 1{d in x} / sum_{d' in x} pi_d'(z)

With menu_constrained=False every meeting is treated as having the full menu {E,U,T} (standard ordered probit).
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import optimize, stats

from .data import LETTER

DECISIONS = (-1, 0, 1)

SPECS = {
    "macro": ["infl_l1", "infl_l2", "unemp_l1", "unemp_l2", "ip_l1", "ip_l2"],
    "macro+fff": ["fff", "infl_l1", "infl_l2", "unemp_l1", "unemp_l2", "ip_l1", "ip_l2"],
    # Romer-Romer (2004) information set: the staff forecasts the FOMC saw, plus the market surprise
    "greenbook": ["fff", "gb_gdp_q0", "gb_gdp_q1", "gb_defl_q0", "gb_defl_q1", "gb_unemp_q0"],
}
LABELS = {
    "fff": "Fed funds futures surprise", "infl_l1": "Inflation, lag 1", "infl_l2": "Inflation, lag 2",
    "unemp_l1": r"$\Delta$ Unemployment, lag 1", "unemp_l2": r"$\Delta$ Unemployment, lag 2",
    "ip_l1": "IP growth, lag 1", "ip_l2": "IP growth, lag 2",
    "gb_gdp_q0": "GB real GDP growth, $q_0$", "gb_gdp_q1": "GB real GDP growth, $q_1$",
    "gb_defl_q0": "GB inflation, $q_0$", "gb_defl_q1": "GB inflation, $q_1$", "gb_unemp_q0": "GB unemployment, $q_0$",
}


def _menu_mask(menus: pd.Series) -> np.ndarray:
    return np.array([[LETTER[d] in m for d in DECISIONS] for m in menus], dtype=float)


def _cell_probs(params: np.ndarray, Z: np.ndarray) -> np.ndarray:
    k = Z.shape[1]
    beta, c1, c2 = params[:k], params[k], params[k] + np.exp(params[k + 1])
    xb = Z @ beta
    F1, F2 = stats.norm.cdf(c1 - xb), stats.norm.cdf(c2 - xb)
    return np.clip(np.column_stack([F1, F2 - F1, 1 - F2]), 1e-300, None)


def _probs(params, Z, mask):
    p = _cell_probs(params, Z) * mask
    return p / p.sum(axis=1, keepdims=True)


@dataclass
class PScore:
    covariates: list
    menu_constrained: bool
    params: np.ndarray
    cov: np.ndarray
    llf: float
    nobs: int

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        mask = _menu_mask(df["menu"]) if self.menu_constrained else np.ones((len(df), 3))
        p = _probs(self.params, df[self.covariates].to_numpy(float), mask)
        return pd.DataFrame(p, index=df.index, columns=list(DECISIONS))

    def ame(self, df: pd.DataFrame, decision: int = 1, eps: float = 1e-5) -> pd.DataFrame:
        """Average marginal effect of each covariate on P(D=decision), with delta-method standard errors."""
        mask = _menu_mask(df["menu"]) if self.menu_constrained else np.ones((len(df), 3))
        Z = df[self.covariates].to_numpy(float)
        j = DECISIONS.index(decision)

        def ame_vec(params):
            base = _probs(params, Z, mask)[:, j]
            out = []
            for i in range(Z.shape[1]):
                Zi = Z.copy()
                Zi[:, i] += eps
                out.append(np.mean((_probs(params, Zi, mask)[:, j] - base) / eps))
            return np.array(out)

        est = ame_vec(self.params)
        G = np.column_stack([(ame_vec(self.params + e) - est) / 1e-5 for e in np.eye(len(self.params)) * 1e-5])
        se = np.sqrt(np.diag(G @ self.cov @ G.T))
        return pd.DataFrame({"ame": est, "se": se}, index=self.covariates)


def fit(df: pd.DataFrame, covariates: list, menu_constrained: bool = True) -> PScore:
    Z = df[covariates].to_numpy(float)
    y = np.array([DECISIONS.index(d) for d in df["D"]])
    mask = _menu_mask(df["menu"]) if menu_constrained else np.ones((len(df), 3))

    def nll(params):
        return -np.log(_probs(params, Z, mask)[np.arange(len(y)), y]).sum()

    x0 = np.r_[np.zeros(Z.shape[1]), -0.5, 0.0]
    res = optimize.minimize(nll, x0, method="BFGS", options={"maxiter": 10_000, "gtol": 1e-7})
    if not res.success and res.status != 2:  # status 2 = precision loss at optimum, harmless here
        raise RuntimeError(f"propensity score did not converge: {res.message}")
    H = _hessian(nll, res.x)
    return PScore(covariates, menu_constrained, res.x, np.linalg.pinv(H), -res.fun, len(y))


def _hessian(f, x, h=1e-4):
    n = len(x)
    H = np.empty((n, n))
    I = np.eye(n) * h
    for i in range(n):
        for j in range(i, n):
            H[i, j] = H[j, i] = (f(x + I[i] + I[j]) - f(x + I[i] - I[j]) - f(x - I[i] + I[j]) + f(x - I[i] - I[j])) / (4 * h * h)
    return H
