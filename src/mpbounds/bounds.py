"""Bounds on theta_d(h) = E[Y_{t,h}(d) - Y_{t,h}(0)] when overlap fails structurally through the policy menu X_t.

Decompose theta_d = sum_x P(X=x) theta_{d,x}. For each menu x:
  {d,0} in x      : theta_{d,x} point-identified (IPW within x)
  0 in x, d not   : E[Y(0)|x] identified, E[Y(d)|x] in [K1,K2]  -> theta_{d,x} in [K1 - m0, K2 - m0]
  d in x, 0 not   : E[Y(d)|x] identified, E[Y(0)|x] in [K1,K2]  -> theta_{d,x} in [md - K2, md - K1]
  neither         : theta_{d,x} in [K1 - K2, K2 - K1]
Summing gives the sharp worst-case identified set. Under the sensitivity assumption
  |theta_{d,x} - theta_{d,A}| <= lam  for menus outside the overlap region A = {x : {d,0} in x}
the set shrinks to theta_A +/- lam * (1 - P(A)) (intersected with the worst case).
"""
import numpy as np
import pandas as pd

from .data import LETTER, MENUS


def hajek_mean(y: np.ndarray, treated: np.ndarray, p: np.ndarray) -> float:
    w = treated / p
    return float((w * y).sum() / w.sum()) if w.sum() > 0 else np.nan  # nan: no units with this decision in the cell


def cell_quantities(s: pd.DataFrame, p: pd.DataFrame, y: str, d: int) -> pd.DataFrame:
    """Per-menu shares and identified conditional means for policy d versus 0."""
    rows = []
    n = len(s)
    for x in MENUS:
        m = (s["menu"] == x).to_numpy()
        yx, Dx, px = s.loc[m, y].to_numpy(), s.loc[m, "D"].to_numpy(), p.loc[m]
        has_d, has_0 = LETTER[d] in x, "U" in x
        m_d = hajek_mean(yx, Dx == d, px[d].to_numpy()) if has_d and m.any() else np.nan
        m_0 = hajek_mean(yx, Dx == 0, px[0].to_numpy()) if has_0 and m.any() else np.nan
        rows.append({"menu": x, "share": m.sum() / n, "n": int(m.sum()), "n_d": int((Dx == d).sum()),
                     "has_d": has_d, "has_0": has_0, "m_d": m_d, "m_0": m_0})
    return pd.DataFrame(rows).set_index("menu")


def identified_set(cells: pd.DataFrame, K1: float, K2: float, lam: float | None = None) -> dict:
    A = cells.has_d & cells.has_0
    P_A = cells.loc[A, "share"].sum()
    theta_x = (cells.m_d - cells.m_0).where(A)
    theta_A = float((theta_x[A] * cells.share[A]).sum() / P_A)

    lo_x = pd.Series(np.select([A, cells.has_0, cells.has_d], [theta_x, K1 - cells.m_0, cells.m_d - K2], K1 - K2), cells.index)
    hi_x = pd.Series(np.select([A, cells.has_0, cells.has_d], [theta_x, K2 - cells.m_0, cells.m_d - K1], K2 - K1), cells.index)
    if lam is not None:
        lo_x = lo_x.where(A, np.maximum(lo_x, theta_A - lam))
        hi_x = hi_x.where(A, np.minimum(hi_x, theta_A + lam))
    return {"theta_A": theta_A, "P_A": float(P_A),
            "lo": float((lo_x * cells.share).sum()), "hi": float((hi_x * cells.share).sum())}


def outcome_support(panel: pd.DataFrame, y: str) -> tuple[float, float]:
    """Assumed support [K1, K2] of the potential outcome: the range of the realised h-month change over the
    full 1989-2010 panel (which includes the 2008-09 recession)."""
    v = panel[y].dropna()
    return float(v.min()), float(v.max())


def breakdown_lambda(theta_A: float, P_A: float) -> float:
    """Smallest lam at which theta_A +/- lam (1 - P_A) contains zero."""
    return abs(theta_A) / (1 - P_A)
