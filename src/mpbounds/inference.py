"""Moving-block bootstrap over meetings (re-estimating the propensity score in every draw) and
Imbens-Manski (2004) confidence intervals for the partially identified parameter."""
import numpy as np
import pandas as pd
from scipy import optimize, stats

from . import bounds, pscore


def estimate_all(s: pd.DataFrame, covariates: list, K: dict, outcomes, horizons, lam=None) -> pd.DataFrame:
    """Point estimates for every (outcome, policy, horizon). K maps outcome column -> (K1, K2)."""
    p = pscore.fit(s, covariates).predict(s)
    rows = []
    for y in outcomes:
        for d in (1, -1):
            for h in horizons:
                col = f"{y}_h{h}"
                cells = bounds.cell_quantities(s, p, col, d)
                r = bounds.identified_set(cells, *K[col])
                row = {"outcome": y, "policy": d, "h": h, **r}
                if lam is not None:
                    rl = bounds.identified_set(cells, *K[col], lam=lam)
                    row.update(lo_lam=rl["lo"], hi_lam=rl["hi"])
                rows.append(row)
    return pd.DataFrame(rows).set_index(["outcome", "policy", "h"])


def block_bootstrap(s: pd.DataFrame, covariates: list, K: dict, outcomes, horizons,
                    B: int = 499, block: int = 8, seed: int = 0) -> tuple[pd.DataFrame, int]:
    """Returns the stacked bootstrap estimates and the number of failed draws (non-convergence or an empty cell)."""
    rng = np.random.default_rng(seed)
    n = len(s)
    starts = np.arange(n - block + 1)
    draws, failed = [], 0
    for b in range(B):
        idx = np.concatenate([np.arange(k, k + block) for k in rng.choice(starts, int(np.ceil(n / block)))])[:n]
        sb = s.iloc[idx]
        try:
            est = estimate_all(sb, covariates, K, outcomes, horizons)
        except (RuntimeError, np.linalg.LinAlgError):
            failed += 1
            continue
        if est[["theta_A", "lo", "hi"]].isna().any().any():
            failed += 1
            continue
        draws.append(est.assign(draw=b))
    return pd.concat(draws), failed


def imbens_manski(lo, hi, se_lo, se_hi, level=0.90):
    """CI covering the true parameter (not the whole set) with probability `level`."""
    sig = max(se_lo, se_hi)
    delta = hi - lo
    f = lambda c: stats.norm.cdf(c + delta / sig) - stats.norm.cdf(-c) - level
    c = optimize.brentq(f, 0, 10)
    return lo - c * se_lo, hi + c * se_hi


def summarise(point: pd.DataFrame, draws: pd.DataFrame, level=0.90) -> pd.DataFrame:
    a = (1 - level) / 2
    g = draws.groupby(level=["outcome", "policy", "h"])
    se = g[["theta_A", "lo", "hi"]].std()
    q = g["theta_A"].quantile([a, 1 - a]).unstack()
    out = point.copy()
    out["se_theta_A"] = se["theta_A"]
    # basic (reverse-percentile) bootstrap interval for theta_A
    out["ci_theta_lo"] = 2 * point["theta_A"] - q[1 - a]
    out["ci_theta_hi"] = 2 * point["theta_A"] - q[a]
    im = [imbens_manski(r.lo, r.hi, se.loc[i, "lo"], se.loc[i, "hi"], level) for i, r in point.iterrows()]
    out["im_lo"], out["im_hi"] = zip(*im)
    return out
