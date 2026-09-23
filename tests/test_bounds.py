"""Synthetic check: with known menu-specific effects the estimator recovers theta_A and the identified set
contains the true ATE."""
import numpy as np
import pandas as pd
import pytest

from mpbounds import bounds, pscore
from mpbounds.data import LETTER, MENUS

TAU = {"EU": 1.0, "UT": 2.0, "EUT": 3.0, "E": 0.5, "U": -1.0, "T": 4.0}  # true E[Y(1)-Y(0) | x]; also used for d=-1
SHARES = [0.05, 0.05, 0.1, 0.25, 0.3, 0.25]


def simulate(n=40_000, seed=1):
    rng = np.random.default_rng(seed)
    menu = rng.choice(MENUS, size=n, p=SHARES)
    z = rng.normal(size=n)
    params = np.array([0.8, -0.6, np.log(1.2)])  # beta, c1, log(c2 - c1)
    mask = np.array([[LETTER[d] in m for d in (-1, 0, 1)] for m in menu], float)
    p = pscore._probs(params, z[:, None], mask)
    D = np.array([rng.choice([-1, 0, 1], p=row) for row in p])
    tau = np.array([TAU[m] for m in menu])
    y = 0.5 * z + tau * (D != 0) + rng.normal(scale=0.5, size=n)  # Y(0) depends on z (confounding); Y(+-1) = Y(0) + tau_x
    df = pd.DataFrame({"menu": menu, "D": D, "z": z, "y": y})
    return df, pd.DataFrame(p, columns=[-1, 0, 1])


@pytest.fixture(scope="module")
def sim():
    return simulate()


@pytest.mark.parametrize("d", [1, -1])
def test_theta_A_recovered_with_true_pscore(sim, d):
    df, p = sim
    cells = bounds.cell_quantities(df, p, "y", d)
    r = bounds.identified_set(cells, K1=-10, K2=10)
    A = [x for x in MENUS if LETTER[d] in x and "U" in x]
    truth = sum(TAU[x] * cells.share[x] for x in A) / cells.share[A].sum()
    assert r["theta_A"] == pytest.approx(truth, abs=0.05)


def test_identified_set_contains_ate_and_has_sharp_width(sim):
    df, p = sim
    cells = bounds.cell_quantities(df, p, "y", 1)
    K1, K2 = df["y"].min(), df["y"].max()
    r = bounds.identified_set(cells, K1, K2)
    ate = sum(TAU[x] * cells.share[x] for x in MENUS)
    assert r["lo"] <= ate <= r["hi"]
    s = cells.share  # d=+1: one-sided cells are {U},{E,U} (0 only) and {T} (d only); {E} has neither
    width = (K2 - K1) * (s["U"] + s["EU"] + s["T"]) + 2 * (K2 - K1) * s["E"]
    assert r["hi"] - r["lo"] == pytest.approx(width, rel=1e-9)


def test_lambda_zero_collapses_to_theta_A(sim):
    df, p = sim
    cells = bounds.cell_quantities(df, p, "y", 1)
    r = bounds.identified_set(cells, -50, 50, lam=0.0)
    assert r["lo"] == pytest.approx(r["theta_A"]) and r["hi"] == pytest.approx(r["theta_A"])


def test_estimated_pscore_recovers_theta_A(sim):
    df, _ = sim
    df = df.iloc[:8000]
    p = pscore.fit(df, ["z"]).predict(df)
    r = bounds.identified_set(bounds.cell_quantities(df, p, "y", 1), -10, 10)
    cells = bounds.cell_quantities(df, p, "y", 1)
    truth = (TAU["UT"] * cells.share["UT"] + TAU["EUT"] * cells.share["EUT"]) / (cells.share["UT"] + cells.share["EUT"])
    assert r["theta_A"] == pytest.approx(truth, abs=0.1)
