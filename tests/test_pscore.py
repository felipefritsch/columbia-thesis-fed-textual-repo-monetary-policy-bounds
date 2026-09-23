import numpy as np
import pytest
from statsmodels.miscmodels.ordinal_model import OrderedModel

from mpbounds import pscore
from mpbounds.data import LETTER, analysis_sample, build_panel


@pytest.fixture(scope="module")
def sample():
    return analysis_sample(build_panel())


def test_unconstrained_matches_statsmodels(sample):
    cov = pscore.SPECS["macro+fff"]
    ours = pscore.fit(sample, cov, menu_constrained=False)
    ref = OrderedModel(sample["D"], sample[cov], distr="probit").fit(method="bfgs", disp=0, maxiter=5000)
    assert ours.llf == pytest.approx(ref.llf, abs=1e-3)


def test_structural_zeros(sample):
    p = pscore.fit(sample, pscore.SPECS["greenbook"]).predict(sample)
    assert np.allclose(p.sum(axis=1), 1)
    for d in (-1, 0, 1):
        outside = ~sample["menu"].str.contains(LETTER[d])
        assert (p.loc[outside, d] == 0).all()


def test_menu_constraint_improves_fit(sample):
    cov = pscore.SPECS["macro+fff"]
    assert pscore.fit(sample, cov, True).llf > pscore.fit(sample, cov, False).llf + 10
