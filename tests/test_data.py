import pandas as pd
import pytest

from mpbounds.data import LETTER, MENUS, analysis_sample, build_panel


@pytest.fixture(scope="module")
def sample():
    return analysis_sample(build_panel())


def test_sample_size(sample):
    assert len(sample) == 133


def test_menu_counts(sample):
    ct = pd.crosstab(sample["menu"], sample["D"]).reindex(index=MENUS, fill_value=0)
    assert ct.loc["UT", 1] == 19 and ct.loc["EUT", 1] == 1 and ct.loc["EU", -1] == 17


def test_decision_always_in_menu(sample):
    assert all(LETTER[d] in m for d, m in zip(sample["D"], sample["menu"]))


def test_covariates_complete(sample):
    cols = ["fff", "infl_l1", "unemp_l1", "ip_l1", "gb_gdp_q0", "gb_defl_q1", "gb_unemp_q0", "ip_h24", "pce_h24", "unrate_h24"]
    assert sample[cols].notna().all().all()


def test_ip_growth_alignment():
    """IP growth from the text panel must line up with growth computed from the AJK IP index."""
    p = build_panel()
    akj = 100 * (p["ip"] / p["ip"].shift(12) - 1)
    both = pd.concat([akj, p["ip_l1"].shift(-1)], axis=1).dropna()
    assert both.corr().iloc[0, 1] > 0.99
