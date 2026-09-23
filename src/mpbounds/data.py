"""Assemble the monthly analysis panel from the raw inputs in data/raw/.

Sources (see data/README.md for provenance):
  akj_outcomes.csv          Angrist-Jorda-Kuersteiner (2018) monthly outcomes: FFED, PCE index, IP index, UNRATE
  akj_pscore_vars.xls       AJK propensity-score variables (target change, FFF surprise, lagged inflation/unemployment)
  bluebook_policy_menus.csv Policy alternatives per FOMC meeting, extracted from Bluebooks
  fomc_text_panel.csv       Monthly macro panel from the FOMCTextAnalysis project (used for IP growth + Figures 1-2)
  greenbook_forecasts.csv   Staff (Greenbook) forecasts per FOMC meeting (Philadelphia Fed Greenbook data set)
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"

SAMPLE_START = "1989-08"  # first month in the AJK files
SAMPLE_END = "2006-01"    # end of the Greenspan chairmanship
HORIZONS = range(1, 25)
MENUS = ["E", "U", "T", "EU", "UT", "EUT"]  # canonical order; letters: E=ease(-1), U=unchanged(0), T=tighten(+1)
LETTER = {-1: "E", 0: "U", 1: "T"}
OUTCOMES = {"ip": "Industrial production", "unrate": "Unemployment rate", "pce": "PCE price index"}


def _monthly(dates) -> pd.PeriodIndex:
    return pd.PeriodIndex(pd.to_datetime(dates), freq="M")


def load_akj() -> pd.DataFrame:
    out = pd.read_csv(RAW / "akj_outcomes.csv")
    out.index = _monthly(pd.to_datetime(out[["Year", "Month", "Day"]]))
    out = out.rename(columns={"FFED": "ffed", "PCEH": "pce", "IP": "ip", "UNRATE": "unrate"})[["ffed", "pce", "ip", "unrate"]]

    ps = pd.read_excel(RAW / "akj_pscore_vars.xls")
    ps.index = _monthly(ps["Date"])
    ps = ps.rename(columns={
        "Target Change": "target_change", "DEAJKold": "fff", "FOMC Meetings": "meeting",
        "PCEH": "infl_l1", "PCEH(-1)": "infl_l2", "UNRATE": "unemp_l1", "UNRATE(-1)": "unemp_l2",
    })[["target_change", "fff", "meeting", "infl_l1", "infl_l2", "unemp_l1", "unemp_l2"]]
    return out.join(ps, how="inner")


def load_menus() -> pd.Series:
    """Menu of policy alternatives per month, as a string of letters in {E,U,T} (e.g. 'UT')."""
    m = pd.read_csv(RAW / "bluebook_policy_menus.csv", index_col=0)
    m = m[m["date"] != "0"]  # three malformed trailing rows in the source file
    m.index = _monthly(m["date"])
    assert not m.index.duplicated().any(), "at most one scheduled meeting per month"
    menu = m.apply(lambda r: "".join(k for k, c in [("E", "d_dec"), ("U", "d_unc"), ("T", "d_inc")] if r[c] == 1), axis=1)
    return menu.rename("menu")


def load_text_panel() -> pd.DataFrame:
    """FOMCTextAnalysis monthly panel. Dates in the file are m/d/yy (ambiguous century); rows run Jan 1920 - Aug 2019."""
    p = pd.read_csv(RAW / "fomc_text_panel.csv", usecols=["INDPRO_PC1", "PCEPI_PCA", "PCEPI"])
    p.index = pd.period_range("1920-01", periods=len(p), freq="M")
    return p.rename(columns={"INDPRO_PC1": "ip_yoy", "PCEPI_PCA": "pce_ann", "PCEPI": "pce_index"})


GREENBOOK_VARS = {"gRGDP": "gdp", "gPGDP": "defl", "UNEMP": "unemp"}


def load_greenbook() -> pd.DataFrame:
    """Romer-Romer (2004) style forecast covariates, one row per Greenbook (indexed by its release date):
    current-quarter (q0) and next-quarter (q1) forecasts of real GDP growth and GDP-deflator inflation, and the
    current-quarter unemployment forecast. Quarters are relative to the Greenbook date."""
    g = pd.read_csv(RAW / "greenbook_forecasts.csv")
    g = g[g["macro_variable"].isin(GREENBOOK_VARS)]
    gb_date = pd.to_datetime(g["meeting_date"].astype(str), format="%Y%m%d")  # the file's "meeting_date" is the Greenbook date
    q_gb = gb_date.dt.year * 4 + gb_date.dt.quarter - 1
    year, q = np.divmod(np.round(g["forecast_date"] * 10).astype(int), 10)
    g = g.assign(gb_date=gb_date, rel=(year * 4 + q - 1) - q_gb)
    g = g[g["rel"].isin([0, 1])]
    wide = g.pivot_table(index="gb_date", columns=["macro_variable", "rel"], values="projection", aggfunc="last")
    wide.columns = [f"gb_{GREENBOOK_VARS[v]}_q{r}" for v, r in wide.columns]
    return wide[["gb_gdp_q0", "gb_gdp_q1", "gb_defl_q0", "gb_defl_q1", "gb_unemp_q0"]].sort_index()


def greenbook_by_meeting() -> pd.DataFrame:
    """Attach to each FOMC meeting the latest Greenbook released at most 3 weeks before it; index = meeting month."""
    m = pd.read_csv(RAW / "bluebook_policy_menus.csv", index_col=0)
    meetings = pd.DataFrame({"meeting_date": pd.to_datetime(m.loc[m["date"] != "0", "date"])}).sort_values("meeting_date")
    gb = load_greenbook().reset_index()
    out = pd.merge_asof(meetings, gb, left_on="meeting_date", right_on="gb_date", direction="backward",
                        tolerance=pd.Timedelta(days=21))
    out.index = _monthly(out["meeting_date"])
    return out.drop(columns=["meeting_date", "gb_date"])


def build_panel() -> pd.DataFrame:
    """Full monthly panel (1989-08 .. 2010-12) with treatment, menu, covariates and cumulative outcomes."""
    df = load_akj()
    text = load_text_panel()
    # IP growth known at decision time (year-on-year, lagged). The 2020 thesis used the contemporaneous IP *level*,
    # which is trending and not observed when the FOMC meets. Lags are taken on the long series before joining.
    text["ip_l1"], text["ip_l2"] = text["ip_yoy"].shift(1), text["ip_yoy"].shift(2)
    df = df.join(load_menus()).join(text[["ip_l1", "ip_l2"]]).join(greenbook_by_meeting())
    df["menu"] = df["menu"].fillna("")
    df["D"] = np.sign(df["target_change"]).astype(int)
    for letter, col in [("E", "menu_E"), ("U", "menu_U"), ("T", "menu_T")]:
        df[col] = df["menu"].str.contains(letter).astype(float)
    # Cumulative outcomes from the decision month t to t+h (percent for logs, percentage points for unemployment)
    logs = {"ip": 100 * np.log(df["ip"]), "pce": 100 * np.log(df["pce"]), "unrate": df["unrate"]}
    for name, s in logs.items():
        for h in HORIZONS:
            df[f"{name}_h{h}"] = s.shift(-h) - s
    return df


def analysis_sample(df: pd.DataFrame) -> pd.DataFrame:
    """Scheduled-meeting months in the Greenspan window that have a Bluebook menu."""
    s = df.loc[SAMPLE_START:SAMPLE_END]
    s = s[s["menu"] != ""].copy()
    bad = s[[LETTER[d] not in m for d, m in zip(s["D"], s["menu"])]]
    assert bad.empty, f"decision outside menu in {list(bad.index)}"  # structural-zero assumption holds in the data
    return s


if __name__ == "__main__":
    panel = build_panel()
    out = ROOT / "data" / "processed" / "panel.csv"
    panel.to_csv(out)
    s = analysis_sample(panel)
    print(f"wrote {out}  ({len(panel)} months); analysis sample: {len(s)} meetings")
    print(pd.crosstab(s["menu"], s["D"]).reindex(MENUS))
