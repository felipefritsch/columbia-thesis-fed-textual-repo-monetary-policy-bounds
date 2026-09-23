"""Rebuild the result tables from results/estimates_*.csv without re-running the bootstrap."""
import pandas as pd

from mpbounds import pscore, report
from mpbounds.data import analysis_sample, build_panel

s = analysis_sample(build_panel())
for tag in ("main", "alt"):
    report.table_results(pd.read_csv(report.ROOT / "results" / f"estimates_{tag}.csv", index_col=[0, 1, 2]), f"results_{tag}.tex")
report.table_pscore([("Macro", pscore.fit(s, pscore.SPECS["macro"], False)),
                     ("+ FFF", pscore.fit(s, pscore.SPECS["macro+fff"], False)),
                     ("+ Menus", pscore.fit(s, pscore.SPECS["macro+fff"], True)),
                     ("Greenbook", pscore.fit(s, pscore.SPECS["greenbook"], True))], s)
report.table_menus(s)
