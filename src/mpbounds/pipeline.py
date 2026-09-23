"""End-to-end run: data -> propensity scores -> bounds -> bootstrap -> tables/figures.

    python -m mpbounds.pipeline            # full run (B=499 bootstrap draws per specification, ~15 min)
    python -m mpbounds.pipeline --quick    # B=49, for checking that everything runs
"""
import argparse
import json

import numpy as np
import pandas as pd

from . import bounds, inference, pscore, report
from .data import HORIZONS, OUTCOMES, ROOT, analysis_sample, build_panel, load_text_panel

RESULTS = ROOT / "results"
MAIN_SPEC, ALT_SPEC = "greenbook", "macro+fff"


def main(B: int):
    panel = build_panel()
    panel.to_csv(ROOT / "data" / "processed" / "panel.csv")
    s = analysis_sample(panel)
    s.to_csv(ROOT / "data" / "processed" / "analysis_sample.csv")
    Y, H = list(OUTCOMES), list(HORIZONS)
    K = {f"{y}_h{h}": bounds.outcome_support(panel, f"{y}_h{h}") for y in Y for h in H}

    # Propensity-score table: the 2020 progression (1)-(2), then the menu constraint (3), then the Greenbook set (4)
    models = [("Macro", pscore.fit(s, pscore.SPECS["macro"], False)),
              ("+ FFF", pscore.fit(s, pscore.SPECS["macro+fff"], False)),
              ("+ Menus", pscore.fit(s, pscore.SPECS["macro+fff"], True)),
              ("Greenbook", pscore.fit(s, pscore.SPECS["greenbook"], True))]
    report.table_pscore(models, s)
    report.table_menus(s)

    summary = {"n_meetings": len(s), "bootstrap_draws": B, "llf": {t: round(m.llf, 2) for t, m in models}}
    for spec, tag in [(MAIN_SPEC, "main"), (ALT_SPEC, "alt")]:
        cov = pscore.SPECS[spec]
        point = inference.estimate_all(s, cov, K, Y, H)
        draws, failed = inference.block_bootstrap(s, cov, K, Y, H, B=B)
        res = inference.summarise(point, draws)
        res["lam_star"] = [bounds.breakdown_lambda(r.theta_A, r.P_A) for r in res.itertuples()]
        res.to_csv(RESULTS / f"estimates_{tag}.csv")
        report.table_results(res, f"results_{tag}.tex")
        report.fig_responses(res, f"responses_{tag}.pdf")
        summary[f"failed_draws_{tag}"] = failed

        # Sensitivity to effect heterogeneity outside the overlap region, h = 12
        p = pscore.fit(s, cov).predict(s)
        rows = []
        for y in Y:
            col = f"{y}_h12"
            lam_max = 3 * res.xs(12, level="h").loc[y, "lam_star"].max()
            for d in (1, -1):
                cells = bounds.cell_quantities(s, p, col, d)
                for lam in np.linspace(0, lam_max, 61):
                    r = bounds.identified_set(cells, *K[col], lam=lam)
                    rows.append({"outcome": y, "policy": d, "lam": lam, "lo": r["lo"], "hi": r["hi"]})
        report.fig_sensitivity(pd.DataFrame(rows).set_index(["outcome", "policy", "lam"]), f"sensitivity_{tag}.pdf")

    # Overlap diagnostics for the main specification
    p = pscore.fit(s, pscore.SPECS[MAIN_SPEC]).predict(s)
    p_obs = np.array([p.loc[i, d] for i, d in zip(s.index, s["D"])])
    summary["min_p_observed_decision"] = round(float(p_obs.min()), 4)
    summary["share_p_observed_below_0.05"] = round(float((p_obs < 0.05).mean()), 3)

    report.fig_state_space(panel, s, load_text_panel())
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    main(B=49 if ap.parse_args().quick else 499)
