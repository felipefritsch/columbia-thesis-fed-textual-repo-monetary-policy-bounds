# Bounds on the Dynamic Causal Effects of Monetary Policy

Code, data and LaTeX source for my Columbia B.A. honors thesis in economics (advisor: Prof. José Luis Montiel Olea),
**revised edition, September 2026**. The paper is `paper/main.pdf`.

**Idea.** Before every FOMC meeting the Bluebook lists the policy alternatives on the table. In 133 meetings
(1989–2006) the Committee never chose an action that was not on its menu, so the overlap assumption behind
propensity-score estimates of monetary policy effects (Angrist, Jordà & Kuersteiner 2018) fails, and the text
tells us exactly where. The repo estimates a menu-constrained propensity score, the sharp identified set for the
average effect of a rate increase/decrease, a sensitivity analysis in the heterogeneity across menus, and
block-bootstrap / Imbens–Manski inference.

> **About the 2020 version.** The revised edition rebuilds the estimation from the raw data: a menu-constrained
> propensity score, sharper bounds, outcome bounds calibrated to the data, no post-estimation scaling step, and
> bootstrap inference. Several conclusions change. See [`docs/REVISION_NOTES.md`](docs/REVISION_NOTES.md) and Appendix C of the paper.

## Reproduce

```bash
pip install -e ".[dev]"      # Python ≥ 3.10
make test                    # 13 tests, ~10 s
make results                 # estimates, 499-draw bootstrap, tables, figures (~15 min); `make quick` for 49 draws
make paper                   # LaTeX -> paper/main.pdf (needs a TeX distribution with latexmk)
```

## Layout

```
data/raw/            inputs, read-only (provenance in data/README.md)
data/processed/      generated monthly panel and analysis sample
src/mpbounds/
  data.py            build the panel: decisions, Bluebook menus, covariates, Greenbook forecasts, cumulative outcomes
  pscore.py          ordered probit renormalised over the menu (structural zeros), marginal effects
  bounds.py          within-menu Hájek IPW, sharp identified set, λ-sensitivity, breakdown values
  inference.py       moving-block bootstrap over meetings, Imbens–Manski intervals
  report.py          LaTeX tables and PDF figures -> results/
  pipeline.py        end-to-end run (python -m mpbounds.pipeline)
scripts/             helpers (rebuild tables from saved estimates)
tests/               synthetic-DGP checks of the estimator, data and propensity-score checks
results/             generated tables (.tex), figures (.pdf), estimates (.csv), summary.json
paper/               LaTeX source; \input's results/ directly, so tables in the paper are never retyped
docs/REVISION_NOTES.md  what changed from the 2020 version
```

## Main results (Greenbook propensity score)

| | 12 months after a **rate increase** | 12 months after a **rate decrease** |
|---|---|---|
| Industrial production, overlap region | +1.89% [0.85, 4.02] | −1.42% [−3.20, −0.49] |
| Unemployment, overlap region | −0.60 pp [−1.18, −0.39] | +0.65 pp [0.42, 1.14] |
| PCE prices, overlap region | +0.20% [−0.07, 0.39] | +0.80% [0.51, 1.66] |

90% moving-block bootstrap intervals. Worst-case identified sets contain zero for every outcome and horizon. The
real-activity effects have the wrong sign, which points to a failure of unconfoundedness rather than of overlap.

## Sources

Angrist–Jordà–Kuersteiner (2018) replication files; Bluebook menus extracted as described in §2 of the paper, building
on [jm4474/FOMCTextAnalysis](https://github.com/jm4474/FOMCTextAnalysis); Philadelphia Fed Greenbook data set.
