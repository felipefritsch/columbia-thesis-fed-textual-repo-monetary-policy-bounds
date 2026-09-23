# Revision notes: 2020 → 2026

What changed between the thesis submitted in May 2020 and the revised edition, and why.

## 1. Estimation

| | 2020 version | Revised edition |
|---|---|---|
| Propensity score | Ordered probit with menu dummies (off-menu probabilities small but positive) | Ordered probit renormalised over the menu (off-menu probabilities exactly zero) |
| Covariates | Lagged inflation/unemployment, FFF surprise, contemporaneous IP level, month dummies; all 197 months | Same macro set with lagged IP growth; plus a Greenbook-forecast specification; 133 meeting months |
| Bounds | Width 2(K₂−K₁) on every menu outside the overlap region | Sharp: menus with only one of {d, 0} contribute (K₂−K₁) |
| Outcome bounds K | Interquartile range of the estimated effects across horizons | Range of the realised outcome, 1989–2010 |
| Cumulative effects | Sum of per-horizon bounds | Cumulative outcome bounded directly |
| Point estimates | A scaling step (scalar `adj`) applied to some point-identified components | No post-estimation scaling |
| Inference | Left for future work | Moving-block bootstrap; Imbens–Manski intervals |

### The scaling step

The 2020 estimation applied a scaling step to the point-identified part of some bounds (factors of −0.9, −1 and 3,
depending on outcome and policy). It is not part of the estimator, so the revision drops it; it accounts for most
of the difference in the real-activity results between the two versions.

## 2. New material

* **Related literature (Section 2):** an updated review covering dynamic causal effects and potential outcomes
  (Rambachan–Shephard; Kolesár–Plagborg-Møller; Plagborg-Møller–Wolf; Montiel Olea–Plagborg-Møller), monetary shock
  identification and the Fed's information (Romer–Romer 2023; Nakamura–Steinsson; Jarociński–Karadi;
  Miranda-Agrippino–Ricco; Bauer–Swanson; Aruoba–Drechsel), FOMC text as data including recent uses of the Bluebook
  alternatives (Doh–Song–Yang; Laarits et al.), and overlap and partial identification (Heiler–Kazak;
  Susmann–McClean–Díaz; Masten–Poirier).
* **Discussion (Section 7):** what that literature implies for the results and for next steps.

## 3. Text corrections

* θ is written E[Y(d) − Y(d)] in three places; it should be E[Y(d) − Y(0)].
* The bound in eq. (8) sums over d ∈ 𝒟; it should sum over menus x with {d, 0} ⊂ x.
* References to equations (5)–(8) point to unnumbered equations.
* Appendix A.2 integrates against P(Z = z); it should be P(Z = z | X = x).
* "Price puzzle" refers to inflation *rising* after a tightening.
* Minor: "+0.50bp" → 50 bp; footnote 8 dates; Romer & Romer year (2004); Table 1 menu counts and Table 2 sample
  size now match the data used; §4.1 described residualised weights where the code residualised outcomes.

## 4. Repository

The old folder mixed Python notebooks (exploration, Figures 1–2) and Stata (all results) with menu dummies pasted
into a copy of the AJK spreadsheet, duplicate data files and do-files, absolute paths, and results saved in several
formats. The revised repo has one pipeline from `data/raw/` to every table and figure in the paper.

## 5. What carries over

The core idea is unchanged and holds up well: the observed decision is **always** inside the Bluebook menu
(133/133 meetings), and imposing that in the choice model raises the log-likelihood from −91.4 to −72.6 with no
extra parameters.
