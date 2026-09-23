"""LaTeX tables and PDF figures written to results/ (the paper \\input's them directly)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data import MENUS, OUTCOMES, ROOT
from .pscore import LABELS

TABLES = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures"
REPORT_H = [6, 12, 18, 24]
POLICY = {1: "Rate increase", -1: "Rate decrease"}
UNITS = {"ip": "%", "pce": "%", "unrate": "pp"}
SHORT = {"ip": "industrial production", "unrate": "unemployment", "pce": "PCE prices"}

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, SET_FILL = "#2a78d6", "#eb6834", "#d9d8d3"

plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False,
})


def _write(name: str, body: str):
    TABLES.mkdir(parents=True, exist_ok=True)
    (TABLES / name).write_text(body)


def _f(x, nd=2):
    """Number for use inside math mode."""
    return f"{x:.{nd}f}"


def table_menus(s: pd.DataFrame):
    ct = pd.crosstab(s["menu"], s["D"]).reindex(index=MENUS, columns=[-1, 0, 1], fill_value=0)
    pretty = {"E": r"$\{E\}$", "U": r"$\{U\}$", "T": r"$\{T\}$", "EU": r"$\{U,E\}$", "UT": r"$\{U,T\}$", "EUT": r"$\{U,T,E\}$"}
    lines = [rf"{pretty[x]} & {r[-1]} & {r[0]} & {r[1]} & {r.sum()} \\" for x, r in ct.iterrows()]
    tot = ct.sum()
    body = "\n".join([
        r"\begin{tabular}{lrrrr}", r"\toprule",
        r"Bluebook menu $X_t$ & Ease & Unchanged & Tighten & Meetings \\", r"\midrule", *lines, r"\midrule",
        rf"Total & {tot[-1]} & {tot[0]} & {tot[1]} & {tot.sum()} \\", r"\bottomrule", r"\end{tabular}"])
    _write("menus.tex", body)


def table_pscore(models: list, s: pd.DataFrame):
    """models: list of (column title, fitted PScore)."""
    covs = list(dict.fromkeys(c for _, m in models for c in m.covariates))
    ames = [m.ame(s, decision=1) for _, m in models]
    rows = []
    for c in covs:
        est, se = [], []
        for a in ames:
            if c in a.index:
                z = abs(a.loc[c, "ame"] / a.loc[c, "se"])
                star = "^{***}" if z > 2.576 else "^{**}" if z > 1.96 else "^{*}" if z > 1.645 else ""
                est.append(f"${_f(a.loc[c, 'ame'], 3)}{star}$")
                se.append(f"({a.loc[c, 'se']:.3f})")
            else:
                est.append("")
                se.append("")
        rows += [f"{LABELS[c]} & " + " & ".join(est) + r" \\", " & " + " & ".join(se) + r" \\[2pt]"]
    k = len(models)
    body = "\n".join([
        r"\begin{tabular}{l" + "c" * k + "}", r"\toprule",
        " & " + " & ".join(f"({i + 1})" for i in range(k)) + r" \\",
        " & " + " & ".join(t for t, _ in models) + r" \\", r"\midrule", *rows, r"\midrule",
        "Menu constraint & " + " & ".join("Yes" if m.menu_constrained else "No" for _, m in models) + r" \\",
        "Log likelihood & " + " & ".join(f"${_f(m.llf)}$" for _, m in models) + r" \\",
        "Meetings & " + " & ".join(str(m.nobs) for _, m in models) + r" \\", r"\bottomrule", r"\end{tabular}"])
    _write("pscore.tex", body)


def table_results(res: pd.DataFrame, name: str):
    lines = []
    for y, label in OUTCOMES.items():
        lines.append(rf"\multicolumn{{6}}{{l}}{{\textit{{{label} ({UNITS[y].replace('%', chr(92) + '%')})}}}} \\")
        for d in (1, -1):
            for h in REPORT_H:
                r = res.loc[(y, d, h)]
                lines.append(
                    f"{POLICY[d] if h == REPORT_H[0] else ''} & {h} & ${_f(r.theta_A)}$ "
                    f"& $[{_f(r.ci_theta_lo)},\\ {_f(r.ci_theta_hi)}]$ & $[{_f(r.lo, 1)},\\ {_f(r.hi, 1)}]$ "
                    f"& ${_f(r.lam_star)}$ \\\\")
            lines.append(r"\addlinespace[2pt]")
        lines.append(r"\midrule")
    lines = lines[:-1]
    body = "\n".join([
        r"\begin{tabular}{llrccc}", r"\toprule",
        r"Policy & $h$ & $\hat\theta_A$ & 90\% CI & Identified set & $\lambda^\ast$ \\", r"\midrule",
        *lines, r"\bottomrule", r"\end{tabular}"])
    _write(name, body)


def fig_responses(res: pd.DataFrame, name: str):
    fig, axes = plt.subplots(3, 2, figsize=(6.3, 7.4), sharex=True)
    for i, y in enumerate(OUTCOMES):
        for j, d in enumerate((1, -1)):
            ax = axes[i, j]
            r = res.xs((y, d), level=["outcome", "policy"])
            h = r.index.to_numpy()
            ax.fill_between(h, r.lo, r.hi, color=SET_FILL, lw=0, label="Identified set (worst case)")
            ax.fill_between(h, r.ci_theta_lo, r.ci_theta_hi, color=BLUE, alpha=0.18, lw=0, label=r"90% CI for $\theta_A$")
            ax.plot(h, r.theta_A, color=BLUE, lw=2, label=r"$\hat\theta_A$ (overlap region)")
            ax.axhline(0, color=INK2, lw=0.8)
            ax.set_title(f"{POLICY[d]} on {SHORT[y]}", fontsize=9)
            ax.set_ylabel(f"Cumulative change ({UNITS[y]})" if j == 0 else "")
            if i == 2:
                ax.set_xlabel("Months after decision")
            ax.set_xticks([0, 6, 12, 18, 24])
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, -0.005))
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / name)
    plt.close(fig)


def fig_state_space(panel: pd.DataFrame, s: pd.DataFrame, text: pd.DataFrame):
    """Figures 1-2 of the thesis, rebuilt: lagged inflation vs IP growth (deviation from sample mean) at meetings,
    marked by the decision taken and by the menu offered."""
    infl = (100 * (text["pce_index"] / text["pce_index"].shift(12) - 1)).shift(1)
    ipg = text["ip_yoy"].shift(1)
    z = pd.DataFrame({"infl": infl, "ip": ipg}).reindex(s.index)
    z = z - z.mean()
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.4), sharey=True)
    groups = [
        [("Tighten", s.D == 1, BLUE, "o"), ("Ease", s.D == -1, ORANGE, "s")],
        [("Menu excludes easing", ~s.menu.str.contains("E"), BLUE, "o"),
         ("Menu excludes tightening", ~s.menu.str.contains("T"), ORANGE, "s")],
    ]
    for ax, g, title in zip(axes, groups, ["(a) Decisions taken", "(b) Alternatives offered"]):
        ax.scatter(z.ip, z.infl, s=10, color=GRID, lw=0, zorder=1)
        for lab, m, c, mk in g:
            ax.scatter(z.ip[m], z.infl[m], s=22, color=c, marker=mk, edgecolor="white", lw=0.6, label=lab, zorder=2)
        ax.axhline(0, color=INK2, lw=0.8)
        ax.axvline(0, color=INK2, lw=0.8)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("IP growth, y/y (deviation from mean, pp)")
        ax.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, handletextpad=0.2)
    axes[0].set_ylabel("PCE inflation, y/y (deviation, pp)")
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "state_space.pdf")
    plt.close(fig)


def fig_sensitivity(sens: pd.DataFrame, name: str):
    """Bounds at h=12 as a function of lambda (effect heterogeneity outside the overlap region)."""
    fig, axes = plt.subplots(3, 2, figsize=(6.3, 6.4))
    for i, y in enumerate(OUTCOMES):
        for j, d in enumerate((1, -1)):
            ax = axes[i, j]
            r = sens.xs((y, d), level=["outcome", "policy"])
            lam = r.index.to_numpy()
            ax.fill_between(lam, r.lo, r.hi, color=BLUE, alpha=0.25, lw=0)
            ax.plot(lam, r.lo, color=BLUE, lw=1.5)
            ax.plot(lam, r.hi, color=BLUE, lw=1.5)
            ax.axhline(0, color=INK2, lw=0.8)
            ax.set_title(f"{POLICY[d]}: {SHORT[y]}", fontsize=8.5)
            if i == 2:
                ax.set_xlabel(r"$\lambda$ (" + UNITS[y] + ")")
            if j == 0:
                ax.set_ylabel(f"Bounds ({UNITS[y]})")
    fig.tight_layout()
    fig.savefig(FIGURES / name)
    plt.close(fig)
