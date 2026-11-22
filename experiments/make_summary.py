"""Summary table and figures across the three games.

Reads experiments/<game>/results.json (run each game's run.py first) and writes
  experiments/summary.md                 values and decay rates r_n = -ln(v_n) / n
  experiments/figures/values_by_game.png
  experiments/figures/joint_vs_dimension.png

    python experiments/make_summary.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).parent
GAMES = {"chsh": "CHSH", "feige": "Feige", "magic_square": "Magic square"}

# categorical slots, fixed order: classical, quantum independent, quantum joint, NPA bound
COLORS = {"classical": "#2a78d6", "independent": "#eb6834", "joint": "#1baf7a", "npa": "#eda100"}
LABELS = {
    "classical": "classical ω(Gⁿ)",
    "independent": "quantum, independent",
    "joint": "quantum, joint (see-saw)",
    "npa": "NPA upper bound",
}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def load() -> dict:
    return {k: json.loads((HERE / k / "results.json").read_text(encoding="utf-8")) for k in GAMES}


def best_joint(res: dict) -> float | None:
    vals = [j["value"] for j in res["joint_n2_seesaw"]]
    return max(vals) if vals else None


def rows(data: dict) -> list[dict]:
    out = []
    for key, res in data.items():
        for r in res["repetitions"]:
            joint = best_joint(res) if r["n"] == 2 else None
            q_lower = max(v for v in (r["quantum_independent"], joint) if v is not None)
            # omega* is at least the classical value; NPA is an upper bound
            out.append(
                dict(
                    game=GAMES[key],
                    n=r["n"],
                    classical=r["classical"],
                    independent=r["quantum_independent"],
                    joint=joint,
                    npa=r["npa_upper"],
                    q_lower=q_lower,
                )
            )
    return out


def rate(v: float | None, n: int) -> float | None:
    return None if v is None else -math.log(v) / n


def write_summary(table: list[dict]) -> None:
    f = lambda v: "–" if v is None else f"{v:.4f}"  # noqa: E731
    lines = [
        "# Summary across games",
        "",
        "Winning probabilities. Quantum lower bound = best of independent and joint strategies; "
        "NPA = upper bound on ω*(Gⁿ). When the two coincide the quantum value is known exactly.",
        "",
        "| Game | n | classical ω(Gⁿ) | quantum independent | quantum joint | NPA upper bound |",
        "|---|---|---|---|---|---|",
    ]
    for r in table:
        lines.append(
            f"| {r['game']} | {r['n']} | {f(r['classical'])} | {f(r['independent'])} | {f(r['joint'])} | {f(r['npa'])} |"
        )
    lines += [
        "",
        "Effective decay rate r_n = −ln(v_n) / n (larger = faster decay; 0 = no decay).",
        "",
        "| Game | n | classical | quantum (from lower bound) | quantum (from NPA bound) |",
        "|---|---|---|---|---|",
    ]
    for r in table:
        lines.append(
            f"| {r['game']} | {r['n']} | {f(rate(r['classical'], r['n']))} | "
            f"{f(rate(r['q_lower'], r['n']))} | {f(rate(r['npa'], r['n']))} |"
        )
    (HERE / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def style(ax) -> None:
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def fig_values(table: list[dict], out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), sharey=True)
    keys = ["classical", "independent", "joint", "npa"]
    width = 0.19
    for ax, game in zip(axes, GAMES.values()):
        sub = [r for r in table if r["game"] == game]
        for i, r in enumerate(sub):
            present = [k for k in keys if r[k] is not None]
            start = i - width * (len(present) - 1) / 2
            for j, k in enumerate(present):
                x = start + j * width
                ax.bar(x, r[k], width * 0.9, color=COLORS[k], label=LABELS[k])
                ax.text(x, r[k] + 0.012, f"{round(r[k], 5):.4g}", ha="center", va="bottom", fontsize=7, color=INK, rotation=90)
        ax.set_xticks(range(len(sub)), [f"n = {r['n']}" for r in sub])
        ax.set_title(game, fontsize=11, color=INK)
        ax.set_ylim(0, 1.18)
        style(ax)
    axes[0].set_ylabel("winning probability", color=INK)
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[k]) for k in keys]
    fig.legend(handles, [LABELS[k] for k in keys], loc="lower center", ncol=4, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out, dpi=200)
    plt.close(fig)


def fig_joint(data: dict, table: list[dict], out: Path) -> None:
    games = ["feige", "magic_square"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, key in zip(axes, games):
        res = data[key]
        r2 = next(r for r in table if r["game"] == GAMES[key] and r["n"] == 2)
        dims = [j["local_dim"] for j in res["joint_n2_seesaw"]]
        vals = [j["value"] for j in res["joint_n2_seesaw"]]
        ax.plot(dims, vals, color=COLORS["joint"], linewidth=2, marker="o", markersize=7, label=LABELS["joint"])
        for d, v in zip(dims, vals):
            ax.annotate(f"{v:.3f}", (d, v), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8, color=INK)
        refs = [
            ("classical", r2["classical"], f"classical ω(G²) = {r2['classical']:.3f}"),
            (
                "independent",
                r2["independent"],
                f"quantum, independent = {r2['independent']:.3f} (local dim "
                f"{res['repetitions'][1]['quantum_independent_local_dim']})",
            ),
        ]
        if r2["npa"] is not None:
            upper, src = r2["npa"], "NPA upper bound"
        else:
            upper, src = 1.0, "trivial upper bound"
        if abs(upper - r2["classical"]) < 1e-6:
            # the two dashed lines would overlap: name both in one entry
            refs = refs[1:]
            refs.append(("npa", upper, f"{src} = classical ω(G²) = {upper:.3f}"))
        else:
            refs.append(("npa", upper, f"{src} = {upper:.3f}"))
        for k, v, lab in refs:
            ax.axhline(v, color=COLORS[k], linewidth=1.5, linestyle="--", label=lab)
        ax.set_xticks(dims)
        ax.set_xlabel("local dimension per player (both copies together)", color=INK, fontsize=9)
        ax.set_title(f"{GAMES[key]}, n = 2", fontsize=11, color=INK)
        ax.set_ylim(0.2 if key == "feige" else 0.7, 1.05)
        style(ax)
        ax.legend(fontsize=7.5, frameon=False, loc="upper right" if key == "feige" else "lower right")
    axes[0].set_ylabel("winning probability", color=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def main() -> None:
    data = load()
    table = rows(data)
    write_summary(table)
    figs = HERE / "figures"
    figs.mkdir(exist_ok=True)
    fig_values(table, figs / "values_by_game.png")
    fig_joint(data, table, figs / "joint_vs_dimension.png")
    print((HERE / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
