"""Weeks 1-4 baselines: classical values, independent quantum strategies and
NPA upper bounds for CHSH, Feige and the magic square, n = 1, 2.

Writes results/baselines.json and results/baselines.md.

    python scripts/run_baselines.py            # everything (~10 minutes)
    python scripts/run_baselines.py --quick    # skip the slow NPA / MILP runs
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import warnings

from qpr import (
    chsh,
    chsh_strategy,
    classical_value,
    feige,
    magic_square,
    magic_square_strategy,
    npa_upper_bound,
    parallel,
    seesaw,
    tensor_power,
)

warnings.filterwarnings("ignore", message="Solution may be inaccurate")
OUT = Path(__file__).resolve().parents[1] / "results"


def single_game_strategies():
    """Best known single-copy strategy for each game."""
    feige_best = seesaw(feige(), 3, restarts=5, seed=0).strategy  # 9/16 in d = 3
    return {"CHSH": chsh_strategy(), "Feige": feige_best, "MagicSquare": magic_square_strategy()}


def main(quick: bool) -> None:
    games = {"CHSH": chsh(), "Feige": feige(), "MagicSquare": magic_square()}
    # NPA level per (game, n); None = skipped.  Level 1+AB for CHSH^2 takes ~5 min.
    npa_levels = {
        ("CHSH", 1): "1+AB",
        ("CHSH", 2): None if quick else "1+AB",
        ("Feige", 1): "1+AB",
        ("Feige", 2): "1",
        ("MagicSquare", 1): "1+AB",
        ("MagicSquare", 2): None,  # omega* = 1 already reached by the explicit strategy
    }
    strategies = single_game_strategies()
    rows = []
    for name, g1 in games.items():
        single_q = strategies[name].value(g1)
        single_c = classical_value(g1).value
        for n in (1, 2):
            g = parallel(g1, n)
            t = time.time()
            if quick and name == "MagicSquare" and n == 2:
                cl, cl_opt = None, None
            else:
                res = classical_value(g)
                cl, cl_opt = res.value, res.proven_optimal
            t_cl = time.time() - t

            q_ind = tensor_power(strategies[name], n).value(g)

            t = time.time()
            lvl = npa_levels[(name, n)]
            npa = npa_upper_bound(g, lvl).value if lvl else None
            t_npa = time.time() - t

            rows.append(
                dict(
                    game=name,
                    n=n,
                    classical=cl,
                    classical_proven=cl_opt,
                    classical_independent=single_c**n,
                    quantum_independent=q_ind,
                    quantum_dim=strategies[name].dims[0] ** n,
                    npa_upper=npa,
                    npa_level=lvl,
                    single_quantum_n=single_q**n,
                    seconds_classical=round(t_cl, 1),
                    seconds_npa=round(t_npa, 1),
                )
            )
            print(rows[-1])

    OUT.mkdir(exist_ok=True)
    (OUT / "baselines.json").write_text(json.dumps(rows, indent=2))

    fmt = lambda v: "n/a" if v is None else f"{v:.6f}"  # noqa: E731
    lines = [
        "| Game | n | ω(G^n) | ω(G)^n | quantum, independent | local dim | NPA upper bound (level) |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['game']} | {r['n']} | {fmt(r['classical'])} | {fmt(r['classical_independent'])} | "
            f"{fmt(r['quantum_independent'])} | {r['quantum_dim']} | "
            f"{fmt(r['npa_upper'])}{'' if r['npa_level'] is None else ' (' + r['npa_level'] + ')'} |"
        )
    (OUT / "baselines.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    main(ap.parse_args().quick)
