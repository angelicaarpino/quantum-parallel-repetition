"""Week 5 starting point: joint quantum strategies for n = 2.

A joint strategy is simply a strategy for the repeated game G^2 in which each
player measures all their questions together on one shared state of local
dimension d.  We optimise it with the same see-saw used for single games and
compare it with the best independent value omega*(G)^2.

Minimal setup on purpose: small dimensions, few restarts.  Results are lower
bounds only (see-saw can stop in a local optimum).

    python scripts/joint_n2.py --game feige --dims 3 4 --restarts 3
"""

from __future__ import annotations

import argparse
import warnings

from qpr import chsh, feige, magic_square, parallel, seesaw

warnings.filterwarnings("ignore", message="Solution may be inaccurate")

GAMES = {"chsh": chsh, "feige": feige, "magic_square": magic_square}
# best single-copy quantum values (CHSH: Tsirelson; Feige: arXiv:2510.08484; MS: 1)
SINGLE = {"chsh": 0.8535533905932737, "feige": 9 / 16, "magic_square": 1.0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", choices=GAMES, default="feige")
    ap.add_argument("--dims", type=int, nargs="+", default=[2, 3])
    ap.add_argument("--restarts", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    g2 = parallel(GAMES[args.game](), 2)
    indep = SINGLE[args.game] ** 2
    print(f"{g2.name}: independent baseline omega*(G)^2 = {indep:.6f}")
    for d in args.dims:
        res = seesaw(g2, d, restarts=args.restarts, seed=args.seed)
        print(f"  joint see-saw, local dim {d}: {res.value:.6f} (iterations {len(res.history)})")


if __name__ == "__main__":
    main()
