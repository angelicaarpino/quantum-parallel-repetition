"""Run the full set of experiments for one game and save the results.

For n = 1, 2 this computes:
  * the classical value omega(G^n) (exact) and the independent baseline omega(G)^n
  * the independent quantum value (best single-copy strategy on each copy)
  * an NPA upper bound on omega*(G^n)
and for n = 2 a joint quantum strategy found by see-saw in a given local
dimension.  Results go to <folder>/results.json and <folder>/results.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import time
from typing import Callable
import warnings

from .classical import classical_value
from .games import Game, parallel
from .npa import npa_upper_bound
from .quantum import QuantumStrategy, tensor_power
from .seesaw import seesaw

warnings.filterwarnings("ignore", message="Solution may be inaccurate")


@dataclass
class ExperimentConfig:
    game: Callable[[], Game]
    single_strategy: Callable[[], QuantumStrategy]
    npa_levels: dict[int, str | None]  # n -> NPA level, None = skip
    joint_dims: list[int] = field(default_factory=lambda: [4])
    joint_restarts: int = 3
    seed: int = 0


def run_experiments(cfg: ExperimentConfig, folder: Path, quick: bool = False) -> dict:
    g1 = cfg.game()
    strat = cfg.single_strategy()
    omega_1 = classical_value(g1).value
    rows = []
    for n in (1, 2):
        g = parallel(g1, n)
        t = time.time()
        cl = classical_value(g)
        t_cl = time.time() - t

        lvl = None if (quick and n == 2) else cfg.npa_levels.get(n)
        t = time.time()
        npa = npa_upper_bound(g, lvl).value if lvl else None
        t_npa = time.time() - t

        rows.append(
            dict(
                n=n,
                classical=cl.value,
                classical_method=cl.method,
                classical_proven_optimal=cl.proven_optimal,
                classical_independent=omega_1**n,
                quantum_independent=tensor_power(strat, n).value(g),
                quantum_independent_local_dim=strat.dims[0] ** n,
                npa_upper=npa,
                npa_level=lvl,
                seconds_classical=round(t_cl, 1),
                seconds_npa=round(t_npa, 1),
            )
        )
        print(rows[-1], flush=True)

    g2 = parallel(g1, 2)
    joint = []
    for d in cfg.joint_dims:
        t = time.time()
        res = seesaw(g2, d, restarts=cfg.joint_restarts, seed=cfg.seed)
        joint.append(dict(local_dim=d, value=res.value, restarts=cfg.joint_restarts, seconds=round(time.time() - t, 1)))
        print(joint[-1], flush=True)

    out = {"game": g1.name, "repetitions": rows, "joint_n2_seesaw": joint}
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "results.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    (folder / "results.md").write_text(to_markdown(out), encoding="utf-8")
    return out


def to_markdown(out: dict) -> str:
    f = lambda v: "n/a" if v is None else f"{v:.6f}"  # noqa: E731
    lines = [
        f"# {out['game']}",
        "",
        "| n | ω(G^n) | ω(G)^n | quantum, independent (local dim) | NPA upper bound (level) |",
        "|---|---|---|---|---|",
    ]
    for r in out["repetitions"]:
        npa = f(r["npa_upper"]) + ("" if r["npa_level"] is None else f" ({r['npa_level']})")
        lines.append(
            f"| {r['n']} | {f(r['classical'])} | {f(r['classical_independent'])} | "
            f"{f(r['quantum_independent'])} ({r['quantum_independent_local_dim']}) | {npa} |"
        )
    lines += [
        "",
        "Joint strategy for n = 2 (see-saw, lower bound):",
        "",
        "| local dim | value | restarts |",
        "|---|---|---|",
    ]
    for j in out["joint_n2_seesaw"]:
        lines.append(f"| {j['local_dim']} | {f(j['value'])} | {j['restarts']} |")
    return "\n".join(lines) + "\n"
