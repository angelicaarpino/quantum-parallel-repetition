"""Feige's game: classical, independent quantum, NPA and joint n = 2.

Game as in arXiv:2510.08484 (answers {0, 1, ⊥}, uniform questions).
The single-copy strategy is found by see-saw in local dimension 3, where the
paper's optimal value 9/16 is attained.

    python experiments/feige/run.py
"""

import argparse
from pathlib import Path

from qpr import feige, seesaw
from qpr.experiments import ExperimentConfig, run_experiments


def best_single_strategy():
    return seesaw(feige(), 3, restarts=5, seed=0).strategy  # 9/16


CONFIG = ExperimentConfig(
    game=feige,
    single_strategy=best_single_strategy,
    npa_levels={1: "1+AB", 2: "1"},  # level 1 already gives the exact 1/2 for n = 2
    joint_dims=[2, 3, 4],
)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    run_experiments(CONFIG, Path(__file__).parent, quick=ap.parse_args().quick)
