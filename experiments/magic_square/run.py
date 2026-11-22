"""Magic square: classical, independent quantum, NPA and joint n = 2.

The classical value for n = 2 is computed exactly with a MILP (~2.5 minutes).
The independent quantum strategy (Mermin-Peres) already wins with probability
1, using local dimension 4 per copy (16 for n = 2), so the n = 2 NPA bound is
skipped.  The joint run uses local dimension 4 for both copies together, i.e.
half the qubits of independent play (a first resource-limited data point).

    python experiments/magic_square/run.py
"""

import argparse
from pathlib import Path

from qpr import magic_square, magic_square_strategy
from qpr.experiments import ExperimentConfig, run_experiments

CONFIG = ExperimentConfig(
    game=magic_square,
    single_strategy=magic_square_strategy,
    npa_levels={1: "1+AB", 2: None},
    joint_dims=[4],
)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    run_experiments(CONFIG, Path(__file__).parent, quick=ap.parse_args().quick)
