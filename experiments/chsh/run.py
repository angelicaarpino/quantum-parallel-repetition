"""CHSH (benchmark): classical, independent quantum, NPA and joint n = 2.

    python experiments/chsh/run.py           # ~6 minutes (NPA 1+AB for n = 2)
    python experiments/chsh/run.py --quick   # skip the n = 2 NPA bound
"""

import argparse
from pathlib import Path

from qpr import chsh, chsh_strategy
from qpr.experiments import ExperimentConfig, run_experiments

CONFIG = ExperimentConfig(
    game=chsh,
    single_strategy=chsh_strategy,  # Tsirelson, cos^2(pi/8)
    npa_levels={1: "1+AB", 2: "1+AB"},
    joint_dims=[4],
)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    run_experiments(CONFIG, Path(__file__).parent, quick=ap.parse_args().quick)
