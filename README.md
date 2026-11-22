
# Parallel Repetition for Quantum Non-Local Games

Research project developed as part of the MSc in Data Science for Decision Making, Quantum Computing specialization, at Maastricht University.

## Project Overview

This project investigates how the winning probabilities of quantum non-local games change under parallel repetition. We compare independent and joint strategies and study the effects of quantum resource constraints.

## Research Objectives

- Investigate the winning probabilities of selected non-local games under parallel repetition.
- Compare independent and joint quantum strategies.
- Analyze the effects of restrictions on local Hilbert-space dimensions.
- Explore the relationship between shared entanglement and winning probability.

## Methodology

Our computational approach includes classical strategy enumeration, explicit quantum strategies, numerical optimization, and semidefinite programming relaxations using the NPA hierarchy.

CHSH serves as the initial benchmark for validating the implementation. Additional non-XOR games will be selected based on theoretical relevance and computational feasibility.

## Code

Games studied: CHSH (benchmark), Feige's game and the Mermin-Peres magic square.
Rules and question distributions follow the original papers (see the docstrings in `src/qpr/games.py`).

```
src/qpr/
  games.py      game definitions and n-fold parallel repetition
  classical.py  classical value: exhaustive enumeration and exact MILP
  quantum.py    explicit quantum strategies, tensor products (independent play)
  seesaw.py     see-saw optimisation in fixed local dimension (lower bounds)
  npa.py        NPA hierarchy levels 1, 1+AB, 2 (upper bounds)
  experiments.py  runs every experiment for one game and saves the results
experiments/
  chsh/           benchmark
  feige/
  magic_square/
    each: run.py, results.md, results.json, <game>.ipynb
tests/            checks against known values
```

For each game and n = 1, 2: exact classical value, independent quantum value,
NPA upper bound, and (n = 2) a joint quantum strategy found by see-saw.

Setup and run:

```
python -m venv .venv
.venv\Scripts\activate
pip install -e .[dev]
pytest
python experiments/chsh/run.py
python experiments/feige/run.py
python experiments/magic_square/run.py
```

## Project Status

Current phase: initial implementation and benchmark validation.

## Team

 Quantum research project 3 Maastricht University

**Supervisors:** Dr. David Mestel and Dr. Georgios Stamoulis
