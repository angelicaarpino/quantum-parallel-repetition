
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

CHSH serves as the benchmark for validating the implementation. The non-XOR games selected are Feige's game and the Mermin-Peres magic square.

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

## Results so far

Winning probabilities for n parallel repetitions (full tables in `experiments/<game>/results.md`).

| Game | n | Classical ω(Gⁿ) | Quantum, independent | Quantum, joint (see-saw, local dim) | NPA upper bound |
|---|---|---|---|---|---|
| CHSH | 1 | 0.750 | 0.854 | – | 0.854 |
| CHSH | 2 | 0.625 | 0.729 | 0.729 (4) | 0.729 |
| Feige | 1 | 0.500 | 0.5625 | – | 0.5625 |
| Feige | 2 | 0.500 | 0.316 | 0.500 (3, 4) | 0.500 |
| Magic square | 1 | 0.889 | 1 | – | 1 |
| Magic square | 2 | 0.815 | 1 (dim 16) | 0.869 (4) | – |

All single-game values match the literature. Joint play does not help for CHSH (perfect parallel
repetition for XOR games); for Feige it improves on independent quantum play but only reaches the
classical value 1/2; for the magic square, local dimension 4 for both copies stays below 1.

## Project Status

- Done (Period 2, weeks 1–4): game selection, game definitions, classical values, independent
  quantum strategies for n = 1, 2, NPA upper bounds, tests.
- Started (week 5): first joint-strategy runs for n = 2.
- Next: complete joint strategies for n = 2 and n = 3 where feasible; resource-limited runs with
  local dimension d ≤ 2^m; report outline.

## Team

 Quantum research project 3 Maastricht University

**Supervisors:** Dr. David Mestel and Dr. Georgios Stamoulis
