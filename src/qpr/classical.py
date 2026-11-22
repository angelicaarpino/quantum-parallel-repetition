"""Classical values omega(G).

Shared randomness never helps, so omega(G) is a maximum over deterministic
strategies.  Two exact methods are provided:

* :func:`classical_value_enum` enumerates every deterministic strategy of
  Alice and lets Bob best-respond.  Cost ~ A^X, fine for small games.
* :func:`classical_value_milp` solves the same problem as a mixed-integer
  linear program (HiGHS via SciPy).  Needed when A^X is too large, e.g. the
  two-fold magic square (16^9 Alice strategies).
"""

from __future__ import annotations

from dataclasses import dataclass
import itertools

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

from .games import Game


@dataclass
class ClassicalResult:
    value: float
    alice: np.ndarray  # alice[x] = answer
    bob: np.ndarray  # bob[y] = answer
    method: str
    proven_optimal: bool = True


def _bob_best_response(W: np.ndarray, alice: np.ndarray) -> tuple[float, np.ndarray]:
    # W[x, y, a, b] = pi * V
    X = W.shape[0]
    T = W[np.arange(X), :, alice, :].sum(axis=0)  # (Y, B)
    return float(T.max(axis=1).sum()), T.argmax(axis=1)


def classical_value_enum(game: Game, max_strategies: int = 5_000_000, chunk: int = 8192) -> ClassicalResult:
    W = game.pi[:, :, None, None] * game.V
    X, Y = game.num_questions
    A, B = game.num_answers
    total = A**X
    if total > max_strategies:
        raise ValueError(f"{total} Alice strategies, use classical_value_milp instead")

    best_val, best_alice = -1.0, None
    all_strats = itertools.product(range(A), repeat=X)
    while True:
        block = np.array(list(itertools.islice(all_strats, chunk)), dtype=np.int64)
        if block.size == 0:
            break
        # T[n, y, b] = sum_x W[x, y, S[n, x], b]
        T = np.zeros((block.shape[0], Y, B))
        for x in range(X):
            T += W[x][:, block[:, x], :].transpose(1, 0, 2)
        vals = T.max(axis=2).sum(axis=1)
        i = int(vals.argmax())
        if vals[i] > best_val + 1e-12:
            best_val, best_alice = float(vals[i]), block[i].copy()

    val, bob = _bob_best_response(W, best_alice)
    return ClassicalResult(val, best_alice, bob, "enumeration")


def classical_value_milp(game: Game, time_limit: float | None = None) -> ClassicalResult:
    """Exact MILP formulation.

    Binary a[x, alpha], b[y, beta] pick one answer per question; continuous
    z[x, y] in [0, 1] is the win indicator, constrained by
        z[x, y] <= sum_{beta wins with alpha} b[y, beta] + 1 - a[x, alpha].
    """
    X, Y = game.num_questions
    A, B = game.num_answers
    qpairs = [(x, y) for x in range(X) for y in range(Y) if game.pi[x, y] > 0]
    na, nb, nz = X * A, Y * B, len(qpairs)
    nvar = na + nb + nz
    ia = lambda x, a: x * A + a  # noqa: E731
    ib = lambda y, b: na + y * B + b  # noqa: E731

    c = np.zeros(nvar)
    for k, (x, y) in enumerate(qpairs):
        c[na + nb + k] = -game.pi[x, y]

    rows = X + Y + nz * A
    M = lil_matrix((rows, nvar))
    lo = np.empty(rows)
    hi = np.empty(rows)
    r = 0
    for x in range(X):
        for a in range(A):
            M[r, ia(x, a)] = 1
        lo[r] = hi[r] = 1
        r += 1
    for y in range(Y):
        for b in range(B):
            M[r, ib(y, b)] = 1
        lo[r] = hi[r] = 1
        r += 1
    for k, (x, y) in enumerate(qpairs):
        for a in range(A):
            # z - sum_win b + a <= 1
            M[r, na + nb + k] = 1
            for b in np.flatnonzero(game.V[x, y, a]):
                M[r, ib(y, b)] = -1
            M[r, ia(x, a)] += 1
            lo[r], hi[r] = -np.inf, 1
            r += 1

    integrality = np.r_[np.ones(na + nb), np.zeros(nz)]
    options = {"disp": False}
    if time_limit is not None:
        options["time_limit"] = time_limit
    res = milp(
        c,
        constraints=LinearConstraint(M.tocsr(), lo, hi),
        integrality=integrality,
        bounds=Bounds(0, 1),
        options=options,
    )
    if res.x is None:
        raise RuntimeError(f"MILP failed: {res.message}")
    xa = res.x[:na].reshape(X, A).argmax(axis=1)
    xb = res.x[na : na + nb].reshape(Y, B).argmax(axis=1)
    p = deterministic_behaviour(game, xa, xb)
    return ClassicalResult(game.value(p), xa, xb, "milp", proven_optimal=res.status == 0)


def deterministic_behaviour(game: Game, alice: np.ndarray, bob: np.ndarray) -> np.ndarray:
    X, Y = game.num_questions
    A, B = game.num_answers
    p = np.zeros((X, Y, A, B))
    for x in range(X):
        for y in range(Y):
            p[x, y, alice[x], bob[y]] = 1
    return p


def classical_value(game: Game, **kwargs) -> ClassicalResult:
    X, _ = game.num_questions
    A, _ = game.num_answers
    if A**X <= 2_000_000:
        return classical_value_enum(game)
    return classical_value_milp(game, **kwargs)
