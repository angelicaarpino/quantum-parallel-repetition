"""See-saw optimisation of quantum strategies in fixed local dimension.

Alternately optimises Alice's POVMs (an SDP per question), Bob's POVMs and
the shared state (top eigenvector of the game operator).  Every step can only
increase the value, so it converges, but possibly to a local optimum:
results are lower bounds on the dimension-restricted value and we use several
random restarts.
"""

from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp
import numpy as np

from .games import Game
from .quantum import QuantumStrategy


def _random_povms(num_q: int, num_a: int, d: int, rng: np.random.Generator) -> np.ndarray:
    """Random projective measurements (a random basis split into blocks)."""
    out = np.zeros((num_q, num_a, d, d), dtype=complex)
    for q in range(num_q):
        G = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
        U, _ = np.linalg.qr(G)
        # spread basis vectors over outcomes so no measurement starts trivial
        labels = rng.permutation(np.arange(d) % num_a) if d >= num_a else rng.integers(num_a, size=d)
        for k in range(d):
            v = U[:, k : k + 1]
            out[q, labels[k]] += v @ v.conj().T
    return out


def _optimal_povm(W: np.ndarray, solver: str | None) -> np.ndarray:
    """max sum_a tr(E_a W_a) over POVMs {E_a}."""
    n, d, _ = W.shape
    E = [cp.Variable((d, d), hermitian=True) for _ in range(n)]
    cons = [e >> 0 for e in E] + [sum(E) == np.eye(d)]
    obj = cp.Maximize(cp.real(sum(cp.trace(E[a] @ W[a]) for a in range(n))))
    cp.Problem(obj, cons).solve(solver=solver)
    out = np.array([e.value for e in E])
    # tidy up small numerical errors
    out = (out + out.conj().transpose(0, 2, 1)) / 2
    return out


def _best_state(game: Game, A: np.ndarray, B: np.ndarray) -> np.ndarray:
    dA, dB = A.shape[-1], B.shape[-1]
    W = game.pi[:, :, None, None] * game.V
    K = np.einsum("xyab,xaij,ybkl->ikjl", W, A, B, optimize=True).reshape(dA * dB, dA * dB)
    K = (K + K.conj().T) / 2
    _, vecs = np.linalg.eigh(K)
    return vecs[:, -1].reshape(dA, dB)


@dataclass
class SeesawResult:
    value: float
    strategy: QuantumStrategy
    history: list[float]


def seesaw(
    game: Game,
    dA: int,
    dB: int | None = None,
    restarts: int = 10,
    max_iter: int = 200,
    tol: float = 1e-8,
    seed: int = 0,
    solver: str | None = "CLARABEL",
) -> SeesawResult:
    dB = dA if dB is None else dB
    X, Y = game.num_questions
    nA, nB = game.num_answers
    W = game.pi[:, :, None, None] * game.V  # (X, Y, A, B)
    rng = np.random.default_rng(seed)
    best: SeesawResult | None = None

    for _ in range(restarts):
        A = _random_povms(X, nA, dA, rng)
        B = _random_povms(Y, nB, dB, rng)
        M = _best_state(game, A, B)
        history: list[float] = []
        for _ in range(max_iter):
            # Alice: W_A[x, a] = sum_{y,b} W[x,y,a,b] M B[y,b]^T M^dag
            WA = np.einsum("xyab,ij,ybkj,lk->xail", W, M, B, M.conj(), optimize=True)
            A = np.array([_optimal_povm(WA[x], solver) for x in range(X)])
            # Bob: W_B[y, b] = sum_{x,a} W[x,y,a,b] (M^dag A[x,a] M)^T
            WB = np.einsum("xyab,ij,xaik,kl->ybjl", W, M.conj(), A, M, optimize=True).transpose(0, 1, 3, 2)
            B = np.array([_optimal_povm(WB[y], solver) for y in range(Y)])
            M = _best_state(game, A, B)
            val = QuantumStrategy(M, A, B).value(game)
            history.append(val)
            if len(history) > 5 and abs(history[-1] - history[-2]) < tol:
                break
        res = SeesawResult(history[-1], QuantumStrategy(M, A, B), history)
        if best is None or res.value > best.value:
            best = res
    assert best is not None
    return best
