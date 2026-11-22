"""NPA hierarchy upper bounds on the quantum value omega*(G).

Operators are Alice's projectors A_a^x and Bob's B_b^y with the last outcome
of every question dropped (it equals I minus the others).  A monomial is a
pair (alice_word, bob_word) of reduced words; the moment matrix
Gamma[i, j] = <psi| m_i^dag m_j |psi> is required to be PSD.

Supported levels:
  "1"     monomials {1, A, B}
  "1+AB"  monomials {1, A, B, AB}   (the usual "almost quantum" level)
  "2"     monomials {1, A, B, AA, AB, BB}

Because the game coefficients are real we may take Gamma real symmetric,
which identifies each word with its reverse.
"""

from __future__ import annotations

from dataclasses import dataclass
import itertools

import cvxpy as cp
import numpy as np
import scipy.sparse as sp

from .games import Game

Letter = tuple[int, int]  # (question, answer)
Word = tuple[Letter, ...]
Mono = tuple[Word, Word]


def _reduce(word: Word) -> Word | None:
    """Apply E_a^x E_a'^x = delta_{aa'} E_a^x.  Returns None for zero."""
    out: list[Letter] = []
    for q, a in word:
        if out and out[-1][0] == q:
            if out[-1][1] == a:
                continue
            return None
        out.append((q, a))
    return tuple(out)


def _canonical(m: Mono) -> Mono:
    rev = (m[0][::-1], m[1][::-1])
    return min(m, rev)


def _monomials(game: Game, level: str) -> list[Mono]:
    X, Y = game.num_questions
    nA, nB = game.num_answers
    As: list[Word] = [((x, a),) for x in range(X) for a in range(nA - 1)]
    Bs: list[Word] = [((y, b),) for y in range(Y) for b in range(nB - 1)]
    monos: list[Mono] = [((), ())]
    monos += [(a, ()) for a in As] + [((), b) for b in Bs]
    if level in ("1+AB", "2"):
        monos += [(a, b) for a in As for b in Bs]
    if level == "2":
        for w in (_reduce(a1 + a2) for a1, a2 in itertools.product(As, As)):
            if w is not None and len(w) == 2:
                monos.append((w, ()))
        for w in (_reduce(b1 + b2) for b1, b2 in itertools.product(Bs, Bs)):
            if w is not None and len(w) == 2:
                monos.append(((), w))
    elif level != "1" and level != "1+AB":
        raise ValueError(f"unknown level {level!r}")
    return monos


@dataclass
class NPAResult:
    value: float
    level: str
    size: int
    status: str


def npa_upper_bound(game: Game, level: str = "1+AB", solver: str = "CLARABEL", **solver_kwargs) -> NPAResult:
    monos = _monomials(game, level)
    n = len(monos)

    # Gamma is a PSD variable.  Entries holding the same canonical moment are
    # tied together by sparse linear equalities, and zero words are set to 0.
    # first[key] is the flat position (i * n + j) where a moment first appears.
    first: dict[Mono, int] = {}
    crow, ccol, cval = [], [], []
    m = 0
    for i in range(n):
        for j in range(i, n):
            pos = i * n + j
            aw = _reduce(monos[i][0][::-1] + monos[j][0])
            bw = _reduce(monos[i][1][::-1] + monos[j][1])
            if aw is None or bw is None:
                crow.append(m), ccol.append(pos), cval.append(1.0)
                m += 1
                continue
            key = _canonical((aw, bw))
            if key not in first:
                first[key] = pos
                continue
            crow += [m, m]
            ccol += [pos, first[key]]
            cval += [1.0, -1.0]
            m += 1
    C = sp.csr_matrix((cval, (crow, ccol)), shape=(m, n * n))

    G = cp.Variable((n, n), symmetric=True)
    g = cp.vec(G, order="C")  # G is symmetric, so the order does not matter
    cons = [G >> 0, G[0, 0] == 1]
    if m:
        cons.append(C @ g == 0)

    X, Y = game.num_questions
    nA, nB = game.num_answers

    def expand(q: int, ans: int, num: int) -> list[tuple[float, Word]]:
        if ans < num - 1:
            return [(1.0, ((q, ans),))]
        return [(1.0, ())] + [(-1.0, ((q, a),)) for a in range(num - 1)]

    c = np.zeros(n * n)
    for x in range(X):
        for yq in range(Y):
            if game.pi[x, yq] == 0:
                continue
            for a in range(nA):
                for b in range(nB):
                    w = game.pi[x, yq] * game.V[x, yq, a, b]
                    if w == 0:
                        continue
                    for ca, aw in expand(x, a, nA):
                        for cb, bw in expand(yq, b, nB):
                            c[first[_canonical((aw, bw))]] += w * ca * cb

    prob = cp.Problem(cp.Maximize(c @ g), cons)
    prob.solve(solver=solver, **solver_kwargs)
    return NPAResult(float(prob.value), level, n, prob.status)
