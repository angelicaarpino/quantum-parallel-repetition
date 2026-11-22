"""Explicit quantum strategies (lower bounds on omega*(G)).

A strategy is a shared pure state |psi> in C^dA (x) C^dB plus POVMs
``A[x, a]`` (dA x dA) and ``B[y, b]`` (dB x dB).  The state is stored as a
dA x dB matrix ``M`` with |psi> = sum_ij M[i, j] |i>|j>, so that

    p(a, b | x, y) = <psi| A[x,a] (x) B[y,b] |psi> = tr(M^dag A M B^T).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .games import Game, magic_square_bits

I2 = np.eye(2)
X_ = np.array([[0, 1], [1, 0]], dtype=complex)
Y_ = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z_ = np.array([[1, 0], [0, -1]], dtype=complex)


@dataclass
class QuantumStrategy:
    M: np.ndarray  # (dA, dB) state coefficients, Frobenius norm 1
    A: np.ndarray  # (X, A, dA, dA)
    B: np.ndarray  # (Y, B, dB, dB)

    @property
    def dims(self) -> tuple[int, int]:
        return self.M.shape

    def behaviour(self) -> np.ndarray:
        M = self.M
        p = np.einsum("ij,xaik,kl,ybjl->xyab", M.conj(), self.A, M, self.B, optimize=True)
        return p.real

    def value(self, game: Game) -> float:
        return game.value(self.behaviour())

    def check(self, atol: float = 1e-8) -> None:
        """Raise if the state or the POVMs are invalid."""
        assert np.isclose(np.linalg.norm(self.M), 1, atol=atol)
        for ops in (self.A, self.B):
            d = ops.shape[-1]
            for povm in ops:
                assert np.allclose(povm.sum(axis=0), np.eye(d), atol=atol)
                for E in povm:
                    assert np.allclose(E, E.conj().T, atol=atol)
                    assert np.linalg.eigvalsh(E).min() > -atol


def maximally_entangled(d: int) -> np.ndarray:
    return np.eye(d, dtype=complex) / np.sqrt(d)


def tensor(s1: QuantumStrategy, s2: QuantumStrategy) -> QuantumStrategy:
    """Play s1 on the first copy and s2 on the second, independently.

    Alice's space is A1 (x) A2 and Bob's is B1 (x) B2, so the state matrix is
    the Kronecker product of the two state matrices.
    """

    def kron_ops(O1, O2):
        Q1, R1, d1, _ = O1.shape
        Q2, R2, d2, _ = O2.shape
        out = np.einsum("xaij,ybkl->xyabikjl", O1, O2)
        return out.reshape(Q1 * Q2, R1 * R2, d1 * d2, d1 * d2)

    return QuantumStrategy(np.kron(s1.M, s2.M), kron_ops(s1.A, s2.A), kron_ops(s1.B, s2.B))


def tensor_power(s: QuantumStrategy, n: int) -> QuantumStrategy:
    out = s
    for _ in range(n - 1):
        out = tensor(out, s)
    return out


def _projectors(observable: np.ndarray) -> np.ndarray:
    """Projectors onto the +1 and -1 eigenspaces (outcome 0 and 1)."""
    d = observable.shape[0]
    return np.stack([(np.eye(d) + observable) / 2, (np.eye(d) - observable) / 2])


# ---------------------------------------------------------------------------
# Known optimal strategies
# ---------------------------------------------------------------------------


def chsh_strategy() -> QuantumStrategy:
    """Tsirelson's strategy, value cos^2(pi/8) ~ 0.8536."""
    A = np.stack([_projectors(Z_), _projectors(X_)])
    B = np.stack([_projectors((Z_ + X_) / np.sqrt(2)), _projectors((Z_ - X_) / np.sqrt(2))])
    return QuantumStrategy(maximally_entangled(2), A, B)


def magic_square_strategy() -> QuantumStrategy:
    """Mermin-Peres strategy on two EPR pairs, value 1.

    Row products are +I and column products are -I.  Alice measures the
    row observables; Bob measures the transposes of the column observables
    (for the maximally entangled state <A (x) B^T> = tr(A B) / d).
    """
    k = np.kron
    square = [
        [k(I2, Z_), k(Z_, I2), k(Z_, Z_)],
        [k(X_, I2), k(I2, X_), k(X_, X_)],
        [-k(X_, Z_), -k(Z_, X_), k(Y_, Y_)],
    ]
    d = 4
    A = np.zeros((3, 4, d, d), dtype=complex)
    B = np.zeros((3, 4, d, d), dtype=complex)
    for x in range(3):
        for ans in range(4):
            P = np.eye(d, dtype=complex)
            for col, bit in enumerate(magic_square_bits(ans, 0)):
                P = P @ (np.eye(d) + (-1) ** bit * square[x][col]) / 2
            A[x, ans] = P
    for y in range(3):
        for ans in range(4):
            P = np.eye(d, dtype=complex)
            for row, bit in enumerate(magic_square_bits(ans, 1)):
                P = P @ (np.eye(d) + (-1) ** bit * square[row][y].T) / 2
            B[y, ans] = P
    return QuantumStrategy(maximally_entangled(d), A, B)


def classical_as_quantum(game: Game, alice: np.ndarray, bob: np.ndarray) -> QuantumStrategy:
    """Embed a deterministic classical strategy (dimension 1)."""
    X, Y = game.num_questions
    A, B = game.num_answers
    Aop = np.zeros((X, A, 1, 1), dtype=complex)
    Bop = np.zeros((Y, B, 1, 1), dtype=complex)
    Aop[np.arange(X), alice] = 1
    Bop[np.arange(Y), bob] = 1
    return QuantumStrategy(np.ones((1, 1), dtype=complex), Aop, Bop)
