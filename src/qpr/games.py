"""Two-player non-local games and their parallel repetition.

A game is stored as two arrays:

* ``pi[x, y]``        probability that the referee sends questions (x, y)
* ``V[x, y, a, b]``   1 if answers (a, b) win on questions (x, y), else 0

Indices are plain integers.  For the n-fold parallel repetition the question
and answer tuples are flattened lexicographically, e.g. for n = 2 the pair
(x1, x2) becomes ``x1 * X + x2``.  This is the same ordering as ``np.kron``,
so tensor-product strategies line up with the repeated game automatically.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Game:
    name: str
    pi: np.ndarray  # shape (X, Y)
    V: np.ndarray  # shape (X, Y, A, B), entries in {0, 1}

    def __post_init__(self) -> None:
        if not np.isclose(self.pi.sum(), 1.0):
            raise ValueError("question distribution must sum to 1")
        if self.V.shape[:2] != self.pi.shape:
            raise ValueError("V and pi have incompatible question dimensions")

    @property
    def num_questions(self) -> tuple[int, int]:
        return self.pi.shape

    @property
    def num_answers(self) -> tuple[int, int]:
        return self.V.shape[2], self.V.shape[3]

    def value(self, p: np.ndarray) -> float:
        """Winning probability of a behaviour ``p[x, y, a, b]``."""
        return float(np.einsum("xy,xyab,xyab->", self.pi, self.V, p))


def product(g1: Game, g2: Game) -> Game:
    """Play g1 and g2 in parallel; the players must win both."""
    X1, Y1 = g1.num_questions
    X2, Y2 = g2.num_questions
    A1, B1 = g1.num_answers
    A2, B2 = g2.num_answers
    pi = np.einsum("ac,bd->abcd", g1.pi, g2.pi).reshape(X1 * X2, Y1 * Y2)
    V = np.einsum("xyab,XYAB->xXyYaAbB", g1.V, g2.V).reshape(
        X1 * X2, Y1 * Y2, A1 * A2, B1 * B2
    )
    return Game(f"{g1.name} x {g2.name}", pi, V)


def parallel(game: Game, n: int) -> Game:
    """The n-fold parallel repetition G^{(x)n}."""
    if n < 1:
        raise ValueError("n must be >= 1")
    g = game
    for _ in range(n - 1):
        g = product(g, game)
    return Game(f"{game.name}^{n}" if n > 1 else game.name, g.pi, g.V)


# ---------------------------------------------------------------------------
# Concrete games
# ---------------------------------------------------------------------------


def chsh() -> Game:
    """CHSH game (Clauser, Horne, Shimony, Holt 1969; game form as in Cleve,
    Høyer, Toner, Watrous 2004): uniform bits x, y; win iff a XOR b = x AND y.
    omega = 3/4, omega* = cos^2(pi/8)."""
    pi = np.full((2, 2), 1 / 4)
    V = np.zeros((2, 2, 2, 2))
    for x in range(2):
        for y in range(2):
            for a in range(2):
                for b in range(2):
                    V[x, y, a, b] = (a ^ b) == (x & y)
    return Game("CHSH", pi, V)


BOT = 2  # answer index of the symbol "⊥" in Feige's game


def feige() -> Game:
    """Feige's game (Feige 1991), in the form used by Schmidt, Storgaard,
    Walter and Zhao, "Guess your neighbor's input: Quantum advantage in
    Feige's game", arXiv:2510.08484.

    X = Y = {0, 1} with the uniform distribution pi(x, y) = 1/4, and
    A = B = {0, 1, ⊥} (⊥ is encoded as 2).  The players win iff
    (a, b) = (⊥, x) or (a, b) = (y, ⊥): exactly one player answers ⊥ and the
    other guesses that player's input.

    Known values (arXiv:2510.08484): omega = 1/2, omega* = 9/16 (dimension 3),
    omega_ns = 2/3; for n = 2 all three equal 1/2.
    """
    pi = np.full((2, 2), 1 / 4)
    V = np.zeros((2, 2, 3, 3))
    for x in range(2):
        for y in range(2):
            V[x, y, BOT, x] = 1
            V[x, y, y, BOT] = 1
    return Game("Feige", pi, V)


def magic_square_bits(answer: int, parity: int) -> tuple[int, int, int]:
    """Decode a magic-square answer index into its three bits.

    The first two bits are ``divmod(answer, 2)`` and the third is fixed by
    the parity constraint (0 = even for Alice's rows, 1 = odd for Bob's
    columns).
    """
    b0, b1 = divmod(answer, 2)
    return b0, b1, b0 ^ b1 ^ parity


def magic_square() -> Game:
    """Mermin-Peres magic square game (Mermin 1990, Peres 1990; game form as
    in Cleve, Høyer, Toner, Watrous 2004 and Aravind 2002).

    The referee picks a row x and a column y uniformly and independently,
    pi(x, y) = 1/9.  Alice fills row x with three bits of even parity; Bob
    fills column y with three bits of odd parity.  They win iff they agree on
    the shared cell: a[y] == b[x].  Only valid fillings are allowed, so each
    player has 4 answers.  omega = 8/9, omega* = 1.
    """
    pi = np.full((3, 3), 1 / 9)
    V = np.zeros((3, 3, 4, 4))
    for x in range(3):
        for y in range(3):
            for a in range(4):
                for b in range(4):
                    row = magic_square_bits(a, 0)
                    col = magic_square_bits(b, 1)
                    V[x, y, a, b] = row[y] == col[x]
    return Game("MagicSquare", pi, V)


GAMES = {"chsh": chsh, "feige": feige, "magic_square": magic_square}
