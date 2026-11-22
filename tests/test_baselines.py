"""Checks against values known from the literature."""

import numpy as np
import pytest

from qpr import (
    chsh,
    chsh_strategy,
    classical_value_enum,
    classical_value_milp,
    feige,
    magic_square,
    magic_square_strategy,
    npa_upper_bound,
    parallel,
    seesaw,
    tensor_power,
)

TSIRELSON = np.cos(np.pi / 8) ** 2


def test_parallel_shapes():
    g = parallel(magic_square(), 2)
    assert g.num_questions == (9, 9)
    assert g.num_answers == (16, 16)
    assert np.isclose(g.pi.sum(), 1)


@pytest.mark.parametrize(
    "game, n, expected",
    [
        (chsh, 1, 3 / 4),
        (chsh, 2, 10 / 16),  # known: omega(CHSH^2) = 0.625 > (3/4)^2
        (feige, 1, 1 / 2),
        (feige, 2, 1 / 2),  # Feige: omega(G^2) = omega(G)
        (magic_square, 1, 8 / 9),
    ],
)
def test_classical_values(game, n, expected):
    assert np.isclose(classical_value_enum(parallel(game(), n)).value, expected)


@pytest.mark.parametrize("game", [chsh, feige, magic_square])
def test_milp_matches_enumeration(game):
    g = game()
    assert np.isclose(classical_value_milp(g).value, classical_value_enum(g).value)


def test_milp_chsh2():
    assert np.isclose(classical_value_milp(parallel(chsh(), 2)).value, 10 / 16)


def test_explicit_strategies():
    s = chsh_strategy()
    s.check()
    assert np.isclose(s.value(chsh()), TSIRELSON)
    assert np.isclose(tensor_power(s, 2).value(parallel(chsh(), 2)), TSIRELSON**2)

    m = magic_square_strategy()
    m.check()
    assert np.isclose(m.value(magic_square()), 1)
    assert np.isclose(tensor_power(m, 2).value(parallel(magic_square(), 2)), 1)


def test_npa():
    assert np.isclose(npa_upper_bound(chsh(), "1").value, TSIRELSON, atol=1e-6)
    assert np.isclose(npa_upper_bound(magic_square(), "1+AB").value, 1, atol=1e-6)
    # arXiv:2510.08484: level 1+AB already certifies omega*(Feige) <= 9/16
    assert np.isclose(npa_upper_bound(feige(), "1+AB").value, 9 / 16, atol=1e-6)


def test_seesaw_feige_dim3():
    # arXiv:2510.08484: omega*(Feige) = 9/16, attained in local dimension 3
    res = seesaw(feige(), 3, restarts=2, seed=0)
    assert np.isclose(res.value, 9 / 16, atol=1e-5)


def test_seesaw_chsh():
    res = seesaw(chsh(), 2, restarts=3, seed=1)
    res.strategy.check(atol=1e-6)
    assert np.isclose(res.value, TSIRELSON, atol=1e-5)
