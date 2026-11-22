"""qpr: parallel repetition of two-player non-local games."""

from .classical import classical_value, classical_value_enum, classical_value_milp
from .games import Game, chsh, feige, magic_square, parallel, product
from .npa import npa_upper_bound
from .quantum import (
    QuantumStrategy,
    chsh_strategy,
    classical_as_quantum,
    magic_square_strategy,
    tensor,
    tensor_power,
)
from .seesaw import seesaw

__all__ = [
    "Game",
    "QuantumStrategy",
    "chsh",
    "chsh_strategy",
    "classical_as_quantum",
    "classical_value",
    "classical_value_enum",
    "classical_value_milp",
    "feige",
    "magic_square",
    "magic_square_strategy",
    "npa_upper_bound",
    "parallel",
    "product",
    "seesaw",
    "tensor",
    "tensor_power",
]
