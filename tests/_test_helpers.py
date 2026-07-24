"""Shared helpers for optimization tests.

Pass wrappers adapt single-arg passes (dead_code_elimination, temp_reuse)
to the OptimizeFn signature ``(ops, consts) -> (ops, consts)`` so they
can be passed to ``Program.optimize()``.
"""

from material_proxy.flatten import FlatOp
from material_proxy.optimize import dead_code_elimination
from material_proxy.optimize import temp_reuse


def dce_only(
    ops: list[FlatOp], consts: dict[float, str]
) -> tuple[list[FlatOp], dict[float, str]]:
    return dead_code_elimination(ops), consts


def reuse_only(
    ops: list[FlatOp], consts: dict[float, str]
) -> tuple[list[FlatOp], dict[float, str]]:
    return temp_reuse(ops), consts


def last_temp_result(state: dict[str, float]) -> float:
    """Value of the highest-numbered ``$tmp_N`` variable (or first value)."""
    key = max(
        (k for k in state if k.startswith('$tmp_')),
        key=lambda k: int(k.removeprefix('$tmp_')),
        default=None,
    )
    return state[key] if key is not None else next(iter(state.values()))


def max_temp_index(ops: list[FlatOp]) -> int:
    """Maximum ``$tmp_N`` index in *ops* (0 if none)."""
    n = 0
    for op in ops:
        if op.result.startswith('$tmp_'):
            n = max(n, int(op.result.removeprefix('$tmp_')))
    return n
