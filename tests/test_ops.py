# ruff: noqa: ANN201, D103

from pytest import approx

from material_proxy import Abs
from material_proxy import Add
from material_proxy import Clamp
from material_proxy import Const
from material_proxy import CurrentTime
from material_proxy import Div
from material_proxy import Equals
from material_proxy import EvalContext
from material_proxy import Exp
from material_proxy import Expr
from material_proxy import Frac
from material_proxy import Int
from material_proxy import LessOrEqual
from material_proxy import Mul
from material_proxy import PlayerSpeed
from material_proxy import SelectFirstIfNonZero
from material_proxy import Sub
from material_proxy import Var
from material_proxy import WrapMinMax
from material_proxy import compile_to_vmt
from material_proxy import interpret_vmt


def check(expr: Expr, ctx: EvalContext, expected: float, result_var: str = ''):
    """Compile *expr*, interpret the VMT, and assert the result matches."""
    vmt = compile_to_vmt(expr)
    state = interpret_vmt(vmt, ctx)

    if not result_var:
        result_var = max(k for k in state if k.startswith('$tmp_'))

    assert state[result_var] == approx(expected, rel=1e-6), (
        f'\n  expr: {expr.pretty_repr()}\n  state: {state}'
    )


def test_add():
    check(Add(Var('x'), Var('y')), EvalContext(vars={'$x': 3.0, '$y': 4.0}), 7.0)


def test_sub():
    check(Sub(Var('x'), Var('y')), EvalContext(vars={'$x': 10.0, '$y': 3.0}), 7.0)


def test_mul():
    check(Mul(Var('x'), Var('y')), EvalContext(vars={'$x': 3.0, '$y': 4.0}), 12.0)


def test_div():
    check(Div(Var('x'), Const(2.0)), EvalContext(vars={'$x': 7.0}), 3.5)


def test_abs():
    check(Abs(Var('x')), EvalContext(vars={'$x': -3.7}), 3.7)


def test_frac():
    check(Frac(Var('x')), EvalContext(vars={'$x': 4.23}), 0.23)


def test_int():
    check(Int(Var('x')), EvalContext(vars={'$x': 4.23}), 4.0)


def test_clamp_above():
    check(Clamp(Var('x'), 0, 1), EvalContext(vars={'$x': 5.0}), 1.0)


def test_clamp_below():
    check(Clamp(Var('x'), 0, 1), EvalContext(vars={'$x': -2.0}), 0.0)


def test_clamp_inside():
    check(Clamp(Var('x'), 0, 1), EvalContext(vars={'$x': 0.5}), 0.5)


def test_wrap_minmax():
    check(WrapMinMax(Var('x'), 0, 10), EvalContext(vars={'$x': 12.0}), 2.0)


def test_wrap_minmax_negative():
    check(WrapMinMax(Var('x'), 0, 10), EvalContext(vars={'$x': -3.0}), 7.0)


def test_wrap_minmax_inverted():
    check(
        WrapMinMax(Var('x'), 10, 5), EvalContext(vars={'$x': 7.0}), 10.0
    )  # min >= max → returns min


def test_less_or_equal_true():
    check(
        LessOrEqual(Var('x'), Var('y'), Var('le'), Var('gr')),
        EvalContext(vars={'$x': 3.0, '$y': 5.0, '$le': 10.0, '$gr': 20.0}),
        10.0,
    )


def test_less_or_equal_false():
    check(
        LessOrEqual(Var('x'), Var('y'), Var('le'), Var('gr')),
        EvalContext(vars={'$x': 7.0, '$y': 5.0, '$le': 10.0, '$gr': 20.0}),
        20.0,
    )


def test_select_first_nonzero():
    check(
        SelectFirstIfNonZero(Var('x'), Var('y')),
        EvalContext(vars={'$x': 3.0, '$y': 99.0}),
        3.0,
    )


def test_select_first_zero():
    check(
        SelectFirstIfNonZero(Var('x'), Var('y')),
        EvalContext(vars={'$x': 0.0, '$y': 99.0}),
        99.0,
    )


def test_exp_basic():
    import math

    expected = 2.0 * math.exp(1.0 + 0.5)
    check(Exp(Var('x'), offset=0.5, scale=2.0), EvalContext(vars={'$x': 1.0}), expected)


def test_exp_clamped():
    check(
        Exp(Var('x'), scale=1, min_val=0, max_val=1),
        EvalContext(vars={'$x': 100.0}),
        1.0,
    )  # clamped at max


def test_current_time():
    check(CurrentTime(), EvalContext(time=42.0), 42.0)


def test_player_speed():
    check(PlayerSpeed(scale=2.0), EvalContext(speed=5.0), 10.0)


def test_equals():
    """Equals is a pass-through — output = input."""
    check(Equals(Var('x')), EvalContext(vars={'$x': -3.7}), -3.7)


def test_neg():
    """Unary negation via ``__neg__``."""
    check(-Var('x'), EvalContext(vars={'$x': 7.0}), -7.0)
    check(-Var('x'), EvalContext(vars={'$x': 0.0}), 0.0)
    check(-Var('x'), EvalContext(vars={'$x': -3.0}), 3.0)
