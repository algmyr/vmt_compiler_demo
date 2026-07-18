# ruff: noqa: ANN201, D103

from material_proxy import Abs
from material_proxy import Add
from material_proxy import Clamp
from material_proxy import Const
from material_proxy import CurrentTime
from material_proxy import Div
from material_proxy import Equals
from material_proxy import Exp
from material_proxy import Frac
from material_proxy import Int
from material_proxy import LessOrEqual
from material_proxy import Mul
from material_proxy import PlayerPosition
from material_proxy import PlayerSpeed
from material_proxy import SelectFirstIfNonZero
from material_proxy import Sub
from material_proxy import Var
from material_proxy import WrapMinMax


def test_repr_compact_leaf():
    assert repr(Const(3.0)) == '3.0'
    assert repr(Var('x')) == 'x'
    assert repr(CurrentTime()) == 'curtime()'


def test_repr_pretty_leaf():
    assert Const(3.0).pretty_repr() == '3.0'
    assert Var('x').pretty_repr() == 'x'
    assert CurrentTime().pretty_repr() == 'curtime()'


def test_repr_compact_unary():
    assert repr(Abs(Const(-4.0))) == 'Abs(-4.0)'
    assert repr(Frac(Const(4.5))) == 'Frac(4.5)'
    assert repr(Int(Const(4.5))) == 'Int(4.5)'


def test_repr_pretty_unary():
    r = Abs(Const(-4.0)).pretty_repr()
    assert r == 'Abs(\n  -4.0)'


def test_repr_compact_binary():
    assert repr(Add(Const(1.0), Const(2.0))) == 'Add(1.0, 2.0)'
    assert repr(Sub(Const(5.0), Const(3.0))) == 'Sub(5.0, 3.0)'
    assert repr(Mul(Const(2.0), Const(3.0))) == 'Mul(2.0, 3.0)'
    assert repr(Div(Const(6.0), Const(2.0))) == 'Div(6.0, 2.0)'


def test_repr_pretty_binary():
    r = Add(Const(1.0), Const(2.0)).pretty_repr()
    assert r == 'Add(\n  1.0,\n  2.0)'


def test_repr_compact_clamp():
    assert repr(Clamp(Var('x'), Const(0.0), Const(1.0))) == 'Clamp(x, 0.0, 1.0)'


def test_repr_pretty_clamp():
    r = Clamp(Var('x'), Const(0.0), Const(1.0)).pretty_repr()
    assert r == 'Clamp(\n  x,\n  0.0,\n  1.0)'


def test_repr_compact_wrap():
    assert (
        repr(WrapMinMax(Var('x'), Const(0.0), Const(1.0))) == 'WrapMinMax(x, 0.0, 1.0)'
    )


def test_repr_compact_less_or_equal():
    assert (
        repr(LessOrEqual(Var('x'), Var('y'), Var('a'), Var('b')))
        == 'LessOrEqual(x, y, a, b)'
    )


def test_repr_compact_select():
    assert (
        repr(SelectFirstIfNonZero(Var('x'), Var('y'))) == 'SelectFirstIfNonZero(x, y)'
    )


def test_repr_compact_equals():
    assert repr(Equals(Var('x'))) == 'Equals(x)'


def test_repr_compact_player_pos():
    assert repr(PlayerPosition()) == 'PlayerPos(1.0)'
    assert repr(PlayerPosition(scale=Var('s'))) == 'PlayerPos(s)'


def test_repr_compact_player_speed():
    assert repr(PlayerSpeed()) == 'PlayerSpeed(1.0)'
    assert repr(PlayerSpeed(scale=Var('s'))) == 'PlayerSpeed(s)'


def test_repr_compact_exp():
    assert repr(Exp(Var('x'))) == 'Exp(x)'
    assert repr(Exp(Var('x'), offset=0.5, scale=2.0)) == 'Exp(x, offset=0.5, scale=2.0)'


def test_repr_compact_nested():
    expr = Abs(Mul(Add(Var('x'), Const(1.0)), Const(2.0)))
    assert repr(expr) == 'Abs(Mul(Add(x, 1.0), 2.0))'


def test_repr_pretty_nested():
    expr = Abs(Mul(Add(Var('x'), Const(1.0)), Const(2.0)))
    r = expr.pretty_repr()
    assert r == 'Abs(\n  Mul(\n    Add(\n      x,\n      1.0),\n    2.0))'
