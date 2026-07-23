"""Tests for expression composition, constant wrapping, and compiler CSE."""
# ruff: noqa: ANN201

import math

from pytest import approx

from material_proxy import Add
from material_proxy import Const
from material_proxy import CurrentTime
from material_proxy import Div
from material_proxy import EvalContext
from material_proxy import Expr
from material_proxy import Mul
from material_proxy import Program
from material_proxy import Sub
from material_proxy import Var
from material_proxy import compile_to_vmt
from material_proxy import interpret_vmt


def _last_tmp(state: dict[str, float]) -> float:
    key = max(
        (k for k in state if k.startswith('$tmp_')),
        key=lambda k: int(k.removeprefix('$tmp_')),
    )
    return state[key]


def test_nested_expression():
    """Composition: Add(Mul(x, y), z)."""
    vmt = compile_to_vmt(Add(Mul(Var('x'), Var('y')), Var('z')))
    state = interpret_vmt(vmt, EvalContext(vars={'$x': 3.0, '$y': 4.0, '$z': 5.0}))
    assert _last_tmp(state) == approx(17.0)


def test_mixed_constants():
    """Mixing Var references and literal Const values."""
    expr = Mul(Add(Var('x'), 1), Sub(Var('y'), 2))
    ctx = EvalContext(vars={'$x': 3.0, '$y': 7.0})
    vmt = compile_to_vmt(expr)
    state = interpret_vmt(vmt, ctx)
    assert _last_tmp(state) == approx((3 + 1) * (7 - 2))


def test_dag_reuse():
    """Same sub-expression used by multiple parents compiles once."""
    shared = Div(200, 3)
    expr = shared * CurrentTime()
    ctx = EvalContext(time=10.0)
    vmt = compile_to_vmt(expr)
    state = interpret_vmt(vmt, ctx)
    assert state['$tmp_3'] == approx((200 / 3) * 10.0)
    assert 'Multiply' in str(vmt)


def test_diamond_dag():
    """Explicit object sharing — a+b used twice (one Mul)."""
    s = Add(Var('x'), Var('y'))
    expr = Mul(s, s)
    vmt = compile_to_vmt(expr)
    ctx = EvalContext(vars={'$x': 3.0, '$y': 4.0})
    state = interpret_vmt(vmt, ctx)
    assert _last_tmp(state) == approx(49.0)


def test_deep_binary_tree():
    """Balanced binary tree of depth 5 — computes x^32."""

    def build(d: int) -> Expr:
        if d == 0:
            return Var('x')
        return Mul(build(d - 1), build(d - 1))

    expr = build(5)
    vmt = compile_to_vmt(expr)
    ctx = EvalContext(vars={'$x': 1.01})
    state = interpret_vmt(vmt, ctx)
    assert _last_tmp(state) == approx(1.01**32, rel=1e-6)


def test_newton_sqrt():
    """Newton iteration computing sqrt(x), written as float-like code."""

    def sqrt_newton(x: Expr, n_iters: int = 6) -> Expr:
        t: Expr = (x + 1.0) * 0.5
        for _ in range(n_iters):
            t = (t + x / t) * 0.5
        return t

    expr = sqrt_newton(Var('x'), n_iters=6)
    ctx = EvalContext(vars={'$x': 25.0})
    vmt = compile_to_vmt(expr)
    state = interpret_vmt(vmt, ctx)
    assert _last_tmp(state) == approx(5.0, rel=1e-4)


def test_horner_polynomial():
    """Horner's method: ax³ + bx² + cx + d."""
    a, b, c, d = Var('a'), Var('b'), Var('c'), Var('d')
    x = Var('x')
    poly = ((a * x + b) * x + c) * x + d

    a_val, b_val, c_val, d_val, x_val = 2.0, -3.0, 1.0, -5.0, 4.0
    vmt = compile_to_vmt(poly)
    ctx = EvalContext(
        vars={'$a': a_val, '$b': b_val, '$c': c_val, '$d': d_val, '$x': x_val}
    )
    state = interpret_vmt(vmt, ctx)
    expected = a_val * x_val**3 + b_val * x_val**2 + c_val * x_val + d_val
    assert _last_tmp(state) == approx(expected)


def test_program_single_output():
    """Program with one output produces equivalent VMT to compile_to_vmt."""
    p = Program()
    p.output('$result', Add(Var('x'), Var('y')))
    vmt = p.compile()
    ctx = EvalContext(vars={'$x': 3.0, '$y': 4.0})
    state = interpret_vmt(vmt, ctx)
    assert state['$result'] == approx(7.0)


def test_program_multi_output():
    """Program with multiple outputs, sharing CSE."""
    p = Program()
    s = Add(Var('x'), Var('y'))
    p.output('$sum', s)
    p.output('$double', Mul(s, Const(2.0)))
    vmt = p.compile()
    ctx = EvalContext(vars={'$x': 3.0, '$y': 4.0})
    state = interpret_vmt(vmt, ctx)
    assert state['$sum'] == approx(7.0)
    assert state['$double'] == approx(14.0)


def test_program_constant_folding():
    """Program folds constants."""
    p = Program()
    p.output('$result', Mul(Add(Const(1.0), Const(2.0)), Const(5.0)))
    vmt = p.compile()
    ctx = EvalContext()
    state = interpret_vmt(vmt, ctx)
    assert state['$result'] == approx(15.0)
    assert 'Multiply' not in vmt
    assert 'Add' not in vmt


def test_newton_sqrt_convergence():
    """Newton sqrt with tighter convergence check across different values."""

    def sqrt_newton(x: Expr, n_iters: int = 10) -> Expr:
        t: Expr = (x + 1.0) * 0.5
        for _ in range(n_iters):
            t = (t + x / t) * 0.5
        return t

    for val in [2.0, 3.0, 10.0, 0.25, 100.0, 1e-4]:
        expr = sqrt_newton(Var('x'), n_iters=10)
        ctx = EvalContext(vars={'$x': val})
        vmt = compile_to_vmt(expr)
        state = interpret_vmt(vmt, ctx)
        assert _last_tmp(state) == approx(math.sqrt(val), rel=1e-6), (
            f'Failed for x={val}'
        )
