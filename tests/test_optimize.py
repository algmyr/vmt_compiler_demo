# ruff: noqa: ANN201

"""Integration tests that exercise multiple optim passes together."""

from pytest import approx

from material_proxy import Abs
from material_proxy import Add
from material_proxy import Const
from material_proxy import Div
from material_proxy import EvalContext
from material_proxy import Expr
from material_proxy import LessOrEqual
from material_proxy import Mul
from material_proxy import Program
from material_proxy import Sub
from material_proxy import Var
from material_proxy import constant_fold
from material_proxy import full_optimize
from material_proxy import interpret_vmt


def test_identity_full_pipeline_example():
    """The motivating example: (1+x)*4.3e-08*0 + x → x after full optimize."""
    tmp = Mul(Mul(Add(Const(1.0), Var('x')), Const(4.3e-08)), Const(0.0))
    expr = Add(tmp, Var('x'))
    prog = Program.from_tree(result=expr)
    folded = prog.optimize(constant_fold)
    eq = next(op for op in folded.ops if op.proxy == 'Equals')
    assert eq.params['srcVar1'] == '$x'
    full = prog.optimize(full_optimize)
    assert len(full.ops) == 1
    assert full.ops[0].proxy == 'Equals'
    vmt = full.emit()
    state = interpret_vmt(vmt, EvalContext(vars={'$x': 3.0}))
    assert state['$result'] == approx(3.0)


def test_reuse_pipeline_correctness_randomish():
    """Various expression shapes survive the full fold → DCE → reuse pipeline."""
    cases: list[tuple[Expr, float]] = [
        (Add(Var('x'), Var('y')), 3.0),
        (Mul(Add(Var('a'), Const(2.0)), Sub(Var('b'), Const(3.0))), 5.0),
        (Add(Abs(Var('v')), Const(42.0)), 47.0),
        (Div(Const(1.0), Add(Var('t'), Const(1.0))), 1.0 / 7.0),
        (Add(Mul(Var('p'), Var('q')), Mul(Var('r'), Var('s'))), 146.0),
        (
            Sub(
                Mul(Add(Var('a'), Var('b')), Add(Var('c'), Var('d'))),
                Add(Var('e'), Var('f')),
            ),
            134.0,
        ),
    ]
    ctx = EvalContext(
        vars={
            '$x': 1.0,
            '$y': 2.0,
            '$a': 3.0,
            '$b': 4.0,
            '$v': -5.0,
            '$t': 6.0,
            '$p': 7.0,
            '$q': 8.0,
            '$r': 9.0,
            '$s': 10.0,
            '$c': 11.0,
            '$d': 12.0,
            '$e': 13.0,
            '$f': 14.0,
        }
    )
    for expr, expected in cases:
        prog = Program.from_tree(result=expr)
        vmt = prog.compile(optimize=full_optimize)
        state = interpret_vmt(vmt, ctx)
        assert state['$result'] == approx(expected)


def test_peephole_full_pipeline():
    """Fused result survives the full optimize pipeline."""
    cond = Var('x')
    tr = Const(10.0)
    fr = Const(20.0)
    truthy = LessOrEqual(cond, Const(0), Const(0), Const(1))
    outer = LessOrEqual(truthy, Const(0), fr, tr)
    prog = Program.from_tree(result=outer)
    vmt = prog.compile(optimize=full_optimize)
    state = interpret_vmt(vmt, EvalContext(vars={'$x': -5.0}))
    assert state['$result'] == approx(20.0)
    state = interpret_vmt(vmt, EvalContext(vars={'$x': 5.0}))
    assert state['$result'] == approx(10.0)
