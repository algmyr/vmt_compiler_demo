# ruff: noqa: ANN201

from pytest import approx

from material_proxy import Abs
from material_proxy import Add
from material_proxy import Const
from material_proxy import Div
from material_proxy import EvalContext
from material_proxy import Mul
from material_proxy import Program
from material_proxy import Var
from material_proxy import compile_to_vmt
from material_proxy import constant_fold
from material_proxy import interpret_vmt
from material_proxy.emit import emit_vmt
from material_proxy.flatten import FlatOp
from tests._test_helpers import last_temp_result


def _fold_and_emit(ops: list[FlatOp], consts: dict[float, str]) -> str:
    folded_ops, folded_consts = constant_fold(list(ops), dict(consts))
    return emit_vmt(folded_ops, folded_consts)


def _fold_ops(ops: list[FlatOp], consts: dict[float, str]) -> list[FlatOp]:
    folded_ops, _ = constant_fold(list(ops), dict(consts))
    return folded_ops


# ---- Constant folding -------------------------------------------------------


def test_fold_single_binary():
    """Div(6.0, 3.0) — constant folded, only Equals remains."""
    prog = Program.from_tree(result=Div(Const(6.0), Const(3.0)))
    folded = prog.optimize(constant_fold)
    assert len(folded.ops) == 1
    assert folded.ops[0].proxy == 'Equals'
    vmt = folded.emit()
    state = interpret_vmt(vmt, EvalContext())
    assert state['$result'] == approx(2.0)


def test_fold_full_tree():
    """Entirely constant expression fully folded."""
    prog = Program.from_tree(result=Mul(Add(Const(1.0), Const(2.0)), Const(5.0)))
    folded = prog.optimize(constant_fold)
    assert len(folded.ops) == 1
    assert folded.ops[0].proxy == 'Equals'
    assert 15.0 in folded.consts


def test_fold_partial():
    """Mixed constant and variable — only constant parts fold."""
    prog = Program.from_tree(result=Add(Mul(Const(2.0), Var('x')), Const(5.0)))
    folded = prog.optimize(constant_fold)
    assert len(folded.ops) == 3
    ctx = EvalContext(vars={'$x': 3.0})
    folded_vmt = folded.emit()
    state = interpret_vmt(folded_vmt, ctx)
    assert state['$result'] == approx(11.0)


def test_fold_abs_neg():
    """Abs folded at compile time."""
    prog = Program.from_tree(result=Abs(Const(-3.0)))
    folded = prog.optimize(constant_fold)
    assert len(folded.ops) == 1
    assert folded.ops[0].proxy == 'Equals'
    assert 3.0 in folded.consts


def test_fold_chained():
    """Chained constant expressions all fold."""
    inner = Add(Const(1.0), Const(2.0))
    prog = Program.from_tree(result=Mul(inner, Add(Const(3.0), Const(4.0))))
    folded = prog.optimize(constant_fold)
    assert len(folded.ops) == 1
    assert folded.ops[0].proxy == 'Equals'
    assert 21.0 in folded.consts


def test_fold_no_side_effects():
    """Folding produces same numerical result as non-folded compilation."""
    expr = Mul(Add(Const(1.0), Const(2.0)), Var('x'))
    prog = Program.from_tree(result=expr)
    ctx = EvalContext(vars={'$x': 5.0})
    state_raw = interpret_vmt(prog.compile(optimize=constant_fold), ctx)
    state_normal = interpret_vmt(compile_to_vmt(expr), ctx)
    assert state_raw['$result'] == approx(15.0)
    assert last_temp_result(state_normal) == approx(15.0)


# ---- Identity / zero aliasing (inside constant_fold) ------------------------


def test_identity_aliased_result_not_emitted():
    """Add(0, x) — op is removed, result is aliased to x."""
    ops = [FlatOp('Add', {'srcVar1': '$0.0', 'srcVar2': '$x'}, '$tmp_1')]
    result = _fold_ops(ops, {0.0: '$0.0'})
    assert len(result) == 0


def test_identity_propagation():
    """Add(0, x) then Sub(result, 0) — both ops aliased away."""
    ops = [
        FlatOp('Add', {'srcVar1': '$0.0', 'srcVar2': '$x'}, '$tmp_1'),
        FlatOp('Subtract', {'srcVar1': '$tmp_1', 'srcVar2': '$0.0'}, '$tmp_2'),
    ]
    result = _fold_ops(ops, {0.0: '$0.0'})
    assert len(result) == 0


def test_identity_correctness():
    """Add(0, $x) followed by Equals gives $x unchanged."""
    ops = [
        FlatOp('Add', {'srcVar1': '$0.0', 'srcVar2': '$x'}, '$tmp_1'),
        FlatOp('Equals', {'srcVar1': '$tmp_1'}, 'result'),
    ]
    vmt = _fold_and_emit(ops, {0.0: '$0.0'})
    state = interpret_vmt(vmt, EvalContext(vars={'$x': 42.0}))
    assert state.get('$result', state.get('result')) == approx(42.0)


def test_identity_add_zero_left():
    """Add(0.0, x) — aliased to x."""
    ops = [FlatOp('Add', {'srcVar1': '$0.0', 'srcVar2': '$x'}, '$tmp_1')]
    result = _fold_ops(ops, {0.0: '$0.0'})
    assert len(result) == 0


def test_identity_add_zero_right():
    """Add(x, 0.0) — aliased to x."""
    ops = [FlatOp('Add', {'srcVar1': '$x', 'srcVar2': '$0.0'}, '$tmp_1')]
    result = _fold_ops(ops, {0.0: '$0.0'})
    assert len(result) == 0


def test_identity_sub_zero():
    """Sub(x, 0.0) — aliased to x."""
    ops = [FlatOp('Subtract', {'srcVar1': '$x', 'srcVar2': '$0.0'}, '$tmp_1')]
    result = _fold_ops(ops, {0.0: '$0.0'})
    assert len(result) == 0


def test_identity_mul_zero_left():
    """Mul(0.0, x) — aliased to $0.0."""
    ops = [FlatOp('Multiply', {'srcVar1': '$0.0', 'srcVar2': '$x'}, '$tmp_1')]
    result = _fold_ops(ops, {0.0: '$0.0'})
    assert len(result) == 0
    vmt = _fold_and_emit(
        ops + [FlatOp('Equals', {'srcVar1': '$tmp_1'}, 'out')],
        {0.0: '$0.0'},
    )
    state = interpret_vmt(vmt, EvalContext(vars={'$x': 99.0}))
    val = state.get('$out', state.get('out', state.get('$0.0')))
    assert val == approx(0.0)


def test_identity_mul_zero_right():
    """Mul(x, 0.0) — aliased to $0.0."""
    ops = [FlatOp('Multiply', {'srcVar1': '$x', 'srcVar2': '$0.0'}, '$tmp_1')]
    result = _fold_ops(ops, {0.0: '$0.0'})
    assert len(result) == 0


def test_identity_mul_one_left():
    """Mul(1.0, x) — aliased to x."""
    ops = [FlatOp('Multiply', {'srcVar1': '$1.0', 'srcVar2': '$x'}, '$tmp_1')]
    result = _fold_ops(ops, {1.0: '$1.0'})
    assert len(result) == 0


def test_identity_mul_one_right():
    """Mul(x, 1.0) — aliased to x."""
    ops = [FlatOp('Multiply', {'srcVar1': '$x', 'srcVar2': '$1.0'}, '$tmp_1')]
    result = _fold_ops(ops, {1.0: '$1.0'})
    assert len(result) == 0


def test_identity_div_one():
    """Div(x, 1.0) — aliased to x."""
    ops = [FlatOp('Divide', {'srcVar1': '$x', 'srcVar2': '$1.0'}, '$tmp_1')]
    result = _fold_ops(ops, {1.0: '$1.0'})
    assert len(result) == 0


def test_identity_div_zero_num():
    """Div(0.0, x) — aliased to $0.0."""
    ops = [FlatOp('Divide', {'srcVar1': '$0.0', 'srcVar2': '$x'}, '$tmp_1')]
    result = _fold_ops(ops, {0.0: '$0.0'})
    assert len(result) == 0


def test_identity_not_foldable_kept():
    """Add(x, y) — neither identity nor constant fold applies."""
    ops = [FlatOp('Add', {'srcVar1': '$x', 'srcVar2': '$y'}, '$tmp_1')]
    result = _fold_ops(ops, {})
    assert len(result) == 1


def test_identity_chained_through_equals():
    """Add(0, x) then Equals — Equals sees the aliased temp."""
    ops = [
        FlatOp('Add', {'srcVar1': '$0.0', 'srcVar2': '$x'}, '$tmp_1'),
        FlatOp('Equals', {'srcVar1': '$tmp_1'}, 'out'),
    ]
    vmt = _fold_and_emit(ops, {0.0: '$0.0'})
    state = interpret_vmt(vmt, EvalContext(vars={'$x': 7.0}))
    val = state.get('$out', state.get('out'))
    assert val == approx(7.0)
