# ruff: noqa: ANN201

import pytest
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
from material_proxy import compile_to_vmt
from material_proxy import constant_fold
from material_proxy import full_optimize
from material_proxy import interpret_vmt
from material_proxy import peephole_optimize
from material_proxy.flatten import FlatOp
from tests._test_helpers import dce_only
from tests._test_helpers import last_temp_result
from tests._test_helpers import max_temp_index
from tests._test_helpers import reuse_only

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


def _fold_and_emit(ops: list[FlatOp], consts: dict[float, str]) -> str:
    """Helper: run constant_fold on flat ops and emit the result."""
    from material_proxy.emit import emit_vmt

    folded_ops, folded_consts = constant_fold(list(ops), dict(consts))
    return emit_vmt(folded_ops, folded_consts)


def _fold_ops(ops: list[FlatOp], consts: dict[float, str]) -> list[FlatOp]:
    """Helper: run constant_fold and return the remaining ops."""
    folded_ops, _ = constant_fold(list(ops), dict(consts))
    return folded_ops


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


def test_identity_full_pipeline_example():
    """The motivating example: (1+x)*4.3e-08*0 + x → x after full optimize."""
    tmp = Mul(Mul(Add(Const(1.0), Var('x')), Const(4.3e-08)), Const(0.0))
    expr = Add(tmp, Var('x'))
    prog = Program.from_tree(result=expr)
    # After constant_fold alone, Equals should reference $x directly
    folded = prog.optimize(constant_fold)
    eq = next(op for op in folded.ops if op.proxy == 'Equals')
    assert eq.params['srcVar1'] == '$x'
    # After full optimize, only Equals remains
    full = prog.optimize(full_optimize)
    assert len(full.ops) == 1
    assert full.ops[0].proxy == 'Equals'
    # Correctness
    vmt = full.emit()
    state = interpret_vmt(vmt, EvalContext(vars={'$x': 3.0}))
    assert state['$result'] == approx(3.0)


# ---- Peephole (_is_truthy fusion) -------------------------------------------


def test_peephole_fusion_simple():
    """LessOrEqual(cond, 0, 0, 1) + LessOrEqual(result, 0, fr, tr) fuse."""
    params_t = {
        'srcVar1': '$cond',
        'srcVar2': '$0.0',
        'lessEqualVar': '$0.0',
        'greaterVar': '$1.0',
    }
    params_o = {
        'srcVar1': '$tmp_t',
        'srcVar2': '$0.0',
        'lessEqualVar': '$fr',
        'greaterVar': '$tr',
    }
    ops = [
        FlatOp('LessOrEqual', params_t, '$tmp_t'),
        FlatOp('LessOrEqual', params_o, '$tmp_r'),
    ]
    fused, _ = peephole_optimize(ops, {})
    assert len(fused) == 1
    assert fused[0].params['srcVar1'] == '$cond'
    assert fused[0].params['lessEqualVar'] == '$fr'
    assert fused[0].params['greaterVar'] == '$tr'


def test_peephole_fusion_interleaved():
    """Ops between truthy and outer don't block fusion."""
    params_t = {
        'srcVar1': '$cond',
        'srcVar2': '$0.0',
        'lessEqualVar': '$0.0',
        'greaterVar': '$1.0',
    }
    params_o = {
        'srcVar1': '$tmp_t',
        'srcVar2': '$0.0',
        'lessEqualVar': '$inter',
        'greaterVar': '$tr',
    }
    ops = [
        FlatOp('LessOrEqual', params_t, '$tmp_t'),
        FlatOp('Add', {'srcVar1': '$a', 'srcVar2': '$b'}, '$inter'),
        FlatOp('LessOrEqual', params_o, '$tmp_r'),
    ]
    fused, _ = peephole_optimize(ops, {})
    assert len(fused) == 2  # Add kept, fused LessOrEqual kept
    assert any(
        op.proxy == 'LessOrEqual' and op.params['srcVar1'] == '$cond' for op in fused
    )


def test_peephole_no_fusion_wrong_pattern():
    """Non-matching patterns are left alone."""
    # Truthy's lessEqualVar is not $0.0
    params_t = {
        'srcVar1': '$cond',
        'srcVar2': '$0.0',
        'lessEqualVar': '$x',
        'greaterVar': '$1.0',
    }
    params_o = {
        'srcVar1': '$tmp_t',
        'srcVar2': '$0.0',
        'lessEqualVar': '$fr',
        'greaterVar': '$tr',
    }
    ops = [
        FlatOp('LessOrEqual', params_t, '$tmp_t'),
        FlatOp('LessOrEqual', params_o, '$tmp_r'),
    ]
    fused, _ = peephole_optimize(ops, {})
    assert len(fused) == 2  # both kept


def test_peephole_correctness():
    """Fused LessOrEqual(cond, 0, fr, tr) produces the same result."""
    cond = Var('x')
    tr = Const(10.0)
    fr = Const(20.0)
    truthy = LessOrEqual(cond, Const(0), Const(0), Const(1))
    outer = LessOrEqual(truthy, Const(0), fr, tr)
    prog = Program.from_tree(result=outer)
    fused = prog.optimize(peephole_optimize)
    assert fused.ops[0].proxy == 'LessOrEqual'
    assert fused.ops[0].params['srcVar1'] == '$x'


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


def test_dce_removes_unused():
    """DCE removes ops whose result is never used."""
    ops = [
        FlatOp('Add', {'srcVar1': '$1', 'srcVar2': '$2'}, '$tmp_1'),
        FlatOp('Add', {'srcVar1': '$3', 'srcVar2': '$4'}, '$tmp_2'),
        FlatOp('Multiply', {'srcVar1': '$tmp_1', 'srcVar2': '$5'}, '$tmp_3'),
        FlatOp('Equals', {'srcVar1': '$tmp_3'}, 'out'),
    ]
    prog = Program.from_flat(ops, {}).optimize(dce_only)
    result = prog.ops
    assert len(result) == 3
    assert result[0].proxy == 'Add'
    assert result[0].result == '$tmp_1'
    assert result[1].result == '$tmp_3'
    assert result[2].proxy == 'Equals'


def test_dce_keeps_all():
    """All ops kept when every result is referenced."""
    ops = [
        FlatOp('Mul', {'srcVar1': '$x', 'srcVar2': '$y'}, '$tmp_1'),
        FlatOp('Add', {'srcVar1': '$tmp_1', 'srcVar2': '$z'}, '$tmp_2'),
        FlatOp('Equals', {'srcVar1': '$tmp_2'}, 'out'),
    ]
    prog = Program.from_flat(ops, {}).optimize(dce_only)
    assert len(prog.ops) == 3


def test_dce_chain_dead():
    """Mid-chain op removed when its result is unused."""
    ops = [
        FlatOp('Add', {'srcVar1': '$1', 'srcVar2': '$2'}, '$tmp_1'),
        FlatOp('Mul', {'srcVar1': '$tmp_1', 'srcVar2': '$3'}, '$tmp_2'),
        FlatOp('Sub', {'srcVar1': '$4', 'srcVar2': '$5'}, '$tmp_3'),
        FlatOp('Add', {'srcVar1': '$tmp_2', 'srcVar2': '$tmp_3'}, '$tmp_4'),
    ]
    out1 = FlatOp('Equals', {'srcVar1': '$tmp_4'}, 'out')
    prog = Program.from_flat([*ops, out1], {}).optimize(dce_only)
    assert len(prog.ops) == 5

    out2 = FlatOp('Equals', {'srcVar1': '$tmp_2'}, 'out')
    prog2 = Program.from_flat([*ops, out2], {}).optimize(dce_only)
    assert len(prog2.ops) == 3
    assert {op.result for op in prog2.ops} == {'$tmp_1', '$tmp_2', 'out'}


def test_dce_empty_live():
    """No live temps — all ops dead."""
    ops = [FlatOp('Add', {'srcVar1': '$1', 'srcVar2': '$2'}, '$tmp_1')]
    prog = Program.from_flat(ops, {}).optimize(dce_only)
    assert len(prog.ops) == 0


# ---- Temp reuse ------------------------------------------------------------


def test_reuse_independent():
    """Independent temps may share slots when lifetimes don't overlap."""
    ops = [
        FlatOp('Add', {'srcVar1': '$1', 'srcVar2': '$2'}, '$tmp_1'),
        FlatOp('Add', {'srcVar1': '$3', 'srcVar2': '$4'}, '$tmp_2'),
        FlatOp('Multiply', {'srcVar1': '$tmp_1', 'srcVar2': '$tmp_2'}, '$tmp_3'),
    ]
    prog = Program.from_flat(ops, {}).optimize(reuse_only)
    assert max_temp_index(prog.ops) <= 3


def test_reuse_chain():
    """Chained dependent temps — temps reused after last use."""
    ops = [
        FlatOp('Add', {'srcVar1': '$1', 'srcVar2': '$2'}, '$tmp_1'),
        FlatOp('Mul', {'srcVar1': '$tmp_1', 'srcVar2': '$3'}, '$tmp_2'),
        FlatOp('Sub', {'srcVar1': '$tmp_2', 'srcVar2': '$4'}, '$tmp_3'),
    ]
    prog = Program.from_flat(ops, {}).optimize(reuse_only)
    assert max_temp_index(prog.ops) <= 2


def test_reuse_semantics_preserved():
    """Temp reuse doesn't change computed results."""
    prog = Program.from_tree(result=Add(Var('x'), Var('y')))
    ctx = EvalContext(vars={'$x': 3.0, '$y': 4.0})
    raw_state = interpret_vmt(prog.emit(), ctx)
    reused_state = interpret_vmt(prog.optimize(reuse_only).emit(), ctx)
    assert raw_state == reused_state


def test_reuse_all_disjoint():
    """Fully disjoint lifetimes — every temp reuses slot 1."""
    ops = [
        FlatOp('Add', {'srcVar1': '$a', 'srcVar2': '$b'}, '$tmp_1'),
        FlatOp('Add', {'srcVar1': '$c', 'srcVar2': '$d'}, '$tmp_2'),
        FlatOp('Add', {'srcVar1': '$e', 'srcVar2': '$f'}, '$tmp_3'),
        FlatOp('Add', {'srcVar1': '$g', 'srcVar2': '$h'}, '$tmp_4'),
    ]
    prog = Program.from_flat(ops, {}).optimize(reuse_only)
    assert max_temp_index(prog.ops) == 1


def test_reuse_all_overlap():
    """Temps all overlap initially, but in-place reuse applies at the end."""
    ops = [
        FlatOp('Add', {'srcVar1': '$a', 'srcVar2': '$b'}, '$tmp_1'),
        FlatOp('Multiply', {'srcVar1': '$tmp_1', 'srcVar2': '$c'}, '$tmp_2'),
        FlatOp('Subtract', {'srcVar1': '$tmp_1', 'srcVar2': '$tmp_2'}, '$tmp_3'),
        FlatOp('Add', {'srcVar1': '$tmp_1', 'srcVar2': '$tmp_3'}, '$tmp_4'),
        FlatOp('Multiply', {'srcVar1': '$tmp_2', 'srcVar2': '$tmp_4'}, '$tmp_5'),
    ]
    prog = Program.from_flat(ops, {}).optimize(reuse_only)
    assert max_temp_index(prog.ops) < 4


@pytest.mark.parametrize('depth', [20, 50, 100])
def test_reuse_long_chain_correctness(depth: int) -> None:
    """Long chain of dependent ops — correct and ≤ 2 slots."""
    ops = []
    prev = '$input'
    for i in range(1, depth + 1):
        ops.append(FlatOp('Add', {'srcVar1': prev, 'srcVar2': '$1'}, f'$tmp_{i}'))
        prev = f'$tmp_{i}'
    prog = Program.from_flat(ops, {1.0: '$1'}).optimize(reuse_only)
    assert max_temp_index(prog.ops) <= 2
    ctx = EvalContext(vars={'$input': 0.0})
    state = interpret_vmt(prog.emit(), ctx)
    assert last_temp_result(state) == approx(float(depth))


def test_reuse_diamond_dag():
    """Diamond DAG: shared sub-expression has short liveness — recycled after."""
    ops = [
        FlatOp('Add', {'srcVar1': '$a', 'srcVar2': '$b'}, '$tmp_1'),
        FlatOp('Add', {'srcVar1': '$c', 'srcVar2': '$d'}, '$tmp_2'),
        FlatOp('Multiply', {'srcVar1': '$tmp_1', 'srcVar2': '$tmp_2'}, '$tmp_3'),
        FlatOp('Subtract', {'srcVar1': '$tmp_3', 'srcVar2': '$e'}, '$tmp_4'),
        FlatOp('Add', {'srcVar1': '$f', 'srcVar2': '$tmp_4'}, '$tmp_5'),
    ]
    prog = Program.from_flat(ops, {}).optimize(reuse_only)
    assert max_temp_index(prog.ops) <= 3
    ctx = EvalContext(
        vars={
            '$a': 1.0,
            '$b': 2.0,
            '$c': 3.0,
            '$d': 4.0,
            '$e': 5.0,
            '$f': 6.0,
        }
    )
    state = interpret_vmt(prog.emit(), ctx)
    values = [v for v in state.values() if isinstance(v, (int, float))]
    assert any(abs(v - 22.0) < 1e-9 for v in values)


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


def test_reuse_flow_graph_with_fork():
    """Fork-join: a value used by two downstream ops — temp freed only after both."""
    ops = [
        FlatOp('Add', {'srcVar1': '$a', 'srcVar2': '$b'}, '$tmp_1'),
        FlatOp('Multiply', {'srcVar1': '$tmp_1', 'srcVar2': '$c'}, '$tmp_2'),
        FlatOp('Subtract', {'srcVar1': '$tmp_1', 'srcVar2': '$d'}, '$tmp_3'),
        FlatOp('Divide', {'srcVar1': '$tmp_2', 'srcVar2': '$tmp_3'}, '$tmp_4'),
    ]
    prog = Program.from_flat(ops, {}).optimize(reuse_only)
    assert max_temp_index(prog.ops) <= 3
    ctx = EvalContext(vars={'$a': 10.0, '$b': 2.0, '$c': 3.0, '$d': 4.0})
    state = interpret_vmt(prog.emit(), ctx)
    values = [v for v in state.values() if isinstance(v, (int, float))]
    assert any(abs(v - 4.5) < 1e-9 for v in values)
