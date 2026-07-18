# ruff: noqa: ANN201

import pytest
from pytest import approx

from material_proxy import Abs
from material_proxy import Add
from material_proxy import Const
from material_proxy import Div
from material_proxy import EvalContext
from material_proxy import Expr
from material_proxy import Mul
from material_proxy import Sub
from material_proxy import Var
from material_proxy import compile_to_vmt
from material_proxy import interpret_vmt
from material_proxy.emit import emit_vmt
from material_proxy.flatten import FlatOp
from material_proxy.flatten import Flattener
from material_proxy.optimize import constant_fold
from material_proxy.optimize import dead_code_elimination
from material_proxy.optimize import temp_reuse


def _last_result(state: dict[str, float]) -> float:
    """Find the highest-numbered ``$tmp_N`` value."""
    key = max(
        (k for k in state if k.startswith('$tmp_')),
        key=lambda k: int(k.removeprefix('$tmp_')),
        default=None,
    )
    return state[key] if key is not None else next(iter(state.values()))


def _max_temp(ops: list[FlatOp]) -> int:
    """Maximum ``$tmp_N`` index in *ops* (0 if none)."""
    n = 0
    for op in ops:
        if op.result.startswith('$tmp_'):
            n = max(n, int(op.result.removeprefix('$tmp_')))
    return n


# ---- Constant folding -------------------------------------------------------


def test_fold_single_binary():
    """Div($6.0, $2.0) → constant folded, no divide in output."""
    expr = Div(Const(6.0), Const(2.0))
    flattener = Flattener()
    flattener.flatten(expr)
    res = flattener.result()
    folded_ops, folded_consts, _ = constant_fold(res.ops, res.consts)
    assert len(folded_ops) == 0
    vmt = emit_vmt(folded_ops, folded_consts)
    state = interpret_vmt(vmt, EvalContext())
    assert state.get('$3.0') == approx(3.0)


def test_fold_full_tree():
    """Entirely constant expression fully folded."""
    expr = Mul(Add(Const(1.0), Const(2.0)), Const(5.0))
    flattener = Flattener()
    flattener.flatten(expr)
    res = flattener.result()
    folded_ops, folded_consts, _ = constant_fold(res.ops, res.consts)
    assert len(folded_ops) == 0
    assert 15.0 in folded_consts


def test_fold_partial():
    """Mixed constant and variable — only constant parts fold."""
    expr = Add(Mul(Const(2.0), Var('x')), Const(5.0))
    flattener = Flattener()
    flattener.flatten(expr)
    res = flattener.result()
    folded_ops, folded_consts, _ = constant_fold(res.ops, res.consts)
    # Neither op is fully constant — Mul($2, $x) depends on $x
    assert len(folded_ops) == 2
    ctx = EvalContext(vars={'$x': 3.0})
    vmt = emit_vmt(folded_ops, folded_consts)
    state = interpret_vmt(vmt, ctx)
    assert _last_result(state) == approx(11.0)


def test_fold_abs_neg():
    """Abs folded at compile time."""
    expr = Abs(Const(-3.0))
    flattener = Flattener()
    flattener.flatten(expr)
    res = flattener.result()
    folded_ops, folded_consts, _ = constant_fold(res.ops, res.consts)
    assert len(folded_ops) == 0
    assert 3.0 in folded_consts


def test_fold_chained():
    """Chained constant expressions all fold."""
    inner = Add(Const(1.0), Const(2.0))
    expr = Mul(inner, Add(Const(3.0), Const(4.0)))
    flattener = Flattener()
    flattener.flatten(expr)
    res = flattener.result()
    folded_ops, folded_consts, _ = constant_fold(res.ops, res.consts)
    assert len(folded_ops) == 0
    assert 21.0 in folded_consts


def test_fold_no_side_effects():
    """Folding produces same numerical result as non-folded compilation."""
    expr = Mul(Add(Const(1.0), Const(2.0)), Var('x'))
    ctx = EvalContext(vars={'$x': 5.0})
    vmt_normal = compile_to_vmt(expr)
    state_normal = interpret_vmt(vmt_normal, ctx)

    flattener = Flattener()
    flattener.flatten(expr)
    res = flattener.result()
    folded_ops, folded_consts, _ = constant_fold(res.ops, res.consts)
    vmt_folded = emit_vmt(folded_ops, folded_consts)
    state_folded = interpret_vmt(vmt_folded, ctx)

    assert _last_result(state_folded) == approx(15.0)
    assert _last_result(state_normal) == approx(15.0)


# ---- Dead code elimination --------------------------------------------------


def test_dce_removes_unused():
    """DCE removes ops whose result is never used."""
    ops = [
        FlatOp('Add', {'srcVar1': '$1', 'srcVar2': '$2'}, '$tmp_1'),
        FlatOp('Add', {'srcVar1': '$3', 'srcVar2': '$4'}, '$tmp_2'),
        FlatOp('Multiply', {'srcVar1': '$tmp_1', 'srcVar2': '$5'}, '$tmp_3'),
    ]
    result = dead_code_elimination(ops, {'$tmp_3'})
    assert len(result) == 2
    assert result[0].proxy == 'Add'
    assert result[0].result == '$tmp_1'
    assert result[1].result == '$tmp_3'


def test_dce_keeps_all():
    """All ops kept when every result is referenced."""
    ops = [
        FlatOp('Mul', {'srcVar1': '$x', 'srcVar2': '$y'}, '$tmp_1'),
        FlatOp('Add', {'srcVar1': '$tmp_1', 'srcVar2': '$z'}, '$tmp_2'),
    ]
    result = dead_code_elimination(ops, {'$tmp_2'})
    assert len(result) == 2


def test_dce_chain_dead():
    """Mid-chain op removed when its result is unused."""
    ops = [
        FlatOp('Add', {'srcVar1': '$1', 'srcVar2': '$2'}, '$tmp_1'),
        FlatOp('Mul', {'srcVar1': '$tmp_1', 'srcVar2': '$3'}, '$tmp_2'),
        FlatOp('Sub', {'srcVar1': '$4', 'srcVar2': '$5'}, '$tmp_3'),
        FlatOp('Add', {'srcVar1': '$tmp_2', 'srcVar2': '$tmp_3'}, '$tmp_4'),
    ]
    result = dead_code_elimination(ops, {'$tmp_4'})
    # $tmp_3 used by tmp_4, so Sub kept; $tmp_1/$tmp_2 used chain; all 4 ops kept
    assert len(result) == 4
    result2 = dead_code_elimination(ops, {'$tmp_2'})
    # Only the chain leading to $tmp_2 kept; $tmp_3/$tmp_4 dead
    assert len(result2) == 2
    assert {op.result for op in result2} == {'$tmp_1', '$tmp_2'}


def test_dce_empty_live():
    """No live temps → all ops dead."""
    ops = [FlatOp('Add', {'srcVar1': '$1', 'srcVar2': '$2'}, '$tmp_1')]
    result = dead_code_elimination(ops, set())
    assert len(result) == 0


# ---- Temp reuse ------------------------------------------------------------


def test_reuse_independent():
    """Independent temps may share slots when lifetimes don't overlap."""
    ops = [
        FlatOp('Add', {'srcVar1': '$1', 'srcVar2': '$2'}, '$tmp_1'),
        FlatOp('Add', {'srcVar1': '$3', 'srcVar2': '$4'}, '$tmp_2'),
        FlatOp('Multiply', {'srcVar1': '$tmp_1', 'srcVar2': '$tmp_2'}, '$tmp_3'),
    ]
    result = temp_reuse(ops)
    assert _max_temp(result) <= 3


def test_reuse_chain():
    """Chained dependent temps — temps reused after last use."""
    ops = [
        FlatOp('Add', {'srcVar1': '$1', 'srcVar2': '$2'}, '$tmp_1'),
        FlatOp('Mul', {'srcVar1': '$tmp_1', 'srcVar2': '$3'}, '$tmp_2'),
        FlatOp('Sub', {'srcVar1': '$tmp_2', 'srcVar2': '$4'}, '$tmp_3'),
    ]
    result = temp_reuse(ops)
    # tmp_1 and tmp_3 can share (non-overlapping), so max ≤ 2
    assert _max_temp(result) <= 2


def test_reuse_semantics_preserved():
    """Temp reuse doesn't change computed results."""
    expr = Add(Var('x'), Var('y'))
    flattener = Flattener()
    flattener.flatten(expr)
    res = flattener.result()
    original = emit_vmt(res.ops, res.consts)
    reused = emit_vmt(temp_reuse(res.ops), res.consts)
    ctx = EvalContext(vars={'$x': 3.0, '$y': 4.0})
    assert interpret_vmt(original, ctx) == interpret_vmt(reused, ctx)


def test_reuse_all_disjoint():
    """Fully disjoint lifetimes — every temp reuses slot 1."""
    ops = [
        FlatOp('Add', {'srcVar1': '$a', 'srcVar2': '$b'}, '$tmp_1'),
        FlatOp('Add', {'srcVar1': '$c', 'srcVar2': '$d'}, '$tmp_2'),
        FlatOp('Add', {'srcVar1': '$e', 'srcVar2': '$f'}, '$tmp_3'),
        FlatOp('Add', {'srcVar1': '$g', 'srcVar2': '$h'}, '$tmp_4'),
    ]
    result = temp_reuse(ops)
    assert _max_temp(result) == 1


def test_reuse_all_overlap():
    """Every temp live simultaneously — no reuse possible."""
    # Each op reads all previously written temps, forcing full overlap.
    ops = [
        FlatOp('Add', {'srcVar1': '$a', 'srcVar2': '$b'}, '$tmp_1'),
        FlatOp('Multiply', {'srcVar1': '$tmp_1', 'srcVar2': '$c'}, '$tmp_2'),
        FlatOp('Subtract', {'srcVar1': '$tmp_1', 'srcVar2': '$tmp_2'}, '$tmp_3'),
        FlatOp('Add', {'srcVar1': '$tmp_1', 'srcVar2': '$tmp_3'}, '$tmp_4'),
        FlatOp('Multiply', {'srcVar1': '$tmp_2', 'srcVar2': '$tmp_4'}, '$tmp_5'),
    ]
    result = temp_reuse(ops)
    assert _max_temp(result) == 4


@pytest.mark.parametrize('depth', [20, 50, 100])
def test_reuse_long_chain_correctness(depth: int) -> None:
    """Long chain of dependent ops — correct and ≤ 2 slots."""
    ops = []
    prev = '$input'
    for i in range(1, depth + 1):
        ops.append(FlatOp('Add', {'srcVar1': prev, 'srcVar2': '$1'}, f'$tmp_{i}'))
        prev = f'$tmp_{i}'
    result = temp_reuse(ops)
    assert _max_temp(result) <= 2
    consts = {1.0: '$1'}
    vmt = emit_vmt(result, consts)
    ctx = EvalContext(vars={'$input': 0.0})
    state = interpret_vmt(vmt, ctx)
    assert _last_result(state) == approx(float(depth))


def test_reuse_diamond_dag():
    """Diamond DAG: shared sub-expression has short liveness — recycled after."""
    #     $a  $b       $c  $d
    #       \ /          \ /
    #       tmp_1       tmp_2       (Add pairs)
    #        |            |
    #       tmp_3 = Mul(tmp_1, tmp_2)   ← last use of tmp_1, tmp_2
    #        |
    #       tmp_4 = Sub(tmp_3, $e)      ← last use of tmp_3
    #  $f --|
    #        \
    #        tmp_5 = Add($f, tmp_4)     ← last use of tmp_4
    #        |
    #       (output)
    ops = [
        FlatOp('Add', {'srcVar1': '$a', 'srcVar2': '$b'}, '$tmp_1'),
        FlatOp('Add', {'srcVar1': '$c', 'srcVar2': '$d'}, '$tmp_2'),
        FlatOp('Multiply', {'srcVar1': '$tmp_1', 'srcVar2': '$tmp_2'}, '$tmp_3'),
        FlatOp('Subtract', {'srcVar1': '$tmp_3', 'srcVar2': '$e'}, '$tmp_4'),
        FlatOp('Add', {'srcVar1': '$f', 'srcVar2': '$tmp_4'}, '$tmp_5'),
    ]
    result = temp_reuse(ops)
    # Intervals: tmp_1=[0,2], tmp_2=[1,2], tmp_3=[2,3], tmp_4=[3,4], tmp_5=[4,4]
    # Greedy needs at least 3 slots: tmp_1, tmp_2, tmp_3 all overlap.
    # tmp_4 uses slot 1, tmp_5 uses slot 2.
    assert _max_temp(result) <= 3
    # Correctness check
    vmt = emit_vmt(result, {})
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
    state = interpret_vmt(vmt, ctx)
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
        flat = Flattener()
        flat.flatten(expr)
        res = flat.result()

        folded_ops, folded_consts, _ = constant_fold(res.ops, res.consts)
        live = set(folded_consts.values())
        for op in folded_ops:
            if op.result.startswith('$') and not op.result.startswith('$tmp_'):
                live.add(op.result)
        if folded_ops:
            live.add(folded_ops[-1].result)
        dce_ops = dead_code_elimination(folded_ops, live)
        final_ops = temp_reuse(dce_ops)
        opt_vmt = emit_vmt(final_ops, folded_consts)
        got = interpret_vmt(opt_vmt, ctx)
        # _last_result is unreliable after temp_reuse renumbers temps,
        # so verify against the known expected value instead.
        vals = [v for v in got.values() if isinstance(v, (int, float))]
        assert expected in vals or any(abs(v - expected) < 1e-9 for v in vals)


def test_reuse_flow_graph_with_fork():
    """Fork-join: a value used by two downstream ops — temp freed only after both."""
    # tmp_1 used by both tmp_2 and tmp_3 → stays alive until both done
    ops = [
        FlatOp('Add', {'srcVar1': '$a', 'srcVar2': '$b'}, '$tmp_1'),
        FlatOp('Multiply', {'srcVar1': '$tmp_1', 'srcVar2': '$c'}, '$tmp_2'),
        FlatOp('Subtract', {'srcVar1': '$tmp_1', 'srcVar2': '$d'}, '$tmp_3'),
        FlatOp('Divide', {'srcVar1': '$tmp_2', 'srcVar2': '$tmp_3'}, '$tmp_4'),
    ]
    result = temp_reuse(ops)
    # Intervals: tmp_1=[0,2], tmp_2=[1,3], tmp_3=[2,3], tmp_4=[3,3]
    # tmp_1 & tmp_2 overlap at [1,2]; tmp_2 & tmp_3 overlap at [2,3].
    # Greedy: tmp_1→1, tmp_2→2, tmp_3→3 (overlaps both), tmp_4→1 (tmp_1 free).
    assert _max_temp(result) <= 3
    vmt = emit_vmt(result, {})
    ctx = EvalContext(vars={'$a': 10.0, '$b': 2.0, '$c': 3.0, '$d': 4.0})
    state = interpret_vmt(vmt, ctx)
    values = [v for v in state.values() if isinstance(v, (int, float))]
    assert any(abs(v - 4.5) < 1e-9 for v in values)
