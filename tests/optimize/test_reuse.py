# ruff: noqa: ANN201

import pytest
from pytest import approx

from material_proxy import Add
from material_proxy import EvalContext
from material_proxy import Program
from material_proxy import Var
from material_proxy import interpret_vmt
from material_proxy.flatten import FlatOp
from tests._test_helpers import last_temp_result
from tests._test_helpers import max_temp_index
from tests._test_helpers import reuse_only


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
