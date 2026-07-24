# ruff: noqa: ANN201

from material_proxy import Program
from material_proxy.flatten import FlatOp
from tests._test_helpers import dce_only


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
