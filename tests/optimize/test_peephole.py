# ruff: noqa: ANN201


from material_proxy import Const
from material_proxy import LessOrEqual
from material_proxy import Program
from material_proxy import Var
from material_proxy import peephole_optimize
from material_proxy.flatten import FlatOp


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
