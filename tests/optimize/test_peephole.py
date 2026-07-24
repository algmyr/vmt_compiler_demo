# ruff: noqa: ANN201

from pytest import approx

from material_proxy import Const
from material_proxy import EvalContext
from material_proxy import LessOrEqual
from material_proxy import Program
from material_proxy import SelectFirstIfNonZero
from material_proxy import Var
from material_proxy import full_optimize
from material_proxy import interpret_vmt
from material_proxy import peephole_optimize
from material_proxy.flatten import FlatOp


def test_identity_identical_branches():
    """LessOrEqual(x, y, c, c) aliases result to c regardless of comparison."""
    ops = [
        FlatOp(
            'LessOrEqual',
            {
                'srcVar1': '$x',
                'srcVar2': '$y',
                'lessEqualVar': '$c',
                'greaterVar': '$c',
            },
            '$tmp',
        ),
    ]
    out, _ = peephole_optimize(ops, {})
    assert out == []


def test_identity_sfi_same_args():
    """SelectFirstIfNonZero(x, x) aliases result to x."""
    ops = [FlatOp('SelectFirstIfNonZero', {'srcVar1': '$x', 'srcVar2': '$x'}, '$tmp')]
    out, _ = peephole_optimize(ops, {})
    assert out == []


def test_identity_sfi_zero_first():
    """SelectFirstIfNonZero(0, x) aliases result to x."""
    ops = [FlatOp('SelectFirstIfNonZero', {'srcVar1': '$0.0', 'srcVar2': '$x'}, '$tmp')]
    out, _ = peephole_optimize(ops, {})
    assert out == []


def test_identity_sfi_nonzero_first_left_alone():
    """SelectFirstIfNonZero(nonzero, x) where nonzero is not 0 is left alone."""
    ops = [
        FlatOp('SelectFirstIfNonZero', {'srcVar1': '$x', 'srcVar2': '$y'}, '$tmp'),
    ]
    out, _ = peephole_optimize(ops, {})
    assert len(out) == 1
    assert out[0].params['srcVar1'] == '$x'
    assert out[0].params['srcVar2'] == '$y'


def test_identity_alias_propagates_to_consumers():
    """Aliased result is resolved in downstream ops' params."""
    ops = [
        FlatOp('SelectFirstIfNonZero', {'srcVar1': '$0.0', 'srcVar2': '$x'}, '$alias'),
        FlatOp('Add', {'srcVar1': '$alias', 'srcVar2': '$y'}, '$sum'),
    ]
    out, _ = peephole_optimize(ops, {})
    assert len(out) == 1
    assert out[0].proxy == 'Add'
    assert out[0].params['srcVar1'] == '$x'


def test_identity_alias_chain():
    """Alias chains are resolved (a -> b -> c -> x)."""
    ops = [
        FlatOp('SelectFirstIfNonZero', {'srcVar1': '$0.0', 'srcVar2': '$a'}, '$t1'),
        FlatOp('SelectFirstIfNonZero', {'srcVar1': '$0.0', 'srcVar2': '$t1'}, '$t2'),
        FlatOp('SelectFirstIfNonZero', {'srcVar1': '$0.0', 'srcVar2': '$t2'}, '$t3'),
        FlatOp('Add', {'srcVar1': '$t3', 'srcVar2': '$b'}, '$sum'),
    ]
    out, _ = peephole_optimize(ops, {})
    assert len(out) == 1
    assert out[0].params['srcVar1'] == '$a'


def test_identity_alias_composes_with_truthy_fusion():
    """Identity alias + truthy fusion — alias resolved before truthy check."""
    ops = [
        # $t_id aliases to $c (identical branches)
        FlatOp(
            'LessOrEqual',
            {
                'srcVar1': '$a',
                'srcVar2': '$b',
                'lessEqualVar': '$c',
                'greaterVar': '$c',
            },
            '$t_id',
        ),
        # truthy($t_id) — should resolve to truthy($c) after alias
        FlatOp(
            'LessOrEqual',
            {
                'srcVar1': '$t_id',
                'srcVar2': '$0.0',
                'lessEqualVar': '$0.0',
                'greaterVar': '$1.0',
            },
            '$t_truthy',
        ),
        # consumer of truthy result
        FlatOp(
            'LessOrEqual',
            {
                'srcVar1': '$t_truthy',
                'srcVar2': '$0.0',
                'lessEqualVar': '$fr',
                'greaterVar': '$tr',
            },
            '$result',
        ),
    ]
    out, _ = peephole_optimize(ops, {})
    # After full peephole: identity alias removes $t_id,
    # truthy($c) → fusion gives LessOrEqual($c, 0, $fr, $tr)
    assert len(out) == 1
    assert out[0].params['srcVar1'] == '$c'
    assert out[0].params['lessEqualVar'] == '$fr'
    assert out[0].params['greaterVar'] == '$tr'


def test_identity_alias_full_pipeline_correctness():
    """Identity alias patterns + full_optimize produce correct results."""
    # SelectFirstIfNonZero(0, x) is an identity: result = x
    sfi = SelectFirstIfNonZero(Const(0.0), Var('v'))
    prog = Program.from_tree(result=sfi)
    vmt = prog.compile(optimize=full_optimize)
    state = interpret_vmt(vmt, EvalContext(vars={'$v': 42.0}))
    assert state['$result'] == approx(42.0)


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
