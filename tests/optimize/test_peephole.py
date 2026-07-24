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


def test_double_not_basic():
    """NOT(NOT(x)) is eliminated to _is_truthy(x) (inner NOT becomes dead)."""
    ops = [
        FlatOp(
            'LessOrEqual',
            {
                'srcVar1': '$x',
                'srcVar2': '$0.0',
                'lessEqualVar': '$1.0',
                'greaterVar': '$0.0',
            },
            '$tmp_i',
        ),
        FlatOp(
            'LessOrEqual',
            {
                'srcVar1': '$tmp_i',
                'srcVar2': '$0.0',
                'lessEqualVar': '$1.0',
                'greaterVar': '$0.0',
            },
            '$tmp_o',
        ),
    ]
    out, _ = peephole_optimize(ops, {})
    assert len(out) == 2  # inner NOT kept (dead), outer transformed
    truthy_op = next(op for op in out if op.params.get('lessEqualVar') == '$0.0')
    assert truthy_op.params['srcVar1'] == '$x'
    assert truthy_op.params['greaterVar'] == '$1.0'

    # Inner NOT should be first (unchanged)
    inner = out[0]
    assert inner.params['lessEqualVar'] == '$1.0'
    assert inner.params['greaterVar'] == '$0.0'


def test_double_not_with_consumer():
    """Double-NOT result consumed by LessOrEqual select fuses further."""
    ops = [
        FlatOp(
            'LessOrEqual',
            {
                'srcVar1': '$x',
                'srcVar2': '$0.0',
                'lessEqualVar': '$1.0',
                'greaterVar': '$0.0',
            },
            '$tmp_i',
        ),
        FlatOp(
            'LessOrEqual',
            {
                'srcVar1': '$tmp_i',
                'srcVar2': '$0.0',
                'lessEqualVar': '$1.0',
                'greaterVar': '$0.0',
            },
            '$tmp_o',
        ),
        FlatOp(
            'LessOrEqual',
            {
                'srcVar1': '$tmp_o',
                'srcVar2': '$0.0',
                'lessEqualVar': '$fr',
                'greaterVar': '$tr',
            },
            '$result',
        ),
    ]
    out, _ = peephole_optimize(ops, {})
    # The consumer is fused: LessOrEqual(x, 0, $fr, $tr)
    fused = next(op for op in out if op.params.get('lessEqualVar') == '$fr')
    assert fused.params['srcVar1'] == '$x'
    assert fused.params['greaterVar'] == '$tr'
    assert len(out) == 3  # dead NOT(x) + dead _is_truthy(x) kept (DCE cleans)


def test_double_not_correctness():
    """Double NOT yields same result as _is_truthy via full pipeline."""
    x = Var('x')
    not_x = LessOrEqual(x, Const(0), Const(1), Const(0))
    not_not_x = LessOrEqual(not_x, Const(0), Const(1), Const(0))
    prog = Program.from_tree(result=not_not_x)
    vmt = prog.compile(optimize=full_optimize)
    for val in (-5.0, 0.0, 3.0):
        state = interpret_vmt(vmt, EvalContext(vars={'$x': val}))
        # NOT(NOT(x)) should be 1 if x > 0, else 0
        expected = 1.0 if val > 0 else 0.0
        assert state['$result'] == approx(expected)


def test_neg_absorb_basic():
    """NOT(LessOrEqual(x, y, le, gr)) → LessOrEqual(x, y, gr, le)."""
    ops = [
        FlatOp(
            'LessOrEqual',
            {
                'srcVar1': '$x',
                'srcVar2': '$y',
                'lessEqualVar': '$le',
                'greaterVar': '$gr',
            },
            '$tmp_inner',
        ),
        FlatOp(
            'LessOrEqual',
            {
                'srcVar1': '$tmp_inner',
                'srcVar2': '$0.0',
                'lessEqualVar': '$1.0',
                'greaterVar': '$0.0',
            },
            '$tmp_out',
        ),
    ]
    out, _ = peephole_optimize(ops, {})
    assert len(out) == 1
    assert out[0].params['srcVar1'] == '$x'
    assert out[0].params['srcVar2'] == '$y'
    assert out[0].params['lessEqualVar'] == '$gr'
    assert out[0].params['greaterVar'] == '$le'


def test_neg_absorb_inner_has_other_consumer():
    """Negation absorption skipped when inner result is used elsewhere."""
    ops = [
        FlatOp(
            'LessOrEqual',
            {
                'srcVar1': '$x',
                'srcVar2': '$y',
                'lessEqualVar': '$le',
                'greaterVar': '$gr',
            },
            '$tmp_inner',
        ),
        FlatOp(
            'LessOrEqual',
            {
                'srcVar1': '$tmp_inner',
                'srcVar2': '$0.0',
                'lessEqualVar': '$1.0',
                'greaterVar': '$0.0',
            },
            '$tmp_out',
        ),
        FlatOp(
            'Add',
            {'srcVar1': '$tmp_inner', 'srcVar2': '$z'},
            '$other',
        ),
    ]
    out, _ = peephole_optimize(ops, {})
    # Inner kept (has second consumer), NOT remains unchanged
    assert len(out) == 3
    assert any(
        op
        for op in out
        if op.proxy == 'LessOrEqual'
        and op.params.get('lessEqualVar') == '$le'
        and op.params.get('greaterVar') == '$gr'
    )
    assert any(
        op
        for op in out
        if op.proxy == 'LessOrEqual'
        and op.params.get('lessEqualVar') == '$1.0'
        and op.params.get('greaterVar') == '$0.0'
    )


def test_neg_absorb_correctness():
    """NOT(LessOrEqual(x, y, fr, tr)) yields same result via full pipeline."""
    x = Var('x')
    y = Const(5.0)
    loe = LessOrEqual(x, y, Const(10.0), Const(20.0))
    not_loe = LessOrEqual(loe, Const(0), Const(1), Const(0))
    prog = Program.from_tree(result=not_loe)
    vmt = prog.compile(optimize=full_optimize)
    for val in (-5.0, 0.0, 3.0, 10.0):
        state = interpret_vmt(vmt, EvalContext(vars={'$x': val}))
        # NOT(x <= 5): if x <= 5 then NOT(10) = 0 else NOT(20) = 0
        # Actually: LessOrEqual(x, 5, 10, 20) → 10 if x <= 5 else 20
        # NOT: LessOrEqual(result, 0, 1, 0) → 1 if result <= 0 else 0
        # So: 1 if (10 if x <= 5 else 20) <= 0 else 0
        # Since 10 > 0 and 20 > 0, result is always 0.
        # After absorption: LessOrEqual(x, 5, 20, 10) → 20 if x <= 5 else 10
        expected = 20.0 if val <= 5.0 else 10.0
        assert state['$result'] == approx(expected)


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
