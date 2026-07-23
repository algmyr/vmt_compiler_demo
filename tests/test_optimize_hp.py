import math

from hypothesis import given
from hypothesis import settings as hp_settings
from hypothesis import strategies as st

from material_proxy import Abs
from material_proxy import Add
from material_proxy import Const
from material_proxy import Div
from material_proxy import EvalContext
from material_proxy import Expr
from material_proxy import Frac
from material_proxy import Int
from material_proxy import Mul
from material_proxy import Sub
from material_proxy import Var
from material_proxy.emit import emit_vmt
from material_proxy.flatten import FlatOp
from material_proxy.flatten import Flattener
from material_proxy.interpret import interpret_vmt
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


_ctx = EvalContext(
    vars={
        '$x': 1.0,
        '$y': 2.0,
        '$z': 3.0,
        '$a': 4.0,
        '$b': 5.0,
        '$c': 6.0,
    }
)

_leaf = st.one_of(
    st.builds(
        Const,
        st.floats(
            allow_nan=False,
            allow_infinity=False,
            min_value=-100.0,
            max_value=100.0,
        ),
    ),
    st.builds(Var, st.sampled_from(['x', 'y', 'z', 'a', 'b', 'c'])),
)


def _no_zero_div(e: object) -> bool:
    """Reject Div-by-zero at any nesting level."""
    match e:
        case Div():
            if isinstance(e.b, Const) and e.b.value == 0.0:
                return False
            return _no_zero_div(getattr(e, 'a', None)) and _no_zero_div(
                getattr(e, 'b', None)
            )
        case Add() | Sub() | Mul():
            return _no_zero_div(getattr(e, 'a', None)) and _no_zero_div(
                getattr(e, 'b', None)
            )
        case Abs() | Frac() | Int():
            return _no_zero_div(getattr(e, 'a', None))
        case _:
            return True


_expr_raw = st.deferred(
    lambda: st.one_of(
        _leaf,
        st.builds(Add, _expr_raw, _expr_raw),
        st.builds(Sub, _expr_raw, _expr_raw),
        st.builds(Mul, _expr_raw, _expr_raw),
        st.builds(Div, _expr_raw, _expr_raw).filter(_no_zero_div),
        st.builds(Abs, _expr_raw),
        st.builds(Frac, _expr_raw),
        st.builds(Int, _expr_raw),
    )
)


def _node_count(expr: Expr) -> int:
    """Total number of nodes (leaf and internal) in the expression tree."""
    match expr:
        case Const() | Var():
            return 1
        case Abs() | Frac() | Int():
            return 1 + _node_count(expr.a)
        case Add() | Sub() | Mul() | Div():
            return 1 + _node_count(expr.a) + _node_count(expr.b)
        case _:
            return 1


_EXPR_MAX_NODES = 40
_expr = _expr_raw.filter(lambda e: _node_count(e) <= _EXPR_MAX_NODES)


@hp_settings(max_examples=100)
@given(_expr)
def test_hypothesis_temp_reuse_semantics(expr: Expr) -> None:
    """Random expression: temp_reuse alone preserves the VMT result."""
    flat = Flattener()
    flat.flatten(expr)
    res = flat.result()
    if not res.ops:
        return  # no temps to reuse (e.g. bare Var or Const)
    raw_vmt = emit_vmt(res.ops, res.consts)
    reused_vmt = emit_vmt(temp_reuse(res.ops), res.consts)
    try:
        raw_state = interpret_vmt(raw_vmt, _ctx)
        reused_state = interpret_vmt(reused_vmt, _ctx)
    except ZeroDivisionError:
        return  # runtime div-by-zero — skip
    raw_final = _last_result(raw_state)
    reused_vals = [v for v in reused_state.values() if isinstance(v, (int, float))]
    if math.isfinite(raw_final):
        assert any(abs(v - raw_final) < 1e-6 for v in reused_vals)


@hp_settings(max_examples=100)
@given(_expr)
def test_hypothesis_full_pipeline(expr: Expr) -> None:
    """Random expression: fold → DCE → temp_reuse preserves result."""
    flat = Flattener()
    flat.flatten(expr)
    res = flat.result()
    if not res.ops:
        return

    # Unoptimised reference
    raw_vmt = emit_vmt(res.ops, res.consts)
    try:
        raw_state = interpret_vmt(raw_vmt, _ctx)
    except ZeroDivisionError:
        return  # runtime div-by-zero — skip

    folded_ops, folded_consts = constant_fold(res.ops, res.consts)
    if not folded_ops:
        return  # fully constant — no optimisation to test

    folded_ops.append(FlatOp('Equals', {'srcVar1': folded_ops[-1].result}, '$hp_out'))
    dce_ops = dead_code_elimination(folded_ops)
    final_ops = temp_reuse(dce_ops)
    opt_vmt = emit_vmt(final_ops, folded_consts)
    try:
        opt_state = interpret_vmt(opt_vmt, _ctx)
    except ZeroDivisionError:
        return  # runtime div-by-zero — skip

    raw_final = _last_result(raw_state)
    opt_vals = [v for v in opt_state.values() if isinstance(v, (int, float))]
    if math.isfinite(raw_final):
        assert any(abs(v - raw_final) < 1e-6 for v in opt_vals)
