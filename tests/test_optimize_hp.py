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
from material_proxy import Program
from material_proxy import Sub
from material_proxy import Var
from material_proxy import full_optimize
from material_proxy import interpret_vmt
from material_proxy import no_optimize
from tests._test_helpers import reuse_only

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
    prog = Program.from_tree(result=expr)
    # bare Const or Var produce only the Equals op — no temps to reassign
    if all(op.proxy == 'Equals' for op in prog.ops):
        return
    try:
        raw_state = interpret_vmt(prog.emit(), _ctx)
        reused_state = interpret_vmt(prog.optimize(reuse_only).emit(), _ctx)
    except ZeroDivisionError:
        return
    raw_val = raw_state['$result']
    reused_val = reused_state['$result']
    if math.isfinite(raw_val):
        assert abs(reused_val - raw_val) < 1e-6


@hp_settings(max_examples=100)
@given(_expr)
def test_hypothesis_full_pipeline(expr: Expr) -> None:
    """Random expression: fold → DCE → temp_reuse preserves result via Program."""
    prog = Program.from_tree(result=expr)
    try:
        raw_state = interpret_vmt(prog.compile(optimize=no_optimize), _ctx)
        opt_state = interpret_vmt(prog.compile(optimize=full_optimize), _ctx)
    except ZeroDivisionError:
        return

    raw_val = raw_state['$result']
    opt_val = opt_state['$result']
    if math.isfinite(raw_val):
        assert abs(opt_val - raw_val) < 1e-6
