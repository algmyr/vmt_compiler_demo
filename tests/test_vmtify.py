"""Tests for the ``@vmtify`` decorator and AST compiler."""
# ruff: noqa: ANN001, ANN201, ANN202, D103, B017

from pytest import approx
from pytest import raises

# Names recognised by @vmtify's AST compiler; not called at Python level.
# Imported here to satisfy the type checker / linter.
from material_proxy import Abs
from material_proxy import CurrentTime
from material_proxy import EvalContext
from material_proxy import Program
from material_proxy import Var
from material_proxy import full_optimize
from material_proxy import interpret_vmt
from material_proxy import no_optimize
from material_proxy import vmtify


def _compile_and_run(func, var_values: dict[str, float], optimize=full_optimize):
    """Helper: compile @vmtify-wrapped function, emit, interpret."""
    expr = func(*(Var(k) for k in var_values))
    prog = Program.from_tree(result=expr)
    vmt = prog.compile('test', optimize=optimize)
    ctx = EvalContext(vars={f'${k}': v for k, v in var_values.items()})
    return interpret_vmt(vmt, ctx)


# -- comparison encoding ------------------------------------------------


def test_gt():
    @vmtify
    def f(x):
        if x > 0:
            return 1
        return 0

    assert _compile_and_run(f, {'x': 5.0})['$result'] == approx(1.0)
    assert _compile_and_run(f, {'x': -1.0})['$result'] == approx(0.0)
    assert _compile_and_run(f, {'x': 0.0})['$result'] == approx(0.0)


def test_lt():
    @vmtify
    def f(x):
        if x < 5:
            return 1
        return 0

    assert _compile_and_run(f, {'x': 3.0})['$result'] == approx(1.0)
    assert _compile_and_run(f, {'x': 7.0})['$result'] == approx(0.0)
    assert _compile_and_run(f, {'x': 5.0})['$result'] == approx(0.0)


def test_eq():
    @vmtify
    def f(x):
        if x == 3:
            return 1
        return 0

    assert _compile_and_run(f, {'x': 3.0})['$result'] == approx(1.0)
    assert _compile_and_run(f, {'x': 4.0})['$result'] == approx(0.0)


def test_ne():
    @vmtify
    def f(x):
        if x != 3:
            return 1
        return 0

    assert _compile_and_run(f, {'x': 4.0})['$result'] == approx(1.0)
    assert _compile_and_run(f, {'x': 3.0})['$result'] == approx(0.0)


def test_lte():
    @vmtify
    def f(x):
        if x <= 3:
            return 1
        return 0

    assert _compile_and_run(f, {'x': 2.0})['$result'] == approx(1.0)
    assert _compile_and_run(f, {'x': 3.0})['$result'] == approx(1.0)
    assert _compile_and_run(f, {'x': 4.0})['$result'] == approx(0.0)


def test_gte():
    @vmtify
    def f(x):
        if x >= 3:
            return 1
        return 0

    assert _compile_and_run(f, {'x': 3.0})['$result'] == approx(1.0)
    assert _compile_and_run(f, {'x': 4.0})['$result'] == approx(1.0)
    assert _compile_and_run(f, {'x': 2.0})['$result'] == approx(0.0)


# -- basic if / else ---------------------------------------------------


def test_if_return():
    @vmtify
    def f(x):
        if x > 0:
            return x * 2
        return x * 3

    assert _compile_and_run(f, {'x': 5.0})['$result'] == approx(10.0)
    assert _compile_and_run(f, {'x': -2.0})['$result'] == approx(-6.0)


def test_if_no_else():
    """If without else — implicit else continues to code after."""

    @vmtify
    def f(x):
        if x > 0:
            return 10
        return 20

    assert _compile_and_run(f, {'x': 5.0})['$result'] == approx(10.0)
    assert _compile_and_run(f, {'x': -1.0})['$result'] == approx(20.0)
    assert _compile_and_run(f, {'x': 0.0})['$result'] == approx(20.0)


# -- elif chains --------------------------------------------------------


def test_elif_chain():
    @vmtify
    def f(x):
        if x < -5:
            return -1
        if x > 5:
            return 1
        return 0

    assert _compile_and_run(f, {'x': -10.0})['$result'] == approx(-1.0)
    assert _compile_and_run(f, {'x': 10.0})['$result'] == approx(1.0)
    assert _compile_and_run(f, {'x': 0.0})['$result'] == approx(0.0)


def test_elif_multi():
    @vmtify
    def f(x):
        if x < 0:
            return -1
        if x > 10:
            return 2
        if x > 5:
            return 1
        return 0

    assert _compile_and_run(f, {'x': -5.0})['$result'] == approx(-1.0)
    assert _compile_and_run(f, {'x': 12.0})['$result'] == approx(2.0)
    assert _compile_and_run(f, {'x': 7.0})['$result'] == approx(1.0)
    assert _compile_and_run(f, {'x': 3.0})['$result'] == approx(0.0)


# -- nested if ----------------------------------------------------------


def test_nested_if():
    @vmtify
    def f(x):
        if x > 0:
            if x > 10:
                return 10
            return x
        return 0

    assert _compile_and_run(f, {'x': 20.0})['$result'] == approx(10.0)
    assert _compile_and_run(f, {'x': 5.0})['$result'] == approx(5.0)
    assert _compile_and_run(f, {'x': -3.0})['$result'] == approx(0.0)


# -- assignment before if -----------------------------------------------


def test_assign_before_if():
    @vmtify
    def f(x):
        y = x * 2
        if y > 0:
            return y
        return -y

    assert _compile_and_run(f, {'x': 3.0})['$result'] == approx(6.0)
    assert _compile_and_run(f, {'x': -3.0})['$result'] == approx(6.0)


def test_env_merge_after_if():
    """Assignments in both branches are merged for code after if."""

    @vmtify
    def f(x):
        if x > 0:
            y = x * 2
        else:
            y = x * 3
        return y

    assert _compile_and_run(f, {'x': 5.0})['$result'] == approx(10.0)
    assert _compile_and_run(f, {'x': -2.0})['$result'] == approx(-6.0)


# -- composability ------------------------------------------------------


def test_compose():
    """@vmtify functions compose naturally — output of one is input to another."""

    @vmtify
    def clamp01(x):
        if x < 0:
            return 0
        if x > 1:
            return 1
        return x

    expr = clamp01(Var('x')) * 2
    prog = Program.from_tree(result=expr)
    vmt = prog.compile('test', optimize=no_optimize)

    for val, expected in [(-0.5, 0.0), (0.3, 0.6), (1.5, 2.0)]:
        state = interpret_vmt(vmt, EvalContext(vars={'$x': val}))
        assert state['$result'] == approx(expected)


# -- boolean operators --------------------------------------------------


def test_bool_and():
    @vmtify
    def f(a, b):
        if a > 0 and b > 0:
            return 1
        return 0

    assert _compile_and_run(f, {'a': 1.0, 'b': 2.0})['$result'] == approx(1.0)
    assert _compile_and_run(f, {'a': 1.0, 'b': 0.0})['$result'] == approx(0.0)
    assert _compile_and_run(f, {'a': 0.0, 'b': 2.0})['$result'] == approx(0.0)


def test_bool_or():
    @vmtify
    def f(a, b):
        if a > 0 or b > 0:
            return 1
        return 0

    assert _compile_and_run(f, {'a': 1.0, 'b': 0.0})['$result'] == approx(1.0)
    assert _compile_and_run(f, {'a': 0.0, 'b': 2.0})['$result'] == approx(1.0)
    assert _compile_and_run(f, {'a': 0.0, 'b': -1.0})['$result'] == approx(0.0)


# -- recognized proxies -------------------------------------------------


def test_current_time():
    @vmtify
    def f():
        return CurrentTime()

    prog = Program.from_tree(result=f())
    vmt = prog.compile('test', optimize=no_optimize)
    state = interpret_vmt(vmt, EvalContext(time=42.0))
    assert state['$result'] == approx(42.0)


def test_abs_proxy():
    @vmtify
    def f(x):
        return Abs(x)

    assert _compile_and_run(f, {'x': -5.0})['$result'] == approx(5.0)
    assert _compile_and_run(f, {'x': 3.0})['$result'] == approx(3.0)


# -- error handling -----------------------------------------------------


def test_no_return():
    """Function without return is an error."""
    with raises(Exception):

        @vmtify
        def f(x):
            x + 1

        expr = f(Var('x'))  # noqa: F841


def test_partial_return_in_branch():
    """Partial return in branch.

    If one branch has a return and the other doesn't, the rest of the
    function after the if provides the return for the non-returning branch.
    """

    @vmtify
    def f(x):
        if x > 0:
            return 10
        return 20

    assert _compile_and_run(f, {'x': 5.0})['$result'] == approx(10.0)
    assert _compile_and_run(f, {'x': -1.0})['$result'] == approx(20.0)


def test_return_in_branch_no_else():
    """Else branch provides the return in an if with conditional return."""

    @vmtify
    def f(x):
        if x > 0:
            return 10
        return 20

    assert _compile_and_run(f, {'x': 5.0})['$result'] == approx(10.0)
    assert _compile_and_run(f, {'x': -1.0})['$result'] == approx(20.0)


# -- unary operators ----------------------------------------------------


def test_unary_not():
    @vmtify
    def f(x):
        if not x > 0:
            return 1
        return 0

    assert _compile_and_run(f, {'x': -3.0})['$result'] == approx(1.0)
    assert _compile_and_run(f, {'x': 5.0})['$result'] == approx(0.0)


def test_unary_neg():
    @vmtify
    def f(x):
        return -x

    assert _compile_and_run(f, {'x': 5.0})['$result'] == approx(-5.0)
    assert _compile_and_run(f, {'x': -3.0})['$result'] == approx(3.0)


# -- augmented assignment -----------------------------------------------


def test_aug_assign():
    @vmtify
    def f(x):
        y = x
        y += 5
        return y

    assert _compile_and_run(f, {'x': 10.0})['$result'] == approx(15.0)


# -- if expression ------------------------------------------------------


def test_ifexp():
    @vmtify
    def f(x):
        return 1 if x > 0 else 0

    assert _compile_and_run(f, {'x': 5.0})['$result'] == approx(1.0)
    assert _compile_and_run(f, {'x': -2.0})['$result'] == approx(0.0)


# -- clamp01 integration ------------------------------------------------


def test_clamp01():
    @vmtify
    def clamp01(x):
        if x < 0:
            return 0
        if x > 1:
            return 1
        return x

    for val in [-0.5, 0.0, 0.3, 1.0, 1.5]:
        expected = max(0.0, min(1.0, val))
        assert _compile_and_run(clamp01, {'x': val})['$result'] == approx(expected)


# -- optimized vs unoptimized -------------------------------------------


def test_optimized_matches_unoptimized():
    """Optimized and unoptimized compilation produce same results."""

    @vmtify
    def f(x):
        if x > 0:
            return x * 2
        if x < -5:
            return -10
        return x + 1

    for val in [-10.0, -5.0, -2.0, 0.0, 3.0]:
        unopt = _compile_and_run(f, {'x': val}, optimize=no_optimize)
        opt = _compile_and_run(f, {'x': val}, optimize=full_optimize)
        assert unopt['$result'] == approx(opt['$result'])


# -- nesting patterns ---------------------------------------------------
# Patterns tested:
#   v > e > v > e   vmtify uses expr uses vmtify uses expr
#   e > v > e > v   expr uses vmtify uses expr uses vmtify
#   e > v > v > e   expr uses vmtify uses vmtify uses expr
#   v > e > e > v   vmtify uses expr uses expr uses vmtify


def test_nesting_v_any_v_any():
    @vmtify
    def scale(x, s):
        return x * s

    @vmtify
    def f(x):
        return scale(x + 1, 2) - 3

    # f(v) = ((v + 1) * 2) - 3 = 2v - 1
    assert _compile_and_run(f, {'v': 5.0})['$result'] == approx(9.0)
    assert _compile_and_run(f, {'v': 0.0})['$result'] == approx(-1.0)
    assert _compile_and_run(f, {'v': -2.0})['$result'] == approx(-5.0)


def test_nesting_e_any_v_any():
    @vmtify
    def halve(x):
        return x / 2

    @vmtify
    def g(x):
        return halve(x) + 1

    # g(Var('v')) → halve(v) + 1 = v/2 + 1 → then * 3
    expr = g(Var('v')) * 3
    prog = Program.from_tree(result=expr)
    vmt = prog.compile('test')

    for val, expected in [(4.0, 9.0), (0.0, 3.0), (-6.0, -6.0)]:
        state = interpret_vmt(vmt, EvalContext(vars={'$v': val}))
        assert state['$result'] == approx(expected)


def test_nesting_e_any_vv():
    @vmtify
    def inc(x):
        return x + 1

    @vmtify
    def double_inc(x):
        return inc(x) * 2

    # double_inc(Var('v')) → (v + 1) * 2 → then - 1
    expr = double_inc(Var('v')) - 1
    prog = Program.from_tree(result=expr)
    vmt = prog.compile('test')

    for val, expected in [(5.0, 11.0), (0.0, 1.0), (-2.0, -3.0)]:
        state = interpret_vmt(vmt, EvalContext(vars={'$v': val}))
        assert state['$result'] == approx(expected)


def test_nesting_v_any_ee():
    @vmtify
    def square(x):
        return x * x

    @vmtify
    def f(x):
        return square(x + 1) + square(x - 1)

    # f(v) = (v+1)² + (v-1)² = 2v² + 2
    assert _compile_and_run(f, {'v': 3.0})['$result'] == approx(20.0)
    assert _compile_and_run(f, {'v': 0.0})['$result'] == approx(2.0)
    assert _compile_and_run(f, {'v': -2.0})['$result'] == approx(10.0)


def test_nesting_three_deep():
    @vmtify
    def add1(x):
        return x + 1

    @vmtify
    def mul2(x):
        return x * 2

    @vmtify
    def compose(x):
        return add1(mul2(add1(x)))

    # compose(v) = ((v + 1) * 2) + 1 = 2v + 3
    assert _compile_and_run(compose, {'v': 5.0})['$result'] == approx(13.0)
    assert _compile_and_run(compose, {'v': 0.0})['$result'] == approx(3.0)
    assert _compile_and_run(compose, {'v': -2.0})['$result'] == approx(-1.0)
