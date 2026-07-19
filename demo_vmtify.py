"""Demo: interleaving @vmtify (AST control flow) with expression-DSL metaprogramming.

Three crossing patterns shown:
  a) @vmtify uses an Expr built by a Python loop
  b) raw Expr uses a @vmtify function as a building block
  c) nested: meta-loop wraps a @vmtify call that itself uses a meta-built Expr
"""
# ruff: noqa: ANN001, ANN201, D103

from material_proxy import Const
from material_proxy import CurrentTime
from material_proxy import EvalContext
from material_proxy import Expr
from material_proxy import Program
from material_proxy import Var
from material_proxy import interpret_vmt
from material_proxy import vmtify

# ======================================================================
# 1.  AST-compiled control flow  (if / elif / else → LessOrEqual)
# ======================================================================


@vmtify
def clamp01(x):
    if x < 0:
        return 0
    if x > 1:
        return 1
    return x


@vmtify
def smootherstep(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1
    return x * x * x * (x * (x * 6 - 15) + 10)


@vmtify
def sign(x):
    if x < 0:
        return -1
    if x > 0:
        return 1
    return 0


# ======================================================================
# 2.  Expression-DSL metaprogramming  (Python loops unroll Expr trees)
# ======================================================================


def lerp(a: Expr, b: Expr, t: Expr) -> Expr:
    """Linear interpolation built from raw Expr constructors."""
    return a + (b - a) * t


def newton_sqrt(x: Expr, n_iters: int = 6) -> Expr:
    """Newton sqrt — loop unrolled at Python import time."""
    t: Expr = (x + Const(1)) * Const(0.5)
    for _ in range(n_iters):
        t = (t + x / t) * Const(0.5)
    return t


def smooth_pulse(t: Expr, freq: Expr) -> Expr:
    """A smooth oscillating pulse built by composing Expr constructors."""
    raw = (t * freq - Const(0.5)) * Const(2)
    return smootherstep(clamp01(raw))


# ======================================================================
# 3.  Interleaving patterns
# ======================================================================

# ----- a) @vmtify uses an Expr built by a Python loop ---------------

# Newton sqrt is an Expr DAG unrolled by a Python `for` loop.
# clamp01 / smootherstep are @vmtify functions.
# Passing the meta-built Expr into the @vmtify function works because
# both sides produce/consume the same Expr type.

scaled_time = CurrentTime() * Const(0.1)
clamped = smootherstep(clamp01(scaled_time))

# ----- b) raw Expr uses a @vmtify function as a building block --------

# @vmtify functions return Expr trees, so they slot directly into
# raw expression composition.
s = sign(Var('x'))
abs_x = s * Var('x')  # sign(x) * x = abs(x)

# ----- c) nested: meta-loop wraps vmtify which uses meta-built Expr ---

# The pulse is a @vmtify call whose argument comes from a Python loop
# that builds an Expr tree via plain arithmetic.
# Inside the pulse, smootherstep receives the meta-built argument.

pulse = smooth_pulse(CurrentTime(), Const(0.5))

# ======================================================================
# 4.  Putting it all together
# ======================================================================

# Final output: blend the pulse with a computated value
result = pulse * lerp(Const(0.5), Const(2.0), clamped) + abs_x * Const(0.1)

prog = Program.from_tree(
    color=result,
    pulse=pulse,
    clamped=clamped,
)

vmt = prog.compile('UnlitGeneric')
print(vmt)

# ======================================================================
# 5.  Verify against the interpreter
# ======================================================================

print('--- interpretation ---')
state = interpret_vmt(
    vmt,
    EvalContext(time=2.5, vars={'$x': -3.0}),
)
for k in ('$color', '$pulse', '$clamped'):
    print(f'  {k} = {state.get(k, "?")}')
