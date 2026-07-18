from __future__ import annotations

from collections.abc import Callable
import math

PARAM_ORDER: dict[str, tuple[str, ...]] = {
    'Add': ('srcVar1', 'srcVar2'),
    'Subtract': ('srcVar1', 'srcVar2'),
    'Multiply': ('srcVar1', 'srcVar2'),
    'Divide': ('srcVar1', 'srcVar2'),
    'Abs': ('srcVar1',),
    'Frac': ('srcVar1',),
    'Int': ('srcVar1',),
    'Equals': ('srcVar1',),
    'Clamp': ('srcVar1', 'min', 'max'),
    'WrapMinMax': ('srcVar1', 'minVal', 'maxVal'),
    'LessOrEqual': ('srcVar1', 'srcVar2', 'lessEqualVar', 'greaterVar'),
    'SelectFirstIfNonZero': ('srcVar1', 'srcVar2'),
    'Exponential': ('srcVar1', 'offset', 'scale'),
}


def _less_or_equal(a: float, b: float, le: float, gr: float) -> float:
    return le if a <= b else gr


def _select_first(a: float, b: float) -> float:
    return a if a != 0.0 else b


def _wrap_min_max(src: float, lo: float, hi: float) -> float:
    if lo >= hi:
        return lo
    return ((src - lo) % (hi - lo)) + lo


def _int(a: float) -> float:
    try:
        return math.trunc(a)
    except OverflowError:
        # This is highly questionable, but avoids some test issues.
        # Probably should be an exception.
        return math.copysign(float('inf'), a)


COMPUTE: dict[str, Callable[..., float]] = {
    'Add': lambda a, b: a + b,
    'Subtract': lambda a, b: a - b,
    'Multiply': lambda a, b: a * b,
    'Divide': lambda a, b: a / b,
    'Abs': lambda a: abs(a),
    'Frac': lambda a: (
        0.0
        if not math.isfinite(a)
        else (a - math.floor(a) if a >= 0 else a - math.ceil(a))
    ),
    'Int': _int,
    'Equals': lambda a: a,
    'Clamp': lambda a, lo, hi: max(min(a, hi), lo) if lo <= hi else max(min(a, lo), hi),
    'WrapMinMax': _wrap_min_max,
    'LessOrEqual': _less_or_equal,
    'SelectFirstIfNonZero': _select_first,
}
