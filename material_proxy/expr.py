from __future__ import annotations

from collections.abc import Iterable
from typing import Union


def _as_expr(v: Union[Expr, float]) -> Expr:
    if isinstance(v, Expr):
        return v
    return Const(float(v))


class Expr:
    """Base class for all material proxy expressions.

    Supports operator overloading so expressions can be written naturally:
        x = Var("x")
        y = Var("y")
        result = (x + y) * 2
    """

    def __add__(self, other: Expr | float) -> Expr:
        return Add(self, other)

    def __sub__(self, other: Expr | float) -> Expr:
        return Sub(self, other)

    def __mul__(self, other: Expr | float) -> Expr:
        return Mul(self, other)

    def __truediv__(self, other: Expr | float) -> Expr:
        return Div(self, other)

    def __neg__(self) -> Expr:
        return Mul(Const(-1), self)

    def __radd__(self, other: Expr | float) -> Expr:
        return Add(other, self)

    def __rsub__(self, other: Expr | float) -> Expr:
        return Sub(other, self)

    def __rmul__(self, other: Expr | float) -> Expr:
        return Mul(other, self)

    def __rtruediv__(self, other: Expr | float) -> Expr:
        return Div(other, self)

    def __repr__(self) -> str:
        return ' '.join(self._repr(compact=True))

    def pretty_repr(self) -> str:
        """Return a multi-line representation of the expression tree."""
        return '\n'.join(self._repr(compact=False))

    def _repr(self, compact: bool) -> Iterable[str]:
        del compact
        yield f'{type(self).__name__}()'

    def _hash_children(self) -> tuple:
        raise NotImplementedError

    def __hash__(self) -> int:
        try:
            return self._hash_cache
        except AttributeError:
            self._hash_cache = hash((type(self), self._hash_children()))
            return self._hash_cache

    def __gt__(self, other: Expr | float) -> Expr:
        return LessOrEqual(self, other, Const(0), Const(1))

    def __lt__(self, other: Expr | float) -> Expr:
        return LessOrEqual(other, self, Const(0), Const(1))

    def __ge__(self, other: Expr | float) -> Expr:
        return LessOrEqual(other, self, Const(1), Const(0))

    def __le__(self, other: Expr | float) -> Expr:
        return LessOrEqual(self, other, Const(1), Const(0))

    def __eq__(self, other: object) -> bool:
        if type(self) is not type(other):
            return NotImplemented
        assert isinstance(other, Expr)
        return self._hash_children() == other._hash_children()


class Const(Expr):
    """A literal float value."""

    def __init__(self, value: float):
        self.value = float(value)

    def _repr(self, compact: bool) -> Iterable[str]:
        del compact
        yield repr(self.value)

    def _hash_children(self) -> tuple:
        return (round(self.value, 6),)


class Var(Expr):
    """Reference to a named material variable (e.g. ``Var("time")`` → ``$time``)."""

    def __init__(self, name: str):
        self.name = name

    def _repr(self, compact: bool) -> Iterable[str]:
        del compact
        yield self.name

    def _hash_children(self) -> tuple:
        return (self.name,)


class CurrentTime(Expr):
    """The number of seconds the current map has been running."""

    def _hash_children(self) -> tuple:
        return ()

    def _repr(self, compact: bool) -> Iterable[str]:
        del compact
        yield 'curtime()'


def _n_ary_repr(
    name: str, children: list[Expr | tuple[str, Expr]], compact: bool
) -> Iterable[str]:
    """Yield repr lines for a node with ordered *children*.

    Each element is either a plain *Expr* (positional) or a ``(label, Expr)``
    tuple (keyword argument).
    """
    n = len(children)

    def prep(i: int, child: Expr | tuple[str, Expr]) -> list[str]:
        label = None
        if isinstance(child, tuple):
            label, child = child
        parts = list(child._repr(compact))
        if label is not None:
            parts[0] = f'{label}={parts[0]}'
        if i < n - 1:
            parts[-1] += ','
        else:
            parts[-1] += ')'
        return parts

    prepared = (prep(i, child) for i, child in enumerate(children))
    if compact:
        inner = ' '.join(part for parts in prepared for part in parts)
        yield f'{name}({inner}'
    else:
        yield f'{name}('
        for parts in prepared:
            yield from ('  ' + s for s in parts)


class PlayerPosition(Expr):
    """The local player's 3D position."""

    def __init__(self, scale: Union[Expr, float] = 1.0):
        self.scale = _as_expr(scale)

    def _hash_children(self) -> tuple:
        return (self.scale,)

    def _repr(self, compact: bool) -> Iterable[str]:
        return _n_ary_repr('PlayerPos', [self.scale], compact)


class PlayerSpeed(Expr):
    """The local player's speed."""

    def __init__(self, scale: Union[Expr, float] = 1.0):
        self.scale = _as_expr(scale)

    def _hash_children(self) -> tuple:
        return (self.scale,)

    def _repr(self, compact: bool) -> Iterable[str]:
        return _n_ary_repr('PlayerSpeed', [self.scale], compact)


class _BinaryOp(Expr):
    def __init__(self, a: Union[Expr, float], b: Union[Expr, float]):
        self.a = _as_expr(a)
        self.b = _as_expr(b)

    def _hash_children(self) -> tuple:
        return (self.a, self.b)

    def _repr(self, compact: bool) -> Iterable[str]:
        return _n_ary_repr(type(self).__name__, [self.a, self.b], compact)


class Add(_BinaryOp):
    """Returns ``a + b``."""


class Sub(_BinaryOp):
    """Returns ``a - b``."""


class Mul(_BinaryOp):
    """Returns ``a * b``."""


class Div(_BinaryOp):
    """Returns ``a / b``."""


class _UnaryOp(Expr):
    def __init__(self, a: Union[Expr, float]):
        self.a = _as_expr(a)

    def _hash_children(self) -> tuple:
        return (self.a,)

    def _repr(self, compact: bool) -> Iterable[str]:
        return _n_ary_repr(type(self).__name__, [self.a], compact)


class Abs(_UnaryOp):
    """Returns the absolute (unsigned) value."""


class Frac(_UnaryOp):
    """Returns the fractional component (e.g. ``frac(4.23) = 0.23``)."""


class Int(_UnaryOp):
    """Returns the integer component (e.g. ``int(4.23) = 4``)."""


class Equals(Expr):
    """Copies a value to a variable.  Analogous to assignment."""

    def __init__(self, src: Union[Expr, float]):
        self.src = _as_expr(src)

    def _hash_children(self) -> tuple:
        return (self.src,)

    def _repr(self, compact: bool) -> Iterable[str]:
        return _n_ary_repr('Equals', [self.src], compact)


class Clamp(Expr):
    """Clamps a value to ``[min, max]``.

    Note: Unlike most proxies, Clamp does *not* force float conversion —
    integer inputs stay integer.
    """

    def __init__(
        self,
        src: Union[Expr, float],
        min_val: Union[Expr, float],
        max_val: Union[Expr, float],
    ):
        self.src = _as_expr(src)
        self.min = _as_expr(min_val)
        self.max = _as_expr(max_val)

    def _hash_children(self) -> tuple:
        return (self.src, self.min, self.max)

    def _repr(self, compact: bool) -> Iterable[str]:
        return _n_ary_repr('Clamp', [self.src, self.min, self.max], compact)


class WrapMinMax(Expr):
    """Constrains a value to ``[min, max]``, wrapping cyclically.

    Bug: If the ``$bumpmap`` is missing, use on a bumpmap will crash Hammer
    or produce rendering artifacts.
    """

    def __init__(
        self,
        src: Union[Expr, float],
        min_val: Union[Expr, float],
        max_val: Union[Expr, float],
    ):
        self.src = _as_expr(src)
        self.min_val = _as_expr(min_val)
        self.max_val = _as_expr(max_val)

    def _hash_children(self) -> tuple:
        return (self.src, self.min_val, self.max_val)

    def _repr(self, compact: bool) -> Iterable[str]:
        return _n_ary_repr(
            'WrapMinMax', [self.src, self.min_val, self.max_val], compact
        )


class LessOrEqual(Expr):
    """If ``a <= b`` then ``less_equal`` else ``greater``.

    Equivalent to::

        resultVar = a <= b ? less_equal : greater
    """

    def __init__(
        self,
        a: Union[Expr, float],
        b: Union[Expr, float],
        less_equal: Union[Expr, float],
        greater: Union[Expr, float],
    ):
        self.a = _as_expr(a)
        self.b = _as_expr(b)
        self.less_equal = _as_expr(less_equal)
        self.greater = _as_expr(greater)

    def _hash_children(self) -> tuple:
        return (self.a, self.b, self.less_equal, self.greater)

    def _repr(self, compact: bool) -> Iterable[str]:
        return _n_ary_repr(
            'LessOrEqual',
            [self.a, self.b, self.less_equal, self.greater],
            compact,
        )


class SelectFirstIfNonZero(Expr):
    """If ``a != 0`` return ``a``, otherwise return ``b``."""

    def __init__(self, a: Union[Expr, float], b: Union[Expr, float]):
        self.a = _as_expr(a)
        self.b = _as_expr(b)

    def _hash_children(self) -> tuple:
        return (self.a, self.b)

    def _repr(self, compact: bool) -> Iterable[str]:
        return _n_ary_repr('SelectFirstIfNonZero', [self.a, self.b], compact)


class _NoClampType:
    """Sentinel indicating Exp has no clamping bound."""

    def __repr__(self) -> str:
        return 'NoClamp'


_NO_CLAMP = _NoClampType()


class Exp(Expr):
    """``scale * e^(src + offset)``, clamped to ``[min_val, max_val]``."""

    _MIN_SENTINEL: _NoClampType = _NO_CLAMP
    _MAX_SENTINEL: _NoClampType = _NO_CLAMP

    def __init__(
        self,
        src: Union[Expr, float],
        offset: Union[Expr, float] = 0.0,
        scale: Union[Expr, float] = 1.0,
        min_val: Union[Expr, float, None] = None,
        max_val: Union[Expr, float, None] = None,
    ):
        self.src = _as_expr(src)
        self.offset = _as_expr(offset)
        self.scale = _as_expr(scale)
        self.min_val = _as_expr(min_val) if min_val is not None else None
        self.max_val = _as_expr(max_val) if max_val is not None else None

    def _hash_children(self) -> tuple:
        return (
            self.src,
            self.offset,
            self.scale,
            self.min_val if self.min_val is not None else Exp._MIN_SENTINEL,
            self.max_val if self.max_val is not None else Exp._MAX_SENTINEL,
        )

    def _repr(self, compact: bool) -> Iterable[str]:
        children: list[Expr | tuple[str, Expr]] = [self.src]
        if not (isinstance(self.offset, Const) and self.offset.value == 0.0):
            children.append(('offset', self.offset))
        if not (isinstance(self.scale, Const) and self.scale.value == 1.0):
            children.append(('scale', self.scale))
        if self.min_val is not None:
            assert self.max_val is not None
            children.append(('min', self.min_val))
            children.append(('max', self.max_val))
        return _n_ary_repr('Exp', children, compact)
