from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field

from material_proxy.expr import Abs
from material_proxy.expr import Add
from material_proxy.expr import Clamp
from material_proxy.expr import Const
from material_proxy.expr import CurrentTime
from material_proxy.expr import Div
from material_proxy.expr import Equals
from material_proxy.expr import Exp
from material_proxy.expr import Expr
from material_proxy.expr import Frac
from material_proxy.expr import Int
from material_proxy.expr import LessOrEqual
from material_proxy.expr import Mul
from material_proxy.expr import PlayerPosition
from material_proxy.expr import PlayerSpeed
from material_proxy.expr import SelectFirstIfNonZero
from material_proxy.expr import Sub
from material_proxy.expr import Var
from material_proxy.expr import WrapMinMax

_PROXY_NAMES: dict[type, str] = {
    Add: 'Add',
    Sub: 'Subtract',
    Mul: 'Multiply',
    Div: 'Divide',
    Abs: 'Abs',
    Frac: 'Frac',
    Int: 'Int',
    Equals: 'Equals',
    Clamp: 'Clamp',
    WrapMinMax: 'WrapMinMax',
    LessOrEqual: 'LessOrEqual',
    SelectFirstIfNonZero: 'SelectFirstIfNonZero',
    Exp: 'Exponential',
}


@dataclass
class FlatOp:
    """A single proxy operation in the linear IR."""

    proxy: str
    params: dict[str, str]
    result: str


@dataclass
class FlattenResult:
    """Result of flattening one or more expressions."""

    ops: list[FlatOp] = field(default_factory=list)
    consts: dict[float, str] = field(default_factory=dict)


class Flattener:
    """Walk an expression tree and emit a linear list of ``FlatOp``s."""

    def __init__(self) -> None:
        self._temp_counter = 0
        self._cse: dict[Expr, str] = {}
        self._consts: dict[float, str] = {}
        self._ops: list[FlatOp] = []

    def _new_temp(self) -> str:
        self._temp_counter += 1
        return f'$tmp_{self._temp_counter}'

    def _const_var(self, value: float) -> str:
        if value not in self._consts:
            name = f'${self._format_const(value)}'
            self._consts[value] = name
        return self._consts[value]

    @staticmethod
    def _format_const(value: float) -> str:
        """Format a constant value cleanly — no trailing ``.0`` for integers."""
        return repr(value)

    def _emit(self, proxy_name: str, params: dict[str, str]) -> str:
        dst = self._new_temp()
        self._ops.append(FlatOp(proxy_name, params, dst))
        return dst

    def flatten(self, expr: Expr) -> str:
        """Flatten *expr* and return the temp name holding its result."""
        cached = self._cse.get(expr)
        if cached is not None:
            return cached

        result = self._flatten_node(expr)
        self._cse[expr] = result
        return result

    def _flatten_node(self, expr: Expr) -> str:
        match expr:
            case Const():
                return self._const_var(expr.value)
            case Var():
                return f'${expr.name}'
            case CurrentTime():
                return self._emit('CurrentTime', {})
            case PlayerPosition():
                scale = self.flatten(expr.scale)
                return self._emit('PlayerPosition', {'scale': scale})
            case PlayerSpeed():
                scale = self.flatten(expr.scale)
                return self._emit('PlayerSpeed', {'scale': scale})
            case Add() | Sub() | Mul() | Div():
                a = self.flatten(expr.a)
                b = self.flatten(expr.b)
                return self._emit(
                    _PROXY_NAMES[type(expr)], {'srcVar1': a, 'srcVar2': b}
                )
            case Abs() | Frac() | Int():
                a = self.flatten(expr.a)
                return self._emit(_PROXY_NAMES[type(expr)], {'srcVar1': a})
            case Equals():
                src = self.flatten(expr.src)
                return self._emit('Equals', {'srcVar1': src})
            case Clamp():
                src = self.flatten(expr.src)
                mn = self.flatten(expr.min)
                mx = self.flatten(expr.max)
                return self._emit('Clamp', {'srcVar1': src, 'min': mn, 'max': mx})
            case WrapMinMax():
                src = self.flatten(expr.src)
                mn = self.flatten(expr.min_val)
                mx = self.flatten(expr.max_val)
                return self._emit(
                    'WrapMinMax', {'srcVar1': src, 'minVal': mn, 'maxVal': mx}
                )
            case LessOrEqual():
                a = self.flatten(expr.a)
                b = self.flatten(expr.b)
                le = self.flatten(expr.less_equal)
                gr = self.flatten(expr.greater)
                return self._emit(
                    'LessOrEqual',
                    {
                        'srcVar1': a,
                        'srcVar2': b,
                        'lessEqualVar': le,
                        'greaterVar': gr,
                    },
                )
            case SelectFirstIfNonZero():
                a = self.flatten(expr.a)
                b = self.flatten(expr.b)
                return self._emit('SelectFirstIfNonZero', {'srcVar1': a, 'srcVar2': b})
            case Exp():
                src = self.flatten(expr.src)
                off = self.flatten(expr.offset)
                sc = self.flatten(expr.scale)
                params = {'srcVar1': src, 'offset': off, 'scale': sc}
                if expr.min_val is not None:
                    assert expr.max_val is not None
                    params['minVal'] = self.flatten(expr.min_val)
                    params['maxVal'] = self.flatten(expr.max_val)
                return self._emit('Exponential', params)
            case _:
                raise TypeError(f'Cannot flatten {type(expr).__name__}')

    def result(self) -> FlattenResult:
        """Return accumulated ops and constants."""
        return FlattenResult(ops=list(self._ops), consts=dict(self._consts))
