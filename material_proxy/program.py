from __future__ import annotations

from material_proxy.emit import emit_readable
from material_proxy.emit import emit_vmt
from material_proxy.expr import Expr
from material_proxy.flatten import FlatOp
from material_proxy.flatten import Flattener
from material_proxy.optimize import OptimizeFn
from material_proxy.optimize import full_optimize


def compile_to_vmt(expr: Expr, material_name: str = 'UnlitGeneric') -> str:
    """Compile an expression tree and return a complete VMT string."""
    flattener = Flattener()
    flattener.flatten(expr)
    res = flattener.result()
    return emit_vmt(res.ops, res.consts, material_name)


class Program:
    """Container tying flatten + emit together for one or more outputs."""

    def __init__(self) -> None:
        self._outputs: list[tuple[str, Expr]] = []

    def output(self, name: str, expr: Expr) -> None:
        """Declare an output variable bound to *expr*."""
        self._outputs.append((name, expr))

    def compile(
        self,
        material_name: str = 'UnlitGeneric',
        optimize: OptimizeFn = full_optimize,
        format: str = 'vmt',
    ) -> str:
        """Flatten all outputs and emit a single VMT string.

        Each output gets an ``Equals`` proxy that copies the final temp
        to the user-specified variable name.

        *optimize* is a callable ``(ops, consts) -> (ops, consts)`` that
        applies optimization passes.  Defaults to :func:`full_optimize`.
        Pass :func:`no_optimize` to skip optimization.

        *format* can be ``'vmt'`` (default) or ``'readable'``.
        """
        flattener = Flattener()

        final_temps: dict[str, str] = {}
        for name, expr in self._outputs:
            tmp = flattener.flatten(expr)
            final_temps[name] = tmp

        res = flattener.result()
        ops = list(res.ops)
        consts = res.consts

        # Append Equals per output to bind final temps to the user provided names.
        for name, tmp in final_temps.items():
            ops.append(FlatOp('Equals', {'srcVar1': tmp}, name))

        ops, consts = optimize(ops, consts)

        if format == 'readable':
            return emit_readable(ops, consts)
        return emit_vmt(ops, consts, material_name)
