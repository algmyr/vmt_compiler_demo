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
    """Compiled material program backed by a flat op list."""

    def __init__(
        self,
        ops: list[FlatOp],
        consts: dict[float, str],
    ) -> None:
        self._ops = list(ops)
        self._consts = dict(consts)

    @property
    def ops(self) -> list[FlatOp]:
        """Flat op list (copy)."""
        return list(self._ops)

    @property
    def consts(self) -> dict[float, str]:
        """Constant table (copy)."""
        return dict(self._consts)

    @classmethod
    def from_tree(cls, **outputs: Expr) -> Program:
        """Flatten one or more named expression trees into a Program.

        Each keyword argument becomes an output variable — ``$`` is
        prepended to the key automatically:

            prog = Program.from_tree(result=Add(x, y))      # → $result
            prog = Program.from_tree(sum=s, double=Mul(s, 2))  # → $sum, $double
        """
        flattener = Flattener()
        result_outputs: list[tuple[str, str]] = []
        for key, expr in outputs.items():
            name = f'${key}'
            tmp = flattener.flatten(expr)
            result_outputs.append((name, tmp))

        res = flattener.result()
        ops: list[FlatOp] = list(res.ops)
        for name, tmp in result_outputs:
            ops.append(FlatOp('Equals', {'srcVar1': tmp}, name))
        return cls(ops, res.consts)

    @classmethod
    def from_flat(
        cls,
        ops: list[FlatOp],
        consts: dict[float, str],
    ) -> Program:
        """Wrap pre-flattened ops and constants.

        Outputs must already be encoded as ``Equals`` ops in *ops*.
        """
        return cls(ops, consts)

    def optimize(self, optimize_fn: OptimizeFn = full_optimize) -> Program:
        """Apply *optimize_fn* and return a new Program with the result.

        The original program is unchanged.
        """
        ops, consts = optimize_fn(list(self._ops), dict(self._consts))
        return Program(ops, consts)

    def emit(self, material_name: str = 'UnlitGeneric', format: str = 'vmt') -> str:
        """Emit the program as a VMT (default) or readable string."""
        if format == 'readable':
            return emit_readable(self._ops, self._consts)
        return emit_vmt(self._ops, self._consts, material_name)

    def compile(
        self,
        material_name: str = 'UnlitGeneric',
        optimize: OptimizeFn = full_optimize,
        format: str = 'vmt',
    ) -> str:
        """Shorthand for ``prog.optimize(optimize).emit(material_name, format)``."""
        return self.optimize(optimize).emit(material_name, format)
