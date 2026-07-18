from __future__ import annotations

from material_proxy.emit import emit_vmt
from material_proxy.expr import Expr
from material_proxy.flatten import FlatOp
from material_proxy.flatten import Flattener
from material_proxy.optimize import constant_fold
from material_proxy.optimize import dead_code_elimination
from material_proxy.optimize import temp_reuse


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
        optimize: bool = True,
    ) -> str:
        """Flatten all outputs and emit a single VMT string.

        Each output gets an ``Equals`` proxy that copies the final temp
        to the user-specified variable name.
        """
        flattener = Flattener()

        final_temps: dict[str, str] = {}
        for name, expr in self._outputs:
            tmp = flattener.flatten(expr)
            final_temps[name] = tmp

        res = flattener.result()
        ops = list(res.ops)
        consts = res.consts
        temp_to_val: dict[str, float] = {}

        if optimize:
            ops, consts, temp_to_val = constant_fold(ops, consts)

        # Build a reverse lookup: folded temp → constant name
        temp_to_name: dict[str, str] = {
            t: consts[v] for t, v in temp_to_val.items() if v in consts
        }

        # Append an Equals per output to assign the final temp to the user name
        for name, tmp in final_temps.items():
            src = temp_to_name.get(tmp, tmp)
            ops.append(FlatOp('Equals', {'srcVar1': src}, name))

        if optimize:
            # Keep user-variable results and any temps they reference alive
            live_temps: set[str] = set()
            for op in ops:
                is_user_var = op.result.startswith('$') and not op.result.startswith(
                    '$tmp_'
                )
                if op.proxy == 'Equals' or is_user_var:
                    live_temps.add(op.result)
            ops = dead_code_elimination(ops, live_temps)
            ops = temp_reuse(ops)

        return emit_vmt(ops, consts, material_name)
