from __future__ import annotations

from material_proxy.flatten import FlatOp


def _fmt_const_value(value: float) -> str:
    return f'{value!r}'


def emit_vmt(
    ops: list[FlatOp],
    consts: dict[float, str],
    material_name: str = 'UnlitGeneric',
) -> str:
    """Format a list of ``FlatOp``s and constants into a VMT string."""
    lines = [f'"{material_name}" {{']
    for value, name in consts.items():
        lines.append(f'\t"{name}" {_fmt_const_value(value)}')
    if consts:
        lines.append('')
    lines.append('\tProxies {')
    for op in ops:
        inner = ' '.join(f'{k} "{v}"' for k, v in op.params.items())
        inner += f' resultVar "{op.result}"'
        lines.append(f'\t\t{op.proxy} {{ {inner} }}')
    lines.append('\t}')
    lines.append('}')
    return '\n'.join(lines)
