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


def emit_readable(
    ops: list[FlatOp],
    consts: dict[float, str],
) -> str:
    """Format a list of ``FlatOp``s into readable Python-like code."""
    const_names: dict[str, float] = {v: k for k, v in consts.items()}

    def _ref(r: str) -> str:
        if r in const_names:
            return f'{const_names[r]!r}'
        if r.startswith('$tmp_'):
            return r[1:]
        if r.startswith('$'):
            return r[1:]
        return r

    def _result(r: str) -> str:
        if r.startswith('$tmp_'):
            return r[1:]
        return r

    lines: list[str] = []
    for op in ops:
        r = _result(op.result)
        match op.proxy:
            case 'Add':
                a = _ref(op.params['srcVar1'])
                b = _ref(op.params['srcVar2'])
                lines.append(f'{r} = {a} + {b}')
            case 'Subtract':
                a = _ref(op.params['srcVar1'])
                b = _ref(op.params['srcVar2'])
                lines.append(f'{r} = {a} - {b}')
            case 'Multiply':
                a = _ref(op.params['srcVar1'])
                b = _ref(op.params['srcVar2'])
                lines.append(f'{r} = {a} * {b}')
            case 'Divide':
                a = _ref(op.params['srcVar1'])
                b = _ref(op.params['srcVar2'])
                lines.append(f'{r} = {a} / {b}')
            case 'Abs':
                a = _ref(op.params['srcVar1'])
                lines.append(f'{r} = abs({a})')
            case 'Frac':
                a = _ref(op.params['srcVar1'])
                lines.append(f'{r} = frac({a})')
            case 'Int':
                a = _ref(op.params['srcVar1'])
                lines.append(f'{r} = int({a})')
            case 'Equals':
                a = _ref(op.params['srcVar1'])
                lines.append(f'{r} = {a}')
            case 'Clamp':
                a = _ref(op.params['srcVar1'])
                lo = _ref(op.params['min'])
                hi = _ref(op.params['max'])
                lines.append(f'{r} = clamp({a}, {lo}, {hi})')
            case 'WrapMinMax':
                a = _ref(op.params['srcVar1'])
                lo = _ref(op.params['minVal'])
                hi = _ref(op.params['maxVal'])
                lines.append(f'{r} = wrap_min_max({a}, {lo}, {hi})')
            case 'LessOrEqual':
                a = _ref(op.params['srcVar1'])
                b = _ref(op.params['srcVar2'])
                le = _ref(op.params['lessEqualVar'])
                gr = _ref(op.params['greaterVar'])
                lines.append(f'{r} = {le} if {a} <= {b} else {gr}')
            case 'SelectFirstIfNonZero':
                a = _ref(op.params['srcVar1'])
                b = _ref(op.params['srcVar2'])
                lines.append(f'{r} = {a} if {a} != 0 else {b}')
            case 'Exponential':
                a = _ref(op.params['srcVar1'])
                off = _ref(op.params['offset'])
                sc = _ref(op.params['scale'])
                parts: list[str] = []
                if off != '0.0':
                    parts.append(f'offset={off}')
                if sc != '1.0':
                    parts.append(f'scale={sc}')
                if 'minVal' in op.params:
                    lo = _ref(op.params['minVal'])
                    hi = _ref(op.params['maxVal'])
                    parts.append(f'min={lo}, max={hi}')
                if parts:
                    lines.append(f'{r} = exp({a}, {", ".join(parts)})')
                else:
                    lines.append(f'{r} = exp({a})')
            case 'CurrentTime':
                lines.append(f'{r} = curtime()')
            case 'PlayerPosition':
                sc = _ref(op.params.get('scale', '1'))
                lines.append(f'{r} = player_pos(scale={sc})')
            case 'PlayerSpeed':
                sc = _ref(op.params.get('scale', '1'))
                lines.append(f'{r} = player_speed(scale={sc})')
            case _:
                lines.append(f'{r} = {op.proxy}({", ".join(op.params.values())})')
    return '\n'.join(lines)
