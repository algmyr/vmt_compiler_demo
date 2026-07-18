from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
import math
import re
from typing import Any

from material_proxy._ops import COMPUTE
from material_proxy._ops import PARAM_ORDER


@dataclass
class EvalContext:
    """Provides values for source proxies during VMT interpretation.

    Example::

        ctx = EvalContext(vars={"$x": 3.0, "$y": 4.0})
        interpret_vmt(vmt, ctx)
    """

    vars: dict[str, float] = field(default_factory=dict)
    time: float = 0.0
    speed: float = 0.0
    pos: tuple = (0.0, 0.0, 0.0)


def _parse_proxy_block(vmt_text: str) -> list[tuple[str, dict[str, str]]]:
    """Extract proxy statements from the ``Proxies { ... }`` block."""
    m = re.search(r'Proxies\s*\{', vmt_text)
    if not m:
        return []

    start = m.end()
    depth = 1
    i = start
    while i < len(vmt_text) and depth > 0:
        if vmt_text[i] == '{':
            depth += 1
        elif vmt_text[i] == '}':
            depth -= 1
        i += 1
    body = vmt_text[start : i - 1]

    proxies = []
    for m2 in re.finditer(r'(\w+)\s*\{\s*([^}]*?)\s*\}', body):
        name = m2.group(1)
        pairs = re.finditer(r'(\w+)\s+"([^"]*)"', m2.group(2))
        params = {p.group(1): p.group(2) for p in pairs}
        proxies.append((name, params))
    return proxies


def _parse_constants(vmt_text: str) -> dict[str, float]:
    """Extract ``"$name" value`` declarations from the material header."""
    consts: dict[str, float] = {}
    for m in re.finditer(r'"(\$\S+)"\s+(-inf|inf|nan|[\d.eE+-]+)', vmt_text):
        consts[m.group(1)] = float(m.group(2))
    return consts


Env = dict[str, Any]


def _resolve(val: str, env: Env) -> float:
    """Resolve a proxy parameter to a float.

    * ``$var`` → lookup in *env*
    * ``$var[idx]`` → lookup vector component (requires ``$var`` to be a tuple)
    * Otherwise → parse as literal float.
    """
    if val.startswith('$'):
        bracket = val.find('[')
        if bracket != -1:
            name = val[:bracket]
            idx = int(val[bracket + 1 : val.find(']')])
            v = env[name]
            if isinstance(v, tuple):
                return float(v[idx])
            return float(v)
        raw = env[val]
        return float(raw) if not isinstance(raw, float) else raw
    return float(val)


def _execute_proxy(
    name: str, params: dict[str, str], env: Env, ctx: EvalContext
) -> None:
    dst = params.get('resultVar')
    if dst is None:
        return

    match name:
        case 'CurrentTime':
            env[dst] = ctx.time

        case 'PlayerSpeed':
            scale_raw = params.get('scale', '1')
            scale = (
                _resolve(scale_raw, env)
                if scale_raw.startswith('$')
                else float(scale_raw)
            )
            env[dst] = ctx.speed * scale

        case 'PlayerPosition':
            scale_raw = params.get('scale', '1')
            scale = (
                _resolve(scale_raw, env)
                if scale_raw.startswith('$')
                else float(scale_raw)
            )
            env[dst] = tuple(v * scale for v in ctx.pos)

        case 'Exponential':
            src = _resolve(params['srcVar1'], env)
            off = _resolve(params.get('offset', '0'), env)
            sc = _resolve(params.get('scale', '1'), env)
            result = sc * math.exp(src + off)
            if 'minVal' in params:
                lo = _resolve(params['minVal'], env)
                hi = _resolve(params['maxVal'], env)
                result = max(lo, min(hi, result))
            env[dst] = result

        case _:
            fn = COMPUTE.get(name)
            if fn is None:
                raise ValueError(f'Unknown proxy: {name}')
            args = tuple(
                _resolve(params.get(k, '0'), env) for k in PARAM_ORDER.get(name, ())
            )
            env[dst] = fn(*args)


def interpret_vmt(vmt_text: str, ctx: EvalContext) -> dict[str, float]:
    """Execute the ``Proxies { ... }`` block from a VMT string.

    Returns the final state of all variables after every proxy has run.
    """
    proxies = _parse_proxy_block(vmt_text)
    consts = _parse_constants(vmt_text)

    env: Env = {}
    env.update(consts)
    env.update(ctx.vars)

    for name, params in proxies:
        _execute_proxy(name, params, env, ctx)

    return {k: v for k, v in env.items() if isinstance(v, (int, float))}
