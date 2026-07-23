from __future__ import annotations

from material_proxy._ops import COMPUTE
from material_proxy._ops import PARAM_ORDER
from material_proxy.flatten import FlatOp


def _const_name(value: float) -> str:
    """Format a constant value as a VMT variable name (with ``$``)."""
    return f'${value!r}'


def constant_fold(
    ops: list[FlatOp],
    consts: dict[float, str],
) -> tuple[list[FlatOp], dict[float, str]]:
    """Fold ops where all sources are compile-time constants.

    Returns ``(folded_ops, updated_consts)``.
    """
    name_to_val: dict[str, float] = {v: k for k, v in consts.items()}
    temp_to_val: dict[str, float] = {}
    new_consts: dict[float, str] = dict(consts)
    result_ops: list[FlatOp] = []

    for op in ops:
        fn = COMPUTE.get(op.proxy)
        param_keys = PARAM_ORDER.get(op.proxy)

        if fn is None or param_keys is None:
            # Source proxies (CurrentTime, PlayerPosition, etc.) can't fold
            _patch_params(op.params, temp_to_val, name_to_val, new_consts)
            result_ops.append(op)
            continue

        if op.proxy == 'Equals':
            # Equals is a name binding — never fold it away.
            # _patch_params resolves folded temps in its src param.
            _patch_params(op.params, temp_to_val, name_to_val, new_consts)
            result_ops.append(op)
            continue

        # Resolve args — all must be constant
        args: list[float] = []
        foldable = True
        for key in param_keys:
            val = op.params.get(key)
            if val is None:
                foldable = False
                break
            if val in temp_to_val:
                args.append(temp_to_val[val])
            elif val in name_to_val:
                args.append(name_to_val[val])
            else:
                foldable = False
                break

        if foldable:
            try:
                computed = fn(*args)
                temp_to_val[op.result] = computed
                # Ensure the folded constant is in the constants table
                if computed not in new_consts:
                    name = _const_name(computed)
                    new_consts[computed] = name
                    name_to_val[name] = computed
            except (ZeroDivisionError, OverflowError, ValueError):
                foldable = False

        if not foldable:
            _patch_params(op.params, temp_to_val, name_to_val, new_consts)
            result_ops.append(op)

    return result_ops, new_consts


def _is_user_var(name: str) -> bool:
    return name.startswith('$') and not name.startswith('$tmp_')


def dead_code_elimination(ops: list[FlatOp]) -> list[FlatOp]:
    """Remove ops whose result temp is never used as a source.

    Walk ops in reverse: an op is kept if its result is in the live set,
    and its source temps are then added to the live set.

    ``Equals`` ops and user-var results seed the live set — callers that
    work with bare FlatOps (no ``Equals``) should append an ``Equals``
    targeting the intended output temp before calling this function.
    """
    live = {op.result for op in ops if op.proxy == 'Equals' or _is_user_var(op.result)}
    result: list[FlatOp] = []
    for op in reversed(ops):
        if op.result in live:
            result.append(op)
            for v in op.params.values():
                live.add(v)
    result.reverse()
    return result


def temp_reuse(ops: list[FlatOp]) -> list[FlatOp]:
    """Reduce temp count by reassigning slots based on liveness intervals.

    Temps whose lifetimes don't overlap share the same ``$tmp_N`` slot.
    """
    if not ops:
        return ops

    # Collect defined temps (only $tmp_N, not user vars or constants)
    def_temps: set[str] = set()
    for op in ops:
        if op.result.startswith('$tmp_'):
            def_temps.add(op.result)

    # Compute liveness interval for each defined temp.
    # The *end* of a temp's liveness is its *last read* position (param use),
    # not its last write (result).  This allows a later temp to reuse the slot
    # of a temp whose last read is the current operation (in-place reuse).
    intervals: list[tuple[int, int, str]] = []
    for t in def_temps:
        start = -1
        end = -1
        for i, op in enumerate(ops):
            if op.result == t:
                if start == -1:
                    start = i
            for v in op.params.values():
                if v == t:
                    if start == -1:
                        start = i
                    end = i
        if end == -1:
            # Defined but never read — its last use is its last definition.
            for i, op in enumerate(ops):
                if op.result == t:
                    end = i
        intervals.append((start, end, t))

    # Sort by start position
    intervals.sort()

    # Greedy slot assignment
    # A slot is reusable when its free-at ≤ the current interval's start,
    # enabling in-place reuse (write a temp's slot with a new value once
    # the old value's last read has passed).
    slots: list[int] = []
    old_to_new: dict[str, str] = {}
    for start, end, t in intervals:
        placed = False
        for slot_idx, free_at in enumerate(slots):
            if free_at <= start:
                slots[slot_idx] = end
                old_to_new[t] = f'$tmp_{slot_idx + 1}'
                placed = True
                break
        if not placed:
            slots.append(end)
            old_to_new[t] = f'$tmp_{len(slots)}'

    # Apply renaming
    result: list[FlatOp] = []
    for op in ops:
        new_params = {k: old_to_new.get(v, v) for k, v in op.params.items()}
        new_result = old_to_new.get(op.result, op.result)
        result.append(FlatOp(op.proxy, new_params, new_result))
    return result


def _patch_params(
    params: dict[str, str],
    temp_to_val: dict[str, float],
    name_to_val: dict[str, float],
    new_consts: dict[float, str],
) -> None:
    """Rewrite param values that now refer to folded constants."""
    for k, v in list(params.items()):
        if v in temp_to_val:
            val = temp_to_val[v]
            name = new_consts.get(val)
            if name is None:
                name = _const_name(val)
                new_consts[val] = name
                name_to_val[name] = val
            params[k] = name
