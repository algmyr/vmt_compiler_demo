from __future__ import annotations

import ast
from collections.abc import Callable
import inspect
from typing import Any

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


class CompileError(Exception):
    """Raised when the input cannot be compiled to a VMT expression."""


# Mapping of Python function names to their Expr constructors.
# These are recognised inside @vmtify function bodies.
_PROXY_MAP: dict[str, type[Expr]] = {
    'Abs': Abs,
    'Clamp': Clamp,
    'CurrentTime': CurrentTime,
    'Equals': Equals,
    'Exp': Exp,
    'exp': Exp,
    'Frac': Frac,
    'Int': Int,
    'LessOrEqual': LessOrEqual,
    'PlayerPosition': PlayerPosition,
    'PlayerSpeed': PlayerSpeed,
    'SelectFirstIfNonZero': SelectFirstIfNonZero,
    'Var': Var,
    'WrapMinMax': WrapMinMax,
}

# ---------------------------------------------------------------------------
# Comparison helpers
#
# LessOrEqual(a, b, le, gr)  →  le if a <= b else gr
#
# a <  b   →  LessOrEqual(right, left, 0, 1)
# a <= b   →  LessOrEqual(left,  right, 1, 0)
# a >  b   →  LessOrEqual(left,  right, 0, 1)
# a >= b   →  LessOrEqual(right, left,  1, 0)
# a == b   →  (a <= b) AND (b <= a)
# a != b   →  NOT (a == b)
# ---------------------------------------------------------------------------


def _cmp_lt(left: Expr, right: Expr) -> Expr:
    return LessOrEqual(right, left, Const(0), Const(1))


def _cmp_lte(left: Expr, right: Expr) -> Expr:
    return LessOrEqual(left, right, Const(1), Const(0))


def _cmp_gt(left: Expr, right: Expr) -> Expr:
    return LessOrEqual(left, right, Const(0), Const(1))


def _cmp_gte(left: Expr, right: Expr) -> Expr:
    return LessOrEqual(right, left, Const(1), Const(0))


def _cmp_eq(left: Expr, right: Expr) -> Expr:
    return Mul(
        LessOrEqual(left, right, Const(1), Const(0)),
        LessOrEqual(right, left, Const(1), Const(0)),
    )


def _cmp_ne(left: Expr, right: Expr) -> Expr:
    return Sub(Const(1), _cmp_eq(left, right))


_CMP_MAP: dict[type, Callable[[Expr, Expr], Expr]] = {
    ast.Lt: _cmp_lt,
    ast.LtE: _cmp_lte,
    ast.Gt: _cmp_gt,
    ast.GtE: _cmp_gte,
    ast.Eq: _cmp_eq,
    ast.NotEq: _cmp_ne,
}


def _is_truthy(expr: Expr) -> Expr:
    """Convert any Expr to a 0/1 predicate (1 if the value is truthy)."""
    return LessOrEqual(expr, Const(0), Const(0), Const(1))


# ---------------------------------------------------------------------------
# Compiler
# ---------------------------------------------------------------------------


class VMTCompiler:
    """Compiles a Python function AST into an expression tree."""

    def __init__(self) -> None:
        self._inline_registry: dict[str, Callable[..., Expr]] = {}
        self._inline_registry.update(_registry)

    def register(self, name: str, func: Callable[..., Expr]) -> None:
        """Register *func* as an inline-able function under *name*."""
        self._inline_registry[name] = func

    def compile_function(
        self,
        func_ast: ast.FunctionDef,
        param_exprs: list[Expr],
    ) -> Expr:
        """Compile a function AST given parameter bindings."""
        param_names = [arg.arg for arg in func_ast.args.args]
        env: dict[str, Expr] = {}
        for name, expr in zip(param_names, param_exprs):
            env[name] = expr

        result = self._compile_body(func_ast.body, env)
        if result is None:
            raise CompileError('Function must have a return statement on every path')
        return result

    def _compile_body(
        self,
        stmts: list[ast.stmt],
        env: dict[str, Expr],
    ) -> Expr | None:
        """Compile a list of statements.

        Returns ``None`` when no ``return`` was encountered (normal
        fall-through).  Returns an ``Expr`` when a ``return`` was hit.
        """
        for i, stmt in enumerate(stmts):
            match stmt:
                case ast.Return(value=None):
                    return Const(0.0)
                case ast.Return():
                    return self.compile_expr(stmt.value, env)
                case ast.If():
                    rest = stmts[i + 1 :]
                    result = self._compile_if(stmt, env, rest)
                    if result is not None:
                        return result
                    continue
                case ast.Assign():
                    self._compile_assign(stmt, env)
                    continue
                case ast.AugAssign():
                    self._compile_augassign(stmt, env)
                    continue
                case ast.Expr() | ast.Pass():
                    continue
                case _:
                    raise CompileError(f'Unsupported statement: {type(stmt).__name__}')

        return None

    def _compile_if(
        self,
        stmt: ast.If,
        env: dict[str, Expr],
        rest: list[ast.stmt],
    ) -> Expr | None:
        """Compile an ``if``/``elif``/``else`` statement.

        *rest* is the list of statements that follow the ``if`` in the
        enclosing block.  It is appended to each branch so that partial
        returns are handled naturally: a branch that returns early never
        processes ``rest``, while a non-returning branch picks up
        ``rest`` as if it were part of the branch body.
        """
        cond = _is_truthy(self.compile_expr(stmt.test, env))

        # -- true branch ---------------------------------------------------
        true_env = env.copy()
        true_result = self._compile_body(stmt.body + rest, true_env)

        # -- false branch ---------------------------------------------------
        false_env = env.copy()
        false_stmts: list[ast.stmt]
        if stmt.orelse:
            false_stmts = stmt.orelse + rest
        else:
            false_stmts = rest
        false_result = self._compile_body(false_stmts, false_env)

        # -- combine results ------------------------------------------------
        if true_result is None and false_result is None:
            self._merge_env(env, true_env, false_env, cond)
            return None

        tr: Expr = true_result if true_result is not None else Const(0.0)
        fr: Expr = false_result if false_result is not None else Const(0.0)
        return LessOrEqual(cond, Const(0), fr, tr)

    def _merge_env(
        self,
        env: dict[str, Expr],
        true_env: dict[str, Expr],
        false_env: dict[str, Expr],
        cond: Expr,
    ) -> None:
        """Merge two branch environments into *env* using *cond*."""
        for var in set(true_env) | set(false_env):
            t = true_env.get(var)
            f = false_env.get(var)
            val: Expr
            if t is None:
                assert f is not None
                val = f
            elif f is None or t is f:
                val = t
            else:
                val = LessOrEqual(cond, Const(0), f, t)
            env[var] = val

    # -- assignments -------------------------------------------------------

    def _compile_assign(self, stmt: ast.Assign, env: dict[str, Expr]) -> None:
        if len(stmt.targets) != 1:
            raise CompileError('Only single-target assignment is supported')
        target = stmt.targets[0]
        if not isinstance(target, ast.Name):
            raise CompileError('Only simple variable targets are supported')
        env[target.id] = self.compile_expr(stmt.value, env)

    def _compile_augassign(self, stmt: ast.AugAssign, env: dict[str, Expr]) -> None:
        if not isinstance(stmt.target, ast.Name):
            raise CompileError('Augmented assignment target must be a simple variable')
        target_id = stmt.target.id
        lhs = self._resolve_name(target_id, env)
        rhs = self.compile_expr(stmt.value, env)

        _aug_map: dict[type, Callable[[Expr, Expr], Expr]] = {
            ast.Add: Add,
            ast.Sub: Sub,
            ast.Mult: Mul,
            ast.Div: Div,
        }
        op = _aug_map.get(type(stmt.op))
        if op is None:
            raise CompileError(
                f'Unsupported augmented assignment: {type(stmt.op).__name__}'
            )
        env[target_id] = op(lhs, rhs)

    # -- expressions -------------------------------------------------------

    def compile_expr(self, node: ast.expr, env: dict[str, Expr]) -> Expr:
        """Compile an AST expression node to an ``Expr`` tree."""
        match node:
            case ast.Constant():
                return self._compile_constant(node)
            case ast.Name():
                return self._resolve_name(node.id, env)
            case ast.UnaryOp():
                return self._compile_unary(node, env)
            case ast.BinOp():
                return self._compile_binop(node, env)
            case ast.BoolOp():
                return self._compile_boolop(node, env)
            case ast.Compare():
                return self._compile_compare(node, env)
            case ast.Call():
                return self._compile_call(node, env)
            case ast.IfExp():
                return self._compile_ifexp(node, env)
            case _:
                raise CompileError(f'Unsupported expression: {type(node).__name__}')

    @staticmethod
    def _compile_constant(node: ast.Constant) -> Expr:
        match node.value:
            case bool():
                return Const(1.0 if node.value else 0.0)
            case int() | float():
                return Const(float(node.value))
            case str():
                raise CompileError(
                    'String constants are not supported outside of Var()'
                )
            case _:
                raise CompileError(f'Unsupported constant: {type(node.value).__name__}')

    def _resolve_name(self, name: str, env: dict[str, Expr]) -> Expr:
        return env.get(name, Var(f'${name}'))

    def _compile_unary(self, node: ast.UnaryOp, env: dict[str, Expr]) -> Expr:
        operand = self.compile_expr(node.operand, env)
        if isinstance(node.op, ast.Not):
            return LessOrEqual(operand, Const(0), Const(1), Const(0))
        if isinstance(node.op, ast.USub):
            return Mul(Const(-1), operand)
        raise CompileError(f'Unsupported unary op: {type(node.op).__name__}')

    def _compile_binop(self, node: ast.BinOp, env: dict[str, Expr]) -> Expr:
        left = self.compile_expr(node.left, env)
        right = self.compile_expr(node.right, env)
        _binop_map: dict[type, Callable[[Expr, Expr], Expr]] = {
            ast.Add: Add,
            ast.Sub: Sub,
            ast.Mult: Mul,
            ast.Div: Div,
        }
        op = _binop_map.get(type(node.op))
        if op is None:
            raise CompileError(f'Unsupported binary op: {type(node.op).__name__}')
        return op(left, right)

    def _compile_boolop(self, node: ast.BoolOp, env: dict[str, Expr]) -> Expr:
        result = self.compile_expr(node.values[0], env)
        for val in node.values[1:]:
            right = self.compile_expr(val, env)
            if isinstance(node.op, ast.And):
                result = LessOrEqual(result, Const(0), Const(0), right)
            elif isinstance(node.op, ast.Or):
                result = LessOrEqual(result, Const(0), right, result)
            else:
                raise CompileError(f'Unsupported bool op: {type(node.op).__name__}')
        return result

    def _compile_compare(self, node: ast.Compare, env: dict[str, Expr]) -> Expr:
        if len(node.ops) != 1:
            raise CompileError('Chained comparisons are not supported')
        left = self.compile_expr(node.left, env)
        right = self.compile_expr(node.comparators[0], env)
        op = node.ops[0]
        func = _CMP_MAP.get(type(op))
        if func is None:
            raise CompileError(f'Unsupported comparison: {type(op).__name__}')
        return func(left, right)

    def _compile_call(self, node: ast.Call, env: dict[str, Expr]) -> Expr:
        match node.func:
            case ast.Name(id=name):
                if name in self._inline_registry:
                    args = [self.compile_expr(a, env) for a in node.args]
                    kwargs = {
                        kw.arg: self.compile_expr(kw.value, env)
                        for kw in node.keywords
                        if kw.arg is not None
                    }
                    return self._inline_registry[name](*args, **kwargs)

                proxy_cls = _PROXY_MAP.get(name)
                if proxy_cls is not None:
                    if proxy_cls is Var:
                        if (
                            len(node.args) == 1
                            and isinstance(node.args[0], ast.Constant)
                            and isinstance(node.args[0].value, str)
                        ):
                            return Var(node.args[0].value)
                        raise CompileError('Var() requires a literal string constant')
                    return self._construct_proxy(proxy_cls, node, env)

                raise CompileError(f'Unknown function: {name}')

            case ast.Attribute():
                raise CompileError('Method calls are not supported')

            case _:
                raise CompileError(
                    f'Unsupported call target: {type(node.func).__name__}'
                )

    def _construct_proxy(
        self,
        proxy_cls: type[Expr],
        node: ast.Call,
        env: dict[str, Expr],
    ) -> Expr:
        args = [self.compile_expr(a, env) for a in node.args]
        kwargs: dict[str, Expr] = {}
        for kw in node.keywords:
            if kw.arg is not None:
                kwargs[kw.arg] = self.compile_expr(kw.value, env)

        if proxy_cls is Var:
            if len(args) == 1 and isinstance(args[0], Const):
                return Var(str(args[0].value))
            raise CompileError('Var() requires a literal string constant')

        return proxy_cls(*args, **kwargs)

    def _compile_ifexp(self, node: ast.IfExp, env: dict[str, Expr]) -> Expr:
        cond = _is_truthy(self.compile_expr(node.test, env))
        body = self.compile_expr(node.body, env)
        orelse = self.compile_expr(node.orelse, env)
        return LessOrEqual(cond, Const(0), orelse, body)


# ---------------------------------------------------------------------------
# public decorator
# ---------------------------------------------------------------------------


class VMTFunction:
    """A compiled ``@vmtify`` function that produces ``Expr`` trees."""

    def __init__(self, func: Callable[..., Any]) -> None:
        import textwrap

        self._func = func
        self._name = func.__name__
        source = inspect.getsource(func)
        lines = source.split('\n')
        while lines and lines[0].strip().startswith('@'):
            lines.pop(0)
        source_clean = textwrap.dedent('\n'.join(lines))
        self._source = source_clean
        self._ast = ast.parse(source_clean)
        assert isinstance(self._ast, ast.Module)
        func_def = self._ast.body[0]
        assert isinstance(func_def, ast.FunctionDef)
        self._func_def = func_def
        self._param_names = [arg.arg for arg in func_def.args.args]

    def __call__(self, *args: Expr | float, **kwargs: Expr | float) -> Expr:
        """Compile the ``@vmtify`` function with the given argument expressions."""
        from material_proxy.expr import _as_expr

        compiler = VMTCompiler()

        # Build positional args from *args and **kwargs
        resolved: list[Expr] = []
        seen_kw: set[str] = set()
        for i, name in enumerate(self._param_names):
            if i < len(args):
                resolved.append(_as_expr(args[i]))
            elif name in kwargs:
                resolved.append(_as_expr(kwargs[name]))
                seen_kw.add(name)
            else:
                raise TypeError(f'{self._name} missing required argument: {name}')

        unexpected = set(kwargs) - seen_kw
        if unexpected:
            raise TypeError(
                f'{self._name} got unexpected keyword argument(s): '
                f'{", ".join(sorted(unexpected))}'
            )

        return compiler.compile_function(self._func_def, resolved)

    def __repr__(self) -> str:
        return f'<@vmtify {self._name}>'


_registry: dict[str, VMTFunction] = {}


def vmtify(func: Callable[..., Any]) -> VMTFunction:
    """Decorator that compiles a Python function to an ``Expr`` factory.

    Inside the decorated function, use ordinary Python control flow
    (``if``/``elif``/``else``, arithmetic, comparisons) and recognised
    proxy calls (``CurrentTime()``, ``Clamp(...)``,  etc.).  The compiled
    function returns an ``Expr`` tree that composes with the rest of
    the expression system.
    """
    wrapped = VMTFunction(func)
    _registry[func.__name__] = wrapped
    return wrapped
