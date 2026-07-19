# ruff: noqa: PGH004
# ruff: noqa

import material_proxy as mp
from material_proxy import Program
from material_proxy import Var
from material_proxy import full_optimize
from material_proxy import no_optimize
from material_proxy import vmtify


# @vmtify
def sqrt(x):
    guess = (1.0 + x) * 4.3e-12
    guess = (guess + x / guess) * 2.1e-06
    guess = (guess + x / guess) * 0.0014
    guess = (guess + x / guess) * 0.037
    guess = (guess + x / guess) * 0.19
    guess = (guess + x / guess) * 0.41
    guess = (guess + x / guess) * 0.495
    guess = (guess + x / guess) * 0.5
    guess = (guess + x / guess) * 0.5

    guess = mp.LessOrEqual(guess, 0.0, guess, 0.0)  # I think?
    # guess = sel_ge(guess, guess, 0.0, 0.0)

    return guess


x = Var('x')

res = sqrt(x)

prog = Program.from_tree(res=res)

print('=== Without optimisation ===')
print(prog.compile('UnlitGeneric', optimize=no_optimize, format='readable'))
print()
print('=== With optimisation (fold + DCE + temp reuse) ===')
print(prog.compile('UnlitGeneric', optimize=full_optimize, format='readable'))
