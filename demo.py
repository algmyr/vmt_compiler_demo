"""Express example.vmt logic using the material proxy expression system."""

from material_proxy import Abs
from material_proxy import Clamp
from material_proxy import Const
from material_proxy import CurrentTime
from material_proxy import Frac
from material_proxy import Int
from material_proxy import LessOrEqual
from material_proxy import PlayerPosition
from material_proxy import Program
from material_proxy import Var
from material_proxy import full_optimize
from material_proxy import no_optimize

time = CurrentTime()
pos = PlayerPosition(scale=1)

z = Clamp(Const(0), pos, pos)

subtick = Const(200) / 3 * time
tick = Int(subtick)

prev_outlier_z = Var('prev_outlier_z')
prev_outlier_tick = Var('prev_outlier_tick')

diff = Abs(prev_outlier_z - z)
z_changed = LessOrEqual(diff, diff * -1, Const(0), Const(1))

near_tick = LessOrEqual(tick - subtick, 1 - 0.9, Const(1), Const(0))

guard = LessOrEqual(z_changed, near_tick, Const(1), Const(0))

delta = LessOrEqual(guard, 0.5, Var('delta'), tick - prev_outlier_tick)
vz_raw = (z - prev_outlier_z) / delta - 0.09 * delta
vz = LessOrEqual(guard, 0.5, Var('vz'), vz_raw)

vz_div = vz / 0.18
round_adj = LessOrEqual(0.5, Frac(vz_div), Const(1), Const(0))
delta_peak = LessOrEqual(guard, 0.5, Var('delta_peak'), Int(vz_div) + round_adj)

peak_z = prev_outlier_z + vz * delta_peak - 0.09 * delta_peak * delta_peak
out_max_z = LessOrEqual(guard, 0.5, Var('out_max_z'), peak_z)

prog = Program.from_tree(
    out_max_z=out_max_z,
    vz=vz,
    delta_peak=delta_peak,
)

print('=== Without optimisation ===')
print(prog.compile('UnlitGeneric', optimize=no_optimize))
print()
print('=== With optimisation (fold + DCE + temp reuse) ===')
print(prog.compile('UnlitGeneric', optimize=full_optimize))
