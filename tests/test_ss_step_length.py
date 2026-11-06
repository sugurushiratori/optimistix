"""SSBFGS / SSBroyden must compute b_k with the step length actually taken.

After the fix, the dense `optx.SSBFGS` / `optx.SSBroyden` (use_inverse=True) reproduce
the dense reference of `test_lssbroyden.py`, which recovers alpha from the step
geometry, when run with the same line search.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import optimistix as optx
import pytest
from optimistix._solver.quasi_newton import _taken_step_length

from .test_lssbroyden import DenseSSBroydenRef


jax.config.update("jax_enable_x64", True)


def _rosen(x, _):
    return jnp.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)


@pytest.mark.parametrize(
    "solver, update, search",
    [
        (optx.SSBFGS(1e-14, 1e-14), "ssbfgs", optx.Zoom(initial_guess_strategy="one")),
        (optx.SSBroyden(1e-14, 1e-14), "ssbroyden", optx.BacktrackingArmijo()),
    ],
)
def test_fork_solvers_match_reference(solver, update, search):
    x0 = jnp.full(10, -1.2)
    steps = 150
    ref = eqx.tree_at(lambda s: s.search, DenseSSBroydenRef(update), search)
    a = optx.minimise(_rosen, solver, x0, max_steps=steps, throw=False)
    b = optx.minimise(_rosen, ref, x0, max_steps=steps, throw=False)
    rel = jnp.linalg.norm(a.value - b.value) / jnp.linalg.norm(b.value)
    assert rel < 1e-8, rel


def test_taken_step_length_recovers_alpha():
    import lineax as lx
    from optimistix._search import FunctionInfo

    H = jnp.array([[2.0, 0.3], [0.3, 1.0]])
    g = jnp.array([0.7, -1.1])
    p = -H @ g
    f_info = FunctionInfo.EvalGradHessianInv(
        jnp.array(0.0), g, lx.MatrixLinearOperator(H, lx.positive_semidefinite_tag)
    )
    assert jnp.allclose(_taken_step_length(0.37 * p, g, f_info, True), 0.37)
