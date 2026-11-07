"""Tests for optimistix.LSSBroyden.

1. With history_length >= #updates and h0="identity", LSSBroyden must reproduce a dense
   SSBroyden reference (same search, same update rule) iterate by iterate.
2. update="bfgs" must reproduce a dense BFGS reference likewise.
3. Works on a pytree (equinox MLP) variable.
"""

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.flatten_util as jfu
import lineax as lx
import optimistix as optx
import pytest
from optimistix._misc import default_verbose
from optimistix._search import FunctionInfo
from optimistix._solver.gauss_newton import NewtonDescent
from optimistix._solver.limited_memory_ssbroyden import ss_tau_phi
from optimistix._solver.quasi_newton import AbstractQuasiNewton


jax.config.update("jax_enable_x64", True)


class _DenseState(eqx.Module):
    H: jax.Array


class DenseSSBroydenRef(AbstractQuasiNewton):
    """Dense reference on flat 1-D `y` (same alpha-from-geometry convention)."""

    rtol: float
    atol: float
    norm: object
    use_inverse: bool
    descent: NewtonDescent
    search: optx.Zoom
    update: str
    verbose: Callable[..., None]

    def __init__(self, update="ssbroyden"):
        self.rtol = self.atol = 1e-14
        self.norm = optx.max_norm
        self.use_inverse = True
        self.descent = NewtonDescent()
        self.search = optx.Zoom(initial_guess_strategy="one")
        self.update = update
        self.verbose = default_verbose(False)

    def _op(self, H):
        return lx.MatrixLinearOperator(H, lx.positive_semidefinite_tag)

    def init_hessian(self, y, f, grad):
        H = jnp.eye(y.size, dtype=y.dtype)
        return FunctionInfo.EvalGradHessianInv(f, grad, self._op(H)), _DenseState(H)

    def update_hessian(self, y, y_eval, f_info, f_eval_info, st, step_size):
        H = st.H
        s = y_eval - y
        yv = f_eval_info.grad - f_info.grad
        sy = s @ yv
        ok = sy > jnp.finfo(sy.dtype).eps * jnp.sqrt((s @ s) * (yv @ yv))

        def upd(H):
            sy_s = jnp.where(ok, sy, 1.0)
            Hy = H @ yv
            p = -H @ f_info.grad
            alpha = (p @ s) / (p @ p)
            b = -alpha * (s @ f_info.grad) / sy_s
            yHy = yv @ Hy
            tau, phi = ss_tau_phi(b, yHy / sy_s, y.size, self.update)
            v = s / sy_s - Hy / yHy
            Hn = (H - jnp.outer(Hy, Hy) / yHy + phi * yHy * jnp.outer(v, v)) / tau
            Hn = Hn + jnp.outer(s, s) / sy_s
            return 0.5 * (Hn + Hn.T)

        H = jax.lax.cond(ok, upd, lambda H: H, H)
        return (
            FunctionInfo.EvalGradHessianInv(f_eval_info.f, f_eval_info.grad, self._op(H)),
            _DenseState(H),
        )


def _rosen(x, _):
    return jnp.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)


def _mlp_problem():
    key = jax.random.PRNGKey(0)
    net = eqx.nn.MLP(1, 1, 10, 1, activation=jnp.tanh, key=key)
    net = jax.tree_util.tree_map(
        lambda x: x.astype(jnp.float64) if eqx.is_array(x) else x, net
    )
    X = jnp.linspace(-1, 1, 64)[:, None]
    Y = jnp.sin(3 * X)
    params, static = eqx.partition(net, eqx.is_array)

    def loss(p, _):
        model = eqx.combine(p, static)
        return jnp.mean((jax.vmap(model)(X) - Y) ** 2)

    return params, loss


@pytest.mark.parametrize("update", ["ssbroyden", "ssbfgs", "bfgs"])
def test_matches_dense_reference_rosenbrock(update):
    x0 = jnp.full(10, -1.2)
    steps = 80  # optimistix steps (= function evaluations)
    dense = optx.minimise(
        _rosen, DenseSSBroydenRef(update), x0, max_steps=steps, throw=False
    )
    lim = optx.minimise(
        _rosen,
        optx.LSSBroyden(1e-14, 1e-14, history_length=steps, update=update, h0="identity"),
        x0,
        max_steps=steps,
        throw=False,
    )
    rel = jnp.linalg.norm(dense.value - lim.value) / jnp.linalg.norm(dense.value)
    assert rel < 1e-8, rel


@pytest.mark.parametrize("update", ["ssbroyden", "bfgs"])
def test_matches_dense_reference_mlp(update):
    params, loss = _mlp_problem()
    flat0, unravel = jfu.ravel_pytree(params)
    steps = 80
    dense = optx.minimise(
        lambda f, a: loss(unravel(f), a),
        DenseSSBroydenRef(update),
        flat0,
        max_steps=steps,
        throw=False,
    )
    # LSSBroyden directly on the pytree
    lim = optx.minimise(
        loss,
        optx.LSSBroyden(1e-14, 1e-14, history_length=steps, update=update, h0="identity"),
        params,
        max_steps=steps,
        throw=False,
    )
    lim_flat, _ = jfu.ravel_pytree(lim.value)
    rel = jnp.linalg.norm(dense.value - lim_flat) / jnp.linalg.norm(dense.value)
    assert rel < 1e-8, rel


def test_limited_memory_converges():
    params, loss = _mlp_problem()
    sol = optx.minimise(
        loss,
        optx.LSSBroyden(1e-10, 1e-10, history_length=10),
        params,
        max_steps=2000,
        throw=False,
    )
    assert loss(sol.value, None) < 1e-4


def test_ss_tau_phi_large_a():
    # 1 - sqrt(a/(1+a)) underflows to 0 for a >~ 1e16 in the direct form
    import jax.numpy as jnp
    import numpy as np

    for a in [1e6, 1e10, 1e16, 1e20]:
        for b in [1e-3, 0.3, 1.0, 10.0]:
            tau, phi = ss_tau_phi(jnp.asarray(b), jnp.asarray((1.0 + a) / b), 1000, "ssbroyden")
            assert np.isfinite(tau) and np.isfinite(phi) and tau > 0 and phi >= 0
