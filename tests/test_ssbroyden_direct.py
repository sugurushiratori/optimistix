"""SSBroyden with use_inverse=False (direct update of B) must match use_inverse=True.

Both forms describe the same self-scaled Broyden update, so in exact arithmetic the
iterates coincide.  Before the fix of h_k (= y^T B^{-1} y / s^T y), of the exponent
1/(1-N) and of its double application, the direct form did not converge.
"""

import jax
import jax.numpy as jnp
import numpy as np
import optimistix as optx
import pytest


jax.config.update("jax_enable_x64", True)


def _quadratic(n=20, cond=1e3, seed=0):
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    a = jnp.asarray(q @ np.diag(np.logspace(0, np.log10(cond), n)) @ q.T)
    return (lambda x, _: 0.5 * x @ a @ x), jnp.asarray(rng.standard_normal(n))


@pytest.mark.parametrize("seed", [0, 1])
def test_direct_matches_inverse(seed):
    fn, x0 = _quadratic(seed=seed)
    out = {}
    for inv in (True, False):
        solver = optx.SSBroyden(rtol=1e-12, atol=1e-12, use_inverse=inv)
        out[inv] = optx.minimise(fn, solver, x0, max_steps=40, throw=False).value
    rel = jnp.linalg.norm(out[True] - out[False]) / jnp.linalg.norm(out[True])
    assert rel < 1e-6, rel


def test_direct_converges():
    fn, x0 = _quadratic(n=30, cond=1e4, seed=3)
    solver = optx.SSBroyden(rtol=1e-12, atol=1e-12, use_inverse=False)
    sol = optx.minimise(fn, solver, x0, max_steps=500, throw=False)
    assert fn(sol.value, None) < 1e-16
