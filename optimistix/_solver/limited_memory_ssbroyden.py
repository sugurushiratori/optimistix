"""Limited-memory self-scaled Broyden (L-SSBroyden) for Optimistix.

Notation follows Optimistix: `y` is the optimisation variable, so the textbook
`s_k = x_{k+1} - x_k` is called `y_diff` and the textbook `y_k = g_{k+1} - g_k` is
called `grad_diff`.

Inverse update (Kiyani et al., CMAME 446 (2025) 118308, Eq. (8)):

    H+ = (1/tau) [ H - H y y^T H / (y^T H y) + phi (y^T H y) v v^T ] + s s^T / (y^T s)
    v  = s / (y^T s) - H y / (y^T H y)

with (tau, phi) chosen by the SSBroyden rule of Sec. 1.3 of that paper.

Limited-memory representation ("Gram-space compact form"):

    H_k = c * gamma * I + P A P^T,     P = [s_1..s_m, y_1..y_m]

The 2m x 2m matrix A and the scalar c are rebuilt after every accepted step by
replaying the m stored updates using only the Gram matrix G = P^T P and the stored
scalars (tau_i, phi_i). Applying H costs two batched inner products / combinations
with the history, i.e. O(mn), plus O(m^2); rebuilding costs O(m^3).
With `history_length >= #steps` and `h0="identity"` this reproduces the dense
SSBroyden iterates exactly (up to round-off).

`b_k = s^T B s / s^T y` is computed as `-alpha s^T g / s^T y` (Eq. (5)), where alpha is
recovered from the geometry of the step (`alpha = <p, s> / <p, p>`, p = -H g), so it is
correct for any line search, including those whose returned `step_size` is the
proposal for the *next* step rather than the one that produced `y_eval`.
"""

from collections.abc import Callable
from typing import Generic

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import lineax as lx
from equinox import AbstractVar
from equinox.internal import ω
from jaxtyping import Array, Float, Int, PyTree, Scalar

from .._custom_types import Aux, Y
from .._misc import default_verbose, filter_cond, max_norm, tree_dot
from .._search import AbstractSearch, FunctionInfo
from .gauss_newton import NewtonDescent
from .quasi_newton import AbstractQuasiNewton
from .zoom import Zoom


_v_tree_dot = jax.vmap(tree_dot, in_axes=(0, None), out_axes=0)


def _is_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def ss_tau_phi(b, h, n: int, update: str):
    """(tau, phi) for the inverse update. b = s^T B s / s^T y, h = y^T H y / s^T y."""
    one = jnp.ones_like(b)
    if update == "bfgs":
        return one, one
    rho_p = jnp.minimum(1.0, 1.0 / b)
    if update == "ssbfgs":
        return rho_p, one
    if update != "ssbroyden":
        raise ValueError(f"unknown update {update!r}")
    a = jnp.maximum(b * h - 1.0, 0.0)
    small = a <= 1e-12
    a_s = jnp.where(small, 1.0, a)
    c = jnp.sqrt(a_s / (1.0 + a_s))
    rho_m = jnp.minimum(1.0, h * (1.0 - c))
    theta_m = (rho_m - 1.0) / a_s
    theta_p = 1.0 / rho_m
    theta = jnp.maximum(theta_m, jnp.minimum(theta_p, (1.0 - b) / b))
    sigma = 1.0 + a_s * theta
    sig = jnp.abs(sigma) ** (1.0 / (1 - n)) if n > 1 else one
    safe_theta = jnp.where(theta > 0, theta, 1.0)
    tau = jnp.where(
        theta <= 0,
        jnp.minimum(rho_p * sig, sigma),
        rho_p * jnp.minimum(sig, 1.0 / safe_theta),
    )
    phi = (1.0 - theta) / (1.0 + a_s * theta)
    # a == 0: H is exact along y; theta = 0 gives tau = rho_p, phi = 1.
    return jnp.where(small, rho_p, tau), jnp.where(small, 1.0, phi)


class _LSSBroydenState(eqx.Module, Generic[Y]):
    """Ring buffer + compact representation.

    - `count`: number of accepted updates so far; slot of the next update is
      `count % history_length`.
    - `y_diff_history`, `grad_diff_history`: batched pytrees (s_i and y_i).
    - `gram`: Gram matrix of [S; Y] in *physical* (slot) layout, shape (2m, 2m).
    - `tau_history`, `phi_history`: scalars of each stored update.
    - `gamma`: H0 = gamma * I of the current window.
    - `A`, `c`: compact representation in physical layout (unused slots are zero).
    """

    count: Int[Array, ""]
    history_length: int = eqx.field(static=True)
    y_diff_history: PyTree[Y]
    grad_diff_history: PyTree[Y]
    gram: Float[Array, "twom twom"]
    tau_history: Float[Array, " hist"]
    phi_history: Float[Array, " hist"]
    gamma: Scalar
    A: Float[Array, "twom twom"]
    c: Scalar


def _inverse_hessian_fn(v: Y, state: _LSSBroydenState) -> Y:
    m = state.history_length
    coeff = jnp.concatenate(
        [_v_tree_dot(state.y_diff_history, v), _v_tree_dot(state.grad_diff_history, v)]
    )
    w = state.A @ coeff
    comb_s = ω(state.y_diff_history).call(lambda x: jnp.einsum("h,h...->...", w[:m], x))
    comb_y = ω(state.grad_diff_history).call(
        lambda x: jnp.einsum("h,h...->...", w[m:], x)
    )
    return ((state.c * state.gamma) * v**ω + comb_s + comb_y).ω


def _rebuild(gram, taus, phis, gamma, count, m):
    """Replay the window of updates in Gram space (physical layout)."""
    k = jnp.minimum(count, m)
    start = jnp.where(count < m, 0, count % m)
    dtype = gram.dtype
    eye = jnp.eye(2 * m, dtype=dtype)

    def body(i, carry):
        A, c = carry
        slot = (start + i) % m
        valid = i < k
        Gy = gram[:, m + slot]
        a_u = A @ Gy + (c * gamma) * eye[m + slot]  # u = H_i y_i in basis P
        yu = Gy @ a_u  # y^T H_i y
        sy = gram[slot, m + slot]
        yu_s = jnp.where(valid & (yu > 0), yu, 1.0)
        sy_s = jnp.where(valid & (sy > 0), sy, 1.0)
        tau = jnp.where(valid, taus[slot], 1.0)
        phi = phis[slot]
        # (A - u u^T/yu + phi yu w w^T)/tau + e e^T/sy with w = e/sy - u/yu, written as one
        # rank-1 term in u plus a row/column correction at `slot` (fewer full-size ops)
        r = -(phi / sy_s) * a_u
        A_new = A + ((phi - 1.0) / yu_s) * jnp.outer(a_u, a_u)
        A_new = A_new.at[slot, :].add(r).at[:, slot].add(r)
        A_new = A_new.at[slot, slot].add(phi * yu_s / sy_s**2)
        A_new = (A_new / tau).at[slot, slot].add(1.0 / sy_s)
        A = jnp.where(valid, A_new, A)
        c = jnp.where(valid, c / tau, c)
        return A, c

    A, c = jax.lax.fori_loop(
        0, m, body, (jnp.zeros((2 * m, 2 * m), dtype), jnp.ones((), dtype))
    )
    return 0.5 * (A + A.T), c


def _batched_zeros_like(y, m):
    return jtu.tree_map(lambda x: jnp.zeros((m, *x.shape), x.dtype), y)


class AbstractLSSBroyden(
    AbstractQuasiNewton[Y, Aux, FunctionInfo.EvalGradHessianInv, _LSSBroydenState]
):
    """Abstract limited-memory self-scaled Broyden solver (inverse form only)."""

    history_length: AbstractVar[int]
    update: AbstractVar[str]
    h0: AbstractVar[str]

    def _operator(self, y, state):
        return lx.FunctionLinearOperator(
            lambda v: _inverse_hessian_fn(v, state),
            jax.eval_shape(lambda: y),
            tags=lx.positive_semidefinite_tag,
        )

    def init_hessian(self, y: Y, f: Scalar, grad: Y):
        if not self.use_inverse:
            raise ValueError("LSSBroyden only supports `use_inverse=True`.")
        m = self.history_length
        dtype = jnp.result_type(*jtu.tree_leaves(y))
        state = _LSSBroydenState(
            count=jnp.array(0),
            history_length=m,
            y_diff_history=_batched_zeros_like(y, m),
            grad_diff_history=_batched_zeros_like(y, m),
            gram=jnp.zeros((2 * m, 2 * m), dtype),
            tau_history=jnp.ones(m, dtype),
            phi_history=jnp.ones(m, dtype),
            gamma=jnp.asarray(self.h0 if _is_number(self.h0) else 1.0, dtype),
            A=jnp.zeros((2 * m, 2 * m), dtype),
            c=jnp.ones((), dtype),
        )
        f_info = FunctionInfo.EvalGradHessianInv(f, grad, self._operator(y, state))
        return f_info, state

    def update_hessian(
        self,
        y: Y,
        y_eval: Y,
        f_info: FunctionInfo.EvalGradHessianInv,
        f_eval_info: FunctionInfo.EvalGrad,
        hessian_update_state: _LSSBroydenState,
        step_size,
    ):
        del step_size  # alpha is recovered from the step geometry instead
        m = self.history_length
        n = sum(x.size for x in jtu.tree_leaves(y))
        operator = f_info.hessian_inv
        g_old = f_info.grad
        s = (y_eval**ω - y**ω).ω
        yv = (f_eval_info.grad**ω - g_old**ω).ω
        sy = tree_dot(s, yv)
        ss, yy = tree_dot(s, s), tree_dot(yv, yv)
        eps = jnp.finfo(jnp.result_type(sy)).eps
        do_update = sy > eps * jnp.sqrt(ss * yy)

        def no_update(args):
            return eqx.filter(operator, eqx.is_array), hessian_update_state

        def update(args):
            state = hessian_update_state
            sy_s = jnp.where(do_update, sy, 1.0)
            Hy = operator.mv(yv)
            p = (-(operator.mv(g_old) ** ω)).ω  # search direction that produced s
            pp = tree_dot(p, p)
            alpha = tree_dot(p, s) / jnp.where(pp > 0, pp, 1.0)
            b = -alpha * tree_dot(s, g_old) / sy_s
            h = tree_dot(yv, Hy) / sy_s
            tau, phi = ss_tau_phi(b, h, n, self.update)

            slot = state.count % m
            S = jtu.tree_map(lambda X, z: X.at[slot].set(z), state.y_diff_history, s)
            Yh = jtu.tree_map(
                lambda X, z: X.at[slot].set(z), state.grad_diff_history, yv
            )
            rs = jnp.concatenate([_v_tree_dot(S, s), _v_tree_dot(Yh, s)])
            ry = jnp.concatenate([_v_tree_dot(S, yv), _v_tree_dot(Yh, yv)])
            gram = state.gram.at[slot, :].set(rs).at[:, slot].set(rs)
            gram = gram.at[m + slot, :].set(ry).at[:, m + slot].set(ry)
            taus = state.tau_history.at[slot].set(tau)
            phis = state.phi_history.at[slot].set(jnp.maximum(phi, 0.0))
            if self.h0 == "latest":
                gamma = sy_s / jnp.where(yy > 0, yy, 1.0)
            elif self.h0 == "bb1":          # experimental: s^T s / s^T y
                gamma = ss / sy_s
            else:
                gamma = state.gamma
            count = state.count + 1
            A, c = _rebuild(gram, taus, phis, gamma, count, m)
            new_state = _LSSBroydenState(
                count=count,
                history_length=m,
                y_diff_history=S,
                grad_diff_history=Yh,
                gram=gram,
                tau_history=taus,
                phi_history=phis,
                gamma=gamma,
                A=A,
                c=c,
            )
            new_op = self._operator(s, new_state)
            return eqx.filter(new_op, eqx.is_array), new_state

        static_operator = eqx.filter(operator, eqx.is_array, inverse=True)
        dyn_op, new_state = filter_cond(do_update, update, no_update, None)
        hessian_inv = eqx.combine(static_operator, dyn_op)
        new_f_info = FunctionInfo.EvalGradHessianInv(
            f_eval_info.f, f_eval_info.grad, hessian_inv
        )
        return new_f_info, new_state


class LSSBroyden(AbstractLSSBroyden[Y, Aux]):
    """Limited-memory self-scaled Broyden (L-SSBroyden) minimiser.

    Same update rule as [`optimistix.SSBroyden`][] (Kiyani et al., CMAME 2025), but
    the inverse Hessian is kept in a compact limited-memory form, so memory is
    O(history_length * n) instead of O(n^2).

    Supports the following `options`:

    - `autodiff_mode`: `"bwd"` (default) or `"fwd"`.
    """

    rtol: float
    atol: float
    norm: Callable[[PyTree], Scalar]
    use_inverse: bool
    descent: NewtonDescent
    search: AbstractSearch
    history_length: int
    update: str
    h0: str | float
    verbose: Callable[..., None]

    def __init__(
        self,
        rtol: float,
        atol: float,
        norm: Callable[[PyTree], Scalar] = max_norm,
        history_length: int = 20,
        update: str = "ssbroyden",
        h0: str | float = "latest",
        search: AbstractSearch | None = None,
        verbose: bool | Callable[..., None] = False,
    ):
        if update not in ("ssbroyden", "ssbfgs", "bfgs"):
            raise ValueError("update must be 'ssbroyden', 'ssbfgs' or 'bfgs'")
        if _is_number(h0):
            if not h0 > 0:
                raise ValueError("a numeric h0 (fixed gamma) must be positive")
            h0 = float(h0)
        elif h0 not in ("identity", "latest", "bb1"):
            raise ValueError("h0 must be 'identity', 'latest', 'bb1' or a positive number")
        self.rtol = rtol
        self.atol = atol
        self.norm = norm
        self.use_inverse = True
        self.descent = NewtonDescent()
        self.search = Zoom(initial_guess_strategy="one") if search is None else search
        self.history_length = history_length
        self.update = update
        self.h0 = h0
        self.verbose = default_verbose(verbose)


LSSBroyden.__init__.__doc__ = """**Arguments:**

- `rtol`, `atol`: tolerances for the Cauchy termination criterion.
- `norm`: norm used in the termination criterion.
- `history_length`: number of (s, y) pairs kept (m).
- `update`: `"ssbroyden"` (default), `"ssbfgs"` or `"bfgs"` (`"bfgs"` gives L-BFGS).
- `h0`: initial matrix of the window. `"latest"` (default; gamma = s^T y / y^T y of
    the newest pair, as in standard L-BFGS), `"identity"` (H0 = I, the self-scaling
    factors alone set the scale; reproduces dense SSBroyden when nothing is dropped,
    but can stall with short histories), `"bb1"` (experimental; gamma = s^T s / s^T y)
    or a positive number (experimental; fixed gamma).
- `search`: line search. Defaults to `optimistix.Zoom(initial_guess_strategy="one")`
    (strong Wolfe, c1 = 1e-4, c2 = 0.9), as in the paper.
- `verbose`: as for other Optimistix minimisers.
"""
