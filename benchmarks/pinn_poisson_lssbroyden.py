"""2D Poisson PINN: -Δu = 2π² sin(πx) sin(πy) on [0,1]², u = 0 on ∂Ω.

Adam warm-up (optax) -> each quasi-Newton solver from the same weights, compared at
an equal budget of function evaluations (= optimistix steps). float64.

    python benchmarks/pinn_poisson_lssbroyden.py --evals 3000
"""

import argparse
import json
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import optimistix as optx


jax.config.update("jax_enable_x64", True)
pi = jnp.pi


def make_data(n_int, n_bc, key):
    k1, k2, k3 = jax.random.split(key, 3)
    xi = jax.random.uniform(k1, (n_int, 2))
    t = jax.random.uniform(k2, (n_bc,))
    side = jax.random.randint(k3, (n_bc,), 0, 4)
    xb = jnp.stack(
        [
            jnp.where(side == 0, 0.0, jnp.where(side == 1, 1.0, t)),
            jnp.where(side == 2, 0.0, jnp.where(side == 3, 1.0, t)),
        ],
        1,
    )
    g = jnp.linspace(0, 1, 101)
    xt = jnp.stack(jnp.meshgrid(g, g, indexing="ij"), -1).reshape(-1, 2)
    return xi, xb, xt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evals", type=int, default=3000)
    ap.add_argument("--adam", type=int, default=1000)
    ap.add_argument("--width", type=int, default=40)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--no_dense", action="store_true")
    ap.add_argument("--out", default="pinn_poisson_lssbroyden.json")
    a = ap.parse_args()

    key = jax.random.PRNGKey(0)
    net = eqx.nn.MLP(2, "scalar", a.width, a.depth, activation=jnp.tanh, key=key)
    net = jax.tree_util.tree_map(lambda x: x.astype(jnp.float64) if eqx.is_array(x) else x, net)
    params0, static = eqx.partition(net, eqx.is_array)
    xi, xb, xt = make_data(2000, 400, jax.random.PRNGKey(1))
    n = sum(x.size for x in jax.tree_util.tree_leaves(params0))
    print(f"params n = {n}")

    def u_fn(p, x):
        return eqx.combine(p, static)(x)

    def loss(p, _=None):
        def lap(x):
            return jnp.trace(jax.hessian(lambda z: u_fn(p, z))(x))

        f = 2 * pi**2 * jnp.sin(pi * xi[:, 0]) * jnp.sin(pi * xi[:, 1])
        r = -jax.vmap(lap)(xi) - f
        return jnp.mean(r**2) + jnp.mean(jax.vmap(lambda x: u_fn(p, x))(xb) ** 2)

    def rel_l2(p):
        ue = jnp.sin(pi * xt[:, 0]) * jnp.sin(pi * xt[:, 1])
        up = jax.vmap(lambda x: u_fn(p, x))(xt)
        return float(jnp.linalg.norm(up - ue) / jnp.linalg.norm(ue))

    opt = optax.adam(1e-3)
    st = opt.init(params0)

    @jax.jit
    def adam_step(p, st):
        l, g = jax.value_and_grad(loss)(p)
        upd, st = opt.update(g, st)
        return optax.apply_updates(p, upd), st, l

    p = params0
    for _ in range(a.adam):
        p, st, l = adam_step(p, st)
    print(f"after Adam: loss={float(l):.3e} relL2={rel_l2(p):.3e}")
    p_warm = p

    tol = 1e-14
    configs = [
        ("optx.LBFGS m=20 (backtracking)", optx.LBFGS(tol, tol, history_length=20)),
        ("L-BFGS m=20 (zoom)", optx.LSSBroyden(tol, tol, history_length=20, update="bfgs", h0="latest")),
        ("L-SSBroyden m=20", optx.LSSBroyden(tol, tol, history_length=20, h0="latest")),
        ("L-SSBroyden m=100", optx.LSSBroyden(tol, tol, history_length=100, h0="latest")),
        ("L-SSBroyden m=100 H0=I", optx.LSSBroyden(tol, tol, history_length=100, h0="identity")),
    ]
    if not a.no_dense:
        configs += [
            ("SSBFGS (fork, dense, zoom)", optx.SSBFGS(tol, tol)),
            ("SSBroyden (fork, dense, backtracking)", optx.SSBroyden(tol, tol)),
        ]

    results = {}
    for name, solver in configs:
        t0 = time.time()
        sol = optx.minimise(loss, solver, p_warm, max_steps=a.evals, throw=False)
        jax.block_until_ready(sol.value)
        dt = time.time() - t0
        l = float(loss(sol.value))
        e = rel_l2(sol.value)
        acc = int(sol.state.num_accepted_steps)
        results[name] = dict(loss=l, relL2=e, time=dt, accepted=acc)
        print(f"{name:40s} loss={l:.3e} relL2={e:.3e} accepted={acc:5d} time={dt:6.1f}s", flush=True)

    json.dump(dict(n=n, args=vars(a), results=results), open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
