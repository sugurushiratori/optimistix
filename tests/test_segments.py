"""Segment-wise training with changing batches (optx.new_batch, optx.minimise_segments)."""
import jax
import jax.numpy as jnp
import optimistix as optx


def _init_mlp(key, width=16):
    k1, k2, k3 = jax.random.split(key, 3)
    return dict(
        W1=jax.random.normal(k1, (1, width)) * 1.0, b1=jnp.zeros((1, width)),
        W2=jax.random.normal(k2, (width, width)) / jnp.sqrt(width), b2=jnp.zeros((1, width)),
        W3=jax.random.normal(k3, (width, 1)) / jnp.sqrt(width), b3=jnp.zeros((1, 1)),
    )


def _mlp(p, x):
    h = jnp.tanh(x @ p["W1"] + p["b1"])
    h = jnp.tanh(h @ p["W2"] + p["b2"])
    return h @ p["W3"] + p["b3"]


def _sine_loss(p, xb):
    return jnp.mean((_mlp(p, xb) - jnp.sin(3 * xb)) ** 2)


def _sample(key, n):
    return jax.random.uniform(key, (n, 1), minval=-1.0, maxval=1.0)


XV = jnp.linspace(-1, 1, 1001)[:, None]


def test_new_batch_forms_no_pair():
    p = _init_mlp(jax.random.PRNGKey(0))
    solver = optx.LSSBroyden(rtol=0.0, atol=0.0, history_length=20)
    fn = lambda y, a: (_sine_loss(y, a), None)  # noqa: E731
    b1, b2 = _sample(jax.random.PRNGKey(1), 64), _sample(jax.random.PRNGKey(2), 64)
    f_struct = jax.eval_shape(lambda: _sine_loss(p, b1))
    state = solver.init(fn, p, b1, {}, f_struct, None, frozenset())
    y = p
    for _ in range(15):
        y, state, _ = solver.step(fn, y, b1, {}, state, frozenset())
    count = int(state.hessian_update_state.count)
    state = optx.new_batch(state, y)
    y2, state, _ = solver.step(fn, y, b2, {}, state, frozenset())
    assert int(state.hessian_update_state.count) == count          # no pair across batches
    assert jax.tree_util.tree_all(jax.tree_util.tree_map(jnp.array_equal, y2, y))
    assert jnp.allclose(state.f_info.f, _sine_loss(y, b2))          # re-evaluated on b2
    for _ in range(5):
        y2, state, _ = solver.step(fn, y2, b2, {}, state, frozenset())
    assert int(state.hessian_update_state.count) > count


def test_minimise_segments_auto():
    p = _init_mlp(jax.random.PRNGKey(0))
    f0 = _sine_loss(p, XV)
    y, info = optx.minimise_segments(
        _sine_loss, optx.LSSBroyden(rtol=0.0, atol=0.0, history_length=20), p, _sample,
        key=jax.random.PRNGKey(3), n0=16, n_max=512, max_steps=3000, m_min=50)
    assert _sine_loss(y, XV) < 1e-4 * f0
    assert len(info["segments"]) >= 2 and info["batch_size"] > 16


def test_minimise_segments_dense_ssbroyden():
    p = _init_mlp(jax.random.PRNGKey(0))
    f0 = _sine_loss(p, XV)
    y, info = optx.minimise_segments(
        _sine_loss, optx.SSBroyden(rtol=0.0, atol=0.0), p, _sample,
        key=jax.random.PRNGKey(4), n0=128, max_steps=1500, segment_steps=200, probe=False)
    assert _sine_loss(y, XV) < 1e-4 * f0 and len(info["segments"]) >= 5


def _peak(x):
    return jnp.exp(-(((x - 0.3) / 0.05) ** 2))


def _peak_sample(key, n):
    return {"int": jax.random.uniform(key, (n, 1), minval=-1.0, maxval=1.0),
            "end": jnp.array([[-1.0], [1.0]])}


def _peak_loss(p, b):
    return jnp.mean((_mlp(p, b["int"]) - _peak(b["int"])) ** 2) + jnp.mean(_mlp(p, b["end"]) ** 2)


def _peak_res(p, b):
    return {"int": (_mlp(p, b["int"]) - _peak(b["int"]))[:, 0]}


def test_rad_sampler_concentrates():
    p = jax.tree_util.tree_map(jnp.zeros_like, _init_mlp(jax.random.PRNGKey(0)))
    s = optx.RADSampler(_peak_sample, _peak_res, k=2.0, c=0.0, keys=["int"])
    b = s(jax.random.PRNGKey(0), 400, p, None)
    assert b["int"].shape == (400, 1) and b["end"].shape == (2, 1)
    assert jnp.mean(jnp.abs(b["int"] - 0.3) < 0.15) > 0.6


def test_rard_sampler_grows_and_trains():
    p = _init_mlp(jax.random.PRNGKey(0), width=20)
    s = optx.RARDSampler(_peak_sample, _peak_res, n_add=32, keys=["int"])
    y, info = optx.minimise_segments(
        _peak_loss, optx.LSSBroyden(rtol=0.0, atol=0.0, history_length=20), p, _peak_sample,
        key=jax.random.PRNGKey(5), n0=64, max_steps=1500, segment_steps=100, probe=False,
        theta=None, sampler=s)
    assert info["batch_size"] > 66
    xv = jnp.linspace(-1, 1, 2001)[:, None]
    assert jnp.mean((_mlp(y, xv) - _peak(xv)) ** 2) < 1e-3


def test_lssbroyden_h0_options():
    p = _init_mlp(jax.random.PRNGKey(0))
    b = _sample(jax.random.PRNGKey(1), 128)
    for h0 in ("latest", "bb1", 10.0):
        sol = optx.minimise(_sine_loss, optx.LSSBroyden(rtol=0.0, atol=0.0, history_length=10,
                                                        h0=h0),
                            p, b, max_steps=200, throw=False)
        assert _sine_loss(sol.value, b) < 1e-2 * _sine_loss(p, b)
