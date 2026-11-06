"""Segment-wise training with changing data (mini-batches, adaptive residual points).

Quasi-Newton methods need a consistent objective over many iterations: resampling the
collocation points at every iteration stalls them.  The segment method keeps every set of
points for a *segment* of iterations and keeps the curvature memory across segments; at a
change of points the pair that would connect the two objectives is not formed.  This is the
JAX counterpart of ``torch_ssbroyden.SegmentTrainer`` (PyTorch package torch-ssbroyden).

Low level (any quasi-Newton minimiser of Optimistix: LSSBroyden, SSBroyden, BFGS, LBFGS)::

    state = optx.new_batch(state, y)        # after switching `args` to a new batch

High level::

    y, info = optx.minimise_segments(loss, optx.LSSBroyden(rtol=0, atol=0), params,
                                     sample_fn, key=key, n0=4096, n_max=65536,
                                     max_steps=20000)
    # loss(params, batch) -> scalar;  sample_fn(key, n) -> batch (pytree of arrays (n_i, d))

Samplers (``sampler=``): `UniformSampler` (default), `RADSampler`, `RARDSampler`; any
callable ``sampler(key, n, y, batch) -> batch`` works.  A batch whose shapes change triggers a
recompilation of the segment loop (RAR-type samplers recompile once per segment).
"""
from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu

__all__ = ["new_batch", "minimise_segments", "UniformSampler", "RADSampler", "RARDSampler",
           "batch_size"]


def new_batch(state, y):
    """Prepare a quasi-Newton state for a new batch (new `args`).

    The pending trial point is dropped and the next step is treated like a first step: it
    evaluates the loss and gradient at the current accepted point `y` on the new batch,
    accepts it without a line-search test, and forms no curvature pair (s = 0).  The
    curvature memory is kept.
    """
    return eqx.tree_at(lambda s: (s.first_step, s.y_eval), state,
                       (jnp.array(True), y))


def batch_size(batch) -> int:
    return int(sum(x.shape[0] for x in jtu.tree_leaves(batch)))


# ---------------------------------------------------------------------------------------
# samplers
# ---------------------------------------------------------------------------------------
def _as_dict(batch):
    if isinstance(batch, dict):
        return dict(batch), dict
    return {None: batch}, None


def _from_dict(d, kind):
    return d if kind is dict else d[None]


def _density(r, k, c):
    r = jnp.abs(r) if r.ndim == 1 else jnp.linalg.norm(r.reshape(r.shape[0], -1), axis=1)
    e = r ** k
    w = e / jnp.maximum(e.mean(), 1e-300) + c
    w = jnp.where(jnp.isfinite(w), w, 0.0) + 1e-300
    return w / w.sum()


class UniformSampler:
    """Fresh points ``sample_fn(key, n)`` for every segment."""
    respects_n = True

    def __init__(self, sample_fn):
        self.sample_fn = sample_fn
        self.last_cost = 0

    def __call__(self, key, n, y, batch):
        self.last_cost = 0
        return self.sample_fn(key, n)


class RADSampler:
    """Residual-based adaptive distribution, p(x) ~ |r|^k / mean|r|^k + c (Wu et al. 2023).

    ``residual_fn(y, batch)`` returns per-point residuals (a dict with the adapted keys of a
    dict batch, or an array for an array batch).
    """
    respects_n = True

    def __init__(self, sample_fn, residual_fn, k=1.0, c=1.0, factor=10,
                 keys: Sequence | None = None):
        self.sample_fn, self.residual_fn = sample_fn, residual_fn
        self.k, self.c, self.factor, self.keys = k, c, factor, keys
        self.last_cost = 0
        self._res = jax.jit(residual_fn)

    def __call__(self, key, n, y, batch):
        k1, k2, k3 = jax.random.split(key, 3)
        template, kind = _as_dict(self.sample_fn(k1, n))
        cand, _ = _as_dict(self.sample_fn(k2, self.factor * n))
        r, _ = _as_dict(self._res(y, _from_dict(cand, kind)))
        self.last_cost = 0
        for i, (name, x) in enumerate(template.items()):
            if (self.keys is not None and name not in self.keys) or name not in r:
                continue
            p = _density(r[name], self.k, self.c)
            idx = jax.random.choice(jax.random.fold_in(k3, i), cand[name].shape[0],
                                    (x.shape[0],), replace=False, p=p)
            template[name] = cand[name][idx]
            self.last_cost += int(cand[name].shape[0])
        return _from_dict(template, kind)


class RARDSampler:
    """Residual-based adaptive refinement: keep the points, add ``n_add`` new ones per
    segment (fraction of the current size if < 1), drawn with p ~ |r|^k / mean|r|^k + c
    (RAR-D, k = 2, c = 0) or the largest residuals if ``greedy`` (RAR)."""
    respects_n = False

    def __init__(self, sample_fn, residual_fn, n_add=0.1, k=2.0, c=0.0, factor=10,
                 greedy=False, keys: Sequence | None = None, n_max: int | None = None):
        self.sample_fn, self.residual_fn = sample_fn, residual_fn
        self.n_add, self.k, self.c, self.factor = n_add, k, c, factor
        self.greedy, self.keys, self.n_max = greedy, keys, n_max
        self.last_cost = 0
        self._res = jax.jit(residual_fn)

    def __call__(self, key, n, y, batch):
        self.last_cost = 0
        if batch is None:
            return self.sample_fn(key, n)
        cur, kind = _as_dict(batch)
        adds = {}
        for name, x in cur.items():
            if self.keys is not None and name not in self.keys:
                continue
            m = int(self.n_add if self.n_add >= 1 else round(self.n_add * x.shape[0]))
            if self.n_max is not None:
                m = min(m, self.n_max - x.shape[0])
            if m > 0:
                adds[name] = m
        if not adds:
            return batch
        k1, k2 = jax.random.split(key)
        cand, _ = _as_dict(self.sample_fn(k1, self.factor * max(adds.values())))
        r, _ = _as_dict(self._res(y, _from_dict(cand, kind)))
        for i, (name, m) in enumerate(adds.items()):
            if name not in r:
                continue
            m = min(m, cand[name].shape[0])
            if self.greedy:
                mag = jnp.abs(r[name]) if r[name].ndim == 1 else jnp.linalg.norm(
                    r[name].reshape(r[name].shape[0], -1), axis=1)
                idx = jnp.argsort(-mag)[:m]
            else:
                idx = jax.random.choice(jax.random.fold_in(k2, i), cand[name].shape[0],
                                        (m,), replace=False,
                                        p=_density(r[name], self.k, self.c))
            cur[name] = jnp.concatenate([cur[name], cand[name][idx]])
            self.last_cost += int(cand[name].shape[0])
        return _from_dict(cur, kind)


# ---------------------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------------------
@eqx.filter_jit
def _run_chunk(solver, fn, y, state, batch, n_steps, options):
    dynamic, static = eqx.partition(state, eqx.is_array)

    def body(_, carry):
        y_, d_ = carry
        y_, s_, _ = solver.step(fn, y_, batch, options, eqx.combine(static, d_), frozenset())
        return y_, eqx.filter(s_, eqx.is_array)

    y, dynamic = jax.lax.fori_loop(0, n_steps, body, (y, dynamic))
    return y, eqx.combine(static, dynamic)


@eqx.filter_jit
def _loss(fn, y, batch):
    return fn(y, batch)[0]


def minimise_segments(
    fn: Callable[[Any, Any], Any],
    solver,
    y0,
    sample_fn: Callable[[jax.Array, int], Any],
    *,
    key: jax.Array,
    n0: int,
    max_steps: int,
    n_max: int | None = None,
    steps_per_probe: int = 10,
    m_min: int = 200,
    theta: float | None = 0.5,
    patience: int = 3,
    rel_tol: float = 0.01,
    probe_fraction: float = 0.5,
    segment_steps: int | None = None,
    probe: bool = True,
    sampler: Callable | None = None,
    options: dict | None = None,
    callback: Callable | None = None,
):
    """Segment-wise minimisation of ``fn(y, batch)`` with changing batches.

    Every ``steps_per_probe`` steps the loss on an independent uniform probe batch is
    evaluated.  A segment ends when the probe loss has not improved by ``rel_tol`` over
    ``patience`` probes (but not before ``m_min`` steps), or after exactly ``segment_steps``
    steps if given.  At the end of a segment the batch size is doubled (up to ``n_max``) if
    the relative gap (f_probe - f_batch) / f_batch exceeds ``theta`` (``None``: never).

    Returns ``(y, info)``; ``info`` has the segment records, the number of steps (= loss and
    gradient evaluations) and the point evaluations.  ``callback(info, y)`` runs after
    every chunk of ``steps_per_probe`` steps; return True to stop.
    """
    if not probe and segment_steps is None:
        raise ValueError("probe=False needs a fixed segment_steps")
    options = {} if options is None else options
    n_max = n0 if n_max is None else n_max
    sampler = UniformSampler(sample_fn) if sampler is None else sampler

    def wrapped(y, batch):
        return fn(y, batch), None

    n = n0
    key, k = jax.random.split(key)
    batch = sampler(k, n, y0, None)
    f_struct = jax.eval_shape(lambda: fn(y0, batch))
    state = solver.init(wrapped, y0, batch, options, f_struct, None, frozenset())
    y = y0
    info = dict(segments=[], steps=0, point_evals=0, sampler_point_evals=getattr(sampler, "last_cost", 0))

    def new_probe(key, b):
        return sample_fn(key, max(1, int(batch_size(b) * probe_fraction)))

    key, k = jax.random.split(key)
    probe_batch = new_probe(k, batch) if probe else None
    it_in_seg, hist = 0, []
    while info["steps"] < max_steps:
        n_steps = min(steps_per_probe, max_steps - info["steps"])
        y, state = _run_chunk(solver, wrapped, y, state, batch, n_steps, options)
        info["steps"] += n_steps
        info["point_evals"] += n_steps * batch_size(batch)
        it_in_seg += n_steps
        f_batch = float(_loss(wrapped, y, batch))
        hist.append(float(_loss(wrapped, y, probe_batch)) if probe else math.nan)
        info["f_batch"], info["f_probe"], info["batch_size"] = f_batch, hist[-1], batch_size(batch)
        if callback is not None and callback(info, y):
            break
        stalled = (len(hist) > patience and
                   min(hist[-patience:]) > (1 - rel_tol) * min(hist[:-patience]))
        end = (it_in_seg >= segment_steps) if segment_steps is not None else (
            it_in_seg >= m_min and stalled)
        if end:
            gap = (hist[-1] - f_batch) / max(abs(f_batch), 1e-300)
            info["segments"].append(dict(steps=it_in_seg, batch_size=batch_size(batch),
                                         f_batch=f_batch, f_probe=hist[-1], gap=gap))
            if (theta is not None and getattr(sampler, "respects_n", True)
                    and math.isfinite(gap) and gap > theta and n < n_max):
                n = min(2 * n, n_max)
            key, k1, k2 = jax.random.split(key, 3)
            batch = sampler(k1, n, y, batch)
            info["sampler_point_evals"] += getattr(sampler, "last_cost", 0)
            if probe:
                probe_batch = new_probe(k2, batch)
            state = new_batch(state, y)
            it_in_seg, hist = 0, []
    return y, info
