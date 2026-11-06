# LSSBroyden: limited-memory self-scaled Broyden

`optx.LSSBroyden` is a limited-memory version of `optx.SSBroyden`. It uses the same
update rule (Kiyani et al., CMAME 446 (2025) 118308, Sec. 1.3, Eqs. (4)–(8)), but keeps
the inverse Hessian in compact form, so the state uses O(m·n) memory instead of O(n²).

```python
import jax
jax.config.update("jax_enable_x64", True)   # recommended
import optimistix as optx

solver = optx.LSSBroyden(rtol=1e-12, atol=1e-12, history_length=50)
sol = optx.minimise(loss_fn, solver, params, max_steps=20_000, throw=False)
```

| argument | default | meaning |
|---|---|---|
| `history_length` | 20 | number of stored (s, y) pairs, m |
| `update` | `"ssbroyden"` | `"ssbroyden"`, `"ssbfgs"`, or `"bfgs"` (`"bfgs"` gives L-BFGS) |
| `h0` | `"latest"` | `"latest"`: γ = sᵀy/yᵀy (standard L-BFGS choice). `"identity"`: H0 = I. `"bb1"`: γ = sᵀs/sᵀy (experimental). A positive number: fixed γ (experimental) |
| `search` | `Zoom(initial_guess_strategy="one")` | strong Wolfe, c1 = 1e-4, c2 = 0.9 (as in the paper) |

Tested with jax 0.4.38 / equinox 0.11.12 / lineax 0.0.8 and with jax 0.12 / equinox 0.13.8 /
lineax 0.1.1 / optax 0.2.8 (CPU). jax 0.4.38 has no CUDA build for aarch64 (e.g. GH200), so
GPU use there needs the newer versions.

The variable `params` can be any pytree, for example an Equinox model partitioned with
`eqx.partition`.

## How it works

- **Representation.** H = c·γ·I + P A Pᵀ, where P = [s₁…s_m, y₁…y_m].
  - After every accepted step, A (2m×2m) and c are rebuilt by replaying the m stored
    updates. The replay only needs the Gram matrix G = PᵀP and the stored (τᵢ, φᵢ),
    and costs O(m³).
  - Applying H costs two batched inner products/combinations with the history, i.e.
    O(m·n). This parallelises well on accelerators. If the history is sharded, the only
    communication needed is a reduction of a 2m-vector.
- **Computing b_k.** b_k = sᵀBs / sᵀy is computed as −α sᵀg / sᵀy (Eq. (5)). The step
  size α is recovered from the step itself: α = ⟨p, s⟩ / ⟨p, p⟩ with p = −H g.
  - This is needed because the `step_size` that Optimistix passes to `update_hessian`
    is the search's proposal for the *next* step. With `Zoom(initial_guess_strategy="one")`
    and `BacktrackingArmijo` that proposal is 1.0 after every accepted step, not the α
    that produced `y_eval`.
- **Truncation.** When the history is truncated, each stored pair keeps the τᵢ and φᵢ
  computed when it was added, and φᵢ is clipped at 0 so that H stays positive definite.

## Tests

`tests/test_lssbroyden.py` checks that, with `history_length >= #steps` and
`h0="identity"`, LSSBroyden reproduces a dense SSBroyden/SSBFGS/BFGS reference iterate
by iterate (relative difference < 1e-8). The check is run on Rosenbrock and on an
Equinox MLP given as a pytree.

## Benchmark (`benchmarks/pinn_poisson_lssbroyden.py`)

Setup: 2D Poisson PINN, MLP 2-40-40-40-1 (n = 3441), float64. 1000 Adam steps, then
3000 function evaluations per solver. CPU, single seed.

| solver | rel. L2 error | accepted steps | time |
|---|---|---|---|
| optx.LBFGS m=20 (backtracking) | 1.5e-03 | 2713 | 134 s |
| LSSBroyden update="bfgs" m=20 | 1.7e-03 | 2795 | 269 s |
| LSSBroyden m=20 | 1.3e-03 | 1827 | 236 s |
| LSSBroyden m=100 | 9.0e-05 | 1958 | 267 s |
| LSSBroyden m=100, h0="identity" | 2.0e-05 | 2033 | 275 s |
| dense SSBroyden, Zoom (reference of `tests/test_lssbroyden.py`) | 1.2e-06 | 1761 | 524 s |
| dense SSBFGS, Zoom (reference of `tests/test_lssbroyden.py`) | 2.5e-06 | 1902 | 530 s |

## Changes to the dense `SSBroyden` in this branch

- The `use_inverse=False` branch (direct update of B) now follows Kiyani et al. (2025):
  h_k is divided by sᵀy, and the σ^(1/(1−N)) term uses consistent exponents.
  `tests/test_ssbroyden_direct.py` checks it against the inverse form.
- `tests/test_ss_step_length.py` checks that the dense `optx.SSBFGS` / `optx.SSBroyden`
  reproduce the dense reference of `tests/test_lssbroyden.py` when run with the same
  line search.
- `SSBroyden` uses `BacktrackingArmijo` by default; the paper (and `SSBFGS`) use a Wolfe
  line search.

## Changing data: segment-wise training (mini-batches, adaptive points)

Quasi-Newton methods stall if the collocation points change at every iteration. Keep each
set of points for a segment of iterations and keep the curvature memory across segments:

```python
y, info = optx.minimise_segments(loss, optx.LSSBroyden(rtol=0.0, atol=0.0, history_length=100),
                                 params, sample_fn, key=key, n0=2000, max_steps=20_000,
                                 segment_steps=200, probe=False, theta=None,
                                 sampler=optx.RADSampler(sample_fn, residual_fn, keys=["pde"]))
# loss(params, batch) -> scalar; sample_fn(key, n) -> batch; residual_fn(params, batch) -> dict
```

Without `segment_steps`, the segment length and the batch size are set automatically from an
independent probe batch (as in `SegmentTrainer` of the PyTorch package
[torch-ssbroyden](https://github.com/Shiratori-Lab/torch-ssbroyden)). In a custom loop, call
`state = optx.new_batch(state, y)` right after switching `args` to a new batch: the next step
re-evaluates the loss at `y` on the new batch, accepts it, and forms no curvature pair.
Works with any quasi-Newton minimiser of Optimistix (LSSBroyden, SSBroyden, SSBFGS, BFGS).
