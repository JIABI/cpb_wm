# Exp 2 — DreamerV3 Baseline (Brief B Phase A)

Trains a DreamerV3 world-model agent on stochastic `dm_control` environments
and integrates it with the NMS certified-stochasticity framework.

## Phases

| Phase | Goal | Status |
|-------|------|--------|
| A | Baseline training (σ=0), verify pipeline | ✅ implemented |
| B | Stochastic wrappers + bandwidth estimation | 🔧 wm_violation.py / cpb_dreamer.py (TODO) |
| C | 5 envs × 3 noise types × 45 settings | ⏳ after Phase B |
| D | OOD bandwidth collapse hero | ⏳ after Phase C |

## Quick start

```bash
# Install extras (gymnasium, dm_control, shimmy, scikit-learn)
pip install -e ".[exp2]"

# Install DreamerV3 (not on standard PyPI)
pip install dreamerv3-torch
# Or from source: pip install git+https://github.com/zhaoyi11/dreamerv3-torch

# Phase A smoke run (~5 min on CPU)
python -m experiments.exp2_dreamer.train_baseline \
    env.suite=cheetah env.task=run \
    training.total_steps=10000 training.eval_every=5000

# Full run (Hydra config in configs/exp2_rl/config.yaml)
python -m experiments.exp2_dreamer.train_baseline
```

## Architecture

```
env (dm_control via shimmy)
  └─ StochasticDynamicsWrapper(σ_obs, σ_reward, σ_dynamics)
       └─ DreamerNMSWrapper
            ├─ KMeansActionDiscretizer  ← NO raw continuous actions
            ├─ DreamerV3 (dreamerv3-torch)
            │    └─ RSSM (K=3 ensemble members)
            └─ estimate_violation_rate()  ← Phase B hook
```

## Key constraints

- **No raw continuous actions** — all actions go through `KMeansActionDiscretizer`.
- **No NMSOperator interface changes** — wrapper is a separate class.
- **No conformal calibration here** — calibration is done via `core/certificate.py`.
- **K=3 ensemble** for Phase A/B; scalable to K=5 for Phase C.
- **RNG** uses `np.random.default_rng(seed)`, never `numpy.random.seed`.

## Stub mode

If `dreamerv3-torch` is not installed, `DreamerNMSWrapper` operates in stub mode:
- `act()` returns a random action (decoded through k-means centroids)
- `imagine()` returns copies of the current latent state
- A `UserWarning` is issued at construction time

This lets CI run without the heavy dreamerv3 dependency.

## Files

| File | Purpose |
|------|---------|
| `train_baseline.py` | Hydra training entry point |
| `../../src/nms/wm/dreamer_wrapper.py` | `DreamerNMSWrapper` + `KMeansActionDiscretizer` |
| `../../src/nms/envs/stochastic_dm_control.py` | Four stochastic env wrappers |
| `../../configs/exp2_rl/config.yaml` | Hydra config (env, noise, dreamer, training) |
| `../../tests/test_wm/test_dreamer_wrapper.py` | Unit tests (no dreamerv3 needed) |
| `../../tests/test_envs/test_stochastic_dm_control.py` | Unit tests (no gymnasium needed) |
