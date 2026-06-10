# NMS — Noise-Matched Stochasticity

Research scaffold for the NMS paper.

## Quick start

```bash
# Install in editable mode with dev + experiment 1 deps (required before running tests):
pip install -e ".[dev,exp1]"

# If you prefer not to install, set PYTHONPATH instead:
PYTHONPATH=src python -m pytest
```

## Optional extras

| Extra | Contents |
|-------|----------|
| `.[exp1]` | No extra deps (pure numpy/scipy). |
| `.[exp2]` | RL/control: gymnasium, dm_control, ale-py. |
| `.[exp3]` | Multi-agent: pettingzoo + dm-meltingpot (from GitHub — see note below). |
| `.[exp4]` | LLM: transformers, accelerate, vllm (GPU optional). |
| `.[dev]` | pytest, ruff, mypy, pre-commit, jupyterlab. |

> **dm-meltingpot (Exp 3)** is installed from source:
> `pip install "git+https://github.com/google-deepmind/meltingpot.git"`

## Running Exp 1 (noisy IPD)

```bash
# Smoke run (~30 s on CPU):
python -m experiments.exp1_noisy_games.run_bandwidth \
  --noise-levels 0.0 0.1 0.2 \
  --sigma-min 0.0 --sigma-max 0.8 --sigma-n 11 \
  --horizon 50 --episodes 200 --seeds 3 \
  --out results/runs/exp1_noisy_games/smoke.csv

# Full sweep (~1 h on CPU with --seeds 10):
python -m experiments.exp1_noisy_games.run_bandwidth \
  --noise-levels 0.0 0.05 0.1 0.15 0.2 0.3 \
  --sigma-min 0.0 --sigma-max 0.8 --sigma-n 41 \
  --horizon 200 --episodes 2000 --seeds 10 \
  --out results/runs/exp1_noisy_games/bandwidth_ipd.csv

# Plot (Fig 1 + Fig 2):
python -m experiments.exp1_noisy_games.plot_bandwidth \
  --csv results/runs/exp1_noisy_games/bandwidth_ipd.csv \
  --out-fig1 results/figures/exp1_bandwidth_ipd.pdf \
  --out-fig2 results/figures/exp1_lemma1_proj.pdf \
  --horizon 200 --alpha 0.1 \
  --train-nus 0.05 0.15 0.30 --test-nus 0.10 0.20 \
  --response-form best
```

## CLI demo

```bash
python -m nms.cli demo
```

## Quality checks

```bash
ruff format src tests
ruff check src tests
mypy --strict src/nms/core
pytest -q
```
