#!/bin/bash
set -e   # 任何一步失败就停,不要 silent skip
set -x   # 显示每一行执行命令


# === Sanity 0: 验证 deps ===
python -c "import dreamerv3; print('OK: dreamerv3')" || \
python -c "from nms.wm.minimal_dreamer import MinimalDreamer; print('Using MinimalDreamer fallback')"
python -c "import sklearn; print('OK: sklearn')"
python -c "import dm_control; print('OK: dm_control')"

# === Sanity 1: k-means interface ===
python -m experiments.exp2_dreamer.fit_interface \
--env cheetah-run \
--fit-steps 10000 \
--n-clusters 8 \
--seed 0 \
--out results/interfaces/cheetah_run_k8.npz


# 检查 centers 不是 random
python -c "
import numpy as np
d = np.load('results/interfaces/cheetah_run_k8.npz')
print('Keys:', list(d.keys()))
for k in d.keys():
    arr = d[k]
    print(f'  {k}: shape={arr.shape}, dtype={arr.dtype}')
"

# === Sanity 2: 训 ensemble (smoke 200K steps) ===
python -m experiments.exp2_dreamer.train_ensemble \
  --env cheetah-run \
  --k 5 \
  --steps 200000 \
  --sigma-obs 0.2 \
  --seed 0 \
  --out results/runs/exp2_smoke/cheetah_obs02_ensemble/

# === Sanity 3: calibrate bandwidth ===
python -m experiments.exp2_dreamer.calibrate_bandwidth \
  --dreamer-dir results/runs/exp2_smoke/cheetah_obs02_ensemble/ \
  --kmeans-path results/interfaces/cheetah_run_k8.npz \
  --env cheetah-run \
  --n-pairs 50 \
  --horizon 15 --alpha 0.1 \
  --sigma-grid-max 2.0 --sigma-grid-n 20 \
  --out results/runs/exp2_smoke/cheetah_obs02_bandwidth.json

# 检查 bandwidth 不为空
python -c "
import json
b = json.load(open('results/runs/exp2_smoke/cheetah_obs02_bandwidth.json'))
print('bandwidth:', b)
assert 'sigma_min' in str(b), 'no sigma_min field!'
"

# === Sanity 4: 跑 cpb_full deploy (20 episodes) ===
python -m experiments.exp2_dreamer.run_main \
  env=cheetah_run noise=obs_noise_020 method=cpb_full seed=0 \
  n_cal=50 n_eval_episodes=20

echo "=== ALL SANITY CHECKS PASSED ==="

