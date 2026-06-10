"""Random Network Distillation (RND) exploration baseline (Phase C).

Adds RND intrinsic reward to any policy's reward signal.

RND trains a predictor network to predict the output of a fixed random
target network.  High prediction error → high novelty → high intrinsic reward.

Reference: Burda et al. "Exploration by Random Network Distillation" (2018).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F  # noqa: N812
    _TORCH = True
except ModuleNotFoundError:  # pragma: no cover
    _TORCH = False


class _RNDNet(nn.Module if _TORCH else object):  # type: ignore[misc]
    def __init__(self, obs_dim: int, out_dim: int, hidden: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RNDWrapper:
    """Wraps any policy with RND intrinsic exploration bonus.

    Parameters
    ----------
    base_policy   : Any policy with sample_action(obs).
    obs_dim       : Observation dimension.
    rnd_dim       : RND encoding dimension.
    bonus_scale   : Scale factor for intrinsic reward β.
    lr            : Learning rate for predictor network.
    device        : PyTorch device.
    seed          : RNG seed.
    """

    def __init__(
        self,
        base_policy: Any,
        obs_dim: int,
        rnd_dim: int = 64,
        bonus_scale: float = 0.1,
        lr: float = 1e-4,
        device: str = "cpu",
        seed: int = 0,
    ) -> None:
        if not _TORCH:
            raise ImportError("PyTorch is required for RNDWrapper.")

        self._base = base_policy
        self._bonus_scale = bonus_scale
        self._device = torch.device(device)

        # Fixed random target + trainable predictor
        self._target = _RNDNet(obs_dim, rnd_dim).to(self._device)
        for p in self._target.parameters():
            p.requires_grad_(False)

        self._predictor = _RNDNet(obs_dim, rnd_dim).to(self._device)
        self._opt = torch.optim.Adam(self._predictor.parameters(), lr=lr)

        # Running stats for normalisation
        self._bonus_mean = 0.0
        self._bonus_var = 1.0
        self._bonus_n = 0
        torch.manual_seed(seed)

    def _compute_bonus(self, obs: npt.NDArray[Any]) -> float:
        with torch.no_grad():
            obs_t = torch.tensor(obs.flatten(), dtype=torch.float32, device=self._device).unsqueeze(0) # noqa: E501
            tgt = self._target(obs_t)
            pred = self._predictor(obs_t)
            bonus = float(F.mse_loss(pred, tgt))
        # Update running stats (Welford)
        self._bonus_n += 1
        delta = bonus - self._bonus_mean
        self._bonus_mean += delta / self._bonus_n
        delta2 = bonus - self._bonus_mean
        self._bonus_var += delta * delta2
        std = (self._bonus_var / max(self._bonus_n, 1)) ** 0.5 + 1e-6
        return (bonus - self._bonus_mean) / std

    def _update_predictor(self, obs: npt.NDArray[Any]) -> None:
        obs_t = torch.tensor(obs.flatten(), dtype=torch.float32, device=self._device).unsqueeze(0)
        tgt = self._target(obs_t).detach()
        pred = self._predictor(obs_t)
        loss = F.mse_loss(pred, tgt)
        self._opt.zero_grad()
        loss.backward()
        self._opt.step()

    # ── public interface ──────────────────────────────────────────────────────

    def sample_action(self, obs: npt.NDArray[Any]) -> npt.NDArray[np.float64]:
        return self._base.sample_action(obs)

    def get_actor_temperature(self) -> float:
        return self._base.get_actor_temperature()

    def set_actor_temperature(self, tau: float) -> None:
        self._base.set_actor_temperature(tau)

    def train(self, total_steps: int, log_dir: str = "runs/rnd") -> None:
        self._base.train(total_steps, log_dir)

    def rollout(self, env: Any, horizon: int, seed: int | None = None) -> dict[str, npt.NDArray[Any]]: # noqa: E501
        kw: dict[str, Any] = {"seed": seed} if seed is not None else {}
        obs, _ = env.reset(**kw)
        obs_list, act_list, rew_list = [obs], [], []
        for _ in range(horizon):
            action = self.sample_action(obs)
            obs, rew, term, trunc, _ = env.step(action)
            obs_list.append(obs)
            act_list.append(action)
            bonus = self._compute_bonus(obs)
            self._update_predictor(obs)
            rew_list.append(float(rew) + self._bonus_scale * bonus)
            if term or trunc:
                break
        act_arr = np.stack(act_list) if act_list else np.empty((0,))
        return {"obs": np.stack(obs_list), "actions": act_arr, "rewards": np.array(rew_list)}

    def train_on_env(self, env: Any, total_steps: int, log_dir: str = "runs/rnd") -> None:
        """Training loop that augments reward with RND bonus."""
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        step = 0
        episode = 0
        while step < total_steps:
            obs, _ = env.reset()
            done = False
            ep_ret = 0.0
            while not done and step < total_steps:
                action = self.sample_action(obs)
                next_obs, reward, terminated, truncated, _ = env.step(action)
                bonus = self._compute_bonus(next_obs)
                self._update_predictor(next_obs)
                ep_ret += float(reward) + self._bonus_scale * bonus
                obs = next_obs
                done = terminated or truncated
                step += 1
            episode += 1
            if episode % 20 == 0:
                print(f"  step={step:>8d} ep={episode:>5d} return={ep_ret:.2f}")
