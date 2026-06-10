"""PPO with entropy bonus baseline for Exp 2 (Phase C).

A minimal PyTorch PPO implementation with configurable entropy coefficient β.
Uses an MLP policy (no RSSM) as a direct baseline against DreamerV3.

Exposes the same interface as DreamerVariant:
  train(total_steps, log_dir)
  sample_action(obs)
  get_actor_temperature() / set_actor_temperature(tau)
  rollout(env, horizon, seed)
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


class _MLPPolicy(nn.Module if _TORCH else object):  # type: ignore[misc]
    """MLP Gaussian policy: obs → (mean, log_std)."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.mean = nn.Linear(hidden, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.net(obs)
        mean = self.mean(h)
        std = self.log_std.exp().expand_as(mean)
        return mean, std

    def get_dist(self, obs: torch.Tensor) -> torch.distributions.Normal:
        mean, std = self.forward(obs)
        return torch.distributions.Normal(mean, std)


class _MLPValue(nn.Module if _TORCH else object):  # type: ignore[misc]
    def __init__(self, obs_dim: int, hidden: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)


class PPOEntropyBaseline:
    """PPO with entropy bonus.

    Parameters
    ----------
    obs_space      : gymnasium observation space.
    act_space      : gymnasium action space.
    entropy_coef   : Entropy bonus β.
    lr             : Learning rate for both policy and value net.
    clip_eps       : PPO clipping parameter.
    n_epochs       : Number of optimization epochs per PPO update.
    batch_size     : Mini-batch size for gradient steps.
    rollout_steps  : Number of env steps per PPO update.
    gamma          : Discount factor.
    lam            : GAE lambda.
    device         : PyTorch device.
    seed           : RNG seed.
    """

    def __init__(
        self,
        obs_space: Any,
        act_space: Any,
        entropy_coef: float = 0.01,
        lr: float = 3e-4,
        clip_eps: float = 0.2,
        n_epochs: int = 10,
        batch_size: int = 64,
        rollout_steps: int = 2048,
        gamma: float = 0.99,
        lam: float = 0.95,
        device: str = "cpu",
        seed: int = 0,
    ) -> None:
        if not _TORCH:
            raise ImportError("PyTorch is required for PPOEntropyBaseline.")

        self.entropy_coef = entropy_coef
        self.clip_eps = clip_eps
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.rollout_steps = rollout_steps
        self.gamma = gamma
        self.lam = lam
        self.device = torch.device(device)
        self._rng = np.random.default_rng(seed)
        if seed is not None:
            torch.manual_seed(seed)

        obs_dim = int(np.prod(obs_space.shape))
        act_dim = int(np.prod(act_space.shape))
        self._obs_dim = obs_dim
        self._act_dim = act_dim
        self._act_low = np.asarray(act_space.low, dtype=np.float32).flatten()
        self._act_high = np.asarray(act_space.high, dtype=np.float32).flatten()
        self._tau: float = 1.0

        self._policy = _MLPPolicy(obs_dim, act_dim).to(self.device)
        self._value = _MLPValue(obs_dim).to(self.device)
        self._opt = torch.optim.Adam(
            list(self._policy.parameters()) + list(self._value.parameters()),
            lr=lr,
        )

    # ── public interface ──────────────────────────────────────────────────────

    def get_actor_temperature(self) -> float:
        return self._tau

    def set_actor_temperature(self, tau: float) -> None:
        self._tau = float(tau)

    def sample_action(self, obs: npt.NDArray[Any]) -> npt.NDArray[np.float64]:
        with torch.no_grad():
            obs_t = torch.tensor(obs.flatten(), dtype=torch.float32, device=self.device).unsqueeze(0) # noqa: E501
            dist = self._policy.get_dist(obs_t)
            action = dist.sample().squeeze(0).cpu().numpy()
        return np.clip(action, self._act_low, self._act_high).astype(np.float64)

    def rollout(self, env: Any, horizon: int, seed: int | None = None) -> dict[str, npt.NDArray[Any]]: # noqa: E501
        kw: dict[str, Any] = {"seed": seed} if seed is not None else {}
        obs, _ = env.reset(**kw)
        obs_list, act_list, rew_list = [obs], [], []
        for _ in range(horizon):
            action = self.sample_action(obs)
            obs, rew, term, trunc, _ = env.step(action)
            obs_list.append(obs)
            act_list.append(action)
            rew_list.append(float(rew))
            if term or trunc:
                break
        act_arr = np.stack(act_list) if act_list else np.empty((0, self._act_dim))
        return {"obs": np.stack(obs_list), "actions": act_arr, "rewards": np.array(rew_list)}

    def train(self, total_steps: int, log_dir: str = "runs/ppo_entropy") -> None:
        """Train PPO for total_steps env interactions."""
        raise NotImplementedError(
            "PPOEntropyBaseline.train() requires an env. "
            "Call train_on_env(env, total_steps, log_dir) instead."
        )

    def train_on_env(self, env: Any, total_steps: int, log_dir: str = "runs/ppo_entropy") -> None:
        """Train PPO with an explicit environment."""
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        step = 0
        episode = 0

        while step < total_steps:
            # Collect rollout
            obs_buf, act_buf, rew_buf, val_buf, lp_buf, don_buf = [], [], [], [], [], []
            obs, _ = env.reset()
            for _ in range(self.rollout_steps):
                obs_t = torch.tensor(obs.flatten(), dtype=torch.float32, device=self.device).unsqueeze(0) # noqa: E501
                with torch.no_grad():
                    dist = self._policy.get_dist(obs_t)
                    action_t = dist.sample()
                    lp = dist.log_prob(action_t).sum(-1)
                    val = self._value(obs_t)

                action = action_t.squeeze(0).cpu().numpy()
                action = np.clip(action, self._act_low, self._act_high)
                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                obs_buf.append(obs.flatten())
                act_buf.append(action)
                rew_buf.append(float(reward))
                val_buf.append(float(val))
                lp_buf.append(float(lp))
                don_buf.append(done)

                obs = next_obs
                step += 1
                if done:
                    obs, _ = env.reset()
                    episode += 1
                if step >= total_steps:
                    break

            # GAE
            with torch.no_grad():
                last_val = float(self._value(
                    torch.tensor(obs.flatten(), dtype=torch.float32, device=self.device).unsqueeze(0) # noqa: E501
                ))
            advantages, returns = self._gae(rew_buf, val_buf, don_buf, last_val)

            # PPO update
            obs_t = torch.tensor(np.stack(obs_buf), dtype=torch.float32, device=self.device)
            act_t = torch.tensor(np.stack(act_buf), dtype=torch.float32, device=self.device)
            adv_t = torch.tensor(advantages, dtype=torch.float32, device=self.device)
            ret_t = torch.tensor(returns, dtype=torch.float32, device=self.device)
            old_lp_t = torch.tensor(lp_buf, dtype=torch.float32, device=self.device)

            adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

            for _ in range(self.n_epochs):
                idx = self._rng.permutation(len(obs_t))
                for start in range(0, len(idx), self.batch_size):
                    mb = idx[start : start + self.batch_size]
                    dist = self._policy.get_dist(obs_t[mb])
                    new_lp = dist.log_prob(act_t[mb]).sum(-1)
                    ent = dist.entropy().sum(-1).mean()
                    ratio = (new_lp - old_lp_t[mb]).exp()
                    surr1 = ratio * adv_t[mb]
                    surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv_t[mb]
                    pol_loss = -torch.min(surr1, surr2).mean()
                    val_pred = self._value(obs_t[mb])
                    val_loss = F.mse_loss(val_pred, ret_t[mb])
                    loss = pol_loss + 0.5 * val_loss - self.entropy_coef * ent
                    self._opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        list(self._policy.parameters()) + list(self._value.parameters()), 0.5
                    )
                    self._opt.step()

            if episode % 10 == 0 and episode > 0:
                print(f"  step={step:>8d} ep={episode:>5d} last_return={sum(rew_buf):.2f}")

    def _gae(
        self, rewards: list[float], values: list[float], dones: list[bool], last_val: float
    ) -> tuple[list[float], list[float]]:
        n = len(rewards)
        advantages = [0.0] * n
        returns = [0.0] * n
        gae = 0.0
        next_val = last_val
        for t in reversed(range(n)):
            mask = 1.0 - float(dones[t])
            delta = rewards[t] + self.gamma * next_val * mask - values[t]
            gae = delta + self.gamma * self.lam * mask * gae
            advantages[t] = gae
            returns[t] = gae + values[t]
            next_val = values[t]
        return advantages, returns
