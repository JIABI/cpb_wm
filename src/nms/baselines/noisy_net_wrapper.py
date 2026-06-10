"""NoisyNet action-exploration wrapper (Phase C baseline).

Wraps any policy (DreamerWrapper / PPO) with NoisyNet-style parameter noise:
at each step, Gaussian noise is added to a linear projection of the observation,
producing exploration diversity without modifying the base policy weights.

Reference: Fortunato et al. "Noisy Networks for Exploration" (2017).

Exposes the same interface as DreamerVariant.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

try:
    import torch
    import torch.nn as nn
    _TORCH = True
except ModuleNotFoundError:  # pragma: no cover
    _TORCH = False


class NoisyLinear(nn.Module if _TORCH else object):  # type: ignore[misc]
    """Factorised NoisyLinear layer.

    y = (μ_w + σ_w ⊙ ε_w) x + (μ_b + σ_b ⊙ ε_b)
    """

    def __init__(self, in_features: int, out_features: int, sigma_init: float = 0.5) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight_mu = nn.Parameter(torch.zeros(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.full((out_features, in_features), sigma_init / (in_features**0.5)))  # noqa: E501
        self.register_buffer("weight_eps", torch.zeros(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.zeros(out_features))
        self.bias_sigma = nn.Parameter(torch.full((out_features,), sigma_init / (out_features**0.5)))  # noqa: E501
        self.register_buffer("bias_eps", torch.zeros(out_features))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = 1.0 / (self.in_features**0.5)
        nn.init.uniform_(self.weight_mu, -bound, bound)
        nn.init.uniform_(self.bias_mu, -bound, bound)

    def sample_noise(self) -> None:
        def f(x: torch.Tensor) -> torch.Tensor:
            return x.sign() * x.abs().sqrt()
        p = f(torch.randn(self.in_features))
        q = f(torch.randn(self.out_features))
        self.weight_eps.copy_(q.ger(p))  # type: ignore[attr-defined]
        self.bias_eps.copy_(q)           # type: ignore[attr-defined]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight_mu + self.weight_sigma * self.weight_eps  # type: ignore[attr-defined]
        b = self.bias_mu + self.bias_sigma * self.bias_eps        # type: ignore[attr-defined]
        return torch.nn.functional.linear(x, w, b)


class NoisyNetWrapper:
    """Wraps any policy with a NoisyNet exploration head.

    A single NoisyLinear layer maps obs → noisy_obs before passing to the
    base policy.  This adds parameter noise without changing base policy weights.

    Parameters
    ----------
    base_policy  : Any policy with sample_action(obs).
    obs_dim      : Observation dimension.
    sigma_init   : Initial noise magnitude.
    device       : PyTorch device.
    seed         : RNG seed.
    """

    def __init__(
        self,
        base_policy: Any,
        obs_dim: int,
        sigma_init: float = 0.5,
        device: str = "cpu",
        seed: int = 0,
    ) -> None:
        if not _TORCH:
            raise ImportError("PyTorch is required for NoisyNetWrapper.")

        self._base = base_policy
        self._device = torch.device(device)
        self._rng = np.random.default_rng(seed)

        # NoisyLinear identity-like projection obs_dim → obs_dim
        self._noisy = NoisyLinear(obs_dim, obs_dim, sigma_init=sigma_init).to(self._device)

    def _noisy_obs(self, obs: npt.NDArray[Any]) -> npt.NDArray[Any]:
        self._noisy.sample_noise()
        with torch.no_grad():
            obs_t = torch.tensor(obs.flatten(), dtype=torch.float32, device=self._device).unsqueeze(0) # noqa: E501
            noisy_t = self._noisy(obs_t)
        return noisy_t.squeeze(0).cpu().numpy()

    def sample_action(self, obs: npt.NDArray[Any]) -> npt.NDArray[np.float64]:
        return self._base.sample_action(self._noisy_obs(obs))

    def train(self, total_steps: int, log_dir: str = "runs/noisynet") -> None:
        self._base.train(total_steps, log_dir)

    def get_actor_temperature(self) -> float:
        return self._base.get_actor_temperature()

    def set_actor_temperature(self, tau: float) -> None:
        self._base.set_actor_temperature(tau)

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
        act_arr = np.stack(act_list) if act_list else np.empty((0, len(env.action_space.shape)))
        return {"obs": np.stack(obs_list), "actions": act_arr, "rewards": np.array(rew_list)}
