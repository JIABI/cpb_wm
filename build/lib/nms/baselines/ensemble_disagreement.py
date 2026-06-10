"""Ensemble disagreement baseline for Exp 2 (Phase C).

Uses EnsembleRSSM disagreement as an intrinsic exploration bonus.
The bonus is added to the extrinsic reward during training.

Exposes the same interface as DreamerVariant.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from nms.wm.ensemble_rssm import EnsembleRSSM


class EnsembleDisagreementBaseline:
    """DreamerWrapper + ensemble disagreement intrinsic reward.

    Parameters
    ----------
    dreamer       : DreamerWrapper used as the policy.
    ensemble      : Trained EnsembleRSSM for disagreement estimation.
    bonus_scale   : Intrinsic reward coefficient β.
    """

    def __init__(
        self,
        dreamer: Any,
        ensemble: EnsembleRSSM,
        bonus_scale: float = 0.1,
    ) -> None:
        self._dreamer = dreamer
        self._ensemble = ensemble
        self._bonus_scale = bonus_scale

    def get_actor_temperature(self) -> float:
        return self._dreamer.get_actor_temperature()

    def set_actor_temperature(self, tau: float) -> None:
        self._dreamer.set_actor_temperature(tau)

    def sample_action(self, obs: npt.NDArray[Any]) -> npt.NDArray[np.float64]:
        return self._dreamer.sample_action(obs)

    def intrinsic_bonus(self, obs: npt.NDArray[Any], horizon: int = 1) -> float:
        """Compute disagreement bonus for current obs."""
        try:
            return self._bonus_scale * self._ensemble.disagreement(obs, horizon=horizon)
        except Exception:
            return 0.0

    def train(self, total_steps: int, log_dir: str = "runs/ensemble_disagreement") -> None:
        self._dreamer.train(total_steps, log_dir)

    def rollout(self, env: Any, horizon: int, seed: int | None = None) -> dict[str, npt.NDArray[Any]]: # noqa: E501
        kw: dict[str, Any] = {"seed": seed} if seed is not None else {}
        obs, _ = env.reset(**kw)
        obs_list, act_list, rew_list = [obs], [], []
        for _ in range(horizon):
            action = self.sample_action(obs)
            obs, rew, term, trunc, _ = env.step(action)
            obs_list.append(obs)
            act_list.append(action)
            # Augment reward with intrinsic bonus
            bonus = self.intrinsic_bonus(obs)
            rew_list.append(float(rew) + bonus)
            if term or trunc:
                break
        act_arr = np.stack(act_list) if act_list else np.empty((0,))
        return {"obs": np.stack(obs_list), "actions": act_arr, "rewards": np.array(rew_list)}
