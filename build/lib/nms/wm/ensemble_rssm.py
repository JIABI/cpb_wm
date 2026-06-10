"""Ensemble of RSSM world models for epistemic/aleatoric uncertainty estimation.

Used in Phase B/C for:
  - Epistemic uncertainty: variance of per-member latent means
  - Aleatoric uncertainty: mean of per-member latent variances
  - Disagreement-based violation detection

K=3 for Phase A/B, K=5 for Phase C (set via constructor).
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt


class EnsembleRSSM:
    """Ensemble of K independent RSSM world models.

    Each member is a ``DreamerWrapper`` (or ``MinimalDreamer``) trained on
    the same environment but with a different random seed.

    Parameters
    ----------
    obs_space  : gymnasium-compatible observation space.
    act_space  : gymnasium-compatible action space.
    k          : Number of ensemble members.
    device     : PyTorch device string.
    seed       : Base seed; member i uses seed + i.
    config_path: YAML config path forwarded to each DreamerWrapper.
    """

    def __init__(
        self,
        obs_space: Any,
        act_space: Any,
        k: int = 3,
        device: str = "cpu",
        seed: int = 0,
        config_path: str = "",
    ) -> None:
        if k < 1:
            raise ValueError(f"k must be ≥ 1, got {k}.")
        self.k = k
        self.device = device
        self.obs_space = obs_space
        self.act_space = act_space
        self._seed = seed
        self._config_path = config_path
        self._members: list[Any] = []  # list of DreamerWrapper or MinimalDreamer

    # ── construction helpers ──────────────────────────────────────────────────

    def _make_member(self, idx: int) -> Any:
        """Build member idx as a DreamerWrapper (lazy backend selection)."""
        from nms.wm.dreamer_wrapper import DreamerWrapper

        class _FakeEnv:
            """Minimal env shim for DreamerWrapper constructor."""
            def __init__(self, obs_space: Any, act_space: Any) -> None:
                self.observation_space = obs_space
                self.action_space = act_space

        fake_env = _FakeEnv(self.obs_space, self.act_space)
        return DreamerWrapper(
            config_path=self._config_path,
            env=fake_env,
            device=self.device,
            seed=self._seed + idx,
        )

    # ── training ──────────────────────────────────────────────────────────────

    def train_all(self, env_factory: Any, total_steps: int, log_dir: str = "runs/ensemble") -> None:
        """Train all K members independently.

        Parameters
        ----------
        env_factory  : Callable() → fresh gymnasium env for each member.
        total_steps  : Total env steps per member.
        log_dir      : Output directory for checkpoints; subfolder per member.
        """
        self._members = []
        for i in range(self.k):
            print(f"[EnsembleRSSM] Training member {i+1}/{self.k} ...")
            env = env_factory()
            member = self._make_member(i)
            member_dir = str(Path(log_dir) / f"member_{i}")
            member.train(total_steps, log_dir=member_dir)
            self._members.append(member)
            env.close()

    def save_all(self, save_dir: str | Path) -> None:
        """Stub: individual member checkpoints are saved during train_all."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        # Each member's DreamerWrapper/MinimalDreamer saves to its own log_dir
        # during training.  This method records a manifest.
        import json
        manifest = {
            "k": self.k,
            "device": self.device,
            "seed": self._seed,
            "config_path": self._config_path,
        }
        (save_dir / "ensemble_manifest.json").write_text(json.dumps(manifest, indent=2))

    def load_all(self, save_dir: str | Path) -> None:
        """Rebuild members from a saved ensemble directory."""
        save_dir = Path(save_dir)
        import json
        manifest = json.loads((save_dir / "ensemble_manifest.json").read_text())
        self.k = manifest["k"]
        self._seed = manifest["seed"]
        self._config_path = manifest.get("config_path", "")
        self._members = [self._make_member(i) for i in range(self.k)]
        warnings.warn(
            "load_all: ensemble members rebuilt from config only (no weight restore "
            "implemented for MinimalDreamer yet).  Re-train if weights are needed.",
            stacklevel=2,
        )

    # ── uncertainty estimation ────────────────────────────────────────────────

    def _posteriors(self, obs: npt.NDArray[Any]) -> list[tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]]: # noqa: E501
        """Return list of (mean, var) posteriors from each member."""
        if not self._members:
            raise RuntimeError("No members: call train_all() or load_all() first.")
        return [m.rssm_posterior(obs) for m in self._members]

    def estimate_aleatoric(self, obs: npt.NDArray[Any]) -> float:
        """Mean of per-member latent variances (aleatoric uncertainty)."""
        posts = self._posteriors(obs)
        return float(np.mean([v.mean() for _, v in posts]))

    def estimate_epistemic(self, obs: npt.NDArray[Any]) -> float:
        """Variance of per-member latent means (epistemic uncertainty)."""
        posts = self._posteriors(obs)
        means = np.stack([m for m, _ in posts])  # (K, latent_dim)
        return float(np.var(means, axis=0).mean())

    def estimate_total(self, obs: npt.NDArray[Any]) -> float:
        """Total uncertainty = aleatoric + epistemic."""
        return self.estimate_aleatoric(obs) + self.estimate_epistemic(obs)

    # ── imagined trajectories for violation detection ─────────────────────────

    def imagine_all(
        self,
        obs: npt.NDArray[Any],
        horizon: int,
    ) -> list[dict[str, npt.NDArray[Any]]]:
        """Imagine H-step trajectories from all K members.

        Returns list of K dicts, each with ``actions`` (H, act_dim).
        """
        if not self._members:
            raise RuntimeError("No members: call train_all() or load_all() first.")
        return [m.rollout_imagined(obs, horizon=horizon, sigma=m.get_actor_temperature()) for m in self._members] # noqa: E501

    def disagreement(
        self,
        obs: npt.NDArray[Any],
        horizon: int,
    ) -> float:
        """Imagined-realized disagreement rate across ensemble members.

        Returns fraction of (member, step) pairs where any two members
        disagree on the argmax discrete action.
        """
        trajs = self.imagine_all(obs, horizon)
        if len(trajs) < 2:
            return 0.0
        # Compare member 0 vs all others at each step
        ref = trajs[0]["actions"]  # (H, act_dim)
        ref_idx = np.argmax(ref, axis=-1) if ref.ndim == 2 else ref.flatten()
        disagree = 0
        total = 0
        for traj in trajs[1:]:
            act = traj["actions"]
            other_idx = np.argmax(act, axis=-1) if act.ndim == 2 else act.flatten()
            n = min(len(ref_idx), len(other_idx))
            disagree += int(np.sum(ref_idx[:n] != other_idx[:n]))
            total += n
        return disagree / max(total, 1)
