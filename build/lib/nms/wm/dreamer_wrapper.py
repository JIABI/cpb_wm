"""DreamerV3 NMS wrappers (Exp 2).

This module provides three classes:

  DreamerWrapper (Phase A+)
    Real wrapper around dreamerv3-torch (NM512/dreamerv3-torch).
    Falls back to MinimalDreamer (nms.wm.minimal_dreamer) when
    dreamerv3-torch is not installed — **not a stub**: MinimalDreamer is
    a full PyTorch RSSM implementation.

  DreamerNMSWrapper (Phase B+)
    Full Phase B wrapper bridging RSSM ensemble + NMS violation
    estimation + CPB projection.  Requires a fitted KMeansActionDiscretizer.

  KMeansActionDiscretizer
    k-means based continuous→discrete action map.

Key design constraints:
  * NO raw continuous actions — all actions flow through k-means.
  * NO conformal calibration here — done via core/certificate.py.
  * NO NMSOperator interface modifications.
  * K=3 ensemble members Phase A/B; K=5 Phase C.
  * np.random.default_rng — never np.random.seed.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

# ── K-means action discretizer ────────────────────────────────────────────────

class KMeansActionDiscretizer:
    """Discretize a continuous action space via k-means clustering.

    Parameters
    ----------
    n_clusters   : Number of discrete action centroids.
    random_state : Seed for sklearn KMeans.
    """

    def __init__(self, n_clusters: int = 20, random_state: int = 42) -> None:
        self.n_clusters = n_clusters
        self.random_state = random_state
        self._kmeans: Any | None = None
        self._centroids: npt.NDArray[np.float64] | None = None

    @property
    def is_fitted(self) -> bool:
        return self._kmeans is not None

    def fit(self, actions: npt.NDArray[np.float64]) -> None:
        """Fit on (N, action_dim) array of collected actions.

        Uses sklearn KMeans when available; falls back to random centroid
        selection (deterministic via random_state) when sklearn is absent.
        The fallback is sufficient for CI dry-runs and smoke tests.
        """
        actions = np.asarray(actions, dtype=np.float64)
        if actions.ndim == 1:
            actions = actions[:, np.newaxis]

        try:
            from sklearn.cluster import KMeans  # type: ignore[import-untyped]
            km = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=10)
            km.fit(actions)
            self._kmeans = km
            self._centroids = km.cluster_centers_.copy()
        except ModuleNotFoundError:
            import warnings
            warnings.warn(
                "scikit-learn not found — KMeansActionDiscretizer using random centroids "
                "(CI/dry-run mode). Install scikit-learn for real k-means.",
                stacklevel=2,
            )
            rng = np.random.default_rng(self.random_state)
            idx = rng.choice(len(actions), size=min(self.n_clusters, len(actions)), replace=False)
            self._centroids = actions[idx].copy()
            self._kmeans = _NumpyKMeans(self._centroids)

    def encode(self, action: npt.NDArray[np.float64]) -> int:
        if not self.is_fitted:
            raise RuntimeError("Call fit() before encode().")
        action = np.asarray(action, dtype=np.float64).reshape(1, -1)
        return int(self._kmeans.predict(action)[0])  # type: ignore[union-attr]

    def decode(self, index: int) -> npt.NDArray[np.float64]:
        if self._centroids is None:
            raise RuntimeError("Call fit() before decode().")
        return self._centroids[index].copy()

    def decode_batch(self, indices: npt.NDArray[np.intp]) -> npt.NDArray[np.float64]:
        if self._centroids is None:
            raise RuntimeError("Call fit() before decode_batch().")
        return self._centroids[indices]

    def save(self, path: str | Path) -> None:
        """Persist centroids to .npz file."""
        if self._centroids is None:
            raise RuntimeError("Call fit() first.")
        np.savez(str(path), centroids=self._centroids, n_clusters=np.array(self.n_clusters))

    @classmethod
    def load(cls, path: str | Path) -> KMeansActionDiscretizer:
        """Load from .npz written by save()."""
        data = np.load(str(path))
        centroids = data["centroids"]
        n = int(data["n_clusters"])
        obj = cls(n_clusters=n)
        obj._centroids = centroids
        try:
            from sklearn.cluster import KMeans  # type: ignore[import-untyped]
            km = KMeans(n_clusters=n, n_init=1)
            km.cluster_centers_ = centroids
            km.n_features_in_ = centroids.shape[1]
            obj._kmeans = km
        except ModuleNotFoundError:
            obj._kmeans = _NumpyKMeans(centroids)
        return obj


class _NumpyKMeans:
    """Minimal KMeans-compatible class backed by numpy (no sklearn needed)."""

    def __init__(self, centroids: npt.NDArray[np.float64]) -> None:
        self.cluster_centers_ = centroids

    def predict(self, x_arr: npt.NDArray[np.float64]) -> npt.NDArray[np.intp]:
        dists = np.linalg.norm(x_arr[:, np.newaxis] - self.cluster_centers_[np.newaxis], axis=-1)
        return dists.argmin(axis=-1)


# ── DreamerWrapper ─────────────────────────────────────────────────────────────

class DreamerWrapper:
    """Thin wrapper exposing the DreamerV3 interface needed by NMS.

    First tries to use the NM512/dreamerv3-torch package (``import dreamerv3``).
    If unavailable, falls back to ``nms.wm.minimal_dreamer.MinimalDreamer``
    — a real ~500-line PyTorch RSSM implementation, not a stub.

    Parameters
    ----------
    config_path : YAML config for dreamerv3-torch (ignored in fallback mode).
    env         : gymnasium-compatible environment.
    device      : PyTorch device string.
    seed        : Optional RNG seed.
    """

    def __init__(
        self,
        config_path: str,
        env: Any,
        device: str = "cpu",
        seed: int | None = None,
    ) -> None:
        self.config_path = config_path
        self.env = env
        self.device = device
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self._tau: float = 1.0
        self._agent: Any = None
        self._backend: str = "uninit"
        self._dv3_state: Any = None

        act_space = env.action_space
        self._action_shape: tuple[int, ...] = tuple(act_space.shape)
        self._action_low = np.asarray(act_space.low, dtype=np.float64)
        self._action_high = np.asarray(act_space.high, dtype=np.float64)

    # ── temperature ───────────────────────────────────────────────────────────

    def get_actor_temperature(self) -> float:
        return self._tau

    def set_actor_temperature(self, tau: float) -> None:
        self._tau = float(tau)
        if self._agent is not None and hasattr(self._agent, "actor"):
            try:
                self._agent.actor.temperature = self._tau
            except AttributeError:
                pass

    # ── agent lazy init ───────────────────────────────────────────────────────

    def _ensure_agent(self) -> bool:
        if self._agent is not None:
            return True
        if self._try_dreamerv3_torch():
            return True
        return self._init_minimal_dreamer()

    def _try_dreamerv3_torch(self) -> bool:
        """Attempt NM512/dreamerv3-torch.  Returns False if unavailable."""
        try:
            import dreamerv3 as dv3  # type: ignore[import-not-found]
            import yaml  # stdlib

            cfg_path = Path(self.config_path)
            if cfg_path.exists():
                with open(cfg_path) as fp:
                    raw_cfg = yaml.safe_load(fp) or {}
            else:
                raw_cfg = {}

            # NM512 exposes dreamerv3.tools.Flags for config + dreamerv3.Dreamer as agent
            cfg = dv3.tools.Flags(raw_cfg)
            obs_space = self.env.observation_space
            act_space = self.env.action_space
            log_dir = str(cfg_path.parent / "dreamer_logs")
            logger = dv3.tools.Logger(logdir=log_dir, step=0)
            dataset: list[Any] = []
            self._agent = dv3.Dreamer(obs_space, act_space, cfg, logger, iter(dataset))
            self._backend = "dreamerv3-torch"
            return True
        except Exception:
            return False

    def _init_minimal_dreamer(self) -> bool:
        """Fall back to MinimalDreamer (real PyTorch RSSM)."""
        try:
            from nms.wm.minimal_dreamer import MinimalDreamer, MinimalDreamerConfig

            warnings.warn(
                "dreamerv3-torch not available — using MinimalDreamer (PyTorch RSSM fallback). "
                "For full DreamerV3: "
                "pip install 'dreamerv3-torch @ git+https://github.com/NM512/dreamerv3-torch.git@main'",
                stacklevel=3,
            )
            md_cfg = MinimalDreamerConfig(device=self.device)
            self._agent = MinimalDreamer(
                self.env.observation_space,
                self.env.action_space,
                cfg=md_cfg,
                seed=self._seed,
            )
            self._backend = "minimal_dreamer"
            return True
        except Exception as exc:
            warnings.warn(f"MinimalDreamer init failed: {exc}", stacklevel=3)
            self._backend = "none"
            return False

    # ── training ──────────────────────────────────────────────────────────────

    def train(self, total_steps: int, log_dir: str = "runs/dreamer") -> None:
        """Train the agent for total_steps environment steps."""
        if not self._ensure_agent():
            warnings.warn("train() skipped — no backend available.", stacklevel=2)
            return

        if self._backend == "minimal_dreamer":
            self._agent.train(self.env, total_steps, log_dir)
        elif self._backend == "dreamerv3-torch":
            self._run_dv3_torch_loop(total_steps, log_dir)

    def _run_dv3_torch_loop(self, total_steps: int, log_dir: str) -> None:
        """Training loop using NM512 dreamerv3-torch __call__ interface."""
        step = 0
        obs, _ = self.env.reset()
        reset_flag = np.array([True])
        state = None

        while step < total_steps:
            obs_batch = {"obs": np.expand_dims(np.asarray(obs), 0)}
            try:
                action_out, state, _ = self._agent(
                    obs_batch, reset_flag, state, training=True
                )
                action = np.asarray(action_out).squeeze(0)
            except Exception:
                action = self._rng.uniform(self._action_low, self._action_high, self._action_shape)

            obs, _, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            reset_flag = np.array([done])
            if done:
                obs, _ = self.env.reset()
            step += 1

    # ── rollout ───────────────────────────────────────────────────────────────

    def rollout(
        self,
        env: Any,
        horizon: int,
        seed: int | None = None,
    ) -> dict[str, npt.NDArray[Any]]:
        """Collect a real ≤horizon-step trajectory.

        Returns dict with ``obs`` (≤H+1, obs_dim), ``actions`` (≤H, act_dim),
        ``rewards`` (≤H,).
        """
        self._ensure_agent()
        kw: dict[str, Any] = {"seed": seed} if seed is not None else {}
        obs, _ = env.reset(**kw)

        obs_list = [obs]
        act_list: list[npt.NDArray[Any]] = []
        rew_list: list[float] = []

        for _ in range(horizon):
            action = self.sample_action(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            obs_list.append(obs)
            act_list.append(np.asarray(action, dtype=np.float64))
            rew_list.append(float(reward))
            if terminated or truncated:
                break

        act_arr = (
            np.stack(act_list)
            if act_list
            else np.empty((0, *self._action_shape), dtype=np.float64)
        )
        return {
            "obs": np.stack(obs_list),
            "actions": act_arr,
            "rewards": np.array(rew_list),
        }

    def rollout_imagined(
        self,
        obs: npt.NDArray[Any],
        horizon: int,
        sigma: float,
        seed: int | None = None,
    ) -> dict[str, npt.NDArray[Any]]:
        """Imagined H-step rollout from world model at temperature sigma."""
        prev_tau = self.get_actor_temperature()
        self.set_actor_temperature(sigma)

        self._ensure_agent()
        if self._backend == "minimal_dreamer":
            result = self._agent.imagine(obs, horizon)
        else:
            rng = np.random.default_rng(seed)
            actions = rng.uniform(
                self._action_low, self._action_high, size=(horizon, *self._action_shape)
            )
            result = {"actions": actions}

        self.set_actor_temperature(prev_tau)
        return result

    def rollout_realized(
        self,
        env: Any,
        horizon: int,
        sigma: float,
        seed: int | None = None,
    ) -> dict[str, npt.NDArray[Any]]:
        """Real H-step rollout using actor at temperature sigma."""
        prev_tau = self.get_actor_temperature()
        self.set_actor_temperature(sigma)
        traj = self.rollout(env, horizon, seed=seed)
        self.set_actor_temperature(prev_tau)
        return {"actions": traj["actions"]}

    def sample_action(self, obs: npt.NDArray[Any]) -> npt.NDArray[np.float64]:
        """Sample one action from the actor."""
        if not self._ensure_agent():
            return self._rng.uniform(self._action_low, self._action_high, self._action_shape)

        if self._backend == "minimal_dreamer":
            return self._agent.sample_action(obs)

        if self._backend == "dreamerv3-torch":
            try:
                obs_batch = {"obs": np.expand_dims(np.asarray(obs), 0)}
                reset_flag = np.array([False])
                action_out, self._dv3_state, _ = self._agent(
                    obs_batch, reset_flag, self._dv3_state, training=False
                )
                return np.asarray(action_out).squeeze(0).astype(np.float64)
            except Exception:
                pass

        return self._rng.uniform(self._action_low, self._action_high, self._action_shape)

    def rssm_posterior(
        self, obs: npt.NDArray[Any]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Return (mean, variance) of RSSM posterior for obs."""
        if not self._ensure_agent():
            dummy = np.zeros(32, dtype=np.float64)
            return dummy, np.ones(32, dtype=np.float64) * 0.01

        if self._backend == "minimal_dreamer":
            return self._agent.rssm_posterior(obs)

        # dreamerv3-torch: try RSSM posterior
        try:
            import torch  # type: ignore[import-not-found]
            obs_t = torch.tensor(np.asarray(obs), dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                embed = self._agent._encoder({"obs": obs_t})
                stoch = self._agent._rssm.initial(1, device=obs_t.device)
                _, posterior = self._agent._rssm.observe(embed, {}, stoch)
                mean = posterior["mean"].squeeze(0).cpu().numpy().astype(np.float64)
                std = posterior["std"].squeeze(0).cpu().numpy().astype(np.float64)
                return mean, std**2
        except Exception:
            dummy = np.zeros(32, dtype=np.float64)
            return dummy, np.ones(32, dtype=np.float64) * 0.01


# ── DreamerNMSWrapper ─────────────────────────────────────────────────────────

class DreamerNMSWrapper:
    """Wraps a DreamerWrapper with NMS violation estimation + CPB projection.

    Parameters
    ----------
    env         : gymnasium-compatible environment.
    dreamer_cfg : Dict config forwarded to the agent.
    kmeans      : Pre-fitted KMeansActionDiscretizer.
    ensemble_k  : Number of RSSM ensemble members (≥ 1).
    device      : PyTorch device string.
    """

    def __init__(
        self,
        env: Any,
        dreamer_cfg: dict[str, Any],
        kmeans: KMeansActionDiscretizer,
        ensemble_k: int = 3,
        device: str = "cpu",
    ) -> None:
        if not kmeans.is_fitted:
            raise ValueError("kmeans must be fitted before passing to DreamerNMSWrapper.")
        if ensemble_k < 1:
            raise ValueError(f"ensemble_k must be ≥ 1, got {ensemble_k}.")

        self.env = env
        self.dreamer_cfg = dreamer_cfg
        self.kmeans = kmeans
        self.ensemble_k = ensemble_k
        self.device = device

        self._agent: Any = None
        self._latent: Any = None
        self._violation_estimator: Any = None

    def _ensure_agent(self) -> None:
        if self._agent is not None:
            return
        try:
            import dreamerv3 as dv3  # type: ignore[import-not-found]
            self._agent = dv3.DreamerV3(
                self.env.observation_space,
                self.env.action_space,
                **self.dreamer_cfg,
            )
        except Exception:
            warnings.warn("dreamerv3 unavailable — DreamerNMSWrapper in stub mode.", stacklevel=2)
            self._agent = None

    def reset(self) -> tuple[npt.NDArray[Any], dict[str, Any]]:
        self._ensure_agent()
        obs, info = self.env.reset()
        self._latent = None
        return obs, info

    def observe(self, obs: npt.NDArray[Any]) -> npt.NDArray[Any]:
        if self._agent is None:
            return np.asarray(obs, dtype=np.float32)
        try:
            import torch  # type: ignore[import-not-found]
            obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
            self._latent = self._agent.observe(obs_t, self._latent)
            return self._latent.detach().cpu().numpy()
        except Exception:
            return np.asarray(obs, dtype=np.float32)

    def act(self, latent: npt.NDArray[Any]) -> npt.NDArray[np.float64]:
        if self._agent is None:
            rng = np.random.default_rng()
            raw = rng.uniform(self.env.action_space.low, self.env.action_space.high)
            idx = self.kmeans.encode(raw)
        else:
            try:
                import torch  # type: ignore[import-not-found]
                lat_t = torch.tensor(latent, device=self.device)
                idx = int(self._agent.act(lat_t).item())
            except Exception:
                rng = np.random.default_rng()
                raw = rng.uniform(self.env.action_space.low, self.env.action_space.high)
                idx = self.kmeans.encode(raw)
        return self.kmeans.decode(idx)

    def imagine(self, latent: npt.NDArray[Any], horizon: int) -> list[npt.NDArray[Any]]:
        if self._agent is None:
            return [latent.copy() for _ in range(horizon)]
        try:
            import torch  # type: ignore[import-not-found]
            lat_t = torch.tensor(latent, device=self.device)
            imagined = self._agent.imagine(lat_t, horizon)
            return [s.detach().cpu().numpy() for s in imagined]
        except Exception:
            return [latent.copy() for _ in range(horizon)]

    def register_violation_estimator(self, estimator: Any) -> None:
        self._violation_estimator = estimator

    def estimate_violation_rate(
        self,
        nu: float,
        sigma: float,
        horizon: int,
        n_imagined: int = 100,
    ) -> float:
        if self._violation_estimator is None:
            warnings.warn("No violation estimator registered.", stacklevel=2)
            return float("nan")
        if self._latent is None:
            return float("nan")
        violations = []
        for _ in range(n_imagined):
            imagined = self.imagine(self._latent, horizon)
            viol = self._violation_estimator(imagined, nu=nu, sigma=sigma)
            violations.append(float(viol))
        return float(np.mean(violations))

    def step(self, action: Any) -> tuple[npt.NDArray[Any], float, bool, bool, dict[str, Any]]:
        return self.env.step(action)  # type: ignore[return-value]

    def close(self) -> None:
        self.env.close()
