"""DreamerV3 NMS wrappers (Exp 2).

This module provides three classes:

  DreamerWrapper
    Primary wrapper around dreamerv3-torch (NM512/dreamerv3-torch).
    Falls back to ``nms.wm.minimal_dreamer.MinimalDreamer`` **only** when
    the ``NMS_ALLOW_MINIMAL_FALLBACK=1`` environment variable is set.
    If dreamerv3-torch is unavailable and the flag is not set, raises
    ``RuntimeError`` — no silent degradation.

  DreamerNMSWrapper
    Full Phase B wrapper bridging RSSM ensemble + NMS violation
    estimation + CPB projection.  Requires a fitted KMeansActionDiscretizer.

  KMeansActionDiscretizer
    k-means based continuous→discrete action map.

Key design constraints:
  * dreamerv3-torch (NM512) is the default backend — MinimalDreamer only when
    NMS_ALLOW_MINIMAL_FALLBACK=1 is explicitly set.
  * NM512 API calls are based on reading the actual source (dreamer.py,
    models.py, networks.py, tools.py, parallel.py).
  * NO raw continuous actions — all actions flow through k-means.
  * NO conformal calibration here — done via core/certificate.py.
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
            km._n_threads = 1  # required by sklearn predict()
            km._n_threads = 1  # required by sklearn predict()
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


# ── NM512 env adapter helpers ─────────────────────────────────────────────────

class _NM512Box:
    """Minimal space descriptor compatible with NM512 WorldModel (duck-typed)."""

    def __init__(self, shape: tuple[int, ...], dtype: Any = np.float32) -> None:
        self.shape = shape
        self.dtype = dtype


class _NM512ObsSpace:
    """Minimal obs-space with .spaces dict (duck-typed, no gym dependency)."""

    def __init__(self, spaces: dict[str, _NM512Box]) -> None:
        self.spaces = spaces


class _NM512EnvAdapter:
    """Wrap a gymnasium env to the NM512 dreamerv3-torch interface.

    NM512 expects:
      * ``reset()``  → obs_dict (not a tuple)
      * ``step(a)``  → (obs_dict, reward, done, info)  ← 4-tuple, not 5
      * ``.id``      → unique string per episode (for cache keying)
      * ``.observation_space.spaces``  → {key: obj_with_shape}
      * ``.action_space``              → gym-compatible space

    The obs_dict always contains:
      ``obs``         float32 (obs_dim,) — actual observation
      ``image``       uint8  (1, 1, 3)  — dummy (NM512 preprocess always
                                          accesses obs["image"])
      ``is_first``    bool scalar
      ``is_terminal`` bool scalar
    """

    def __init__(self, env: Any, seed: int | None = None) -> None:
        import datetime
        import uuid as _uuid

        self._env = env
        self._seed = seed
        self._uuid_mod = _uuid
        self._ts_prefix = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        # id is refreshed on every reset() (mimics NM512's UUID wrapper)
        self.id: str = f"{self._ts_prefix}-{_uuid.uuid4().hex}"

        obs_shape = tuple(env.observation_space.shape)
        self.observation_space = _NM512ObsSpace({
            "obs": _NM512Box(shape=obs_shape, dtype=np.float32),
            "image": _NM512Box(shape=(1, 1, 3), dtype=np.uint8),
        })
        self.action_space = env.action_space

    def reset(self) -> dict[str, Any]:
        # Refresh episode ID so each episode gets its own cache slot (NM512 UUID pattern)
        self.id = f"{self._ts_prefix}-{self._uuid_mod.uuid4().hex}"
        obs, _ = self._env.reset()
        obs_shape = self.observation_space.spaces["obs"].shape
        return {
            "obs": np.asarray(obs, dtype=np.float32).reshape(obs_shape),
            "image": np.zeros((1, 1, 3), dtype=np.uint8),
            "is_first": np.array(True),
            "is_terminal": np.array(False),
        }

    def step(self, action: Any) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        # action may arrive as dict {"action": array, "logprob": ...}
        if isinstance(action, dict):
            a: Any = action.get("action", next(iter(action.values())))
        else:
            a = action
        if hasattr(a, "detach"):
            a = a.detach().cpu().numpy()
        a_np = np.asarray(a, dtype=np.float64)

        obs, reward, terminated, truncated, info = self._env.step(a_np)
        done = bool(terminated) or bool(truncated)
        discount = np.array(1.0 - float(terminated), dtype=np.float32)
        obs_dict = {
            "obs": np.asarray(obs, dtype=np.float32).reshape(
                self.observation_space.spaces["obs"].shape
            ),
            "image": np.zeros((1, 1, 3), dtype=np.uint8),
            "is_first": np.array(False),
            "is_terminal": np.array(terminated),
        }
        out_info = dict(info) if isinstance(info, dict) else {}
        out_info["discount"] = discount
        return obs_dict, float(reward), done, out_info

    def close(self) -> None:
        self._env.close()


# ── DreamerWrapper ─────────────────────────────────────────────────────────────

class DreamerWrapper:
    """Wrapper exposing the DreamerV3 interface needed by NMS.

    Uses NM512/dreamerv3-torch as the primary backend.  MinimalDreamer is only
    used when ``NMS_ALLOW_MINIMAL_FALLBACK=1`` is set **and** dreamerv3-torch
    is unavailable.  Otherwise a missing backend raises ``RuntimeError``.

    Parameters
    ----------
    config_path : Ignored (kept for API compatibility).
    env         : gymnasium-compatible environment.
    device      : PyTorch device string.
    seed        : Optional RNG seed.
    """

    def __init__(
        self,
        config_path: str,
        env: Any,
        device: str = "cuda" if __import__("torch").cuda.is_available() else "cpu",
        seed: int | None = None,
    ) -> None:
        self.config_path = config_path
        self.env = env
        self.device = device
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self._tau: float = 1.0
        self._agent: Any = None          # NM512 Dreamer or MinimalDreamer
        self._backend: str = "uninit"
        self._nm_config: Any = None      # SimpleNamespace config for NM512
        self._dv3_state: Any = None      # RSSM state for inference

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

    def _ensure_agent(self) -> None:
        """Set backend and build NM512 config (lazy).

        Does NOT create the Dreamer agent (that requires a populated replay
        buffer and happens inside train()).  Methods called before train() that
        need the agent fall back to random / stub values.
        """
        if self._backend != "uninit":
            return

        from nms.wm._dreamerv3_path import DREAMERV3_AVAILABLE, DREAMERV3_PATH

        if DREAMERV3_AVAILABLE:
            try:
                self._nm_config = self._build_nm512_config()
                self._backend = "dreamerv3-torch"
                return
            except Exception as exc:
                raise RuntimeError(
                    f"dreamerv3-torch config build failed: {exc}\n"
                    f"NM512 path: {DREAMERV3_PATH}\n"
                    "Set NMS_ALLOW_MINIMAL_FALLBACK=1 to use MinimalDreamer instead."
                ) from exc

        import os

        if os.environ.get("NMS_ALLOW_MINIMAL_FALLBACK") != "1":
            raise RuntimeError(
                "dreamerv3-torch not found and NMS_ALLOW_MINIMAL_FALLBACK is not '1'.\n"
                "Options:\n"
                "  1. Clone https://github.com/NM512/dreamerv3-torch.git to a sibling dir\n"
                "  2. Set NMS_DREAMERV3_PATH=<path> to point at the checkout\n"
                "  3. Set NMS_ALLOW_MINIMAL_FALLBACK=1 to use the MinimalDreamer stub\n"
                "     (paper-quality numbers require dreamerv3-torch)"
            )

        warnings.warn(
            "dreamerv3-torch not available (NMS_ALLOW_MINIMAL_FALLBACK=1). "
            "Using MinimalDreamer — paper-quality numbers require dreamerv3-torch.",
            stacklevel=3,
        )
        self._init_minimal_dreamer()

    def _build_nm512_config(self) -> Any:
        """Build a SimpleNamespace config for NM512 from configs.yaml + overrides.

        Uses ruamel.yaml (the same loader as NM512) so that scientific-notation
        values like ``1e-4`` are parsed as floats, not strings.
        """
        from types import SimpleNamespace

        import ruamel.yaml as ryaml  # type: ignore[import-untyped]

        from nms.wm._dreamerv3_path import DREAMERV3_PATH

        raw = ryaml.YAML(typ="safe").load(  # type: ignore[union-attr]
            (DREAMERV3_PATH / "configs.yaml").read_text()  # type: ignore[union-attr]
        )

        # Start from defaults, overlay dmc_proprio (MLP-only proprio tasks)
        cfg: dict[str, Any] = {}
        cfg.update(raw["defaults"])
        cfg.update(raw["dmc_proprio"])

        act_dim = int(np.prod(self.env.action_space.shape))

        # Numeric scalars that main() divides by action_repeat
        cfg["steps"] = int(float(str(cfg.get("steps", 1_000_000))))
        cfg["log_every"] = int(float(str(cfg.get("log_every", 10_000))))
        cfg["eval_every"] = int(float(str(cfg.get("eval_every", 10_000))))
        cfg["time_limit"] = int(float(str(cfg.get("time_limit", 1_000))))
        cfg["prefill"] = int(float(str(cfg.get("prefill", 2_500))))
        cfg["dataset_size"] = int(float(str(cfg.get("dataset_size", 1_000_000))))

        cfg.update({
            "device": self.device,
            "seed": self._seed or 0,
            "compile": False,
            "video_pred_log": False,
            "precision": 32,
            "action_repeat": 1,
            "num_actions": act_dim,
            "envs": 1,
            "logdir": None,
            "traindir": None,
            "evaldir": None,
            "reward_EMA": True,
            # MLP-only encoder/decoder: encode "obs" key, ignore dummy image
            "encoder": {
                "mlp_keys": "obs",
                "cnn_keys": "$^",
                "act": "SiLU",
                "norm": True,
                "cnn_depth": 32,
                "kernel_size": 4,
                "minres": 4,
                "mlp_layers": 5,
                "mlp_units": 1024,
                "symlog_inputs": True,
            },
            "decoder": {
                "mlp_keys": "obs",
                "cnn_keys": "$^",
                "act": "SiLU",
                "norm": True,
                "cnn_depth": 32,
                "kernel_size": 4,
                "minres": 4,
                "mlp_layers": 5,
                "mlp_units": 1024,
                "cnn_sigmoid": False,
                "image_dist": "mse",
                "vector_dist": "symlog_mse",
                "outscale": 1.0,
            },
        })
        return SimpleNamespace(**cfg)

    def _init_minimal_dreamer(self) -> None:
        """Initialise MinimalDreamer (only called when NMS_ALLOW_MINIMAL_FALLBACK=1)."""
        from nms.wm.minimal_dreamer import MinimalDreamer, MinimalDreamerConfig

        md_cfg = MinimalDreamerConfig(device=self.device)
        self._agent = MinimalDreamer(
            self.env.observation_space,
            self.env.action_space,
            cfg=md_cfg,
            seed=self._seed,
        )
        self._backend = "minimal_dreamer"

    # ── checkpoint loading ────────────────────────────────────────────────────

    def load_from_dir(self, log_dir: str | "Path") -> None:  # type: ignore[name-defined]
        """Load a trained agent from *log_dir*/latest.pt without running any training.

        This is the preferred way to load a checkpoint for evaluation/calibration.
        After this call ``self._agent`` is set and ``sample_action`` / rollout
        methods work correctly.

        Raises
        ------
        FileNotFoundError
            If ``log_dir/latest.pt`` does not exist.
        RuntimeError
            If backend is not dreamerv3-torch (MinimalDreamer does not support
            checkpoint loading via this method).
        """
        import pathlib
        import torch

        self._ensure_agent()  # sets up backend, _nm_config, action_space etc.

        if self._backend == "minimal_dreamer":
            raise RuntimeError(
                "load_from_dir() is only supported for the dreamerv3-torch backend. "
                "MinimalDreamer does not support checkpoint loading."
            )

        import networks as nm_networks  # type: ignore[import-not-found]
        import tools as nm_tools  # type: ignore[import-not-found]
        from dreamer import Dreamer, count_steps, make_dataset  # type: ignore[import-not-found]
        from parallel import Damy  # type: ignore[import-not-found]

        # Patch NM512's MLP default device (same as _train_dreamerv3_torch does)
        _orig_mpl_init = nm_networks.MLP.__init__
        _target_device = self.device

        def _patched_mpl_init(  # type: ignore[no-untyped-def]
            self_mpl: Any, *args: Any, device: str = _target_device, **kw: Any
        ) -> None:
            _orig_mpl_init(self_mpl, *args, device=device, **kw)

        nm_networks.MLP.__init__ = _patched_mpl_init  # type: ignore[method-assign]

        logdir = pathlib.Path(log_dir).expanduser()
        ckpt_path = logdir / "latest.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"No checkpoint found at {ckpt_path}. "
                "Train the model first with DreamerWrapper.train()."
            )

        cfg = self._nm_config
        cfg.logdir = logdir
        cfg.traindir = cfg.traindir or (logdir / "train_eps")
        cfg.traindir.mkdir(parents=True, exist_ok=True)

        step = count_steps(cfg.traindir)
        logger = nm_tools.Logger(logdir, cfg.action_repeat * step)

        train_eps: dict = nm_tools.load_episodes(cfg.traindir, limit=cfg.dataset_size)
        train_dataset = iter(make_dataset(train_eps, cfg))

        adapter = _NM512EnvAdapter(self.env, seed=self._seed)
        train_envs = [Damy(adapter)]
        acts = train_envs[0].action_space
        obs_space = train_envs[0].observation_space

        agent: Any = Dreamer(obs_space, acts, cfg, logger, train_dataset).to(cfg.device)
        agent.requires_grad_(requires_grad=False)

        # Load checkpoint
        checkpoint = torch.load(ckpt_path, map_location=cfg.device)
        agent.load_state_dict(checkpoint["agent_state_dict"])
        try:
            nm_tools.recursively_load_optim_state_dict(
                agent, checkpoint["optims_state_dict"]
            )
        except Exception:
            pass
        agent._should_pretrain._once = False

        self._agent = agent
        self._dv3_state = None
        print(f"[DreamerWrapper] Loaded checkpoint from {ckpt_path}")

    # ── training ──────────────────────────────────────────────────────────────

    def train(self, total_steps: int, log_dir: str = "runs/dreamer") -> None:
        """Train the agent for *total_steps* environment steps.

        With the dreamerv3-torch backend, fewer than
        ``batch_size * batch_length`` steps just issue a warning and return
        (not enough data to fill one training batch).
        """
        self._ensure_agent()

        if self._backend == "minimal_dreamer":
            self._agent.train(self.env, total_steps, log_dir)
            return

        # dreamerv3-torch backend
        cfg = self._nm_config
        min_steps = cfg.batch_size * cfg.batch_length  # typically 16*64 = 1024
        if total_steps < min_steps:
            warnings.warn(
                f"DreamerWrapper.train(): total_steps={total_steps} < "
                f"batch_size×batch_length={min_steps}. "
                "Not enough data for one NM512 training batch — skipping.",
                stacklevel=2,
            )
            return
        self._train_dreamerv3_torch(total_steps, log_dir)

    def _train_dreamerv3_torch(self, total_steps: int, log_dir: str) -> None:
        """Full NM512 training loop using tools.simulate."""
        import pathlib

        # NM512 imports (sys.path set up by _dreamerv3_path.py)
        import networks as nm_networks  # type: ignore[import-not-found]
        import tools as nm_tools  # type: ignore[import-not-found]
        import torch
        from dreamer import Dreamer, count_steps, make_dataset  # type: ignore[import-not-found]
        from parallel import Damy  # type: ignore[import-not-found]
        from torch import distributions as torchd  # noqa: F401

        # Patch NM512's MLP default device from "cuda" to our target device.
        # MultiEncoder creates MLP without passing device, so the default "cuda"
        # would crash on CPU-only PyTorch builds (e.g., macOS with CPU torch).
        _orig_mpl_init = nm_networks.MLP.__init__
        _target_device = self.device

        def _patched_mpl_init(  # noqa: E501
            self_mpl: Any, *args: Any, device: str = _target_device, **kw: Any
        ) -> None:
            _orig_mpl_init(self_mpl, *args, device=device, **kw)

        nm_networks.MLP.__init__ = _patched_mpl_init  # type: ignore[method-assign]

        logdir = pathlib.Path(log_dir).expanduser()
        logdir.mkdir(parents=True, exist_ok=True)

        cfg = self._nm_config
        cfg.logdir = logdir
        cfg.traindir = cfg.traindir or (logdir / "train_eps")
        cfg.traindir.mkdir(parents=True, exist_ok=True)

        step = count_steps(cfg.traindir)
        logger = nm_tools.Logger(logdir, cfg.action_repeat * step)

        # Load any episodes already on disk
        train_eps: dict[str, Any] = nm_tools.load_episodes(cfg.traindir, limit=cfg.dataset_size)

        # Wrapped env (Damy makes reset/step return callables, as simulate expects)
        adapter = _NM512EnvAdapter(self.env, seed=self._seed)
        train_envs = [Damy(adapter)]
        acts = train_envs[0].action_space

        # ── prefill with random agent ──────────────────────────────────────
        existing = count_steps(cfg.traindir)
        prefill = max(0, min(cfg.prefill, total_steps // 2) - existing)
        if prefill > 0:
            if hasattr(acts, "n"):
                random_dist: Any = nm_tools.OneHotDist(
                    torch.zeros(cfg.num_actions).repeat(cfg.envs, 1)
                )
            else:
                random_dist = torchd.independent.Independent(
                    torchd.uniform.Uniform(
                        torch.tensor(np.array(acts.low, dtype=np.float32)).repeat(cfg.envs, 1),
                        torch.tensor(np.array(acts.high, dtype=np.float32)).repeat(cfg.envs, 1),
                    ),
                    1,
                )

            def _random_agent(o: Any, d: Any, s: Any) -> tuple[Any, None]:
                act = random_dist.sample()
                logp = random_dist.log_prob(act)
                return {"action": act, "logprob": logp}, None

            nm_tools.simulate(
                _random_agent,
                train_envs,
                train_eps,
                cfg.traindir,
                logger,
                limit=cfg.dataset_size,
                steps=prefill,
            )
            logger.step += prefill * cfg.action_repeat

        if not train_eps:
            warnings.warn(
                "No episodes in replay buffer after prefill — aborting NM512 training.",
                stacklevel=2,
            )
            return

        # ── create dataset + agent ────────────────────────────────────────
        train_dataset = make_dataset(train_eps, cfg)

        obs_space = train_envs[0].observation_space
        agent: Any = Dreamer(obs_space, acts, cfg, logger, train_dataset).to(cfg.device)
        agent.requires_grad_(requires_grad=False)

        # Restore checkpoint if present
        ckpt_path = logdir / "latest.pt"
        if ckpt_path.exists():
            checkpoint = torch.load(ckpt_path, map_location=cfg.device)
            agent.load_state_dict(checkpoint["agent_state_dict"])
            try:
                nm_tools.recursively_load_optim_state_dict(
                    agent, checkpoint["optims_state_dict"]
                )
            except Exception:
                pass
            agent._should_pretrain._once = False

        self._agent = agent
        self._dv3_state = None

        # ── main training simulate loop ───────────────────────────────────
        remaining = max(0, total_steps - prefill)
        nm_tools.simulate(
            agent,
            train_envs,
            train_eps,
            cfg.traindir,
            logger,
            limit=cfg.dataset_size,
            steps=remaining,
        )
        logger.write()

        # Persist checkpoint
        try:
            torch.save(
                {
                    "agent_state_dict": agent.state_dict(),
                    "optims_state_dict": nm_tools.recursively_collect_optim_state_dict(agent),
                },
                logdir / "latest.pt",
            )
        except Exception:
            pass

    # ── policy ────────────────────────────────────────────────────────────────

    def sample_action(self, obs: npt.NDArray[Any]) -> npt.NDArray[np.float64]:
        """Sample one action from the actor.

        Returns a random action if the agent has not yet been trained.
        """
        self._ensure_agent()

        if self._backend == "minimal_dreamer":
            return self._agent.sample_action(obs)  # type: ignore[union-attr]

        # dreamerv3-torch backend
        if self._agent is None:
            # Pre-training: return random action (not a MinimalDreamer fallback)
            return self._rng.uniform(self._action_low, self._action_high, self._action_shape)

        try:
            import torch
            agent = self._agent
            agent.eval()  # disable dropout/batchnorm stochasticity
            wm = agent._wm
            behavior = agent._task_behavior
            obs_arr = np.asarray(obs, dtype=np.float32)
            obs_batch: dict[str, Any] = {
                "obs": obs_arr[np.newaxis],
                "image": np.zeros((1, 1, 1, 3), dtype=np.uint8),
                "is_first": np.ones(1, dtype=np.float32),  # fresh state each call
                "is_terminal": np.zeros(1, dtype=np.float32),
            }
            with torch.no_grad():
                preprocessed = wm.preprocess(obs_batch)
                embed = wm.encoder(preprocessed)
                latent, _ = wm.dynamics.obs_step(None, None, embed, preprocessed["is_first"])
                feat = wm.dynamics.get_feat(latent)
                actor = behavior.actor(feat)
                # deterministic mode + sigma-scaled noise (CIR bandwidth)
                action_t = actor.mode if not callable(getattr(actor, "mode", actor.sample)) else actor.mode()
                _tau = self.get_actor_temperature()
                if _tau > 0:
                    action_t = action_t + _tau * torch.randn_like(action_t)
                    action_t = torch.clamp(action_t, -1.0, 1.0)
            action = action_t.squeeze(0).detach().cpu().numpy()
            return action.astype(np.float64)
        except Exception as e:
            warnings.warn(f"sample_action fallback to random: {e}", stacklevel=2)
            return self._rng.uniform(self._action_low, self._action_high, self._action_shape)

    def reset_state(self) -> None:
        """Reset the RSSM recurrent state (call at episode boundaries)."""
        self._dv3_state = None

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
        if seed is not None:
            import torch as _t
            _t.manual_seed(seed)
            _t.cuda.manual_seed_all(seed)
            _t.backends.cudnn.deterministic = True
        # Reset NM512 internal state for fresh episode
        self._dv3_state = None
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
        """H-step imagined rollout from world model at temperature sigma.

        Uses RSSM img_step when the NM512 agent is available; returns random
        actions otherwise (pre-training or MinimalDreamer stub behaviour).
        """
        prev_tau = self.get_actor_temperature()
        self.set_actor_temperature(sigma)
        self._ensure_agent()

        result: dict[str, npt.NDArray[Any]]

        if self._backend == "minimal_dreamer":
            result = self._agent.imagine(obs, horizon)  # type: ignore[union-attr]
        elif self._backend == "dreamerv3-torch" and self._agent is not None:
            result = self._imagined_rollout_nm512(obs, horizon, seed)
        else:
            rng = np.random.default_rng(seed)
            actions = rng.uniform(
                self._action_low, self._action_high, size=(horizon, *self._action_shape)
            )
            result = {"actions": actions}

        self.set_actor_temperature(prev_tau)
        return result

    def _imagined_rollout_nm512(
        self,
        obs: npt.NDArray[Any],
        horizon: int,
        seed: int | None = None,
    ) -> dict[str, npt.NDArray[Any]]:
        """RSSM img_step imagined rollout using the trained NM512 agent."""
        import torch
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True

        try:
            agent = self._agent
            agent.eval()  # disable dropout/batchnorm stochasticity
            wm = agent._wm
            behavior = agent._task_behavior

            obs_arr = np.asarray(obs, dtype=np.float32)
            obs_batch = {
                "obs": obs_arr[np.newaxis],
                "image": np.zeros((1, 1, 1, 3), dtype=np.uint8),
                "is_first": np.ones(1, dtype=np.float32),
                "is_terminal": np.zeros(1, dtype=np.float32),
            }
            preprocessed = wm.preprocess(obs_batch)
            embed = wm.encoder(preprocessed)
            latent, _ = wm.dynamics.obs_step(
                None, None, embed, preprocessed["is_first"]
            )

            actions_list: list[npt.NDArray[Any]] = []
            with torch.no_grad():
                for _ in range(horizon):
                    feat = wm.dynamics.get_feat(latent)
                    actor = behavior.actor(feat)
                    # paired-rollout determinism via mode (not sample)
                    action = actor.mode if not callable(getattr(actor, "mode", actor.sample)) else actor.mode()
                    # CIR: inject sigma-scaled actuation noise (decorrelated per path)
                    _tau = self.get_actor_temperature()
                    if _tau > 0:
                        action = action + _tau * torch.randn_like(action)
                        action = torch.clamp(action, -1.0, 1.0)
                    actions_list.append(action.squeeze(0).detach().cpu().numpy())
                    latent = wm.dynamics.img_step(latent, action)

            return {"actions": np.stack(actions_list).astype(np.float64)}
        except Exception:
            rng = np.random.default_rng(seed)
            actions = rng.uniform(
                self._action_low, self._action_high, size=(horizon, *self._action_shape)
            )
            return {"actions": actions}

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

    def rssm_posterior(
        self, obs: npt.NDArray[Any]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Return (mean, variance) of the RSSM posterior for *obs*.

        Returns dummy zeros/ones if the agent has not yet been trained.
        """
        self._ensure_agent()

        if self._backend == "minimal_dreamer":
            return self._agent.rssm_posterior(obs)  # type: ignore[union-attr]

        if self._backend == "dreamerv3-torch" and self._agent is not None:
            try:
                import torch

                wm = self._agent._wm
                obs_arr = np.asarray(obs, dtype=np.float32)
                obs_batch = {
                    "obs": obs_arr[np.newaxis],
                    "image": np.zeros((1, 1, 1, 3), dtype=np.uint8),
                    "is_first": np.ones(1, dtype=np.float32),
                    "is_terminal": np.zeros(1, dtype=np.float32),
                }
                with torch.no_grad():
                    preprocessed = wm.preprocess(obs_batch)
                    embed = wm.encoder(preprocessed)
                    post, _ = wm.dynamics.obs_step(
                        None, None, embed, preprocessed["is_first"]
                    )
                # Flatten stoch (B, stoch, discrete) → (B, stoch*discrete) + deter
                feat = wm.dynamics.get_feat(post)
                mean = feat.squeeze(0).cpu().numpy().astype(np.float64)
                # Approximate variance from logit spread (discrete RSSM)
                if "logit" in post:
                    logit = post["logit"].squeeze(0)
                    probs = torch.softmax(logit, dim=-1)
                    var = (probs * (1 - probs)).mean(dim=-1)
                    var_flat = var.cpu().numpy().astype(np.float64)
                    # pad/truncate to match mean
                    var_out = np.full_like(mean, var_flat.mean())
                else:
                    std = post.get("std", torch.ones_like(post["stoch"]))
                    var_out = std.squeeze(0).cpu().numpy().astype(np.float64) ** 2
                return mean, var_out
            except Exception:
                pass

        feat_dim = 32
        return np.zeros(feat_dim, dtype=np.float64), np.ones(feat_dim, dtype=np.float64) * 0.01


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
        device: str = "cuda" if __import__("torch").cuda.is_available() else "cpu",
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
        warnings.warn("DreamerNMSWrapper._agent not set — operating in stub mode.", stacklevel=2)

    def reset(self) -> tuple[npt.NDArray[Any], dict[str, Any]]:
        self._ensure_agent()
        obs, info = self.env.reset()
        self._latent = None
        return obs, info

    def observe(self, obs: npt.NDArray[Any]) -> npt.NDArray[Any]:
        if self._agent is None:
            return np.asarray(obs, dtype=np.float32)
        try:
            import torch

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
                import torch

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
            import torch

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
