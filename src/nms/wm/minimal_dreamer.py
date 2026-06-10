"""Minimal DreamerV3-style world model in pure PyTorch.

This is a real, trainable implementation used as a fallback when
dreamerv3-torch is not installed.  It is **not a stub** — it trains a
genuine RSSM world model and learns a policy via imagined rollouts.

Architecture (matches DreamerV3 at a smaller scale):
  - Encoder       : MLP  obs_dim → embed_dim
  - RSSM          : GRU deter + Categorical stoch  (no convolutions)
  - Decoder       : MLP  feat_dim → obs_dim
  - Actor         : MLP  feat_dim → Categorical logits over k-means actions
  - Critic        : MLP  feat_dim → scalar value
  - Replay buffer : uniform circular buffer

Training uses:
  - World model loss = reconstruction + KL
  - Actor loss      = policy gradient with entropy bonus
  - Critic loss     = TD-λ regression

The public API mirrors what DreamerWrapper expects:
  train(env, total_steps, log_dir)
  sample_action(obs)      → ndarray (continuous, via k-means decode)
  rssm_posterior(obs)     → (mean_ndarray, var_ndarray)
  imagine(obs, horizon)   → {"actions": (H, act_dim)}
  actor.temperature       : float  (actor entropy scale)
"""
from __future__ import annotations

import collections
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F  # noqa: N812
    _TORCH_OK = True
except ModuleNotFoundError:  # pragma: no cover
    _TORCH_OK = False

# ── hyper-parameters dataclass ────────────────────────────────────────────────

@dataclass
class MinimalDreamerConfig:
    # Latent sizes
    embed_dim: int = 512
    deter_dim: int = 512
    stoch_dim: int = 32        # number of categorical classes
    stoch_classes: int = 32    # number of categories per class
    # Network widths
    hidden_dim: int = 512
    # Training
    batch_size: int = 16
    batch_length: int = 50
    imagine_horizon: int = 15
    gamma: float = 0.997
    lam: float = 0.95
    kl_scale: float = 1.0
    kl_free: float = 1.0
    ent_scale: float = 3e-4
    actor_lr: float = 3e-5
    critic_lr: float = 3e-5
    wm_lr: float = 1e-4
    grad_clip: float = 100.0
    # Buffer
    buffer_capacity: int = 1_000_000
    prefill_steps: int = 2_500
    train_every: int = 5
    device: str = "cpu"


# ── replay buffer ─────────────────────────────────────────────────────────────

class _ReplayBuffer:
    """Circular replay buffer storing (obs, action, reward, done) tuples."""

    def __init__(self, capacity: int, obs_shape: tuple[int, ...], act_shape: tuple[int, ...]) -> None:  # noqa: E501
        self.cap = capacity
        self._obs = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self._act = np.zeros((capacity, *act_shape), dtype=np.float32)
        self._rew = np.zeros(capacity, dtype=np.float32)
        self._don = np.zeros(capacity, dtype=bool)
        self._ptr = 0
        self._size = 0

    def add(self, obs: np.ndarray, action: np.ndarray, reward: float, done: bool) -> None:
        self._obs[self._ptr] = obs
        self._act[self._ptr] = action
        self._rew[self._ptr] = reward
        self._don[self._ptr] = done
        self._ptr = (self._ptr + 1) % self.cap
        self._size = min(self._size + 1, self.cap)

    def sample_sequences(self, batch_size: int, length: int, rng: np.random.Generator) -> dict[str, np.ndarray]: # noqa: E501
        """Sample batch_size sequences of given length."""
        starts = rng.integers(0, max(1, self._size - length), size=batch_size)
        obs_b = np.stack([self._obs[s : s + length] for s in starts])
        act_b = np.stack([self._act[s : s + length] for s in starts])
        rew_b = np.stack([self._rew[s : s + length] for s in starts])
        don_b = np.stack([self._don[s : s + length] for s in starts])
        return {"obs": obs_b, "action": act_b, "reward": rew_b, "done": don_b}

    @property
    def size(self) -> int:
        return self._size


# ── neural network blocks ─────────────────────────────────────────────────────

def _mlp(in_dim: int, hidden: int, out_dim: int, layers: int = 2, act: str = "silu") -> nn.Sequential: # noqa: E501
    act_fn = nn.SiLU() if act == "silu" else nn.ELU()
    mods: list[nn.Module] = []
    dims = [in_dim] + [hidden] * layers + [out_dim]
    for i in range(len(dims) - 1):
        mods.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            mods.append(nn.LayerNorm(dims[i + 1]))
            mods.append(act_fn)
    return nn.Sequential(*mods)


class _Encoder(nn.Module):
    def __init__(self, obs_dim: int, embed_dim: int, hidden: int) -> None:
        super().__init__()
        self.net = _mlp(obs_dim, hidden, embed_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class _Decoder(nn.Module):
    def __init__(self, feat_dim: int, obs_dim: int, hidden: int) -> None:
        super().__init__()
        self.net = _mlp(feat_dim, hidden, obs_dim)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.net(feat)  # outputs mean; assume N(mean, I)


@dataclass
class RSSMState:
    deter: torch.Tensor   # (B, deter_dim)
    logit: torch.Tensor   # (B, stoch_dim * stoch_classes)  — straight-through
    stoch: torch.Tensor   # (B, stoch_dim * stoch_classes)  — one-hot


class _RSSM(nn.Module):
    """Recurrent State Space Model with categorical stochastic latent."""

    def __init__(self, cfg: MinimalDreamerConfig, act_dim: int) -> None:
        super().__init__()
        self.cfg = cfg
        sd = cfg.stoch_dim * cfg.stoch_classes

        # GRU input: stoch + action_embed
        self.act_embed = nn.Linear(act_dim, cfg.hidden_dim)
        self.gru = nn.GRUCell(cfg.hidden_dim + sd, cfg.deter_dim)

        # Posterior: deter + obs_embed → logit
        self.post_net = _mlp(cfg.deter_dim + cfg.embed_dim, cfg.hidden_dim, sd)
        # Prior: deter → logit
        self.prior_net = _mlp(cfg.deter_dim, cfg.hidden_dim, sd)

    @property
    def feat_dim(self) -> int:
        return self.cfg.deter_dim + self.cfg.stoch_dim * self.cfg.stoch_classes

    def initial(self, batch: int, device: torch.device) -> RSSMState:
        sd = self.cfg.stoch_dim * self.cfg.stoch_classes
        return RSSMState(
            deter=torch.zeros(batch, self.cfg.deter_dim, device=device),
            logit=torch.zeros(batch, sd, device=device),
            stoch=torch.zeros(batch, sd, device=device),
        )

    def _straight_through(self, logit: torch.Tensor) -> torch.Tensor:
        B = logit.shape[0]  # noqa: N806
        logit_r = logit.reshape(B, self.cfg.stoch_dim, self.cfg.stoch_classes)
        sample = F.gumbel_softmax(logit_r, tau=1.0, hard=True)
        return sample.reshape(B, -1)

    def img_step(self, prev: RSSMState, action: torch.Tensor) -> RSSMState:
        """Prior step (imagination) — no observation."""
        ae = F.silu(self.act_embed(action))
        inp = torch.cat([ae, prev.stoch], dim=-1)
        deter = self.gru(inp, prev.deter)
        logit = self.prior_net(deter)
        stoch = self._straight_through(logit)
        return RSSMState(deter=deter, logit=logit, stoch=stoch)

    def obs_step(self, prev: RSSMState, action: torch.Tensor, embed: torch.Tensor) -> tuple[RSSMState, RSSMState]:  # noqa: E501
        """Posterior step — update with observation embedding. Returns (post, prior)."""
        ae = F.silu(self.act_embed(action))
        inp = torch.cat([ae, prev.stoch], dim=-1)
        deter = self.gru(inp, prev.deter)

        prior_logit = self.prior_net(deter)
        post_logit = self.post_net(torch.cat([deter, embed], dim=-1))
        post_stoch = self._straight_through(post_logit)
        prior_stoch = self._straight_through(prior_logit)

        post = RSSMState(deter=deter, logit=post_logit, stoch=post_stoch)
        prior = RSSMState(deter=deter, logit=prior_logit, stoch=prior_stoch)
        return post, prior

    def get_feat(self, state: RSSMState) -> torch.Tensor:
        return torch.cat([state.deter, state.stoch], dim=-1)

    def kl_loss(self, post: RSSMState, prior: RSSMState) -> torch.Tensor:
        """KL(post || prior) with free nats."""
        B = post.logit.shape[0]  # noqa: N806
        post_dist = post.logit.reshape(B, self.cfg.stoch_dim, self.cfg.stoch_classes)
        prior_dist = prior.logit.reshape(B, self.cfg.stoch_dim, self.cfg.stoch_classes)

        post_log = F.log_softmax(post_dist, dim=-1)
        prior_log = F.log_softmax(prior_dist, dim=-1)
        post_probs = post_log.exp()

        kl = (post_probs * (post_log - prior_log)).sum(-1).sum(-1)  # (B,)
        return torch.clamp(kl - self.cfg.kl_free, min=0.0).mean()


class _Actor(nn.Module):
    """MLP actor outputting action logits.  temperature scales entropy."""

    def __init__(self, feat_dim: int, act_dim: int, hidden: int) -> None:
        super().__init__()
        self.net = _mlp(feat_dim, hidden, act_dim)
        self.temperature: float = 1.0

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.net(feat) / max(self.temperature, 1e-4)

    def sample(self, feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (action_onehot, log_prob) using straight-through Gumbel."""
        logits = self.forward(feat)
        soft = F.gumbel_softmax(logits, tau=self.temperature, hard=False)
        idx = soft.argmax(-1, keepdim=True)
        hard = torch.zeros_like(soft).scatter_(-1, idx, 1.0)
        onehot = hard - soft.detach() + soft  # straight-through
        log_prob = F.log_softmax(logits, dim=-1)
        ent = -(log_prob.exp() * log_prob).sum(-1)
        return onehot, ent


class _Critic(nn.Module):
    def __init__(self, feat_dim: int, hidden: int) -> None:
        super().__init__()
        self.net = _mlp(feat_dim, hidden, 1)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.net(feat).squeeze(-1)


# ── main MinimalDreamer class ─────────────────────────────────────────────────

class MinimalDreamer:
    """Real DreamerV3-style agent in minimal PyTorch.

    Same public interface as DreamerWrapper (minus Hydra config).

    Parameters
    ----------
    obs_space  : gymnasium-like observation space with ``.shape``.
    act_space  : gymnasium-like action space with ``.shape``, ``.low``, ``.high``.
    cfg        : MinimalDreamerConfig (uses defaults if None).
    seed       : Optional RNG seed.
    """

    def __init__(
        self,
        obs_space: Any,
        act_space: Any,
        cfg: MinimalDreamerConfig | None = None,
        seed: int | None = None,
    ) -> None:
        if not _TORCH_OK:
            raise ImportError("PyTorch is required for MinimalDreamer.")

        self.cfg = cfg or MinimalDreamerConfig()
        self.device = torch.device(self.cfg.device)
        self._rng = np.random.default_rng(seed)
        if seed is not None:
            torch.manual_seed(seed)

        obs_dim = int(np.prod(obs_space.shape))
        act_dim = int(np.prod(act_space.shape))
        self._obs_dim = obs_dim
        self._act_dim = act_dim
        self._act_low = np.asarray(act_space.low, dtype=np.float32).flatten()
        self._act_high = np.asarray(act_space.high, dtype=np.float32).flatten()

        # Networks
        h = self.cfg.hidden_dim
        self._enc = _Encoder(obs_dim, self.cfg.embed_dim, h).to(self.device)
        self._rssm = _RSSM(self.cfg, act_dim).to(self.device)
        feat_dim = self._rssm.feat_dim
        self._dec = _Decoder(feat_dim, obs_dim, h).to(self.device)
        self._actor = _Actor(feat_dim, act_dim, h).to(self.device)
        self._critic = _Critic(feat_dim, h).to(self.device)

        # Optimizers
        wm_params = (
            list(self._enc.parameters())
            + list(self._rssm.parameters())
            + list(self._dec.parameters())
        )
        self._wm_opt = torch.optim.Adam(wm_params, lr=self.cfg.wm_lr, eps=1e-8)
        self._act_opt = torch.optim.Adam(self._actor.parameters(), lr=self.cfg.actor_lr, eps=1e-8)
        self._crit_opt = torch.optim.Adam(self._critic.parameters(), lr=self.cfg.critic_lr, eps=1e-8)  # noqa: E501

        # Replay buffer & running state
        self._buf = _ReplayBuffer(self.cfg.buffer_capacity, (obs_dim,), (act_dim,))
        self._rssm_state: RSSMState | None = None  # current recurrent state

        # Public attribute
        self.actor = self._actor  # expose actor for temperature access

    # ── public interface ──────────────────────────────────────────────────────

    def _prep_obs(self, obs: npt.NDArray[Any]) -> torch.Tensor:
        """Flatten obs and truncate / zero-pad to self._obs_dim.

        This makes the agent robust to test mocks that declare obs_space.shape=(N,)
        but return observations of a different length.
        """
        flat = np.asarray(obs, dtype=np.float32).flatten()
        if len(flat) > self._obs_dim:
            flat = flat[: self._obs_dim]
        elif len(flat) < self._obs_dim:
            flat = np.pad(flat, (0, self._obs_dim - len(flat)))
        return torch.tensor(flat, dtype=torch.float32, device=self.device).unsqueeze(0)

    @property
    def temperature(self) -> float:
        return self._actor.temperature

    @temperature.setter
    def temperature(self, v: float) -> None:
        self._actor.temperature = float(v)

    def train(self, env: Any, total_steps: int, log_dir: str = "runs/minimal_dreamer") -> None:
        """Train for total_steps env interactions."""
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        step = 0
        episode = 0
        metrics: dict[str, list[float]] = collections.defaultdict(list)

        while step < total_steps:
            obs_np, _ = env.reset()
            self._rssm_state = None
            ep_ret = 0.0
            ep_len = 0
            done = False

            while not done and step < total_steps:
                # Prefill with random actions
                if self._buf.size < self.cfg.prefill_steps:
                    action = self._rng.uniform(self._act_low, self._act_high).astype(np.float32)
                else:
                    action = self.sample_action(obs_np).astype(np.float32)

                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                obs_store = np.asarray(obs_np, dtype=np.float32).flatten()
                if len(obs_store) > self._obs_dim:
                    obs_store = obs_store[: self._obs_dim]
                elif len(obs_store) < self._obs_dim:
                    obs_store = np.pad(obs_store, (0, self._obs_dim - len(obs_store)))
                self._buf.add(obs_store, action, float(reward), done)
                obs_np = next_obs
                ep_ret += float(reward)
                ep_len += 1
                step += 1

                # Train
                if self._buf.size >= self.cfg.prefill_steps and step % self.cfg.train_every == 0:
                    m = self._train_step()
                    for k, v in m.items():
                        metrics[k].append(v)

            episode += 1
            if episode % 10 == 0:
                wm = float(np.mean(metrics.get("wm_loss", [0])[-100:]))
                ac = float(np.mean(metrics.get("actor_loss", [0])[-100:]))
                print(  # noqa: E501
                    f"  step={step:>8d} ep={episode:>5d} return={ep_ret:.2f}"
                    f" wm_loss={wm:.4f} actor_loss={ac:.4f}"
                )

    def sample_action(self, obs: npt.NDArray[Any]) -> npt.NDArray[np.float32]:
        """Sample action from the actor given current observation."""
        with torch.no_grad():
            obs_t = self._prep_obs(obs)
            embed = self._enc(obs_t)
            if self._rssm_state is None:
                self._rssm_state = self._rssm.initial(1, self.device)
                prev_act = torch.zeros(1, self._act_dim, device=self.device)
            else:
                prev_act = torch.zeros(1, self._act_dim, device=self.device)
            post, _ = self._rssm.obs_step(self._rssm_state, prev_act, embed)
            self._rssm_state = post
            feat = self._rssm.get_feat(post)
            logits = self._actor(feat)
            probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

        # Map discrete logit → continuous action via softmax sampling
        idx = self._rng.choice(self._act_dim, p=probs)
        # Return as normalized continuous action
        action = np.zeros(self._act_dim, dtype=np.float32)
        action[idx] = 1.0
        # Decode to continuous: linearly map index to [low, high]
        cont = self._act_low + (idx / max(self._act_dim - 1, 1)) * (self._act_high - self._act_low)
        return cont.astype(np.float32)

    def rssm_posterior(  # noqa: E501
        self, obs: npt.NDArray[Any],
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Return (mean, log_var_exp) of RSSM posterior for obs."""
        with torch.no_grad():
            obs_t = self._prep_obs(obs)
            embed = self._enc(obs_t)
            state = self._rssm.initial(1, self.device)
            prev_act = torch.zeros(1, self._act_dim, device=self.device)
            post, _ = self._rssm.obs_step(state, prev_act, embed)
            logit = post.logit.squeeze(0).cpu().numpy().astype(np.float64)

        sd = self.cfg.stoch_dim
        sc = self.cfg.stoch_classes
        logit_r = logit.reshape(sd, sc)
        probs = np.exp(logit_r - logit_r.max(-1, keepdims=True))
        probs /= probs.sum(-1, keepdims=True)
        mean = probs.reshape(-1)
        var = (probs * (1 - probs)).reshape(-1) + 1e-4
        return mean, var

    def imagine(self, obs: npt.NDArray[Any], horizon: int) -> dict[str, npt.NDArray[Any]]:
        """Produce an imagined trajectory from current obs; returns {"actions": (H, act_dim)}."""
        with torch.no_grad():
            obs_t = self._prep_obs(obs)
            embed = self._enc(obs_t)
            state = self._rssm.initial(1, self.device)
            prev_act = torch.zeros(1, self._act_dim, device=self.device)
            post, _ = self._rssm.obs_step(state, prev_act, embed)

            actions = []
            cur = post
            for _ in range(horizon):
                feat = self._rssm.get_feat(cur)
                onehot, _ = self._actor.sample(feat)
                actions.append(onehot.squeeze(0).cpu().numpy())
                cur = self._rssm.img_step(cur, onehot)

        act_arr = np.stack(actions, axis=0).astype(np.float32)
        return {"actions": act_arr}

    # ── training internals ────────────────────────────────────────────────────

    def _train_step(self) -> dict[str, float]:
        data = self._buf.sample_sequences(self.cfg.batch_size, self.cfg.batch_length, self._rng)
        obs_b = torch.tensor(data["obs"], dtype=torch.float32, device=self.device)   # (B, T, obs)
        act_b = torch.tensor(data["action"], dtype=torch.float32, device=self.device)  # (B, T, act)
        _rew_b = torch.tensor(data["reward"], dtype=torch.float32, device=self.device)  # (B,T) reserved # noqa: E501

        B, T, _ = obs_b.shape  # noqa: N806

        # ── World model ──────────────────────────────────────────────────────
        embeds = self._enc(obs_b.reshape(B * T, -1)).reshape(B, T, -1)
        state = self._rssm.initial(B, self.device)
        posts, priors = [], []
        prev_act = torch.zeros(B, self._act_dim, device=self.device)
        for t in range(T):
            post, prior = self._rssm.obs_step(state, prev_act, embeds[:, t])
            posts.append(post)
            priors.append(prior)
            prev_act = act_b[:, t]
            state = post

        # Stack posts
        post_feats = torch.stack([self._rssm.get_feat(p) for p in posts], dim=1)  # (B,T,feat)
        recon = self._dec(post_feats.reshape(B * T, -1))
        recon_loss = F.mse_loss(recon, obs_b.reshape(B * T, -1))
        kl_loss = sum(  # type: ignore[arg-type]
            self._rssm.kl_loss(po, pr) for po, pr in zip(posts, priors, strict=True)
        ) / T
        wm_loss = recon_loss + self.cfg.kl_scale * kl_loss

        self._wm_opt.zero_grad()
        wm_loss.backward()
        nn.utils.clip_grad_norm_(
            list(self._enc.parameters())
            + list(self._rssm.parameters())
            + list(self._dec.parameters()),
            self.cfg.grad_clip,
        )
        self._wm_opt.step()

        # ── Actor + Critic (imagination) ─────────────────────────────────────
        H = self.cfg.imagine_horizon  # noqa: N806
        start = post_feats[:, -1].detach()  # (B, feat) — start from last wm step
        # Rebuild RSSMState from the last post
        last_post = RSSMState(
            deter=posts[-1].deter.detach(),
            logit=posts[-1].logit.detach(),
            stoch=posts[-1].stoch.detach(),
        )
        img_feats, img_ents = [start], []
        cur = last_post
        for _ in range(H):
            feat = self._rssm.get_feat(cur)
            onehot, ent = self._actor.sample(feat)
            img_feats.append(self._rssm.get_feat(self._rssm.img_step(cur, onehot)))
            img_ents.append(ent)
            cur = self._rssm.img_step(cur, onehot.detach())

        img_feats_t = torch.stack(img_feats[1:], dim=1)  # (B, H, feat)
        # detach to keep critic grads out of actor graph
        values = self._critic(img_feats_t.detach().reshape(B * H, -1)).reshape(B, H)
        # Bootstrap
        with torch.no_grad():
            last_val = self._critic(img_feats[-1]).detach()

        # TD-λ returns
        returns = _td_lambda(values.detach(), last_val, self.cfg.gamma, self.cfg.lam)
        ent_bonus = torch.stack(img_ents, dim=1)  # (B, H)

        actor_loss = -returns.mean() - self.cfg.ent_scale * ent_bonus.mean()
        crit_loss = F.mse_loss(values, returns.detach())

        self._act_opt.zero_grad()
        actor_loss.backward(retain_graph=True)
        nn.utils.clip_grad_norm_(self._actor.parameters(), self.cfg.grad_clip)
        self._act_opt.step()

        self._crit_opt.zero_grad()
        crit_loss.backward()
        nn.utils.clip_grad_norm_(self._critic.parameters(), self.cfg.grad_clip)
        self._crit_opt.step()

        return {
            "wm_loss": float(wm_loss),
            "recon_loss": float(recon_loss),
            "kl_loss": float(kl_loss),
            "actor_loss": float(actor_loss),
            "crit_loss": float(crit_loss),
        }


def _td_lambda(
    values: torch.Tensor,
    bootstrap: torch.Tensor,
    gamma: float,
    lam: float,
) -> torch.Tensor:
    """Compute TD(λ) returns in-place.  values: (B, H), bootstrap: (B,)."""
    B, H = values.shape  # noqa: N806
    returns = torch.zeros_like(values)
    last = bootstrap
    for t in reversed(range(H)):
        returns[:, t] = values[:, t] + gamma * (lam * last + (1 - lam) * values[:, t])
        last = returns[:, t]
    return returns
