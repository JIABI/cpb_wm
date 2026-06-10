"""Tests for rollout_ipd_with_mixture."""
from __future__ import annotations

import numpy as np

from nms.envs.noisy_games import C, IPDTrajectory, rollout_ipd_with_mixture
from nms.policies.opponents import AlwaysCooperate, AlwaysDefect
from nms.policies.reciprocal import SoftReciprocalPolicy


def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


def test_mixture_returns_ipd_trajectory() -> None:
    """rollout_ipd_with_mixture returns a valid IPDTrajectory."""
    p1 = SoftReciprocalPolicy(0.5)
    mixture = {AlwaysDefect(): 1.0}
    traj = rollout_ipd_with_mixture(p1, mixture, horizon=20, nu=0.0, rng=_rng())
    assert isinstance(traj, IPDTrajectory)
    assert len(traj.a1) == 20


def test_mixture_shape_matches_horizon() -> None:
    """All trajectory arrays have shape (horizon,)."""
    horizon = 13
    p1 = SoftReciprocalPolicy(0.3)
    mixture = {AlwaysDefect(): 0.5, AlwaysCooperate(): 0.5}
    traj = rollout_ipd_with_mixture(p1, mixture, horizon=horizon, nu=0.1, rng=_rng(7))
    for name, arr in [
        ("a1", traj.a1), ("a2", traj.a2),
        ("obs1", traj.obs1), ("obs2", traj.obs2),
        ("r1", traj.r1), ("r2", traj.r2),
        ("false_obs1", traj.false_obs1), ("false_obs2", traj.false_obs2),
    ]:
        assert arr.shape == (horizon,), f"{name}: expected ({horizon},), got {arr.shape}"


def test_mixture_always_defect_opponent_exploits_sigma1() -> None:
    """Against pure AllD, sigma=1 agent always cooperates → always exploited."""
    p1 = SoftReciprocalPolicy(sigma=1.0)  # unconditional cooperator
    mixture = {AlwaysDefect(): 1.0}
    traj = rollout_ipd_with_mixture(p1, mixture, horizon=30, nu=0.0, rng=_rng(0))
    assert (traj.a1 == C).all(), "sigma=1 agent must always cooperate"
    assert (traj.a2 == 1).all(), "AllD must always defect"


def test_mixture_pure_selfplay_uses_reciprocal_opponent() -> None:
    """With reciprocal:1.0, opponent is always a SoftReciprocalPolicy.

    We verify this indirectly: at ν=0 with sigma=0 (= DeterministicTFT),
    two TFTs always cooperate.  Via mixture with only a reciprocal opponent,
    both players should still cooperate at every step.
    """

    rng = _rng(7)
    sigma = 0.0  # DeterministicTFT
    mixture = {SoftReciprocalPolicy(sigma): 1.0}
    n_ep = 50
    all_coop = True
    for _ in range(n_ep):
        traj = rollout_ipd_with_mixture(
            SoftReciprocalPolicy(sigma), mixture, horizon=20, nu=0.0, rng=rng
        )
        if not (traj.a1 == C).all():
            all_coop = False
            break
    assert all_coop, "Two TFTs at nu=0 should always cooperate"


def test_mixture_sampling_distributes_correctly() -> None:
    """With a 50/50 mixture, over many episodes ~50% should be AllD opponent.

    We detect AllD opponent episodes by checking if all of a2 == D.
    (AllC→all a2==C; AllD→all a2==D with sigma=0 agent)
    """
    # Use sigma=0 so the agent defects back at every step; this makes AllD
    # detectable (a2 always D regardless of agent) vs AllC (a2 always C).
    p1 = SoftReciprocalPolicy(sigma=0.0)
    mixture = {AlwaysDefect(): 0.5, AlwaysCooperate(): 0.5}
    rng = _rng(99)
    n_episodes = 400
    alld_count = 0
    for _ in range(n_episodes):
        traj = rollout_ipd_with_mixture(p1, mixture, horizon=10, nu=0.0, rng=rng)
        if (traj.a2 == 1).all():
            alld_count += 1
    rate = alld_count / n_episodes
    assert abs(rate - 0.5) < 0.08, (
        f"Expected ~50% AllD episodes, got {rate:.2%}"
    )
