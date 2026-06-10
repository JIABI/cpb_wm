"""Fixed opponent policies for Exp 1 mixture evaluation.

Phase A.1 of the v4 brief:
  - AlwaysDefect / AlwaysCooperate: simple stateless opponents.
  - sample_opponent(rng, mixture): draw one opponent from a probability dict.
  - parse_mixture_spec(spec, agent_sigma): parse CLI string like
      "reciprocal:0.8,always_defect:0.2"
    into a {policy_object: weight} dict.

Mixture spec format:
  "<type>[:<weight>][,<type>[:<weight>]]..."

Supported type tokens:
  "always_defect"   → AlwaysDefect()
  "always_cooperate"→ AlwaysCooperate()
  "reciprocal"      → SoftReciprocalPolicy(agent_sigma)

The weight after ":" is the *mixture probability*, NOT a policy parameter.
The agent's sigma is threaded in from the outside (agent_sigma kwarg).
"""
from __future__ import annotations

import numpy as np

from nms.envs.noisy_games import C, D

# ── Fixed opponents ──────────────────────────────────────────────────────────

class AlwaysDefect:
    """Unconditional defector (AllD)."""

    def act(self, last_obs_opponent: int, rng: np.random.Generator) -> int:  # noqa: ARG002
        return D


class AlwaysCooperate:
    """Unconditional cooperator (AllC)."""

    def act(self, last_obs_opponent: int, rng: np.random.Generator) -> int:  # noqa: ARG002
        return C


# ── Mixture sampling ─────────────────────────────────────────────────────────

def sample_opponent(
    rng: np.random.Generator,
    mixture: dict[object, float],
) -> object:
    """Draw one opponent policy from a probability-weighted dict.

    Parameters
    ----------
    rng     : numpy Generator.
    mixture : {policy_object: probability} — probabilities should sum to ~1.

    Returns
    -------
    One of the policy objects (same reference, not a copy).
    """
    policies = list(mixture.keys())
    weights = np.asarray(list(mixture.values()), dtype=float)
    weights = weights / weights.sum()  # normalise in case of rounding
    idx = int(rng.choice(len(policies), p=weights))
    return policies[idx]


def parse_mixture_spec(spec: str, agent_sigma: float) -> dict[object, float]:
    """Parse a mixture specification string into a {policy: weight} dict.

    Parameters
    ----------
    spec         : Comma-separated "<type>:<weight>" tokens, e.g.
                   "reciprocal:0.8,always_defect:0.2"
    agent_sigma  : Forgiveness probability forwarded to SoftReciprocalPolicy
                   when the token type is "reciprocal".

    Returns
    -------
    dict mapping policy instances to their mixture weights.

    Examples
    --------
    >>> d = parse_mixture_spec("reciprocal:0.8,always_defect:0.2", agent_sigma=0.3)
    >>> list(d.values())
    [0.8, 0.2]
    """
    # Import here to avoid circular import (opponents ← noisy_games is fine,
    # but reciprocal imports nothing from opponents so this is safe).
    from nms.policies.reciprocal import SoftReciprocalPolicy  # noqa: PLC0415

    result: dict[object, float] = {}
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if ":" in token:
            type_str, weight_str = token.rsplit(":", 1)
            weight = float(weight_str)
        else:
            type_str = token
            weight = 1.0
        type_str = type_str.strip().lower()

        if type_str == "always_defect":
            policy: object = AlwaysDefect()
        elif type_str == "always_cooperate":
            policy = AlwaysCooperate()
        elif type_str == "reciprocal":
            policy = SoftReciprocalPolicy(sigma=agent_sigma)
        else:
            raise ValueError(
                f"Unknown opponent type {type_str!r}. "
                "Choose from: always_defect, always_cooperate, reciprocal."
            )
        result[policy] = weight

    if not result:
        raise ValueError(f"Empty mixture spec: {spec!r}")

    return result
