"""NMS baselines for Exp 2.

Exports
-------
DreamerVariant              : Base wrapper for DreamerV3 with fixed temperature.
make_dreamer_default        : τ=1.0 baseline.
make_dreamer_fixed          : Fixed τ baseline.
AutotunedDreamer            : SAC-style entropy autotuning.
PPOEntropyBaseline          : PPO with entropy bonus.
NoisyNetWrapper             : NoisyNet exploration wrapper.
EnsembleDisagreementBaseline: Ensemble disagreement intrinsic reward.
RNDWrapper                  : Random Network Distillation wrapper.
"""
from nms.baselines.dreamer_variants import (
    AutotunedDreamer,
    DreamerVariant,
    make_dreamer_default,
    make_dreamer_fixed,
)
from nms.baselines.ensemble_disagreement import EnsembleDisagreementBaseline
from nms.baselines.noisy_net_wrapper import NoisyNetWrapper
from nms.baselines.ppo_entropy import PPOEntropyBaseline
from nms.baselines.rnd_wrapper import RNDWrapper

__all__ = [
    "DreamerVariant",
    "make_dreamer_default",
    "make_dreamer_fixed",
    "AutotunedDreamer",
    "PPOEntropyBaseline",
    "NoisyNetWrapper",
    "EnsembleDisagreementBaseline",
    "RNDWrapper",
]
