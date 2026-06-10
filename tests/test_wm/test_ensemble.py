"""Tests for EnsembleRSSM (Phase B)."""
from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from nms.wm.ensemble_rssm import EnsembleRSSM


class _MockSpace:
    shape = (4,)
    low = np.array([-1.0, -1.0, -1.0, -1.0])
    high = np.array([1.0, 1.0, 1.0, 1.0])


class TestEnsembleRSSM:
    def _make(self, k: int = 2) -> EnsembleRSSM:
        return EnsembleRSSM(
            obs_space=_MockSpace(),
            act_space=_MockSpace(),
            k=k,
            device="cpu",
            seed=0,
            config_path="fake_config.yaml",
        )

    def test_k_validation(self) -> None:
        with pytest.raises(ValueError, match="k must be"):
            EnsembleRSSM(_MockSpace(), _MockSpace(), k=0)

    def test_no_members_raises(self) -> None:
        ens = self._make(k=2)
        with pytest.raises(RuntimeError, match="No members"):
            ens.estimate_aleatoric(np.zeros(4))

    def test_make_member_returns_wrapper(self) -> None:
        ens = self._make(k=2)
        m = ens._make_member(0)
        assert hasattr(m, "sample_action")
        assert hasattr(m, "rssm_posterior")

    def test_disagreement_no_members(self) -> None:
        ens = self._make(k=2)
        with pytest.raises(RuntimeError, match="No members"):
            ens.disagreement(np.zeros(4), horizon=5)

    def test_save_manifest(self, tmp_path: Any) -> None:
        import json
        ens = self._make(k=3)
        ens.save_all(tmp_path)
        manifest = json.loads((tmp_path / "ensemble_manifest.json").read_text())
        assert manifest["k"] == 3
        assert manifest["seed"] == 0

    def test_imagine_all_needs_members(self) -> None:
        ens = self._make(k=2)
        with pytest.raises(RuntimeError, match="No members"):
            ens.imagine_all(np.zeros(4), horizon=5)
