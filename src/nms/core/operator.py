from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy.typing as npt

from .envelope import estimate_bandwidth
from .noise_estimation import AleatoricEstimator, NoiseResponseMap
from .projection import project_onto_bandwidth


@dataclass
class NMSOperator:
    """Complete NMS operator mapping history to deployed stochasticity σ_t."""

    aleatoric: AleatoricEstimator
    response_map: NoiseResponseMap
    horizon: int
    alpha: float

    def __call__(
        self,
        history: npt.NDArray[Any],
        risk_estimator: Callable[[float], float],
        sigma_grid: npt.NDArray[Any],
    ) -> float:
        nu_hat = self.aleatoric.estimate(history)
        m_nu = self.response_map(nu_hat)
        bandwidth = estimate_bandwidth(
            sigma_grid=sigma_grid,
            risk_estimator=risk_estimator,
            nu=nu_hat,
            horizon=self.horizon,
            alpha=self.alpha,
        )
        return project_onto_bandwidth(m_nu, bandwidth)
