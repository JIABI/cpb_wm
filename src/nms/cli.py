from __future__ import annotations

import argparse

import numpy as np

from nms.core.envelope import estimate_bandwidth
from nms.core.noise_estimation import linear_response
from nms.core.operator import NMSOperator


class FixedAleatoric:
    def estimate(self, history: np.ndarray) -> float:
        del history
        return 0.1


def _risk_estimator(sigma: float) -> float:
    return 0.05 + (sigma - 0.1) ** 2


def demo() -> None:
    sigma_grid = np.linspace(0.0, 0.3, 31)
    operator = NMSOperator(
        aleatoric=FixedAleatoric(), response_map=linear_response(1.0), horizon=100, alpha=0.1
    )
    history = np.zeros((1,), dtype=float)
    sigma_star = operator(
        history=history, risk_estimator=_risk_estimator, sigma_grid=sigma_grid
    )
    bw = estimate_bandwidth(sigma_grid, _risk_estimator, nu=0.1, horizon=100, alpha=0.1)
    print(
        f"σ^NMS = {sigma_star:.4f} "
        f"(ν=0.10, m(ν)=0.10, bandwidth=[{bw.sigma_min:.2f}, {bw.sigma_max:.2f}])"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="NMS CLI")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("demo", help="run a tiny end-to-end NMS operator demo")
    args = parser.parse_args()
    if args.command == "demo":
        demo()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
