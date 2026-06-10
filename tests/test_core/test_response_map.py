"""Tests for response_map.py (Phase B)."""
from __future__ import annotations

import numpy as np
import pytest

from nms.core.response_map import (
    best_fit_response_map,
    fit_linear_response,
    fit_response_map,
)

# ── fit_response_map ──────────────────────────────────────────────────────────

def test_fit_linear_perfect() -> None:
    """σ* = 2·ν exactly → R² = 1.0, c ≈ 2.0."""
    nu = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
    sigma_star = 2.0 * nu
    m = fit_response_map("linear", nu, sigma_star)
    assert m.params[0] == pytest.approx(2.0, abs=1e-9)
    assert m.r2_train == pytest.approx(1.0, abs=1e-6)
    assert m.form == "linear"


def test_fit_sqrt_perfect() -> None:
    """σ* = 3·√ν exactly → R² = 1.0, c ≈ 3.0."""
    nu = np.array([0.01, 0.04, 0.09, 0.16, 0.25])
    sigma_star = 3.0 * np.sqrt(nu)
    m = fit_response_map("sqrt", nu, sigma_star)
    assert m.params[0] == pytest.approx(3.0, abs=1e-6)
    assert m.r2_train == pytest.approx(1.0, abs=1e-5)


def test_fit_log_perfect() -> None:
    """σ* = 1.5·log(1+ν) exactly → R² = 1.0."""
    nu = np.array([0.0, 0.1, 0.5, 1.0])
    sigma_star = 1.5 * np.log1p(nu)
    m = fit_response_map("log", nu, sigma_star)
    assert m.params[0] == pytest.approx(1.5, abs=1e-9)
    assert m.r2_train == pytest.approx(1.0, abs=1e-6)


def test_fit_linear_response_alias() -> None:
    """fit_linear_response is a shortcut for fit_response_map('linear',...)."""
    nu = np.array([0.1, 0.2, 0.3])
    sigma_star = np.array([0.1, 0.2, 0.3])
    m1 = fit_linear_response(nu, sigma_star)
    m2 = fit_response_map("linear", nu, sigma_star)
    assert m1.params == m2.params
    assert m1.form == m2.form


def test_unknown_form_raises() -> None:
    with pytest.raises(ValueError, match="Unknown form"):
        fit_response_map("polynomial", np.array([0.1]), np.array([0.1]))  # type: ignore[arg-type]


def test_fit_with_test_data_computes_r2_test() -> None:
    """When test data is provided, r2_test reflects test performance."""
    nu_tr = np.array([0.1, 0.2, 0.3])
    ss_tr = 2.0 * nu_tr
    nu_te = np.array([0.15, 0.25])
    ss_te = 2.0 * nu_te  # exact → R²_test = 1.0
    m = fit_response_map("linear", nu_tr, ss_tr, nu_te, ss_te)
    assert m.r2_test == pytest.approx(1.0, abs=1e-6)
    assert set(m.test_nus) == {0.15, 0.25}


def test_fitted_map_callable() -> None:
    """FittedResponseMap is callable: m(ν) returns a float."""
    m = fit_response_map("linear", np.array([0.1, 0.2]), np.array([0.2, 0.4]))
    assert isinstance(m(0.3), float)
    assert m(0.0) == pytest.approx(0.0, abs=1e-9)


def test_train_nus_stored_correctly() -> None:
    nu = np.array([0.05, 0.15, 0.30])
    sigma_star = nu * 1.5
    m = fit_response_map("linear", nu, sigma_star)
    assert set(m.train_nus) == {0.05, 0.15, 0.30}


# ── best_fit_response_map ─────────────────────────────────────────────────────

def test_best_fit_selects_linear_for_linear_data() -> None:
    """Linear data → best_fit should pick 'linear' (highest R²)."""
    nu = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    sigma_star = 1.8 * nu
    m = best_fit_response_map(nu, sigma_star)
    assert m.form == "linear"
    assert m.r2_train > 0.99


def test_best_fit_selects_sqrt_for_sqrt_data() -> None:
    """Sqrt data → best_fit should pick 'sqrt' over linear/log."""
    nu = np.array([0.01, 0.04, 0.09, 0.16, 0.25, 0.36])
    sigma_star = 2.5 * np.sqrt(nu)
    m = best_fit_response_map(nu, sigma_star)
    assert m.form == "sqrt"
    assert m.r2_train > 0.99


def test_best_fit_with_test_data_uses_test_r2() -> None:
    """Selection criterion is test R², not train R², when test data is present."""
    # Train: perfectly sqrt data
    nu_tr = np.array([0.01, 0.04, 0.09, 0.16])
    ss_tr = 2.0 * np.sqrt(nu_tr)
    # Test: also sqrt → sqrt should win over linear by higher test R²
    nu_te = np.array([0.25, 0.36])
    ss_te = 2.0 * np.sqrt(nu_te)
    m = best_fit_response_map(nu_tr, ss_tr, nu_te, ss_te)
    assert m.r2_test > 0.99
