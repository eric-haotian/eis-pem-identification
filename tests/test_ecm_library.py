"""Tests for the ECM model library."""

import numpy as np
import pytest

from eis_pem.ecm_library import (
    SingleRCModel,
    SingleRCPEModel,
    TwoRCPEModel,
    RandlesWarburgModel,
    TwoRCPEWarburgModel,
    ThreeRCPEModel,
    all_ecm_models,
    ecm_model_by_name,
    suggest_models_from_peaks,
)

def test_all_ecm_models_available():
    models = all_ecm_models()
    assert len(models) == 6
    names = [m.name for m in models]
    assert "1RC" in names
    assert "3RCPE" in names

def test_ecm_model_by_name():
    model = ecm_model_by_name("2RCPE")
    assert isinstance(model, TwoRCPEModel)
    with pytest.raises(ValueError):
        ecm_model_by_name("NonExistent")

def test_1rc_simulate():
    model = SingleRCModel()
    freq = np.logspace(6, -2, 30)
    theta = np.array([0.01, 0.05, 1e-3])  # Rs, R1, C1
    z = model.simulate(freq, theta)
    assert z.shape == freq.shape
    # High freq limit is Rs
    assert np.isclose(z[0].real, 0.01, rtol=1e-2)
    # Low freq limit is Rs + R1
    assert np.isclose(z[-1].real, 0.06, rtol=1e-2)

def test_cpe_model_simulate():
    model = SingleRCPEModel()
    freq = np.array([1000.0, 1.0])
    theta = np.array([0.01, 0.05, 1e-3, 0.8])  # Rs, R1, Q1, a1
    z = model.simulate(freq, theta)
    assert np.all(np.isfinite(z))
    assert z.imag[0] <= 0  # capacitive

def test_suggest_models_from_peaks():
    # 1 peak, no diffusion
    m1 = suggest_models_from_peaks(1, False, False)
    assert any(isinstance(m, SingleRCPEModel) for m in m1)
    
    # 2 peaks, with diffusion
    m2 = suggest_models_from_peaks(2, True, False)
    assert any(isinstance(m, TwoRCPEWarburgModel) for m in m2)
    assert any(isinstance(m, RandlesWarburgModel) for m in m2)
    
    # 3 peaks, no diffusion
    m3 = suggest_models_from_peaks(3, False, False)
    assert any(isinstance(m, ThreeRCPEModel) for m in m3)
