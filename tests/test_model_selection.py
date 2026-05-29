"""Tests for model selection metrics and comparison."""

import numpy as np
import pytest

from eis_pem.dataset import EISDataset
from eis_pem.ecm_library import SingleRCModel, TwoRCPEModel
from eis_pem.model_selection import (
    compare_models,
    compute_aic,
    compute_aicc,
    compute_bic,
    durbin_watson,
    f_test_nested,
    ljung_box_p_value,
)

def test_information_criteria():
    n = 100
    k = 5
    rss = 1.0
    
    aic = compute_aic(n, k, rss)
    aicc = compute_aicc(n, k, rss)
    bic = compute_bic(n, k, rss)
    
    assert aic < bic  # BIC penalizes more for n=100
    assert aic < aicc # AICc has extra penalty term
    assert np.isfinite(aic)

def test_residual_diagnostics():
    # White noise
    np.random.seed(42)
    white_noise = np.random.normal(0, 1, 100)
    
    dw = durbin_watson(white_noise)
    assert 1.5 < dw < 2.5  # Ideal is 2.0
    
    lb_p = ljung_box_p_value(white_noise)
    assert lb_p > 0.05  # Fail to reject null hypothesis of whiteness

    # Autocorrelated noise
    correlated = np.cumsum(white_noise)
    dw_corr = durbin_watson(correlated)
    assert dw_corr < 1.5  # Positive autocorrelation
    
    lb_p_corr = ljung_box_p_value(correlated)
    assert lb_p_corr < 0.05  # Reject null, it's not white

def test_f_test():
    n = 100
    rss_simple = 10.0
    k_simple = 3
    
    # Significant improvement
    rss_complex1 = 5.0
    k_complex1 = 4
    p1 = f_test_nested(rss_simple, rss_complex1, k_simple, k_complex1, n)
    assert p1 < 0.05
    
    # Insignificant improvement
    rss_complex2 = 9.9
    p2 = f_test_nested(rss_simple, rss_complex2, k_simple, k_complex1, n)
    assert p2 > 0.05

def test_compare_models():
    # Generate data with 2RCPE
    freq = np.logspace(4, -2, 40)
    model_true = TwoRCPEModel()
    theta_true = np.array([0.01, 0.01, 1e-3, 0.9, 0.05, 1.0, 0.8])
    z_clean = model_true.simulate(freq, theta_true)
    
    # Add noise
    np.random.seed(42)
    noise = np.random.normal(0, 0.001, z_clean.shape) + 1j * np.random.normal(0, 0.001, z_clean.shape)
    z_obs = z_clean + noise
    
    dataset = EISDataset(freq_hz=freq, z_obs=z_obs)
    
    candidates = [SingleRCModel(), TwoRCPEModel()]
    
    comparison = compare_models(dataset, candidates, relative=False, max_nfev=500)
    
    assert len(comparison.candidates) == 2
    # The true model (2RCPE) should win
    assert comparison.best_by_aicc == "2RCPE"
    assert comparison.best_by_bic == "2RCPE"
    
    # F-test should show 2RCPE is significantly better than 1RC
    p_val = comparison.f_test_results.get(("1RC", "2RCPE"))
    assert p_val is not None
    assert p_val < 0.05
