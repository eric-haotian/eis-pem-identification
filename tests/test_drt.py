"""Tests for DRT analysis."""

import numpy as np
import pytest

from eis_pem.dataset import EISDataset
from eis_pem.drt import DRTAnalyzer
from eis_pem.ecm_library import TwoRCPEModel

def test_drt_peak_detection():
    # Generate synthetic 2-RC data
    freq = np.logspace(5, -2, 80)
    model = TwoRCPEModel()
    # Distinct time constants: t1 = 0.05*1e-3 = 5e-5s (~3kHz), t2 = 0.05*1 = 0.05s (~3Hz)
    # Balanced resistances so both peaks are prominent
    theta = np.array([0.01, 0.05, 1e-3, 1.0, 0.05, 1.0, 1.0])
    z_obs = model.simulate(freq, theta)
    
    dataset = EISDataset(freq_hz=freq, z_obs=z_obs)
    
    # Use GCV for reliable automatic lambda selection
    analyzer = DRTAnalyzer(n_basis=60, lambda_method="gcv")
    result = analyzer.analyze(dataset)
    
    assert result.n_peaks == 2
    assert not result.has_diffusion_tail
    assert not result.has_inductive
    assert result.r_inf > 0
    
    # Check sum of all resistances approx Rs + R1 + R2 = 0.11
    assert np.isclose(result.total_resistance() + result.r_inf, 0.11, rtol=1e-1)

def test_drt_diffusion_tail():
    # Fake a diffusion tail by adding Warburg-like low freq
    freq = np.logspace(2, -3, 50)
    z_obs = 0.01 + 0.05/(1 + 1j*freq*2*np.pi*1e-2) + 0.01/np.sqrt(1j*freq*2*np.pi)
    
    dataset = EISDataset(freq_hz=freq, z_obs=z_obs)
    analyzer = DRTAnalyzer(n_basis=60, lambda_method="fixed", lambda_fixed=1e-3)
    result = analyzer.analyze(dataset)
    
    assert result.has_diffusion_tail

def test_drt_inductive():
    freq = np.logspace(5, 1, 40)
    # Add inductive part
    z_obs = 0.01 + 1j*freq*2*np.pi*1e-6 + 0.05/(1 + 1j*freq*2*np.pi*1e-3)
    
    dataset = EISDataset(freq_hz=freq, z_obs=z_obs)
    analyzer = DRTAnalyzer(n_basis=40, lambda_method="fixed", lambda_fixed=1e-3)
    result = analyzer.analyze(dataset)
    
    assert result.has_inductive
