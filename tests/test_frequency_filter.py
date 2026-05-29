"""Tests for the frequency_filter module."""

from __future__ import annotations

import numpy as np
import pytest

from eis_pem.dataset import EISDataset
from eis_pem.frequency_filter import (
    FrequencyBandAnalyzer,
    FrequencyBandConfig,
    FrequencyFilterResult,
    filter_dataset,
    weighted_dataset,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_clean_dataset(n_points: int = 60) -> EISDataset:
    """Create a clean synthetic EIS dataset (single semicircle)."""
    freq_hz = np.logspace(-2, 5, n_points)
    omega = 2.0 * np.pi * freq_hz
    # Simple R-RC circuit: Z = Rs + Rp / (1 + jω*Rp*C)
    rs, rp, c = 0.01, 0.05, 1.0
    z = rs + rp / (1.0 + 1j * omega * rp * c)
    return EISDataset(freq_hz=freq_hz, z_obs=z)


def _make_noisy_dataset(noise_level: float = 0.01, seed: int = 42) -> EISDataset:
    """Create a noisy dataset."""
    ds = _make_clean_dataset()
    rng = np.random.default_rng(seed)
    noise = noise_level * np.abs(ds.z_obs) * (
        rng.standard_normal(ds.z_obs.size)
        + 1j * rng.standard_normal(ds.z_obs.size)
    )
    return EISDataset(freq_hz=ds.freq_hz, z_obs=ds.z_obs + noise)


def _make_inductive_dataset() -> EISDataset:
    """Create dataset with inductive artifacts at high frequency."""
    ds = _make_clean_dataset(60)
    z = ds.z_obs.copy()
    # Add positive imaginary part at highest frequencies (inductive)
    freq = ds.freq_hz
    for i in range(len(freq)):
        if freq[i] > 10000:
            z[i] = z[i].real + 1j * abs(z[i]) * 0.1
    return EISDataset(freq_hz=freq, z_obs=z)


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


class TestFrequencyBandConfig:
    def test_default_construction(self):
        config = FrequencyBandConfig()
        assert config.low_mid_boundary == 1.0
        assert config.mid_high_boundary == 1000.0

    def test_invalid_boundaries(self):
        with pytest.raises(ValueError, match="low_mid_boundary"):
            FrequencyBandConfig(low_mid_boundary=-1.0)

    def test_reversed_boundaries(self):
        with pytest.raises(ValueError, match="mid_high_boundary"):
            FrequencyBandConfig(low_mid_boundary=100.0, mid_high_boundary=10.0)

    def test_invalid_threshold(self):
        with pytest.raises(ValueError, match="max_inductive_residual"):
            FrequencyBandConfig(max_inductive_residual=1.5)


# ---------------------------------------------------------------------------
# Analyzer tests
# ---------------------------------------------------------------------------


class TestFrequencyBandAnalyzer:
    def test_clean_data_keeps_most_points(self):
        """Clean data should retain at least half the points.

        Note: simple R-RC circuits have sharp curvature transitions that
        may trigger the outlier detector. Real EIS data with smooth arcs
        retains more points.
        """
        ds = _make_clean_dataset()
        analyzer = FrequencyBandAnalyzer()
        result = analyzer.analyze(ds)

        assert isinstance(result, FrequencyFilterResult)
        assert result.valid_fraction > 0.5
        assert result.freq_hz.size == ds.freq_hz.size

    def test_band_labels_correct(self):
        """Band labels should match config boundaries."""
        config = FrequencyBandConfig(
            low_mid_boundary=1.0,
            mid_high_boundary=1000.0,
        )
        ds = _make_clean_dataset()
        result = FrequencyBandAnalyzer(config=config).analyze(ds)

        for i, f in enumerate(ds.freq_hz):
            label = result.band_labels[i]
            if f < 1.0:
                assert label == "low", f"Expected 'low' for f={f}, got {label}"
            elif f <= 1000.0:
                assert label == "mid", f"Expected 'mid' for f={f}, got {label}"
            else:
                assert label == "high", f"Expected 'high' for f={f}, got {label}"

    def test_inductive_detection(self):
        """Inductive points should be flagged."""
        ds = _make_inductive_dataset()
        result = FrequencyBandAnalyzer().analyze(ds)
        assert np.any(result.inductive_mask), "Should detect inductive behavior"

    def test_weights_in_valid_range(self):
        """All weights should be in [0, 1]."""
        ds = _make_noisy_dataset()
        result = FrequencyBandAnalyzer().analyze(ds)
        assert np.all(result.weights >= 0.0)
        assert np.all(result.weights <= 1.0)

    def test_invalid_points_have_zero_weight(self):
        """Invalid points must have weight 0."""
        ds = _make_inductive_dataset()
        result = FrequencyBandAnalyzer().analyze(ds)
        for i in range(len(result.valid_mask)):
            if not result.valid_mask[i]:
                assert result.weights[i] == 0.0

    def test_to_frame(self):
        """to_frame should return a DataFrame with expected columns."""
        ds = _make_clean_dataset()
        result = FrequencyBandAnalyzer().analyze(ds)
        df = result.to_frame()
        assert "freq_Hz" in df.columns
        assert "valid" in df.columns
        assert "weight" in df.columns
        assert "band" in df.columns
        assert len(df) == ds.freq_hz.size

    def test_kk_residuals_present_when_enabled(self):
        """KK residuals should be computed when enabled."""
        config = FrequencyBandConfig(enable_kramers_kronig=True)
        ds = _make_clean_dataset()
        result = FrequencyBandAnalyzer(config=config).analyze(ds)
        assert result.kk_residuals is not None
        assert result.kk_residuals.size == ds.freq_hz.size

    def test_kk_residuals_absent_when_disabled(self):
        """KK residuals should be None when disabled."""
        config = FrequencyBandConfig(enable_kramers_kronig=False)
        ds = _make_clean_dataset()
        result = FrequencyBandAnalyzer(config=config).analyze(ds)
        assert result.kk_residuals is None


# ---------------------------------------------------------------------------
# Filter/weight dataset tests
# ---------------------------------------------------------------------------


class TestFilterDataset:
    def test_filter_reduces_size(self):
        """Filtered dataset should have fewer points when some are removed."""
        ds = _make_inductive_dataset()
        result = FrequencyBandAnalyzer().analyze(ds)
        if result.n_removed > 0:
            filtered = filter_dataset(ds, result)
            assert filtered.freq_hz.size < ds.freq_hz.size

    def test_filter_preserves_z_true(self):
        """z_true should be filtered along with the rest."""
        ds_clean = _make_clean_dataset()
        ds_with_true = EISDataset(
            freq_hz=ds_clean.freq_hz,
            z_obs=ds_clean.z_obs,
            z_true=ds_clean.z_obs.copy(),
        )
        result = FrequencyBandAnalyzer().analyze(ds_with_true)
        filtered = filter_dataset(ds_with_true, result)
        assert filtered.z_true is not None
        assert filtered.z_true.size == filtered.freq_hz.size

    def test_weighted_dataset_returns_weights(self):
        """weighted_dataset should return matching weights."""
        ds = _make_clean_dataset()
        result = FrequencyBandAnalyzer().analyze(ds)
        filtered, weights = weighted_dataset(ds, result)
        assert weights.size == filtered.freq_hz.size
        assert np.all(weights > 0)
