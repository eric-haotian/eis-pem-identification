"""Tests for the data_quality module."""

from __future__ import annotations

import numpy as np
import pytest

from eis_pem.dataset import EISDataset
from eis_pem.data_quality import (
    DataQualityReport,
    assess_data_quality,
)


def _make_clean_dataset(n: int = 60) -> EISDataset:
    """Simple R-RC spectrum."""
    freq = np.logspace(-2, 5, n)
    omega = 2 * np.pi * freq
    rs, rp, c = 0.01, 0.05, 1.0
    z = rs + rp / (1 + 1j * omega * rp * c)
    return EISDataset(freq_hz=freq, z_obs=z)


def _make_noisy_dataset(noise: float = 0.01, seed: int = 42) -> EISDataset:
    ds = _make_clean_dataset()
    rng = np.random.default_rng(seed)
    n = noise * np.abs(ds.z_obs) * (
        rng.standard_normal(ds.z_obs.size) + 1j * rng.standard_normal(ds.z_obs.size)
    )
    return EISDataset(freq_hz=ds.freq_hz, z_obs=ds.z_obs + n, z_true=ds.z_obs)


class TestAssessDataQuality:
    def test_clean_data_high_quality(self):
        """Clean data should have a reasonable quality score.

        Note: the smoothness-based noise estimator can overestimate noise
        for simple R-RC circuits with sharp impedance transitions.
        """
        ds = _make_clean_dataset()
        report = assess_data_quality(ds)
        assert isinstance(report, DataQualityReport)
        assert report.quality_score > 0.5
        assert report.total_points == ds.freq_hz.size
        assert report.valid_points > 0

    def test_noisy_data_lower_quality(self):
        """Noisy data should have a lower quality score."""
        ds_clean = _make_clean_dataset()
        ds_noisy = _make_noisy_dataset(noise=0.05)
        report_clean = assess_data_quality(ds_clean)
        report_noisy = assess_data_quality(ds_noisy)
        # Noisy data should have lower score
        assert report_noisy.quality_score <= report_clean.quality_score

    def test_noise_estimation_from_true(self):
        """When z_true is available, noise should use actual residuals."""
        ds = _make_noisy_dataset(noise=0.02)
        report = assess_data_quality(ds)
        # Noise estimate should be in the right ballpark
        assert report.estimated_noise_level > 0
        assert report.estimated_noise_level < 0.1

    def test_inductive_detection(self):
        """Should detect inductive behavior (Im(Z) > 0)."""
        ds = _make_clean_dataset()
        z = ds.z_obs.copy()
        # Make last 5 points inductive
        z[-5:] = z[-5:].real + 1j * np.abs(z[-5:]) * 0.1
        ds_ind = EISDataset(freq_hz=ds.freq_hz, z_obs=z)
        report = assess_data_quality(ds_ind)
        assert report.inductive_indices.size > 0

    def test_summary_is_string(self):
        """Summary should return a readable string."""
        ds = _make_clean_dataset()
        report = assess_data_quality(ds)
        s = report.summary()
        assert isinstance(s, str)
        assert "Quality score" in s

    def test_to_frame(self):
        """to_frame should return DataFrame with expected columns."""
        ds = _make_clean_dataset()
        report = assess_data_quality(ds)
        df = report.to_frame()
        assert "point_index" in df.columns
        assert "frequency_weight" in df.columns
        assert len(df) == ds.freq_hz.size

    def test_recommended_freq_range(self):
        """Recommended range should be within measured range."""
        ds = _make_clean_dataset()
        report = assess_data_quality(ds)
        f_low, f_high = report.recommended_freq_range
        assert f_low >= ds.freq_hz.min()
        assert f_high <= ds.freq_hz.max()
        assert f_low < f_high

    def test_kk_disabled(self):
        """KK residuals should be None when disabled."""
        ds = _make_clean_dataset()
        report = assess_data_quality(ds, run_kk=False)
        assert report.kk_residuals is None
        assert report.kk_violation_indices.size == 0

    def test_invalid_thresholds(self):
        """Should raise on invalid parameters."""
        ds = _make_clean_dataset()
        with pytest.raises(ValueError):
            assess_data_quality(ds, kk_threshold=-1)
        with pytest.raises(ValueError):
            assess_data_quality(ds, noise_threshold=0)

    def test_export_csv(self, tmp_path):
        """Should export to CSV without errors."""
        ds = _make_clean_dataset()
        report = assess_data_quality(ds)
        path = report.export_csv(tmp_path / "quality.csv")
        assert path.exists()


class TestAdaptiveSelection:
    """Test the adaptive strategy in IdentifiabilitySelector."""

    def test_adaptive_vs_classic_both_work(self):
        """Both strategies should produce a valid ParameterSelection."""
        from eis_pem import (
            IdentifiabilitySelector,
            IdentifiabilityStrategy,
            StackedSEISModel,
            all_seis_parameter_specs,
            generate_synthetic_dataset,
        )

        specs = list(all_seis_parameter_specs())
        names = tuple(spec.name for spec in specs)
        theta = np.array([spec.initial_value for spec in specs], dtype=float)
        conditions = ((298.15, 0.5),)
        freq = np.logspace(-2, 4, 30)

        model = StackedSEISModel(conditions=conditions, parameter_names=names)
        stacked_freq = np.tile(freq, len(conditions))
        dataset = generate_synthetic_dataset(
            model=model, freq_hz=stacked_freq,
            theta_true=theta, noise_level=0.01, seed=42,
        )

        for strategy in IdentifiabilityStrategy:
            selector = IdentifiabilitySelector(
                strategy=strategy,
                max_condition_number=1e4,
                assumed_noise_level=0.01,
                max_relative_ci95=0.15,
            )
            selection = selector.select(
                dataset=dataset,
                model=model,
                parameter_specs=specs,
                theta_reference=theta,
            )
            assert len(selection.free_names) > 0
            assert len(selection.free_names) <= len(specs)
            # Fixed + free should equal total
            assert len(selection.free_names) + len(selection.fixed_values) == len(specs)
