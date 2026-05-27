"""Diagnostic figure generation for an EIS identification result."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from .dataset import EISDataset
from .diagnostics import IdentifiabilityReport
from .robust import ParameterSelection
from .results import IdentificationResult


def save_diagnostic_plots(
    dataset: EISDataset,
    result: IdentificationResult,
    output_dir: str | Path,
    filename_prefix: str = "",
    impedance_unit: str = "ohm",
) -> dict[str, Path]:
    """Save Nyquist, Bode and residual plots for the fitted spectrum."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    paths = {
        "nyquist_fit": output_path / f"{filename_prefix}nyquist_fit.png",
        "bode_magnitude_fit": output_path / f"{filename_prefix}bode_magnitude_fit.png",
        "bode_phase_fit": output_path / f"{filename_prefix}bode_phase_fit.png",
        "residuals": output_path / f"{filename_prefix}residuals.png",
    }
    order = np.argsort(dataset.freq_hz)
    freq = dataset.freq_hz[order]
    observed = dataset.z_obs[order]
    fitted = result.z_fit[order]
    truth = dataset.z_true[order] if dataset.z_true is not None else None

    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    ax.scatter(observed.real, -observed.imag, s=18, alpha=0.65, label="Observed")
    ax.plot(fitted.real, -fitted.imag, linewidth=2.0, label="PEM fit")
    if truth is not None:
        ax.plot(truth.real, -truth.imag, "--", linewidth=1.5, label="True")
    ax.set_xlabel(f"Re(Z) [{impedance_unit}]")
    ax.set_ylabel(f"-Im(Z) [{impedance_unit}]")
    ax.set_title("Nyquist Fit")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(paths["nyquist_fit"], dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.8, 4.5))
    ax.loglog(freq, np.abs(observed), "o", markersize=3.5, alpha=0.65, label="Observed")
    ax.loglog(freq, np.abs(fitted), linewidth=2.0, label="PEM fit")
    if truth is not None:
        ax.loglog(freq, np.abs(truth), "--", linewidth=1.5, label="True")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel(f"|Z| [{impedance_unit}]")
    ax.set_title("Bode Magnitude Fit")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(paths["bode_magnitude_fit"], dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.8, 4.5))
    ax.semilogx(
        freq,
        np.angle(observed, deg=True),
        "o",
        markersize=3.5,
        alpha=0.65,
        label="Observed",
    )
    ax.semilogx(freq, np.angle(fitted, deg=True), linewidth=2.0, label="PEM fit")
    if truth is not None:
        ax.semilogx(freq, np.angle(truth, deg=True), "--", linewidth=1.5, label="True")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Phase [degree]")
    ax.set_title("Bode Phase Fit")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(paths["bode_phase_fit"], dpi=160)
    plt.close(fig)

    residuals = result.residuals[order]
    fig, axes = plt.subplots(2, 1, figsize=(6.8, 6.0), sharex=True)
    axes[0].semilogx(freq, residuals.real, "o-", markersize=3.0)
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_ylabel(f"Real residual [{impedance_unit}]")
    axes[1].semilogx(freq, residuals.imag, "o-", markersize=3.0)
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_xlabel("Frequency [Hz]")
    axes[1].set_ylabel(f"Imag residual [{impedance_unit}]")
    fig.suptitle("Prediction Residuals")
    for ax in axes:
        ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(paths["residuals"], dpi=160)
    plt.close(fig)

    return paths


def save_robust_selection_plots(
    result: IdentificationResult,
    selection: ParameterSelection,
    output_dir: str | Path,
    impedance_unit: str = "ohm*m^2",
) -> dict[str, Path]:
    """Save uncertainty, singular-value and residual diagnostics for robust fits."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    paths = {
        "parameter_uncertainty": output_path
        / "robust_all_seis_parameter_uncertainty.png",
        "singular_values": output_path / "robust_all_seis_singular_values.png",
        "residuals": output_path / "robust_all_seis_residuals.png",
    }
    names = selection.parameter_names
    statuses = [selection.statuses[name] for name in names]
    ci95_percent = 100.0 * np.asarray(
        [selection.ci95_relative[name] for name in names], dtype=float
    )
    displayed_ci95 = np.nan_to_num(ci95_percent, nan=0.0)
    colors = [
        {
            "estimated": "#2878b5",
            "fixed_identifiability": "#b0b0b0",
            "protected_high_uncertainty": "#d95f02",
        }[status]
        for status in statuses
    ]

    fig, ax = plt.subplots(figsize=(14.0, 5.5))
    positions = np.arange(len(names))
    ax.bar(positions, displayed_ci95, color=colors)
    ax.axhline(
        100.0 * selection.max_relative_ci95,
        color="#b22222",
        linestyle="--",
        linewidth=1.2,
        label="CI threshold",
    )
    ax.set_xticks(positions, names, rotation=70, ha="right", fontsize=8)
    ax.set_ylabel("Predicted 95% relative interval [%]")
    ax.set_title("Identifiability-Aware Parameter Selection")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(
        handles=[
            Patch(color="#2878b5", label="Estimated"),
            Patch(color="#b0b0b0", label="Fixed by identifiability"),
            Patch(color="#d95f02", label="Protected high uncertainty"),
        ]
    )
    fig.tight_layout()
    fig.savefig(paths["parameter_uncertainty"], dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    ax.semilogy(
        np.arange(1, selection.original_singular_values.size + 1),
        selection.original_singular_values,
        "o-",
        markersize=3.5,
        label="All parameters",
    )
    ax.semilogy(
        np.arange(1, selection.selected_singular_values.size + 1),
        selection.selected_singular_values,
        "o-",
        markersize=3.5,
        label="Selected free parameters",
    )
    ax.set_xlabel("Singular value index")
    ax.set_ylabel("Singular value")
    ax.set_title(
        "Jacobian Spectrum "
        f"(condition {selection.original_condition_number:.2e} to "
        f"{selection.reduced_condition_number:.2e})"
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(paths["singular_values"], dpi=160)
    plt.close(fig)

    scale = np.maximum(np.abs(result.dataset.z_obs), 1e-12)
    normalized_residuals = result.residuals / scale
    sample_index = np.arange(result.dataset.freq_hz.size)
    fig, axes = plt.subplots(2, 1, figsize=(9.0, 6.0), sharex=True)
    axes[0].plot(sample_index, normalized_residuals.real, linewidth=0.8)
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_ylabel("Relative real residual")
    axes[1].plot(sample_index, normalized_residuals.imag, linewidth=0.8)
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_xlabel("Stacked EIS sample index")
    axes[1].set_ylabel("Relative imag residual")
    fig.suptitle(f"Robust PEM Residuals [{impedance_unit}]")
    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(paths["residuals"], dpi=160)
    plt.close(fig)

    return paths


def save_joint_identification_plots(
    result: IdentificationResult,
    report: IdentifiabilityReport,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Save singular-value and residual diagnostics for joint identification."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    paths = {
        "singular_values": output_path / "joint_all_seis_singular_values.png",
        "residuals": output_path / "joint_all_seis_residuals.png",
    }

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    ax.semilogy(
        np.arange(1, report.singular_values.size + 1),
        report.singular_values,
        "o-",
        markersize=3.5,
    )
    ax.set_xlabel("Singular value index")
    ax.set_ylabel("Singular value")
    ax.set_title(f"Joint Jacobian Spectrum (condition {report.condition_number:.2e})")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(paths["singular_values"], dpi=160)
    plt.close(fig)

    scale = np.maximum(np.abs(result.dataset.z_obs), 1e-12)
    normalized_residuals = result.residuals / scale
    sample_index = np.arange(result.dataset.freq_hz.size)
    fig, axes = plt.subplots(2, 1, figsize=(9.0, 6.0), sharex=True)
    axes[0].plot(sample_index, normalized_residuals.real, linewidth=0.8)
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_ylabel("Relative real residual")
    axes[1].plot(sample_index, normalized_residuals.imag, linewidth=0.8)
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_xlabel("Stacked decoupled EIS sample index")
    axes[1].set_ylabel("Relative imag residual")
    fig.suptitle("Joint Decoupled SEIS Prediction Residuals")
    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(paths["residuals"], dpi=160)
    plt.close(fig)

    return paths
