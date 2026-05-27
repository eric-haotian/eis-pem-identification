"""Ablation study comparing parameter identifiability of full-cell only vs. decoupled models."""

from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eis_pem import (
    DecoupledStackedSEISModel,
    IdentifiabilitySelector,
    LeastSquaresOptimizer,
    StackedSEISModel,
    all_seis_parameter_specs,
    evaluate_local_identifiability,
    generate_synthetic_dataset,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
DATA_DIR = PROJECT_ROOT / "data"


def _perturbed_specs(specs):
    result = []
    for index, spec in enumerate(specs):
        lower, upper = spec.bounds
        margin = (upper - lower) * 1e-12
        value = np.clip(
            spec.initial_value * (1.03 if index % 2 == 0 else 0.97),
            lower + margin,
            upper - margin,
        )
        result.append(replace(spec, initial_value=float(value)))
    return result


def main() -> None:
    specs = all_seis_parameter_specs()
    names = tuple(spec.name for spec in specs)
    theta_true = np.array([spec.initial_value for spec in specs], dtype=float)

    # 5 temperatures x 5 SOCs = 25 conditions
    conditions = tuple(
        (temperature_k, soc)
        for temperature_k in (278.15, 288.15, 298.15, 308.15, 318.15)
        for soc in (0.05, 0.25, 0.50, 0.75, 0.95)
    )

    points_per_channel = 60
    base_freq_hz = np.logspace(-3, 5, points_per_channel)

    # --- 1. Full-cell only Stacked SEIS Model ---
    freq_hz_cell = np.tile(base_freq_hz, len(conditions))
    model_cell = StackedSEISModel(
        conditions=conditions,
        parameter_names=names,
    )
    dataset_cell_noisy = generate_synthetic_dataset(
        model=model_cell,
        freq_hz=freq_hz_cell,
        theta_true=theta_true,
        noise_level=0.005,
        seed=42,
    )
    dataset_cell_clean = generate_synthetic_dataset(
        model=model_cell,
        freq_hz=freq_hz_cell,
        theta_true=theta_true,
        noise_level=0.0,
        seed=42,
    )

    # --- 2. Decoupled Stacked SEIS Model ---
    channels = ("neg", "pos", "sep")
    freq_hz_dec = np.tile(base_freq_hz, len(conditions) * len(channels))
    model_dec = DecoupledStackedSEISModel(
        conditions=conditions,
        response_channels=channels,
        parameter_names=names,
    )
    dataset_dec_noisy = generate_synthetic_dataset(
        model=model_dec,
        freq_hz=freq_hz_dec,
        theta_true=theta_true,
        noise_level=0.005,
        seed=42,
    )
    dataset_dec_clean = generate_synthetic_dataset(
        model=model_dec,
        freq_hz=freq_hz_dec,
        theta_true=theta_true,
        noise_level=0.0,
        seed=42,
    )

    # --- 3. Evaluate Identifiability ---
    report_cell = evaluate_local_identifiability(
        model=model_cell,
        dataset=dataset_cell_clean,
        parameter_specs=specs,
        theta=theta_true,
    )
    report_dec = evaluate_local_identifiability(
        model=model_dec,
        dataset=dataset_dec_clean,
        parameter_specs=specs,
        theta=theta_true,
    )

    # --- 4. Parameter Selection ---
    selector = IdentifiabilitySelector(max_condition_number=1e4)
    selection_cell = selector.select(dataset_cell_clean, model_cell, specs, theta_true)
    selection_dec = selector.select(dataset_dec_clean, model_dec, specs, theta_true)

    # --- 5. Fit & Compare Recovery ---
    print("Fitting Full-cell only model (max_nfev=500)...")
    result_cell = LeastSquaresOptimizer(relative=True, max_nfev=500).fit(
        dataset=dataset_cell_noisy,
        model=model_cell,
        parameter_specs=_perturbed_specs(specs),
    )
    print("Fitting Decoupled model (max_nfev=500)...")
    result_dec = LeastSquaresOptimizer(relative=True, max_nfev=500).fit(
        dataset=dataset_dec_noisy,
        model=model_dec,
        parameter_specs=_perturbed_specs(specs),
    )

    theta_fit_cell = np.array([result_cell.theta_best[n] for n in names])
    theta_fit_dec = np.array([result_dec.theta_best[n] for n in names])

    error_cell = np.abs(theta_fit_cell - theta_true) / theta_true
    error_dec = np.abs(theta_fit_dec - theta_true) / theta_true

    # --- 6. Save Data & Generate Plots ---
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    summary_df = pd.DataFrame(
        {
            "Metric": [
                "Total Parameters",
                "Jacobian Rank",
                "Jacobian Condition Number",
                "Selector Free Parameters Count",
                "Selector Fixed Parameters Count",
                "PEM Final Cost",
                "Median Parameter Error [%]",
                "Max Parameter Error [%]",
            ],
            "Full-Cell Only": [
                len(names),
                report_cell.rank,
                report_cell.condition_number,
                len(selection_cell.free_names),
                len(selection_cell.fixed_names),
                result_cell.final_cost,
                np.median(error_cell) * 100.0,
                np.max(error_cell) * 100.0,
            ],
            "Decoupled (Three-Electrode)": [
                len(names),
                report_dec.rank,
                report_dec.condition_number,
                len(selection_dec.free_names),
                len(selection_dec.fixed_names),
                result_dec.final_cost,
                np.median(error_dec) * 100.0,
                np.max(error_dec) * 100.0,
            ],
        }
    )

    summary_path = DATA_DIR / "ablation_full_cell_vs_decoupled.csv"
    summary_df.to_csv(summary_path, index=False)

    print("\n=== ABLATION STUDY COMPARISON ===")
    print(summary_df.to_string(index=False))
    print(f"Summary written to: {summary_path}")

    # Plot 1: Singular value decay
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.semilogy(
        np.arange(1, report_cell.singular_values.size + 1),
        report_cell.singular_values,
        "o-",
        color="#e31a1c",
        label="Full-Cell Only",
    )
    ax.semilogy(
        np.arange(1, report_dec.singular_values.size + 1),
        report_dec.singular_values,
        "s-",
        color="#1f78b4",
        label="Decoupled (Three-Electrode)",
    )
    ax.set_xlabel("Singular value index")
    ax.set_ylabel("Singular value")
    ax.set_title("Singular Value Spectrum Decay Comparison")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    plot1_path = OUTPUT_DIR / "ablation_cell_vs_decoupled_singular_values.png"
    fig.savefig(plot1_path, dpi=160)
    plt.close(fig)
    print(f"Spectrum plot written to: {plot1_path}")

    # Plot 2: Relative parameter errors comparison
    fig, ax = plt.subplots(figsize=(14.0, 6.0))
    positions = np.arange(len(names))
    width = 0.38
    ax.bar(
        positions - width / 2,
        error_cell * 100.0,
        width,
        color="#e31a1c",
        label="Full-Cell Only",
    )
    ax.bar(
        positions + width / 2,
        error_dec * 100.0,
        width,
        color="#1f78b4",
        label="Decoupled",
    )
    ax.set_xticks(positions, names, rotation=70, ha="right", fontsize=8)
    ax.set_ylabel("Relative parameter error [%]")
    ax.set_title("Parameter Estimation Error Comparison (0.5% Noise)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    plot2_path = OUTPUT_DIR / "ablation_cell_vs_decoupled_errors.png"
    fig.savefig(plot2_path, dpi=160)
    plt.close(fig)
    print(f"Error comparison plot written to: {plot2_path}")


if __name__ == "__main__":
    main()
