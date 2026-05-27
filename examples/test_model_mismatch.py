"""Model mismatch test comparing matched fit and mismatched fit (ignoring R_contact/L_ind)."""

from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eis_pem import (
    DecoupledStackedSEISModel,
    LeastSquaresOptimizer,
    all_seis_parameter_specs,
    generate_synthetic_dataset,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
DATA_DIR = PROJECT_ROOT / "data"


class MismatchedSEISModel:
    """Wrapper that forces R_contact and L_ind to 0.0 to simulate model mismatch."""

    def __init__(self, base_model: DecoupledStackedSEISModel):
        self.base_model = base_model
        # Parameter names excluding R_contact and L_ind
        self.parameter_names = tuple(
            name
            for name in base_model.parameter_names
            if name not in ("R_contact", "L_ind")
        )

    def simulate(self, freq_hz: np.ndarray, theta: np.ndarray) -> np.ndarray:
        theta_dict = dict(zip(self.parameter_names, theta, strict=True))
        # Force contact resistance and inductance to practically zero
        theta_dict["R_contact"] = 1e-20
        theta_dict["L_ind"] = 1e-20

        full_theta = np.array(
            [theta_dict[name] for name in self.base_model.parameter_names], dtype=float
        )
        return self.base_model.simulate(freq_hz, full_theta)


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

    channels = ("neg", "pos", "sep")
    points_per_channel = 60
    base_freq_hz = np.logspace(-3, 5, points_per_channel)
    freq_hz = np.tile(base_freq_hz, len(conditions) * len(channels))

    # Full matched model
    model_matched = DecoupledStackedSEISModel(
        conditions=conditions,
        response_channels=channels,
        parameter_names=names,
    )

    # Generate synthetic observations with R_contact = 0.01 and L_ind = 1e-8
    print("Generating synthetic data with R_contact and L_ind active...")
    dataset_noisy = generate_synthetic_dataset(
        model=model_matched,
        freq_hz=freq_hz,
        theta_true=theta_true,
        noise_level=0.005,
        seed=42,
    )

    # 1. Fit with Matched Model (fits all parameters, including R_contact/L_ind)
    print("Running Matched Fit (max_nfev=500)...")
    result_matched = LeastSquaresOptimizer(relative=True, max_nfev=500).fit(
        dataset=dataset_noisy,
        model=model_matched,
        parameter_specs=_perturbed_specs(specs),
    )

    # 2. Fit with Mismatched Model (excludes R_contact and L_ind from specs, forces to 0.0)
    print("Running Mismatched Fit (max_nfev=500)...")
    model_mismatched = MismatchedSEISModel(model_matched)
    mismatched_specs = [
        spec for spec in specs if spec.name not in ("R_contact", "L_ind")
    ]
    result_mismatched = LeastSquaresOptimizer(relative=True, max_nfev=500).fit(
        dataset=dataset_noisy,
        model=model_mismatched,
        parameter_specs=_perturbed_specs(mismatched_specs),
    )

    # 3. Analyze Mismatch Impact
    names_mismatched = tuple(spec.name for spec in mismatched_specs)
    theta_true_mismatched = np.array(
        [
            spec.initial_value
            for spec in specs
            if spec.name not in ("R_contact", "L_ind")
        ]
    )

    theta_fit_matched = np.array(
        [result_matched.theta_best[n] for n in names_mismatched]
    )
    theta_fit_mismatched = np.array(
        [result_mismatched.theta_best[n] for n in names_mismatched]
    )

    error_matched = (
        np.abs(theta_fit_matched - theta_true_mismatched) / theta_true_mismatched
    )
    error_mismatched = (
        np.abs(theta_fit_mismatched - theta_true_mismatched) / theta_true_mismatched
    )

    # Save results summary
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    mismatch_summary = pd.DataFrame(
        {
            "Metric": [
                "Fitted Parameter Count",
                "Final PEM Cost",
                "Median Parameter Error [%]",
                "Max Parameter Error [%]",
            ],
            "Matched Model (with R_contact & L_ind)": [
                len(specs),
                result_matched.final_cost,
                np.median(error_matched) * 100.0,
                np.max(error_matched) * 100.0,
            ],
            "Mismatched Model (forces R_contact & L_ind = 0)": [
                len(mismatched_specs),
                result_mismatched.final_cost,
                np.median(error_mismatched) * 100.0,
                np.max(error_mismatched) * 100.0,
            ],
        }
    )

    summary_path = DATA_DIR / "model_mismatch_results.csv"
    mismatch_summary.to_csv(summary_path, index=False)
    print("\n=== MODEL MISMATCH COMPARISON ===")
    print(mismatch_summary.to_string(index=False))

    # Print worst affected parameters
    worst_mismatched_indices = np.argsort(error_mismatched)[::-1][:5]
    print("\nTop 5 worst mismatched parameter estimation errors:")
    for idx in worst_mismatched_indices:
        p_name = names_mismatched[idx]
        print(
            f"  {p_name:<15}: Mismatched Error = {error_mismatched[idx]:>8.2%}, Matched Error = {error_matched[idx]:>8.2%}"
        )

    # Plot 1: Mismatched parameter errors bar chart
    fig, ax = plt.subplots(figsize=(14.0, 6.0))
    positions = np.arange(len(names_mismatched))
    width = 0.38
    ax.bar(
        positions - width / 2,
        error_matched * 100.0,
        width,
        color="#2ca02c",
        label="Matched Model",
    )
    ax.bar(
        positions + width / 2,
        error_mismatched * 100.0,
        width,
        color="#d62728",
        label="Mismatched Model (ignoring contact)",
    )
    ax.set_xticks(positions, names_mismatched, rotation=70, ha="right", fontsize=8)
    ax.set_ylabel("Relative parameter error [%]")
    ax.set_title("Impact of Model Mismatch on Parameter Estimation Errors")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    plot1_path = OUTPUT_DIR / "mismatch_parameter_errors.png"
    fig.savefig(plot1_path, dpi=160)
    plt.close(fig)
    print(f"Mismatch error plot written to: {plot1_path}")

    # Plot 2: Nyquist comparison showing mismatch at high frequency
    # We choose a single condition (e.g. 298.15 K, 0.50 SOC, channel "neg") to plot
    # and compare the fitted curves.
    cond_temp, cond_soc = 298.15, 0.50
    plot_freq = np.logspace(-3, 5, 100)

    # Simulate single point curves
    from eis_pem.seis_model import SEISComponentModel

    # For matched model
    matched_comp_model = SEISComponentModel(
        temperature_k=cond_temp,
        soc=cond_soc,
        response_channel="neg",
        parameter_names=names,
    )
    z_true = matched_comp_model.simulate(plot_freq, theta_true)
    z_fit_matched = matched_comp_model.simulate(
        plot_freq, np.array([result_matched.theta_best[n] for n in names])
    )

    # For mismatched model
    theta_best_mismatched = {
        n: result_mismatched.theta_best[n] for n in names_mismatched
    }
    theta_best_mismatched["R_contact"] = 1e-20
    theta_best_mismatched["L_ind"] = 1e-20
    z_fit_mismatched = matched_comp_model.simulate(
        plot_freq, np.array([theta_best_mismatched[n] for n in names])
    )

    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    ax.scatter(
        z_true.real,
        -z_true.imag,
        s=20,
        facecolors="none",
        edgecolors="k",
        label="True (with contact terms)",
    )
    ax.plot(
        z_fit_matched.real,
        -z_fit_matched.imag,
        color="#2ca02c",
        linewidth=2.0,
        label="Matched PEM fit",
    )
    ax.plot(
        z_fit_mismatched.real,
        -z_fit_mismatched.imag,
        color="#d62728",
        linestyle="--",
        linewidth=2.0,
        label="Mismatched PEM fit",
    )
    ax.set_xlabel("Re(Z) [ohm*m^2]")
    ax.set_ylabel("-Im(Z) [ohm*m^2]")
    ax.set_title("Nyquist Mismatch Comparison (Neg Electrode, 298.15K, 0.50 SOC)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    plot2_path = OUTPUT_DIR / "mismatch_nyquist_comparison.png"
    fig.savefig(plot2_path, dpi=160)
    plt.close(fig)
    print(f"Nyquist mismatch plot written to: {plot2_path}")


if __name__ == "__main__":
    main()
