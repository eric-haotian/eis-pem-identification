"""High-frequency truncation study for setup inductance identifiability."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eis_pem import (
    LeastSquaresOptimizer,
    ParameterSpec,
    RandlesModel,
    evaluate_local_identifiability,
    generate_synthetic_dataset,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
DATA_DIR = PROJECT_ROOT / "data"


def main() -> None:
    model = RandlesModel()
    theta_true = np.array([0.01, 0.05, 2000.0, 0.8, 1e-6])  # L = 1e-6 H

    parameter_specs = [
        ParameterSpec("Rs", 0.02, (1e-4, 1e-1), "ohm", log_transform=True),
        ParameterSpec("Rct", 0.1, (1e-4, 1.0), "ohm", log_transform=True),
        ParameterSpec("Qdl", 1000.0, (10.0, 1e5), "F", log_transform=True),
        ParameterSpec("alpha", 0.9, (0.5, 1.0), "-", log_transform=False),
        ParameterSpec("L", 2e-6, (1e-10, 1e-4), "H", log_transform=True),
    ]

    caps = [100.0, 500.0, 1000.0, 5000.0, 10000.0, 50000.0, 100000.0]

    sensitivities = []
    estimation_errors = []
    final_costs = []

    for cap in caps:
        # Generate truncated frequency grid
        freq_hz = np.logspace(-2, np.log10(cap), 100)

        # Clean dataset for local identifiability
        dataset_clean = generate_synthetic_dataset(
            model=model,
            freq_hz=freq_hz,
            theta_true=theta_true,
            noise_level=0.0,
            seed=42,
        )

        # Noisy dataset for fit
        dataset_noisy = generate_synthetic_dataset(
            model=model,
            freq_hz=freq_hz,
            theta_true=theta_true,
            noise_level=0.005,
            seed=42,
        )

        # 1. Compute sensitivity norm
        report = evaluate_local_identifiability(
            model=model,
            dataset=dataset_clean,
            parameter_specs=parameter_specs,
            theta=theta_true,
        )
        # Inductance (L) is index 4
        l_sens = report.sensitivity_norms[4]
        sensitivities.append(l_sens)

        # 2. Run Fit
        result = LeastSquaresOptimizer(relative=True, max_nfev=3000).fit(
            dataset=dataset_noisy,
            model=model,
            parameter_specs=parameter_specs,
        )
        l_fit = result.theta_best["L"]
        l_error = abs(l_fit - theta_true[4]) / theta_true[4]

        estimation_errors.append(l_error)
        final_costs.append(result.final_cost)

        print(
            f"Cap: {cap:>8.1f} Hz | Sensitivity Norm: {l_sens:>10.4f} | Inductance Error: {l_error:>10.4%}"
        )

    # Save to CSV
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    summary_df = pd.DataFrame(
        {
            "Max_Frequency_Hz": caps,
            "L_Sensitivity_Norm": sensitivities,
            "L_Estimation_Error": estimation_errors,
            "Final_Cost": final_costs,
        }
    )

    csv_path = DATA_DIR / "high_frequency_truncation_study.csv"
    summary_df.to_csv(csv_path, index=False)
    print(f"Summary saved to: {csv_path}")

    # Plot
    fig, ax1 = plt.subplots(figsize=(7.5, 5.0))

    color = "#1f77b4"
    ax1.set_xlabel("Maximum Frequency Cap [Hz]")
    ax1.set_ylabel("L Sensitivity Norm (Jacobian column norm)", color=color)
    line1 = ax1.semilogx(
        caps,
        sensitivities,
        "o-",
        color=color,
        linewidth=2.0,
        label="L Sensitivity Norm",
    )
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.grid(True, which="both", alpha=0.3)

    ax2 = ax1.twinx()
    color = "#d62728"
    ax2.set_ylabel("L Relative Estimation Error [%]", color=color)
    line2 = ax2.semilogy(
        caps,
        np.array(estimation_errors) * 100.0,
        "s-",
        color=color,
        linewidth=2.0,
        label="L Error [%]",
    )
    ax2.tick_params(axis="y", labelcolor=color)

    # added these lines for combining legends
    lines = line1 + line2
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="upper right")

    plt.title("Impact of High-Frequency Truncation on Inductance Identifiability")
    fig.tight_layout()
    plot_path = OUTPUT_DIR / "truncation_inductance_identifiability.png"
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)
    print(f"Plot written to: {plot_path}")


if __name__ == "__main__":
    main()
