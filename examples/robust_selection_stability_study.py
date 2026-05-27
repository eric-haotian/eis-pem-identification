"""Stability study of robust parameter selection under log-normally perturbed prior parameter guesses."""

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
    all_seis_parameter_specs,
    generate_synthetic_dataset,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
DATA_DIR = PROJECT_ROOT / "data"


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

    model = DecoupledStackedSEISModel(
        conditions=conditions,
        response_channels=channels,
        parameter_names=names,
    )

    # Generate reference clean dataset
    dataset = generate_synthetic_dataset(
        model=model,
        freq_hz=freq_hz,
        theta_true=theta_true,
        noise_level=0.0,
        seed=42,
    )

    selector = IdentifiabilitySelector(max_condition_number=1e4)
    rng = np.random.default_rng(seed=42)
    replications = 20
    perturbation_sigmas = [0.20, 0.50]

    # Store selection outcomes
    # outcomes[sigma][parameter_name] = list of statuses (1 for free, 0 for fixed)
    outcomes = {sigma: {name: [] for name in names} for sigma in perturbation_sigmas}
    condition_numbers = {sigma: [] for sigma in perturbation_sigmas}

    for sigma in perturbation_sigmas:
        print(
            f"\nEvaluating selection stability under prior perturbation sigma = {sigma}..."
        )
        for rep in range(replications):
            perturbed_specs = []
            perturbed_values = []
            for spec in specs:
                # Log-normal perturbation multiplier
                multiplier = rng.lognormal(mean=0.0, sigma=sigma)
                val = spec.initial_value * multiplier
                lower, upper = spec.bounds
                margin = (upper - lower) * 1e-6
                val_clipped = float(np.clip(val, lower + margin, upper - margin))

                perturbed_specs.append(replace(spec, initial_value=val_clipped))
                perturbed_values.append(val_clipped)

            perturbed_theta = np.array(perturbed_values, dtype=float)

            # Run Selector
            selection = selector.select(
                dataset=dataset,
                model=model,
                parameter_specs=perturbed_specs,
                theta_reference=perturbed_theta,
            )

            # Record selection (free vs fixed)
            # Free if status is 'estimated' or 'protected_high_uncertainty'
            for name in names:
                status = selection.statuses[name]
                is_free = (
                    1 if status in ("estimated", "protected_high_uncertainty") else 0
                )
                outcomes[sigma][name].append(is_free)

            condition_numbers[sigma].append(selection.reduced_condition_number)

    # Calculate frequencies
    freq_data = {"parameter": names}
    for sigma in perturbation_sigmas:
        freq_data[f"free_frequency_sigma_{sigma:.2f}"] = [
            np.mean(outcomes[sigma][name]) for name in names
        ]

    summary_df = pd.DataFrame(freq_data)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = DATA_DIR / "robust_selection_stability.csv"
    summary_df.to_csv(csv_path, index=False)
    print(f"\nSelection frequencies saved to: {csv_path}")

    # Plot
    fig, ax = plt.subplots(figsize=(14.0, 6.0))
    positions = np.arange(len(names))
    width = 0.38

    ax.bar(
        positions - width / 2,
        summary_df["free_frequency_sigma_0.20"] * 100.0,
        width,
        color="#2b5c8f",
        label=r"$\sigma = 0.20$ (Moderate Uncertainty)",
    )
    ax.bar(
        positions + width / 2,
        summary_df["free_frequency_sigma_0.50"] * 100.0,
        width,
        color="#d95f02",
        label=r"$\sigma = 0.50$ (High Prior Uncertainty)",
    )

    ax.set_xticks(positions, names, rotation=70, ha="right", fontsize=8)
    ax.set_ylabel("Frequency selected as Free Parameter [%]")
    ax.set_title(
        "Identifiability Selection Stability Under Perturbed Prior Assumptions (20 Replications)"
    )
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    plot_path = OUTPUT_DIR / "prior_stability_selection_frequency.png"
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)
    print(f"Selection stability plot written to: {plot_path}")

    # Print selection stats
    print("\nSummary of selection consistency:")
    for sigma in perturbation_sigmas:
        frequencies = summary_df[f"free_frequency_sigma_{sigma:.2f}"].to_numpy()
        always_free = np.sum(frequencies == 1.0)
        always_fixed = np.sum(frequencies == 0.0)
        variable = np.sum((frequencies > 0.0) & (frequencies < 1.0))
        mean_cond = np.mean(condition_numbers[sigma])

        print(f"  For sigma = {sigma:.2f}:")
        print(f"    Always Free:  {always_free:>2d} / {len(names)}")
        print(f"    Always Fixed: {always_fixed:>2d} / {len(names)}")
        print(f"    Variable:     {variable:>2d} / {len(names)}")
        print(f"    Mean Reduced Condition Number: {mean_cond:.2e}")


if __name__ == "__main__":
    main()
