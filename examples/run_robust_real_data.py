"""Robust real-data EIS parameter identification workflow.

Demonstrates the full pipeline designed for real experimental data:
1. Data quality assessment (noise, inductive tails, drift, KK validation)
2. Physics-aware frequency filtering
3. Adaptive identifiability selection (four-criterion joint evaluation)
4. Multi-start optimization with LHS and optional regularization
5. Post-fit diagnostics with grading

This uses synthetic noisy data to simulate real-data conditions:
- Higher noise (2%)
- Limited frequency range (0.01 Hz – 10 kHz instead of 1 mHz – 100 kHz)
- Inductive artifacts at high frequency
- Low-frequency drift
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Ensure the package is importable
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from eis_pem import (
    AdaptiveLeastSquaresOptimizer,
    EISDataset,
    FrequencyBandAnalyzer,
    FrequencyBandConfig,
    IdentifiabilitySelector,
    IdentifiabilityStrategy,
    ReducedParameterModel,
    StackedSEISModel,
    all_seis_parameter_specs,
    assess_data_quality,
    compute_post_fit_diagnostics,
    default_seis_theta,
    filter_dataset,
    save_diagnostic_plots,
    save_robust_selection_plots,
    weighted_dataset,
)

OUTPUT_DIR = Path(__file__).resolve().parent / "output_robust_real_data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    print("=" * 60)
    print("ROBUST REAL-DATA EIS PARAMETER IDENTIFICATION")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Generate synthetic data simulating real-world conditions
    # ------------------------------------------------------------------
    print("\n[1/6] Generating synthetic real-world-like data...")

    specs = list(all_seis_parameter_specs())
    theta_true = default_seis_theta()

    # Limited conditions (3 temperatures × 3 SOCs = 9 conditions)
    temperatures = [278.15, 298.15, 318.15]
    socs = [0.2, 0.5, 0.8]
    conditions = [(t, s) for t in temperatures for s in socs]

    # Limited frequency range: 0.01 Hz to 10 kHz (4 decades, not 8)
    freq_hz = np.logspace(-2, 4, 40)

    model = StackedSEISModel(
        conditions=conditions,
        freq_hz_per_condition=freq_hz,
    )

    z_true = model.simulate(freq_hz=None, theta=theta_true)

    # Add realistic noise (2%) + inductive artifacts + drift
    rng = np.random.default_rng(42)
    noise_level = 0.02
    noise = noise_level * np.abs(z_true) * (
        rng.standard_normal(z_true.size) + 1j * rng.standard_normal(z_true.size)
    )

    # Add inductive artifact at highest frequencies (simulating wire inductance)
    n_per_cond = len(freq_hz)
    for cond_idx in range(len(conditions)):
        start = cond_idx * n_per_cond
        # Add positive imaginary component at high freq
        for i in range(n_per_cond - 3, n_per_cond):
            freq_i = freq_hz[i - (n_per_cond - n_per_cond)]
            noise[start + i] += 1j * 0.05 * np.abs(z_true[start + i])

    # Add drift at lowest frequencies (simulating SOC drift)
    for cond_idx in range(len(conditions)):
        start = cond_idx * n_per_cond
        for i in range(3):
            noise[start + i] += 0.08 * np.abs(z_true[start + i]) * rng.standard_normal()

    z_obs = z_true + noise

    # Build a single stacked dataset
    stacked_freq = np.tile(freq_hz, len(conditions))
    dataset = EISDataset(
        freq_hz=stacked_freq,
        z_obs=z_obs,
        z_true=z_true,
    )
    print(f"  Total points: {dataset.freq_hz.size}")
    print(f"  Conditions: {len(conditions)} (T × SOC)")
    print(f"  Freq range: {freq_hz[0]:.2f} – {freq_hz[-1]:.0f} Hz")
    print(f"  Noise level: {noise_level * 100:.0f}% + artifacts")

    # ------------------------------------------------------------------
    # 2. Data quality assessment
    # ------------------------------------------------------------------
    print("\n[2/6] Assessing data quality...")
    quality = assess_data_quality(dataset)
    print(quality.summary())
    quality.export_csv(OUTPUT_DIR / "data_quality.csv")

    # ------------------------------------------------------------------
    # 3. Physics-aware frequency filtering
    # ------------------------------------------------------------------
    print("\n[3/6] Filtering frequency bands...")
    config = FrequencyBandConfig(
        low_mid_boundary=0.1,
        mid_high_boundary=5000.0,
        max_low_freq_scatter=0.15,
        kk_residual_threshold=0.03,
    )
    analyzer = FrequencyBandAnalyzer(config=config)
    filter_result = analyzer.analyze(dataset)

    print(f"  Valid points: {np.sum(filter_result.valid_mask)}/{dataset.freq_hz.size}")
    print(f"  Valid fraction: {filter_result.valid_fraction:.1%}")
    if filter_result.removal_reasons:
        for reason, count in filter_result.removal_reasons.items():
            print(f"  Removed ({reason}): {count}")

    filter_result.to_frame().to_csv(OUTPUT_DIR / "frequency_filter.csv", index=False)

    # Apply filtering
    filtered_dataset, weights = weighted_dataset(dataset, filter_result)
    print(f"  Filtered dataset: {filtered_dataset.freq_hz.size} points")

    # ------------------------------------------------------------------
    # 4. Adaptive identifiability selection
    # ------------------------------------------------------------------
    print("\n[4/6] Running adaptive parameter selection...")
    selector = IdentifiabilitySelector(
        strategy=IdentifiabilityStrategy.ADAPTIVE,
        max_condition_number=1e4,
        assumed_noise_level=noise_level,
        max_relative_ci95=0.15,
        max_correlation_threshold=0.95,
        eigenvalue_gap_ratio=100.0,
    )

    # Use filtered model
    filtered_model = StackedSEISModel(
        conditions=conditions,
        freq_hz_per_condition=filtered_dataset.freq_hz[:len(filtered_dataset.freq_hz) // len(conditions)],
    )

    selection = selector.select(
        dataset=filtered_dataset,
        model=filtered_model,
        parameter_specs=specs,
        theta_reference=theta_true,
    )

    n_free = len(selection.free_names)
    n_total = len(specs)
    print(f"  Parameters: {n_free}/{n_total} free")
    print(f"  Condition number: {selection.original_condition_number:.2e} → "
          f"{selection.reduced_condition_number:.2e}")

    # Show fixed parameters with reasons
    fixed_count = 0
    for name in selection.parameter_names:
        if selection.statuses[name] != "estimated":
            fixed_count += 1
            if fixed_count <= 10:
                print(f"    Fixed: {name} ({selection.reasons[name]})")
    if fixed_count > 10:
        print(f"    ... and {fixed_count - 10} more fixed parameters")

    selection.export_csv(OUTPUT_DIR / "parameter_selection.csv")

    # ------------------------------------------------------------------
    # 5. Fit with adaptive optimizer
    # ------------------------------------------------------------------
    print("\n[5/6] Fitting with AdaptiveLeastSquaresOptimizer...")
    reduced_model = ReducedParameterModel(
        full_model=filtered_model,
        selection=selection,
    )

    # Perturb initial guess by ±5%
    free_specs = [s for s in specs if s.name in selection.free_names]
    free_theta_true = np.array([theta_true[specs.index(s)] for s in free_specs])
    rng_init = np.random.default_rng(123)
    perturbation = 1.0 + 0.05 * rng_init.standard_normal(len(free_specs))

    # Update initial values
    perturbed_specs = []
    for s, p in zip(free_specs, perturbation):
        from eis_pem.parameters import ParameterSpec
        perturbed_specs.append(
            ParameterSpec(
                name=s.name,
                initial_value=s.initial_value * p,
                bounds=s.bounds,
                unit=s.unit,
                log_transform=s.log_transform,
            )
        )

    optimizer = AdaptiveLeastSquaresOptimizer(
        relative=True,
        n_starts=3,
        alpha=1e-4,  # Light Tikhonov regularization
        max_nfev=10000,
        seed=42,
    )

    result = optimizer.fit(
        dataset=filtered_dataset,
        model=reduced_model,
        parameter_specs=perturbed_specs,
        weights=weights[: filtered_dataset.freq_hz.size] if weights is not None else None,
    )

    print(f"  Final cost: {result.final_cost:.6e}")
    print(f"  Optimizer: {result.metadata.get('optimizer', 'N/A')}")
    print(f"  Success: {result.metadata.get('success', 'N/A')}")
    print(f"  Function evaluations: {result.metadata.get('nfev', 'N/A')}")

    result.export_fit_csv(OUTPUT_DIR / "fit_results.csv")

    # Compute parameter errors for free parameters
    print("\n  Parameter recovery (free parameters):")
    errors = []
    for name in selection.free_names:
        if name in result.theta_best:
            idx = [s.name for s in specs].index(name)
            true_val = theta_true[idx]
            fit_val = result.theta_best[name]
            rel_err = abs(fit_val - true_val) / abs(true_val)
            errors.append(rel_err)
            if rel_err > 0.10:
                print(f"    {name}: {rel_err:.1%} error (true={true_val:.4e}, fit={fit_val:.4e})")

    if errors:
        print(f"  Median relative error: {np.median(errors):.2%}")
        print(f"  Max relative error: {np.max(errors):.2%}")
        print(f"  Parameters within 10%: {sum(1 for e in errors if e < 0.10)}/{len(errors)}")

    # ------------------------------------------------------------------
    # 6. Post-fit diagnostics
    # ------------------------------------------------------------------
    print("\n[6/6] Computing post-fit diagnostics...")
    diag = compute_post_fit_diagnostics(
        model=reduced_model,
        dataset=filtered_dataset,
        parameter_specs=perturbed_specs,
        theta_fitted=np.array([result.theta_best[s.name] for s in perturbed_specs]),
        relative=True,
    )
    print(diag.graded_summary())

    diag.export_csv(OUTPUT_DIR / "diagnostics.csv")
    diag.export_singular_values_csv(OUTPUT_DIR / "singular_values.csv")

    print(f"\nAll outputs saved to: {OUTPUT_DIR}")
    print("Done!")


if __name__ == "__main__":
    main()
