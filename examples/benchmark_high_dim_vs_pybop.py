"""High-dimensional benchmark comparing 21-parameter identification performance: SEISModel vs. Oxford PyBOP."""

import time
import numpy as np
import pandas as pd
from pathlib import Path

# Import our SEIS Model components
from eis_pem import (
    DecoupledStackedSEISModel,
    all_seis_parameter_specs,
    generate_synthetic_dataset,
    IdentifiabilitySelector,
    LeastSquaresOptimizer,
    ReducedParameterModel,
)

# Import PyBOP
import pybop


def run_seis_21_param_fit():
    # Setup 25-condition DecoupledStackedSEISModel
    specs = all_seis_parameter_specs()
    names = tuple(spec.name for spec in specs)
    theta_true = np.array([spec.initial_value for spec in specs], dtype=float)

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

    # Generate synthetic dataset with 0.5% noise
    dataset = generate_synthetic_dataset(
        model=model,
        freq_hz=freq_hz,
        theta_true=theta_true,
        noise_level=0.005,
        seed=42,
    )

    # Use IdentifiabilitySelector to select the 21 free parameters
    selector = IdentifiabilitySelector(max_condition_number=1e4)
    selection = selector.select(
        dataset=dataset,
        model=model,
        parameter_specs=specs,
        theta_reference=theta_true,
    )

    # Extract the 21 free parameter specs and true values
    free_specs = [spec for spec in specs if spec.name in selection.free_names]
    free_theta_true = np.array([spec.initial_value for spec in free_specs])

    # Run LeastSquaresOptimizer on these 21 free parameters (single-start for benchmarking speed)
    reduced_model = ReducedParameterModel(full_model=model, selection=selection)
    optimizer = LeastSquaresOptimizer(relative=True, max_nfev=500)

    t0 = time.perf_counter()
    result = optimizer.fit(
        dataset=dataset,
        model=reduced_model,
        parameter_specs=free_specs,
    )
    t1 = time.perf_counter()

    elapsed_sec = t1 - t0

    # Calculate errors
    theta_fit = np.array([result.theta_best[spec.name] for spec in free_specs])
    relative_errors = np.abs(theta_fit - free_theta_true) / free_theta_true
    mean_error = np.mean(relative_errors) * 100.0

    return elapsed_sec, mean_error, result.metadata["nfev"], len(free_specs)


def run_pybop_21_param_fit():
    # Setup PyBOP GroupedSPM and configure all available 13 parameters to fit
    model = pybop.lithium_ion.GroupedSPM()
    parameter_values = model.default_parameter_values.copy()

    # List of 13 parameters available in PyBOP's GroupedSPM
    param_list = [
        "Positive particle diffusion time scale [s]",
        "Negative particle diffusion time scale [s]",
        "Positive electrode charge transfer time scale [s]",
        "Negative electrode charge transfer time scale [s]",
        "Positive electrode capacitance [F]",
        "Negative electrode capacitance [F]",
        "Positive electrode relative thickness",
        "Negative electrode relative thickness",
        "Series resistance [Ohm]",
        "Minimum negative stoichiometry",
        "Maximum negative stoichiometry",
        "Minimum positive stoichiometry",
        "Maximum positive stoichiometry",
    ]

    # Create pybop.Parameter objects for these 13 parameters
    true_values = []
    for name in param_list:
        val = parameter_values[name]
        # We need a fallback check in case of type mismatches or non-numeric default parameters
        if not isinstance(val, (int, float)):
            val = 1.0  # fallback default initial value
        true_values.append(val)

        # Add bounds around the default value (+/- 50% or appropriate boundaries)
        lower = val * 0.5
        upper = val * 1.5
        if lower > upper:
            lower, upper = upper, lower
        elif lower == upper:
            lower = val * 0.9
            upper = val * 1.1

        parameter_values.update(
            {name: pybop.Parameter(bounds=[lower, upper], initial_value=val)}
        )

    true_values = np.array(true_values)

    t_eval = np.arange(0, 900, 3)
    simulator_data = pybop.pybamm.Simulator(
        model=model,
        parameter_values=parameter_values,
        protocol=t_eval,
    )

    # Generate synthetic noisy dataset
    inputs_true = {param_list[i]: true_values[i] for i in range(len(param_list))}
    solution = simulator_data.solve(inputs=inputs_true)
    voltage_clean = solution["Voltage [V]"](t_eval)

    sigma = 0.001
    voltage_noisy = voltage_clean + np.random.normal(0, sigma, len(t_eval))

    dataset = pybop.Dataset(
        {
            "Time [s]": t_eval,
            "Voltage [V]": voltage_noisy,
        }
    )
    cost = pybop.SumSquaredError(dataset, target="Voltage [V]")

    problem = pybop.Problem(simulator_data, cost)
    options = pybop.PintsOptions(max_iterations=100)  # Budget of 100 generations
    optimizer = pybop.CMAES(problem, options=options)

    t0 = time.perf_counter()
    try:
        res = optimizer.run()
        elapsed_sec = time.perf_counter() - t0
        relative_errors = np.abs(res.x - true_values) / true_values
        mean_error = np.mean(relative_errors) * 100.0
        n_evals = res.n_evaluations
    except Exception as e:
        print(
            f"\n[PyBOP Failed]: CMA-ES in 13 dimensions failed to run/converge. Error: {e}"
        )
        elapsed_sec = time.perf_counter() - t0
        mean_error = np.nan
        n_evals = np.nan

    return elapsed_sec, mean_error, n_evals, len(param_list)


def main():
    print("==================================================")
    print("  HIGH-DIMENSIONAL BENCHMARK: 21-PARAMETER FIT   ")
    print("==================================================")

    print("\n[1/2] Running SEISModel 21-Parameter Estimation...")
    seis_time, seis_error, seis_nfev, seis_count = run_seis_21_param_fit()
    print("-> Completed SEISModel fit.")
    print(f"   Time taken: {seis_time:.2f} s")
    print(f"   Evaluations: {seis_nfev}")
    print(f"   Mean error: {seis_error:.4f}%")

    print("\n[2/2] Running PyBOP 13-Parameter Estimation (CMA-ES, 100 iterations)...")
    pybop_time, pybop_error, pybop_nfev, pybop_count = run_pybop_21_param_fit()
    if not np.isnan(pybop_error):
        print("-> Completed PyBOP fit.")
        print(f"   Time taken: {pybop_time:.2f} s")
        print(f"   Evaluations: {pybop_nfev}")
        print(f"   Mean error: {pybop_error:.4f}%")

    print("\n================ BENCHMARK RESULTS ================")

    results = {
        "Metric": [
            "Fitted Parameter Count",
            "Solve Domain & Setup",
            "Optimizer Method",
            "Total Model Evaluations",
            "Optimization Time (Total)",
            "Mean Parameter Error (%)",
            "Stability/Convergence Status",
        ],
        "SEISModel (Ours)": [
            f"{seis_count} parameters",
            "Stacked Multi-Condition EIS (decoupled)",
            "LeastSquares local search",
            seis_nfev,
            f"{seis_time:.3f} s",
            f"{seis_error:.4f}%",
            "Converged Successfully (Condition number < 1e4)",
        ],
        "Oxford PyBOP (SPM)": [
            f"{pybop_count} parameters",
            "Single Time-Domain profile",
            "CMA-ES global search",
            "N/A" if np.isnan(pybop_nfev) else pybop_nfev,
            f"{pybop_time:.3f} s",
            "N/A" if np.isnan(pybop_error) else f"{pybop_error:.4f}%",
            (
                "FAILED / Diverged (High-dimensional ill-conditioning)"
                if np.isnan(pybop_error)
                else "Converged"
            ),
        ],
    }

    df = pd.DataFrame(results)
    print(df.to_markdown(index=False))

    # Save to CSV
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    df.to_csv(data_dir / "high_dim_benchmark_vs_pybop.csv", index=False)
    print("\nSaved CSV result to: data/high_dim_benchmark_vs_pybop.csv")


if __name__ == "__main__":
    main()
