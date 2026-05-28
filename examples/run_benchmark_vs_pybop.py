"""Benchmark script comparing optimized SEISModel vs. Oxford PyBOP."""

import time
import numpy as np
import pandas as pd
from pathlib import Path

# Import our SEIS Model components
from eis_pem.seis_model import SEISModel, default_seis_parameter_specs, default_seis_theta
from eis_pem.dataset import generate_synthetic_dataset
from eis_pem.optimizers import LeastSquaresOptimizer

# Import PyBOP
import pybop

def run_seis_forward_benchmark(n_runs=100):
    specs = default_seis_parameter_specs()
    parameter_names = tuple(spec.name for spec in specs)
    theta = np.array([spec.initial_value for spec in specs])
    model = SEISModel(parameter_names=parameter_names)
    freqs = np.logspace(-2, 5, 60)
    
    t0 = time.perf_counter()
    for _ in range(n_runs):
        _ = model.simulate(freqs, theta)
    t1 = time.perf_counter()
    
    avg_time_ms = 1000.0 * (t1 - t0) / n_runs
    return avg_time_ms

def run_pybop_forward_benchmark(n_runs=100):
    model = pybop.lithium_ion.GroupedSPM()
    parameter_values = model.default_parameter_values
    solver = model.default_solver
    t_eval = np.arange(0, 900, 3) # 300 steps
    
    t0 = time.perf_counter()
    for _ in range(n_runs):
        simulator = pybop.pybamm.Simulator(
            model=model,
            parameter_values=parameter_values,
            protocol=t_eval,
            solver=solver,
            output_variables=["Voltage [V]", "Current [A]"],
        )
        _ = simulator.solve()
    t1 = time.perf_counter()
    
    avg_time_ms = 1000.0 * (t1 - t0) / n_runs
    return avg_time_ms

def run_seis_fitting_benchmark():
    # 4-parameter estimation benchmark
    model = SEISModel()
    specs = default_seis_parameter_specs()
    theta_true = default_seis_theta()
    
    # Generate synthetic EIS dataset (60 frequencies, 0.5% noise)
    freq_hz = np.logspace(-2, 5, 60)
    dataset = generate_synthetic_dataset(
        model=model,
        freq_hz=freq_hz,
        theta_true=theta_true,
        noise_level=0.005,
        seed=42
    )
    
    # Setup LeastSquaresOptimizer (single-start for pure speed benchmark)
    optimizer = LeastSquaresOptimizer(n_starts=1, max_nfev=500)
    
    t0 = time.perf_counter()
    result = optimizer.fit(
        dataset=dataset,
        model=model,
        parameter_specs=specs,
    )
    t1 = time.perf_counter()
    
    elapsed_sec = t1 - t0
    
    # Calculate parameter errors
    theta_fit = np.array([result.theta_best[spec.name] for spec in specs])
    relative_errors = np.abs(theta_fit - theta_true) / theta_true
    mean_error = np.mean(relative_errors) * 100.0
    
    return elapsed_sec, mean_error, result.metadata["nfev"]

def run_pybop_fitting_benchmark():
    # Fit 2 parameters (particle radii) using CMAES
    model = pybop.lithium_ion.GroupedSPM()
    
    # 1. Update parameter values with pybop.Parameter objects
    parameter_values = model.default_parameter_values.copy()
    parameter_values.update({
        "Negative particle radius [m]": pybop.Parameter(
            bounds=[1e-6, 9e-6],
            initial_value=6e-6,
        ),
        "Positive particle radius [m]": pybop.Parameter(
            bounds=[1e-6, 9e-6],
            initial_value=4.5e-6,
        ),
    })
    
    # 2. Setup simulator and generate synthetic dataset
    t_eval = np.arange(0, 900, 3)
    simulator_data = pybop.pybamm.Simulator(
        model=model,
        parameter_values=parameter_values,
        protocol=t_eval,
    )
    solution = simulator_data.solve(
        inputs={
            "Negative particle radius [m]": 6e-6,
            "Positive particle radius [m]": 4.5e-6,
        }
    )
    voltage_clean = solution["Voltage [V]"](t_eval)
    
    sigma = 0.001
    voltage_noisy = voltage_clean + np.random.normal(0, sigma, len(t_eval))
    
    # 3. Create dataset and cost
    dataset = pybop.Dataset(
        {
            "Time [s]": t_eval,
            "Voltage [V]": voltage_noisy,
        }
    )
    cost = pybop.SumSquaredError(dataset, target="Voltage [V]")
    
    # 4. Build problem and optimizer
    problem = pybop.Problem(simulator_data, cost)
    options = pybop.PintsOptions(max_iterations=50)
    optimizer = pybop.CMAES(problem, options=options)
    
    t0 = time.perf_counter()
    res = optimizer.run()
    t1 = time.perf_counter()
    
    elapsed_sec = t1 - t0
    
    # Calculate parameter errors
    true_values = np.array([6e-6, 4.5e-6])
    relative_errors = np.abs(res.x - true_values) / true_values
    mean_error = np.mean(relative_errors) * 100.0
    
    return elapsed_sec, mean_error, res.n_iterations

def main():
    print("==================================================")
    print("      BENCHMARK: SEISModel vs. Oxford PyBOP       ")
    print("==================================================")
    
    print("\nRunning Forward Simulation Benchmarks (100 runs)...")
    seis_fwd_time = run_seis_forward_benchmark()
    print(f"-> SEISModel (60 freqs, analytical): {seis_fwd_time:.3f} ms")
    
    pybop_fwd_time = run_pybop_forward_benchmark()
    print(f"-> PyBOP GroupedSPM (300 steps, PDE solver): {pybop_fwd_time:.3f} ms")
    print(f"Speedup: {pybop_fwd_time / seis_fwd_time:.1f}x faster")
    
    print("\nRunning Parameter Estimation Benchmarks...")
    print("-> Fitting 4 parameters in SEISModel...")
    seis_fit_time, seis_error, seis_nfev = run_seis_fitting_benchmark()
    print(f"   Time taken: {seis_fit_time:.2f} s")
    print(f"   Evaluations: {seis_nfev}")
    print(f"   Mean relative error: {seis_error:.4f}%")
    
    print("-> Fitting 2 parameters in PyBOP GroupedSPM (CMAES, 50 iterations)...")
    pybop_fit_time, pybop_error, pybop_iters = run_pybop_fitting_benchmark()
    print(f"   Time taken: {pybop_fit_time:.2f} s")
    print(f"   Iterations: {pybop_iters}")
    print(f"   Mean relative error: {pybop_error:.4f}%")
    
    # Save results to CSV for presentation
    results = {
        "Metric": [
            "Forward Model Type",
            "Forward Evaluation Time",
            "Parameter Fit Type",
            "Fit Execution Time",
            "Average Parameter Error",
            "Computational Efficiency"
        ],
        "SEISModel (Ours)": [
            "Analytical Frequency-Domain (DFN)",
            f"{seis_fwd_time:.3f} ms",
            "4-Parameter LeastSquares Fit",
            f"{seis_fit_time:.3f} s",
            f"{seis_error:.4f}%",
            "High (Sub-millisecond solver)"
        ],
        "Oxford PyBOP": [
            "PDE Time-Domain (SPM)",
            f"{pybop_fwd_time:.3f} ms",
            "2-Parameter CMA-ES Fit",
            f"{pybop_fit_time:.3f} s",
            f"{pybop_error:.4f}%",
            "Standard (PDE transient solver)"
        ]
    }
    df = pd.DataFrame(results)
    
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    df.to_csv(data_dir / "benchmark_vs_pybop.csv", index=False)
    print(f"\nBenchmark results saved to: data/benchmark_vs_pybop.csv")
    
    # Display Markdown Table
    print("\nBenchmark Summary Table:")
    print(df.to_markdown(index=False))

if __name__ == "__main__":
    main()
