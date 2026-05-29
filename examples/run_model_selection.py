"""Example: Automated EIS model selection.

This script demonstrates the full automated pipeline:
1. Generate synthetic EIS data with noise
2. Perform Data Quality checks and Frequency filtering
3. Compute DRT and detect peaks
4. Select candidate circuit models based on DRT
5. Fit all candidates and compare via AICc
6. Output the best model and physical priors validation
"""

import json
import numpy as np

from eis_pem import (
    TwoRCPEModel,
    identify_with_model_selection,
)

def main():
    print("Generating synthetic 2-RCPE data with noise...")
    freq = np.logspace(4, -2, 60)
    
    # True model: Rs + R1||CPE1 + R2||CPE2
    # Distinct arcs: high frequency (SEI) and low frequency (charge transfer)
    true_model = TwoRCPEModel()
    theta_true = np.array([
        0.01,   # Rs
        0.015,  # R1
        2e-4,   # Q1
        0.85,   # alpha1
        0.04,   # R2
        0.05,   # Q2
        0.80,   # alpha2
    ])
    
    z_clean = true_model.simulate(freq, theta_true)
    
    # Add 2% relative noise
    np.random.seed(42)
    noise_level = 0.02
    noise = (
        np.random.normal(0, noise_level, z_clean.shape) +
        1j * np.random.normal(0, noise_level, z_clean.shape)
    ) * np.abs(z_clean)
    
    z_obs = z_clean + noise
    
    # Build request payload
    request = {
        "freq_hz": freq.tolist(),
        "z_obs_real": z_obs.real.tolist(),
        "z_obs_imag": z_obs.imag.tolist(),
        "selection_criterion": "aicc",
        "priors": "lithium_ion",
        # Optionally force candidates:
        # "candidate_models": ["1RC", "2RCPE", "3RCPE"]
    }
    
    print("\nRunning automated model selection pipeline...")
    print("(DRT analysis -> Model suggestion -> Fitting -> AICc comparison)")
    result = identify_with_model_selection(request)
    
    print("\n" + "="*50)
    print("MODEL SELECTION RESULT")
    print("="*50)
    
    print(f"\nBest Model: {result['best_model']}")
    print(f"Circuit:    {result['best_circuit']}")
    
    print("\nTrue Parameters vs Identified:")
    names = true_model.parameter_names
    for name, true_val in zip(names, theta_true):
        if name in result["theta_best"]:
            fit_val = result["theta_best"][name]
            err = abs(fit_val - true_val) / true_val * 100
            print(f"  {name:>6}: {true_val:8.4f} -> {fit_val:8.4f}  ({err:4.1f}% err)")
    
    print("\nDRT Analysis:")
    print(f"  Peaks detected:     {result['drt']['n_peaks']}")
    print(f"  Diffusion tail:     {result['drt']['has_diffusion_tail']}")
    print(f"  Inductive behavior: {result['drt']['has_inductive']}")
    
    print("\nModel Comparison Table:")
    print(f"  {'Model':<12} {'k':>2} {'AICc':>10} {'ΔAIC':>8} {'Weight':>6}")
    print("  " + "-" * 42)
    for c in result["comparison"]:
        print(f"  {c['model']:<12} {c['n_params']:2d} {c['AICc']:10.2f} "
              f"{c['delta_AIC']:8.2f} {c['akaike_weight']:6.3f}")

    if result["prior_violations"]:
        print("\nPhysics Prior Violations:")
        for v in result["prior_violations"]:
            print(f"  - {v}")
    else:
        print("\nPhysics Priors: All satisfied")

if __name__ == "__main__":
    main()
