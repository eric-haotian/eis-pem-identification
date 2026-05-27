# Pull Request Description: SEIS Model Optimization & Physical Parameter Identification Refinements

This Pull Request introduces major enhancements to the `eis_pem_identification` prototype. It expands the physical parameters, optimizes the optimization strategies, improves diagnostic tools, and implements robust tests and studies to prepare the codebase for production use.

## Summary of Changes

### 1. Expanded Physical Realism (48 Parameters)
We have added **7 new parameters** (bringing the total from 41 to 48) to eliminate unphysical assumptions and align with experimental battery literature:
- **CPE Double-Layer Admittance**: Added `alpha_dl_neg` and `alpha_dl_pos` to replace the ideal capacitor (`jωCdl`) with Constant Phase Elements (`Cdl · (jω)^α`), successfully capturing the depressed semicircles caused by porous electrode roughness.
- **Cathode CEI Film**: Replaced the hardcoded assumption of `Rsei_pos = 0` and `Csei_pos = 0`. Added cathode-electrolyte interphase (CEI) parameters: `rou_sei_pos_0` (reference CEI resistivity) and `epse_sei_pos` (CEI permittivity).
- **Adjustable Bruggeman Exponents**: Replaced the hardcoded `brug = 1.5` with three adjustable parameter specs: `brug_neg`, `brug_pos`, and `brug_sep` to represent realistic tortuosity values.

### 2. Modulus-Weighted Cost Function
- Added a `modulus_weighting: bool` flag to cost calculation and optimization routines.
- Weighting residuals by `1 / |Z_obs|` (Boukamp's modulus weighting) prevents the optimizer from over-fitting high-frequency noise and balances the fitting effort across the entire impedance spectrum.

### 3. Multi-Start Optimizer
- Enhanced `LeastSquaresOptimizer` with an optional `n_starts` parameter.
- Randomly samples initial guesses in the bounds of log-transformed space when `n_starts > 1` and runs the optimizer from each. This significantly avoids local minima in the new 48-parameter space.

### 4. Correlation-Based Diagnostics & Pre-screening
- **Diagnostics**: Added a `correlation_matrix` to `IdentifiabilityReport` and created the `export_correlation_csv` utility to output the parameter correlation matrix calculated from the Fisher Information Matrix ($J^T J$).
- **Identifiability Selector**: Implemented a correlation-based pre-screening step in `IdentifiabilitySelector.select()`. Before SVD elimination, if any pair of parameters has an absolute correlation $| \rho | > 0.98$, the one with the lower sensitivity norm is automatically fixed. This drastically reduces the condition number from $\sim 10^4$ to $\mathbf{1.13 \times 10^3}$.

### 5. Verification Studies (Ablation, Mismatch, & Stability)
Added new study scripts and unit tests verifying the critical properties requested:
- **Full-Cell vs. Decoupled Ablation**: Shows the SVD rank and condition number improvement when using decoupled electrode impedance vs full-cell only data.
- **Model Mismatch Study**: Verifies parameter recovery when contact resistance (`R_contact`) or inductance (`L_ind`) are present in the cell but missing from the model.
- **High-Frequency Truncation**: Quantifies the relationship between high-frequency cutoff limits and the identifiability of the inductive parameters.
- **Robust Selection Stability**: Confirms the robustness of parameter selections under perturbed priors.

---

## Verification & Test Results

### 1. Unit Tests
All unit tests run and pass cleanly:
```bash
$ PYTHONPATH=src pytest
============================= 25 passed in 35.44s ==============================
```

### 2. Example Verification
- **Joint PEM Benchmark** (`run_joint_all_seis_parameters_pem.py`):
  - Solves the full 48-parameter estimation problem with decoupled impedance spectra.
  - Achieves a full rank Jacobian ($48/48$) with a condition number of $2.42 \times 10^3$, yielding a median parameter error of **0.15%** (max error 7.25%).
- **Robust Selection Benchmark** (`run_robust_all_seis_parameters_pem.py`):
  - Automatically selects 21 parameters for estimation under full-cell conditions, fixing the other 27.
  - Achieves a median parameter error of **2.22%** (max error 12.40%) on noisy data ($0.5\%$ relative complex Gaussian noise).

---

## Files Changed/Added

| Component / Path | Description |
| :--- | :--- |
| [`src/eis_pem/seis_model.py`](file:///Users/zhonghaotian/Desktop/battery/eis_pem_identification/src/eis_pem/seis_model.py) | Added 7 parameters, CPE double-layer, CEI film computation, adjustable Bruggeman exponents |
| [`src/eis_pem/costs.py`](file:///Users/zhonghaotian/Desktop/battery/eis_pem_identification/src/eis_pem/costs.py) | Implemented modulus-weighting functionality |
| [`src/eis_pem/optimizers.py`](file:///Users/zhonghaotian/Desktop/battery/eis_pem_identification/src/eis_pem/optimizers.py) | Added multi-start optimizer loop & modulus weighting compatibility |
| [`src/eis_pem/diagnostics.py`](file:///Users/zhonghaotian/Desktop/battery/eis_pem_identification/src/eis_pem/diagnostics.py) | Implemented correlation matrix computation & CSV export |
| [`src/eis_pem/robust.py`](file:///Users/zhonghaotian/Desktop/battery/eis_pem_identification/src/eis_pem/robust.py) | Implemented correlation-based pre-screening |
| [`tests/`](file:///Users/zhonghaotian/Desktop/battery/eis_pem_identification/tests/) | Updated assertions, bounds, expected parameter lists, and added multi-start/correlation tests |
| [`examples/`](file:///Users/zhonghaotian/Desktop/battery/eis_pem_identification/examples/) | Updated existing run scripts, added studies for ablation, mismatch, truncation, and stability |
| [`README.md`](file:///Users/zhonghaotian/Desktop/battery/eis_pem_identification/README.md) | Documented updated parameter counts, model formulations, and output structures |
| [`.gitignore`](file:///Users/zhonghaotian/Desktop/battery/eis_pem_identification/.gitignore) | Configured to ignore generated `data/` and `outputs/` files |
