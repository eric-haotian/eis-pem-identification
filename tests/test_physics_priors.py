"""Tests for physics priors system."""

import numpy as np

from eis_pem.physics_priors import (
    BoundPrior,
    OrderingPrior,
    RatioPrior,
    SumPrior,
    check_prior_violations,
    lithium_ion_priors,
    prior_penalty,
    prior_penalty_vector,
)

def test_bound_prior():
    p = BoundPrior("test", "bound", "desc", 1.0, "Rs", 0.0, 1.0)
    
    # Inside bounds
    assert prior_penalty([p], ["Rs"], np.array([0.5])) == 0.0
    
    # Below lower bound
    pen1 = prior_penalty([p], ["Rs"], np.array([-0.1]))
    assert pen1 > 0
    
    # Above upper bound
    pen2 = prior_penalty([p], ["Rs"], np.array([1.5]))
    assert pen2 > 0

def test_ratio_prior():
    p = RatioPrior("test", "ratio", "desc", 1.0, "R1", "R2", (0.1, 10.0))
    
    assert prior_penalty([p], ["R1", "R2"], np.array([1.0, 1.0])) == 0.0
    assert prior_penalty([p], ["R1", "R2"], np.array([0.05, 1.0])) > 0.0
    assert prior_penalty([p], ["R1", "R2"], np.array([20.0, 1.0])) > 0.0

def test_ordering_prior():
    p = OrderingPrior("test", "ordering", "desc", 1.0, "R1", "R2")
    
    # R1 < R2 -> OK
    assert prior_penalty([p], ["R1", "R2"], np.array([1.0, 2.0])) == 0.0
    
    # R1 > R2 -> Violation
    assert prior_penalty([p], ["R1", "R2"], np.array([2.0, 1.0])) > 0.0

def test_sum_prior():
    p = SumPrior("test", "sum", "desc", 1.0, ("Rs", "R1", "R2"), (0.0, 2.0))
    
    # Sum = 1.5 -> OK
    assert prior_penalty([p], ["Rs", "R1", "R2"], np.array([0.5, 0.5, 0.5])) == 0.0
    
    # Sum = 2.5 -> Violation
    assert prior_penalty([p], ["Rs", "R1", "R2"], np.array([1.0, 1.0, 0.5])) > 0.0

def test_missing_parameter():
    p = BoundPrior("test", "bound", "desc", 1.0, "Rs", 0.0, 1.0)
    # If Rs is missing from param_names, penalty is 0 (soft fail)
    assert prior_penalty([p], ["Rct"], np.array([2.0])) == 0.0

def test_lithium_ion_priors():
    priors = lithium_ion_priors()
    
    # Good parameters
    theta_good = np.array([0.01, 0.05, 0.10, 0.85, 0.85])
    names = ["Rs", "R1", "R2", "alpha1", "alpha2"]
    
    pen_good = prior_penalty(priors, names, theta_good)
    assert pen_good == 0.0
    
    # Bad parameters: R1 > R2, total R > 2.0, Rs negative
    theta_bad = np.array([-0.01, 1.5, 0.1, 0.4, 0.4])
    pen_bad = prior_penalty(priors, names, theta_bad)
    assert pen_bad > 0.0
    
    violations = check_prior_violations(priors, names, theta_bad)
    assert len(violations) > 0
    assert any("R1" in v and "R2" in v for v in violations)

def test_penalty_vector():
    priors = [
        BoundPrior("b1", "bound", "desc", 1.0, "Rs", 0.0, 1.0),
        OrderingPrior("o1", "ordering", "desc", 1.0, "R1", "R2"),
    ]
    names = ["Rs", "R1", "R2"]
    # Rs violated (1.5), ordering violated (2.0 > 1.0)
    theta = np.array([1.5, 2.0, 1.0])
    vec = prior_penalty_vector(priors, names, theta)
    
    assert vec.shape == (2,)
    assert vec[0] > 0
    assert vec[1] > 0
