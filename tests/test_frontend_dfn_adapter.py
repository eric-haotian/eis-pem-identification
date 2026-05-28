import json

import numpy as np
import pandas as pd
import pytest

from eis_pem import (
    SEISModel,
    dfn_frontend_response_to_frame,
    simulate_dfn_from_frontend,
)
from eis_pem.seis_model import all_seis_parameter_specs


def test_frontend_dfn_default_cell_request_matches_dfn_model() -> None:
    freq_hz = [0.01, 1.0, 100.0]
    response = simulate_dfn_from_frontend({"freq_hz": freq_hz})
    frame = dfn_frontend_response_to_frame(response)
    specs = all_seis_parameter_specs()
    theta = np.array([spec.initial_value for spec in specs], dtype=float)
    expected = SEISModel(parameter_names=tuple(spec.name for spec in specs)).simulate(
        np.array(freq_hz, dtype=float), theta
    )

    assert response["model"] == "DFN"
    assert response["parameter_count"] == len(specs)
    assert response["impedance_unit"] == "ohm*m^2"
    assert json.loads(json.dumps(response)) == response
    assert list(frame["response_channel"]) == ["cell", "cell", "cell"]
    np.testing.assert_allclose(frame["Zreal_ohm_m2"].to_numpy(), expected.real)
    np.testing.assert_allclose(frame["Zimag_ohm_m2"].to_numpy(), expected.imag)


def test_frontend_dfn_supports_frequency_range_conditions_channels_and_overrides() -> None:
    payload = {
        "frequency": {"min_hz": 0.1, "max_hz": 1000.0, "points": 5},
        "conditions": [
            {"temperature_K": 288.15, "SOC": 0.25},
            {"temperature_K": 308.15, "SOC": 0.75},
        ],
        "response_channels": ["cell", "neg", "pos", "sep"],
        "parameters": {"R_contact": 0.02, "L_ind": 5e-8},
    }

    response = simulate_dfn_from_frontend(payload)
    frame = dfn_frontend_response_to_frame(response)

    assert len(response["spectra"]) == 2 * 4 * 5
    assert len(frame) == len(response["spectra"])
    assert set(frame["response_channel"]) == {"cell", "neg", "pos", "sep"}
    assert set(frame["temperature_K"]) == {288.15, 308.15}
    assert set(frame["SOC"]) == {0.25, 0.75}
    assert response["parameters"]["R_contact"] == pytest.approx(0.02)
    assert response["parameters"]["L_ind"] == pytest.approx(5e-8)
    assert np.isfinite(frame[["Zreal_ohm_m2", "Zimag_ohm_m2"]].to_numpy()).all()


def test_frontend_dfn_rejects_invalid_requests() -> None:
    with pytest.raises(ValueError, match="unknown DFN parameter"):
        simulate_dfn_from_frontend({"parameters": {"not_a_parameter": 1.0}})

    with pytest.raises(ValueError, match="response_channels"):
        simulate_dfn_from_frontend({"response_channels": ["unknown"]})

    with pytest.raises(ValueError, match="freq_hz"):
        simulate_dfn_from_frontend({"freq_hz": [1.0, -2.0]})
