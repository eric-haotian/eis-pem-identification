import numpy as np

from eis_pem import (
    DecoupledStackedSEISModel,
    SEISComponentModel,
    SEISModel,
    default_seis_parameter_specs,
    default_seis_theta,
)


def test_decoupled_region_spectra_reconstruct_full_cell_impedance() -> None:
    freq_hz = np.logspace(-3, 5, 80)
    theta = default_seis_theta()
    z_cell = SEISModel().simulate(freq_hz, theta)
    z_regions = sum(
        SEISComponentModel(response_channel=channel).simulate(freq_hz, theta)
        for channel in ("neg", "sep", "pos")
    )

    np.testing.assert_allclose(z_regions, z_cell, rtol=1e-10, atol=1e-15)


def test_decoupled_stacked_model_matches_condition_channel_block_order() -> None:
    conditions = ((288.15, 0.25), (308.15, 0.75))
    channels = ("cell", "neg", "pos", "sep")
    specs = default_seis_parameter_specs()
    theta = default_seis_theta()
    freq_block = np.logspace(-3, 4, 25)
    freq_hz = np.tile(freq_block, len(conditions) * len(channels))
    model = DecoupledStackedSEISModel(
        conditions=conditions,
        response_channels=channels,
        parameter_names=tuple(spec.name for spec in specs),
    )

    prediction = model.simulate(freq_hz, theta)
    expected = np.concatenate(
        [
            SEISComponentModel(
                temperature_k=temperature_k,
                soc=soc,
                response_channel=channel,
            ).simulate(freq_block, theta)
            for temperature_k, soc in conditions
            for channel in channels
        ]
    )

    np.testing.assert_allclose(prediction, expected)
