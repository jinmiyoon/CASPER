import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from casper.interface.gisic import normalize


@pytest.mark.parametrize("return_points", [False, True])
def test_normalize_with_defaults(return_points):
    # Synthetic spectrum with multiple features for extrema detection
    wavelength = np.linspace(3900.0, 4000.0, 100)  # linearly spaced array of 100 wavelengths from 3900A to 4000A

    # Constructs synthetic flux data with two Gaussian dips centered at 3950 and 3970.
    # -0.3 * and -0.2 * simulate abosrption lines

    flux = (
        1.0
        - 0.3 * np.exp(-0.5 * ((wavelength - 3950.0) / 4.0) ** 2)
        - 0.2 * np.exp(-0.5 * ((wavelength - 3970.0) / 2.5) ** 2)
    )

    # adds a small amount of noise to the flux values
    # simulates real observational noise and helps avoid having a perfectly smooth flux array
    flux += np.random.normal(0, 0.001, size=wavelength.shape)

    result = normalize(
        wavelength=wavelength,
        flux=flux,
        sigma=5,  # less smoothing to preserve dips
        k=2,  # spline degree: safer if few anchor points
        s=12,
        cahk=False,
        band_check=True,
        flux_min=50,  # more lenient threshold
        boost=True,
        return_points=return_points,
    )

    # verifies that the returned points dictionary has the correct keys
    # confirms that both arrays in the dictionary have equal lengths
    if return_points:
        wl_out, flux_norm, continuum, points = result
        assert "wavelength" in points
        assert "flux" in points
        assert len(points["wavelength"]) == len(points["flux"])
    else:
        wl_out, flux_norm, continuum = result

    # verifies that each of the returned arrays has the same shape as the original flux array
    # this ensures nothing got truncated, padded, or reshaped incorrectly.
    assert wl_out.shape == flux.shape
    assert flux_norm.shape == flux.shape
    assert continuum.shape == flux.shape

    # flux_norm values should be [0,2]
    assert np.all((flux_norm >= 0) & (flux_norm <= 2))


# ---------------------------------------------------------------------------
# Property-based test (NFR-6, PBT-03 invariant): normalize()'s returned
# normalized flux must always stay within the documented [0, 2] bounds,
# regardless of the specific dip shape/depth/noise of the input spectrum.
# ---------------------------------------------------------------------------


@given(
    center1=st.floats(min_value=3920.0, max_value=3960.0),
    center2=st.floats(min_value=3960.0, max_value=3980.0),
    depth1=st.floats(min_value=0.05, max_value=0.4),
    depth2=st.floats(min_value=0.05, max_value=0.4),
    width1=st.floats(min_value=2.0, max_value=6.0),
    width2=st.floats(min_value=2.0, max_value=6.0),
    noise_scale=st.floats(min_value=0.0, max_value=0.01),
    seed=st.integers(min_value=0, max_value=10_000),
)
@settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_normalize_bounds_invariant_property(center1, center2, depth1, depth2, width1, width2, noise_scale, seed):
    rng = np.random.default_rng(seed)
    wavelength = np.linspace(3900.0, 4000.0, 100)

    flux = (
        1.0
        - depth1 * np.exp(-0.5 * ((wavelength - center1) / width1) ** 2)
        - depth2 * np.exp(-0.5 * ((wavelength - center2) / width2) ** 2)
    )
    flux += rng.normal(0, noise_scale, size=wavelength.shape)

    _, flux_norm, _ = normalize(
        wavelength=wavelength,
        flux=flux,
        sigma=5,
        k=2,
        s=12,
        cahk=False,
        band_check=True,
        flux_min=50,
        boost=True,
    )

    assert np.all(np.isfinite(flux_norm))
    assert np.all((flux_norm >= 0.0) & (flux_norm <= 2.0))
