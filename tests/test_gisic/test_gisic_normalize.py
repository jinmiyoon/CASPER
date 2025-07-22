import numpy as np
import pytest

from casper.interface.gisic import normalize


@pytest.mark.parametrize("return_points", [False, True])
def test_normalize_with_defaults(return_points):
    # Synthetic spectrum with multiple features for extrema detection
    wavelength = np.linspace(3900, 4000, 100)  # linearly spaced array of 100 wavelengths from 3900A to 4000A

    # Constructs synthetic flux data with two Gaussian dips centered at 3950 and 3970.
    # -0.3 * and -0.2 * simulate abosrption lines

    flux = (
        1.0
        - 0.3 * np.exp(-0.5 * ((wavelength - 3950) / 4) ** 2)
        - 0.2 * np.exp(-0.5 * ((wavelength - 3970) / 2.5) ** 2)
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
    # this ensures nothing got truncated, padded, or reshaped incorrectly.    assert wl_out.shape == flux.shape
    assert flux_norm.shape == flux.shape
    assert continuum.shape == flux.shape
