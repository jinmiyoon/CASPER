import numpy as np
import pytest

from casper.interface.segment import Segment


@pytest.mark.parametrize(
    "wl, flux, expected_wl, expected_flux, expected_midpoint",
    [
        # Case 1: Normal inputs
        ([4000, 5000, 6000], [1.2, 1.5, 1.1], np.array([4000, 5000, 6000]), np.array([1.2, 1.5, 1.1]), 5000.0),
        # Case 2: Empty lists
        ([], [], np.array([]), np.array([]), np.nan),
        # Case 3: None inputs
        (None, None, np.array([]), np.array([]), np.nan),
    ],
)
def test_segment_init(wl, flux, expected_wl, expected_flux, expected_midpoint):
    seg = Segment(wl=wl, flux=flux)

    assert np.array_equal(seg.wl, expected_wl)
    assert np.array_equal(seg.flux, expected_flux)

    if np.isnan(expected_midpoint):
        assert np.isnan(seg.midpoint)
    else:
        assert seg.midpoint == expected_midpoint


@pytest.mark.parametrize(
    "wl, which, expected_midpoint, valid_input",
    [
        ([4000, 5000, 6000], "left", 4000, True),
        ([4000, 5000, 6000], "right", 6000, True),
        ([1, 2, 3, 4], "left", 1, True),
        ([1, 2, 3, 4], "right", 4, True),
        # Invalid inputs
        ([100, 200], "top", 150, False),
        ([100, 200], "bottom", 150, False),
        ([100, 200], "", 150, False),
        ([100, 200], "LEFT", 150, False),
        ([100, 200], None, 150, False),
    ],
)
def test_is_edge(wl, which, expected_midpoint, valid_input):
    seg = Segment(wl=wl, flux=[1] * len(wl))

    seg.midpoint = np.median(wl)

    seg.is_edge(which)

    if valid_input:
        assert seg.midpoint == expected_midpoint
    else:
        assert seg.midpoint == np.median(wl)


@pytest.mark.parametrize(
    "flux, lower, expected_mad, expected_flux_min, expected_flux_max",
    [
        ([1, 2, 3, 4, 5], 85, 1.0, 4.4, 4.92),  # linear data
    ],
)
def test_get_statistics(flux, lower, expected_mad, expected_flux_min, expected_flux_max):
    seg = Segment(wl=np.arange(len(flux)), flux=flux)
    seg.get_statistics(lower=lower)

    assert np.isclose(seg.mad, expected_mad, atol=0.01)
    assert np.isclose(seg.flux_min, expected_flux_min, atol=0.01)
    assert np.isclose(seg.flux_max, expected_flux_max, atol=0.01)

    if not np.isnan(seg.flux_med):
        assert seg.flux_min <= seg.flux_med <= seg.flux_max


@pytest.mark.parametrize(
    "mad, flux_med, flux_max, mad_min, mad_range, expected_mad_normal, expected_cont_point",
    [
        (3.0, 10.0, 20.0, 1.0, 4.0, 0.5, 15.0),
    ],
)
def test_define_cont_point(mad, flux_med, flux_max, mad_min, mad_range, expected_mad_normal, expected_cont_point):
    seg = Segment(wl=[], flux=[])
    seg.mad = mad
    seg.flux_med = flux_med
    seg.flux_max = flux_max

    seg.define_cont_point(mad_min=mad_min, mad_range=mad_range)

    assert np.isclose(seg.mad_normal, expected_mad_normal, atol=0.01)
    assert np.isclose(seg.continuum_point, expected_cont_point, atol=0.01)
