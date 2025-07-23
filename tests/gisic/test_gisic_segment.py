import numpy as np
import pytest

from casper.interface.gisic import Segment


@pytest.mark.parametrize(
    "wl, flux, expected_wl, expected_flux, expected_midpoint",
    [
        ([4000, 5000, 6000], [1.0, 1.2, 1.1], [4000, 5000, 6000], [1.0, 1.2, 1.1], 5000.0),
    ],
)
def test_segment_init(wl, flux, expected_wl, expected_flux, expected_midpoint):
    seg = Segment(wl=wl, flux=flux)

    assert seg.wl == expected_wl
    assert np.array_equal(seg.flux, expected_flux)

    if np.isnan(expected_midpoint):
        assert np.isnan(seg.midpoint)
    else:
        assert seg.midpoint == expected_midpoint


@pytest.mark.parametrize(
    "wl, which, expected_midpoint, valid",
    [
        ([4000, 5000, 6000], "left", 4000, True),
        ([4000, 5000, 6000], "right", 6000, True),
        ([4000, 5000, 6000], "top", 5000, False),
        ([4000, 5000, 6000], "", 5000, False),
    ],
)
def test_is_edge(wl, which, expected_midpoint, valid, capsys):
    seg = Segment(wl=wl, flux=[1.0] * len(wl))
    seg.midpoint = np.median(wl)

    seg.is_edge(which)

    if valid:
        assert seg.midpoint == expected_midpoint
    else:
        # Check that midpoint was not changed
        assert seg.midpoint == np.median(wl)

        # Check that error was printed
        captured = capsys.readouterr()
        assert "Error in edge definition" in captured.out


@pytest.mark.parametrize(
    "flux, flux_min_percentile, expected_mad, expected_mad_normal, expected_flux_min, expected_flux_max",
    [
        ([1, 2, 3, 4, 5], 70, 1.0, 1.0 / 3.0, 3.8, 4.92),
    ],
)
def test_get_statistics(
    flux, flux_min_percentile, expected_mad, expected_mad_normal, expected_flux_min, expected_flux_max
):
    seg = Segment(wl=np.arange(len(flux)), flux=flux)
    seg.get_statistics(flux_min=flux_min_percentile)

    assert np.isclose(seg.mad, expected_mad, atol=0.01)
    assert np.isclose(seg.mad_normal, expected_mad_normal, atol=0.01)
    assert np.isclose(seg.flux_min, expected_flux_min, atol=0.01)
    assert np.isclose(seg.flux_max, expected_flux_max, atol=0.01)

    if not np.isnan(seg.flux_med):
        assert seg.flux_min <= seg.flux_med <= seg.flux_max


@pytest.mark.parametrize(
    "mad_normal, flux_med, flux_max, mad_min, mad_range, boost, expected_mad_relative, expected_cont_point",
    [
        # boost=True
        (0.5, 10.0, 20.0, 0.0, 1.0, True, 0.5, 15.0),
        # boost=False
        (0.75, 12.0, 25.0, 0.0, 1.0, False, 0.75, 12.0),
    ],
)
def test_define_cont_point(
    mad_normal, flux_med, flux_max, mad_min, mad_range, boost, expected_mad_relative, expected_cont_point
):
    seg = Segment(wl=[], flux=[])
    seg.mad_normal = mad_normal
    seg.flux_med = flux_med
    seg.flux_max = flux_max

    seg.define_cont_point(mad_min=mad_min, mad_range=mad_range, boost=boost)

    assert np.isclose(seg.mad_relative, expected_mad_relative, atol=0.01)
    assert np.isclose(seg.continuum_point, expected_cont_point, atol=0.01)
