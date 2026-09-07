import numpy as np
import pytest

from casper.interface.gisic.segment import Segment
from casper.interface.gisic.spectrum import Spectrum


def make_dip_spectrum(low=3900.0, high=4000.0, n=100):
    """Synthetic spectrum with two Gaussian absorption dips (matches test_gisic_normalize.py's fixture)."""
    wavelength = np.linspace(low, high, n)
    flux = (
        1.0
        - 0.3 * np.exp(-0.5 * ((wavelength - 3950.0) / 4.0) ** 2)
        - 0.2 * np.exp(-0.5 * ((wavelength - 3970.0) / 2.5) ** 2)
    )
    return wavelength, flux


# ---------------------------------------------------------------------------
# __init__()
# ---------------------------------------------------------------------------


def test_init_sets_wavelength_flux_and_defaults():
    wavelength, flux = make_dip_spectrum()
    spec = Spectrum(wavelength, flux)

    assert np.array_equal(spec.wavelength, wavelength)
    assert np.array_equal(spec.flux, flux)
    assert spec.segments is None
    assert spec.mad_global is None


# ---------------------------------------------------------------------------
# generate_segments() (bins-based, alternative path)
# ---------------------------------------------------------------------------


def test_generate_segments_bins_and_edges():
    wavelength, flux = make_dip_spectrum(n=100)
    spec = Spectrum(wavelength, flux)
    spec.generate_segments(bins=10, lower=70)

    assert len(spec.segments) == 10
    assert all(isinstance(seg, Segment) for seg in spec.segments)
    # first/last segments are marked as edges: midpoint = first/last wavelength of that chunk
    first_chunk_wave = np.array_split(wavelength, 10)[0]
    last_chunk_wave = np.array_split(wavelength, 10)[-1]
    assert spec.segments[0].midpoint == first_chunk_wave[0]
    assert spec.segments[-1].midpoint == last_chunk_wave[-1]
    # get_statistics() was called on each segment
    assert all(hasattr(seg, "mad_normal") for seg in spec.segments)


# ---------------------------------------------------------------------------
# generate_inflection_segments()
# ---------------------------------------------------------------------------


def test_generate_inflection_segments_produces_segments_and_frame():
    wavelength, flux = make_dip_spectrum()
    spec = Spectrum(wavelength, flux)
    spec.generate_inflection_segments(sigma=5, band_check=False, flux_min=50)

    assert len(spec.segments) >= 3  # at least the two forced edge segments + interior ones
    assert spec.segments[0].midpoint == wavelength[0]
    assert spec.segments[-1].midpoint == wavelength[-1]
    assert list(spec.frame.columns) == ["wave", "flux", "d1", "d2"]
    assert len(spec.ZEROS) > 0


def test_generate_inflection_segments_with_cahk_forces_ca_hk_segments():
    wavelength, flux = make_dip_spectrum()
    spec = Spectrum(wavelength, flux)
    cahkwidth = 2
    spec.generate_inflection_segments(sigma=5, band_check=False, flux_min=50, cahk=True, cahkwidth=cahkwidth)

    # Ca H (3916) and Ca K (3991) segments should be present within the configured ±cahkwidth window.
    midpoints = [seg.midpoint for seg in spec.segments]
    assert any(abs(mp - 3916.0) <= cahkwidth for mp in midpoints)
    assert any(abs(mp - 3991.0) <= cahkwidth for mp in midpoints)


def test_generate_inflection_segments_with_band_check_runs_without_error():
    wavelength, flux = make_dip_spectrum()
    spec = Spectrum(wavelength, flux)
    # Should complete without error when molecular-band exclusion is active.
    # band_check=True means segments inside known molecular bands are skipped,
    # but the routine should still keep the left/right edge segments valid.
    spec.generate_inflection_segments(sigma=5, band_check=True, flux_min=50)

    assert spec.segments[0].midpoint == wavelength[0]
    assert spec.segments[-1].midpoint == wavelength[-1]


# ---------------------------------------------------------------------------
# assess_segment_variation()
# ---------------------------------------------------------------------------


def test_assess_segment_variation_computes_expected_statistics():
    wavelength, flux = make_dip_spectrum(n=60)
    spec = Spectrum(wavelength, flux)
    spec.generate_segments(bins=6, lower=70)
    spec.assess_segment_variation()

    expected_mad_array = np.array([seg.mad_normal for seg in spec.segments], dtype=float)
    assert np.array_equal(spec.mad_array, expected_mad_array)
    assert spec.mad_global == pytest.approx(np.median(expected_mad_array))
    assert spec.mad_min == pytest.approx(min(expected_mad_array))
    assert spec.mad_max == pytest.approx(max(expected_mad_array))
    assert spec.mad_range == pytest.approx(spec.mad_max - spec.mad_min)
    expected_relative = np.divide(expected_mad_array - spec.mad_min, spec.mad_range)
    assert np.allclose(spec.mad_relative_array, expected_relative)


# ---------------------------------------------------------------------------
# define_cont_points() / set_segment_midpoints() / set_segment_continuum()
# ---------------------------------------------------------------------------


def test_define_cont_points_and_collect_midpoints_continuum():
    wavelength, flux = make_dip_spectrum(n=60)
    spec = Spectrum(wavelength, flux)
    spec.generate_segments(bins=6, lower=70)
    spec.assess_segment_variation()
    spec.define_cont_points(boost=True)

    expected_continuum_points = [seg.continuum_point for seg in spec.segments]
    midpoints = spec.set_segment_midpoints()
    fluxpoints = spec.set_segment_continuum()

    assert np.allclose(fluxpoints, expected_continuum_points)
    assert np.allclose(midpoints, [seg.midpoint for seg in spec.segments])
    assert np.array_equal(spec.midpoints, midpoints.tolist()) if isinstance(spec.midpoints, list) else True


# ---------------------------------------------------------------------------
# add_continuum_point() / remove_point()
# ---------------------------------------------------------------------------


def test_add_continuum_point_inserts_sorted():
    spec = Spectrum(np.arange(5), np.ones(5))
    spec.midpoints = [10.0, 30.0, 40.0]
    spec.fluxpoints = [1.0, 3.0, 4.0]

    spec.add_continuum_point((20.0, 2.0))

    assert spec.midpoints == [10.0, 20.0, 30.0, 40.0]
    assert spec.fluxpoints == [1.0, 2.0, 3.0, 4.0]


def test_remove_point_removes_by_index_safely():
    spec = Spectrum(np.arange(5), np.ones(5))
    spec.midpoints = [10.0, 20.0, 30.0, 40.0]
    spec.fluxpoints = [1.0, 2.0, 3.0, 4.0]

    spec.remove_point([0, 2])  # remove indices 0 and 2 (sorted, increasing)

    assert spec.midpoints == [20.0, 40.0]
    assert spec.fluxpoints == [2.0, 4.0]


# ---------------------------------------------------------------------------
# set_wavelength() / set_fluxpoints() / get_continuum_points()
# ---------------------------------------------------------------------------


def test_set_wavelength_and_set_fluxpoints_roundtrip():
    spec = Spectrum(np.arange(5), np.ones(5))
    spec.set_wavelength([1.0, 2.0, 3.0])
    spec.set_fluxpoints([10.0, 20.0, 30.0])

    assert spec.midpoints == [1.0, 2.0, 3.0]
    assert spec.fluxpoints == [10.0, 20.0, 30.0]


def test_get_continuum_points_logs_without_error(caplog):
    spec = Spectrum(np.arange(5), np.ones(5))
    spec.set_wavelength([1.0, 2.0])
    spec.set_fluxpoints([10.0, 20.0])

    with caplog.at_level("INFO"):
        spec.get_continuum_points()

    assert "0: 1.0 10.0" in caplog.text
    assert "1: 2.0 20.0" in caplog.text


# ---------------------------------------------------------------------------
# spline_continuum() / normalize()
# ---------------------------------------------------------------------------


def test_spline_continuum_evaluates_at_input_wavelength():
    wavelength = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    spec = Spectrum(wavelength, flux=np.zeros_like(wavelength))
    spec.midpoints = wavelength.tolist()
    spec.fluxpoints = [2.0, 2.0, 2.0, 2.0, 2.0]

    spec.spline_continuum(k=1, s=0)

    assert spec.continuum.shape == wavelength.shape
    assert np.allclose(spec.continuum, 2.0)


def test_normalize_computes_ratio_of_flux_to_continuum():
    spec = Spectrum(np.arange(5), flux=np.array([2.0, 4.0, 6.0, 8.0, 10.0]))
    spec.continuum = np.array([2.0, 2.0, 2.0, 2.0, 2.0])

    spec.normalize()

    # unclipped ratio would be [1, 2, 3, 4, 5]; values > 2.0 get clipped to 1.0
    assert list(spec.flux_norm) == [1.0, 2.0, 1.0, 1.0, 1.0]


def test_normalize_clips_single_out_of_bounds_point_each_side():
    # Regression test for the >1 -> >0 clipping fix: exactly one point above 2.0
    # and exactly one point below 0.0 must both still be clipped.
    spec = Spectrum(np.arange(5), flux=np.array([5.0, 4.0, 4.0, 4.0, -4.0]))
    spec.continuum = np.array([2.0, 2.0, 2.0, 2.0, 2.0])

    spec.normalize()

    # raw ratios: [2.5, 2.0, 2.0, 2.0, -2.0] -> only index 0 (2.5) and index 4 (-2.0) violate bounds
    assert spec.flux_norm[0] == pytest.approx(1.0)  # clipped from 2.5
    assert spec.flux_norm[4] == pytest.approx(0.0)  # clipped from -2.0
    assert list(spec.flux_norm[1:4]) == [2.0, 2.0, 2.0]  # untouched, exactly at boundary (not > 2.0)
