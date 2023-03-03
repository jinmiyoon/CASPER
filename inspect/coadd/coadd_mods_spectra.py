#!/usr/bin/env python
# coding: utf-8

## This script cross-correlate mods1 and mods2 to align wavelength for coadding the two spectra. 
## No normalization before cross-correlation.

import numpy as np
import astropy.units as u
from specutils import Spectrum1D
from specutils.analysis import correlation
from specutils import SpectralRegion
from specutils.manipulation import noise_region_uncertainty
from astropy.nddata import StdDevUncertainty
import os, sys
from astropy.io import fits, ascii
import matplotlib.pyplot as plt
from ipywidgets import interact
import importlib
import copy
import warnings
import copy
from scipy import interpolate

sys.path.insert(0,'../../interface/')
import GISIC_C as GISIC

spectra_path= "../../inputs/spectra/bf-full-survey-data/"

with open("coadd_list.csv") as f:
    coadd_list = f.read().splitlines()

rv_shift =[]
wave_shift =[]
star_list =[]
for i, star_name in enumerate(coadd_list):
#star_name = "g77-61"
    print("star name = ", star_name)
    print(type(star_name))

    with fits.open(spectra_path+star_name+"_m1b_casper.fits") as spec1:
        # assign wave and flux
        wave1= (np.arange(0, spec1[0].header['NAXIS1'], 1) * spec1[0].header['CD1_1']) + spec1[0].header['CRVAL1']
        flux1 =  spec1[0].data[0].flatten()

    with fits.open(spectra_path+star_name+"_m2b_casper.fits") as spec2:
        wave2= (np.arange(0, spec2[0].header['NAXIS1'], 1) * spec2[0].header['CD1_1']) + spec2[0].header['CRVAL1']
        flux2 =  spec2[0].data[0].flatten()

    # For two spectra having different sized arrays, we need to reframe the arrays.
    # To do so, we redefine the wavelength range and interpolate flux based on the new wave array.

    wave = np.arange(3700, 5000, 0.55)
    reframe_intp1= interpolate.interp1d(wave1,flux1, kind='cubic' )
    reframe_intp2= interpolate.interp1d(wave2,flux2, kind='cubic' )

    reflux1 = reframe_intp1(wave)
    reflux2 = reframe_intp2(wave)

    # normalize spectra using GISIC
    #wave, norm_flux1, continuum1 = GISIC.normalize(wave, reflux1, sigma=30, k=1,
    #                                                cahk=True,band_check=True, boost=True)
    #wave, norm_flux2, continuum2 = GISIC.normalize(wave, reflux2, sigma=30, k=1,
    #                                                cahk=True,band_check=True, boost=True)

    # Using specutils Spectrum1D, trim spectra for reasonable
    uncertainty = StdDevUncertainty(0.1*np.ones(len(wave))*u.photon)

    spec1_utils = Spectrum1D(spectral_axis=wave*u.AA, flux=reflux1*u.photon,uncertainty=uncertainty, velocity_convention='optical', rest_value=4320. *u.AA)
    spec2_utils = Spectrum1D(spectral_axis=wave*u.AA, flux=reflux2*u.photon,uncertainty=uncertainty, velocity_convention='optical')


    # uncertainty calculation
    # this value can be used in Spectrum1D for cross-correlation below.
    noise_region = SpectralRegion([(3900, 3920), (4530, 4550)] * u.AA)
    spec_w_unc = noise_region_uncertainty(spec1_utils, noise_region)
    spec_w_unc.uncertainty

    with warnings.catch_warnings():  # Ignore warnings
        warnings.simplefilter('ignore')
        corr, lag = correlation.template_correlate(spec1_utils, spec2_utils)

    maxind = np.argmax(corr)

    print(lag[maxind])
    print(corr[maxind])

    spec2_shift= copy.deepcopy(spec2_utils)
    spec2_utils.spectral_axis
    spec2_shift.shift_spectrum_to(radial_velocity = lag[maxind])
    spec2_shift.spectral_axis

    wave_shift.append(round((spec2_shift.spectral_axis[-1].value - spec2_utils.spectral_axis[-1].value), 3))
    rv_shift.append(round(lag[maxind].value, 3))
    star_list.append(star_name)
    # shift makes two spectra disaligned over wavelength so we need to reframed wave/flux.

    new_wave = np.arange(3750, 4950, 0.55) * u.AA

    fct_intp1= interpolate.interp1d(spec1_utils.spectral_axis.value, spec1_utils.flux.value, kind = 'cubic')
    fct_intp2= interpolate.interp1d(spec2_shift.spectral_axis.value, spec2_shift.flux.value, kind = 'cubic')


    new_flux1 = fct_intp1(new_wave)
    new_flux2 = fct_intp2(new_wave)

    coadd_flux = (new_flux1+new_flux2)

    new_wave, norm_coadd_flux, continuum_coadd = GISIC.normalize(new_wave, coadd_flux, sigma=30, k=1,
                                                    cahk=True,band_check=True, boost=True)

    plt.plot(spec1_utils.spectral_axis, spec1_utils.flux, lw=0.5, label='m1b')
    plt.plot(spec2_utils.spectral_axis, spec2_utils.flux, lw=0.5, ls='--', label='m2b')
    plt.plot(spec2_shift.spectral_axis, spec2_shift.flux, lw=0.5, label='m2b shift')
    plt.text(4300, 500, "RV shift "+str(round(lag[maxind].value, 3))+ " km/s" )
    plt.xlim(3900, 4400)
    plt.title(star_name+" cross-correlation")
    plt.legend()
    plt.savefig("spectra_plots/"+star_name+'_cross-correlated-shift.pdf')
    plt.close()

    plt.plot(new_wave, coadd_flux,  lw=0.5, label ='coadded')
    plt.plot(new_wave, new_flux1,  lw=0.5, label ='mods1')
    plt.plot(new_wave, new_flux2,  lw=0.5, label ='mods2 shift')

    #plt.axhline(1.0)
    plt.title(star_name+" coaddition of m1b and m2b")
    plt.xlim(3900, 4400)
    plt.legend(loc=4)
    plt.savefig("spectra_plots/"+star_name+'_coadd-spectrum.pdf')
    plt.close()

    ascii.write([new_wave, coadd_flux], "coadded_spectra/"+star_name+"_coadd_spectrum.csv",
        names= ["wave", "flux"], format='csv', overwrite=True)
    #ascii.write([new_wave, norm_coadd_flux], star_name+"_coadd_norm_spectrum.csv",
    #    names= ["wave", "flux"], format='csv', overwrite=True)
ascii.write([star_list, wave_shift, rv_shift],"rv-shift-between-mods1-mods2.csv",
    names=["star_name", "wave_shift", "rv_shift"], format='csv', overwrite=True )
