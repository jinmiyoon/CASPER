################################################################################
### Author: Devin Whitten
### Email: devin.d.whitten@gmail.com
### Institute: University of Notre Dame
### Date: Nov 12, 2016
################################################################################

## For now just the plotting functions


import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate as interp
plt.ion()

plt.style.use("ggplot")
plt.rcParams['font.family'] = "Times New Roman"

def plot(spec, dir):
    fig, ax = plt.subplots(3,1, figsize=(10,6), sharex = True)

    ax[0].plot(spec.wavelength, spec.flux, color="black", alpha=0.75, linewidth=0.75)
    [label.set_xlabel(r"Wavelength $\AA$") for label in ax]

    ax[0].set_ylabel(r"Flux")
    ax[1].set_ylabel(r"Normalized")

    ax[0].scatter(spec.midpoints, spec.fluxpoints, s =10, zorder=3, color="black")
    ax[0].plot(spec.frame.wave, spec.smooth)

    [[label.axvline(midpoint, linestyle="--", color="green", linewidth=0.5) for midpoint in spec.midpoints] for label in ax]
    #[[label.axvline(midpoint, linestyle="--", color="black", linewidth=0.5) for midpoint in spec.ZEROS] for label in ax]

    ### Continuum
    ax[0].plot(spec.wavelength, spec.continuum, zorder=3, color="black", alpha=0.75, linewidth=0.75)

    [ax[0].fill_between(np.linspace(segment.wl[0], segment.wl[-1], 30),
                        np.ones(30)*segment.flux_min, np.ones(30)*segment.flux_max,
                    color="gold", zorder=3, alpha=0.65) for segment in spec.segments]





    ### adding the derivatives here
    ax[1].plot(spec.wavelength, spec.frame.d1, color="black", alpha=0.75, linewidth=0.75, label=r'$\frac{df}{d\lambda}$')
    ax[1].plot(spec.wavelength, spec.frame.d2, color="purple", label=r'$\frac{d^2f}{d\lambda^2}$')
    #ax[1].scatter(spec.wavelength, spec.frame.d2, color="purple", s=10)
    ax[1].legend(loc=1)
    ax[0].set_title(spec.name, fontsize=14)

    ### Normalization
    ax[2].plot(spec.wavelength, spec.flux_norm, color="black", alpha=0.75, linewidth=0.75)
    ax[2].axhline(1., linestyle="--", color="black")
    [label.tick_params(direction="in", length=0) for label in ax]
    [label.axvline(3934, linestyle='--', linewidth=0.75) for label in ax]
    [label.axvline(3969, linestyle='--', linewidth=0.75) for label in ax]

    [label.set_xlim([3800, 5000]) for label in ax]
    ax[2].set_ylim([0., 1.2])
    #plt.savefig(dir + spec.name + ".png")
    plt.show()
    #plt.pause(1.)
    #plt.clf()
    #plt.close()
    input("Press any key to continue")
    plt.close()






### APPEND NORMALIZED FLUX TO FITS
def update_fits(spec, fits):
    ### Precondition: spec needs flux_norm define
    ### we'll add the continuum to the fits file as well.

    #fits[0].header['ARRAY5'] = "FLUX_NORM"
    #fits[0].header['ARRAY6'] = "CONTINUUM"

    #fits[0].data = np.vstack((fits[0].data[0][0],  fits[0].data[1][0],
                              #fits[0].data[2][0], fits[0].data[3][0], spec.flux_norm, spec.continuum))#spec.flux_norm #np.vstack((fits[0].data, spec.flux_norm, spec.continuum))

    #fits[0].data[0] = spec.flux_norm

    return pd.DataFrame({"wave": spec.wavelength,
                         "flux": spec.flux,
                         "norm": spec.flux_norm})


def in_molecular_band(wl, tol=10):
    #print("Checking wavelength for band", wl)
    ### Checks to see if wavelength is within unacceptable limits of known bands
    bands = {"gband": [4200., 4400.],
             "C2":    [5060., 5180.]}

    for band in bands:
        #print(band, bands[band])
        if (wl > bands[band][0]) & (wl < bands[band][1]):
            #print("\t in ", band)
            return True

    return False
