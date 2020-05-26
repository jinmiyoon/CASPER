### Author: Devin Whitten
### Email: devin.d.whitten@gmail.com
### University of Notre Dame
### This is the interface for the archetype assessment routines

import os, sys
import pandas as pd
import numpy as np
#import h5py
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import GISIC as GISIC_C

### This is unnormalized..
#arch_base = h5py.File("inputs/archetypes/arch_base.hdf5", 'r')


def CHI_CAHK(spec, synth_f):

    spec_flux = spec.norm_flux[(spec.wavelength > 3925) & (spec.wavelength < 3980)]
    spec_wave = spec.wavelength[(spec.wavelength > 3925) & (spec.wavelength < 3980)]

    CHI = np.square(np.divide(spec_flux - synth_f(spec_wave), synth_f(spec_wave)))

    return CHI.sum()/len(CHI)


def CHI_GBAND(spec, synth_f):

    spec_flux = spec.norm_flux[(spec.wavelength > 4222) & (spec.wavelength < 4322)]
    spec_wave = spec.wavelength[(spec.wavelength > 4222) & (spec.wavelength < 4322)]

    CHI = np.square(np.divide(spec_flux - synth_f(spec_wave), synth_f(spec_wave)))

    return CHI.sum()/len(CHI)



def assess(spec):
    ## Here is where we will perform the chi2 statistics

    #GROUP 1
    GI_D_HK, GI_D_GBAND = [], []

    group_list = ['G1/dwarf/', 'G1/giant/', 'G2/dwarf/', 'G2/giant/', 'G3/dwarf/','G3/giant/']

    ### initialize chi_frame with empty lists for each group/class
    chi_frame = dict([(title, []) for title in group_list])

    for title in group_list:
        print("... assessing " + title)

        for ele in arch_base[title + 'spec']:
            ### I suppose interpolation should happen here

            synth_f = interp1d(ele[:,0], ele[:,1])
            chi_frame[title].append(np.average([CHI_CAHK(spec, synth_f), CHI_GBAND(spec, synth_f)]))


    return chi_frame


def determine_class(chi_frame):
    ### Determines best
    ##
    medians = []
    for key in chi_frame.keys():
        print(key , np.median(np.array(chi_frame[key])))
        medians.append(np.median(np.array(chi_frame[key])))

    #return group, value
    print('Suggested Group/Class:    ' , list(chi_frame.keys())[np.argsort(medians)[0]])

    return list(chi_frame.keys())[np.argsort(medians)[0]], medians[np.argsort(medians)[0]], np.argsort(medians)[0]



def plot_chi(chi_frame):

    fig = plt.figure(figsize=(6, 8))
    for i, key in enumerate(chi_frame.keys()):
        print(key)
        ax = plt.subplot(3,2,i+1)
        ax.plot(np.array(arch_base[key + 'teff'])[np.argsort(arch_base[key + 'teff'])],
                np.array(chi_frame[key])[np.argsort(arch_base[key + 'teff'])])

        ax.set_title(key.split("/"))

    plt.show()
    ### GROUP1
    return
