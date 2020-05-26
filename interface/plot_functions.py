################################################################################
### Author: Devin Whitten
### Email: devin.d.whitten@gmail.com
### Institute: University of Notre Dame
################################################################################

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import interp1d
import corner
import MCMC_interface
from matplotlib.backends.backend_pdf import PdfPages

#####
### This is just useful for many of the functions. Wanna keep format consistent

#####

plt.ion()

plt.style.use('classic')
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['xtick.labelsize'] = 7
plt.rcParams['ytick.labelsize'] = 7
plt.rcParams['axes.linewidth'] = 0.75

def produce_title(spectrum):
    ## just returns a nice looking string for the plot title
    MCMC_DICT = spectrum.get_mcmc_dict(mode='BOTH')

    return spectrum.get_name() + "  " + spectrum.get_arch_group() + \
    '   Teff : %.0F  [Fe/H] : %.2F   [C/Fe] : %.2F   A(C) : %.2F' % (MCMC_DICT[0]['TEFF'][0] , MCMC_DICT[1]['FEH'][0], MCMC_DICT[1]['CFE'][0], MCMC_DICT[1]['AC'][0]) + \
    "   MODE:  " + spectrum.get_carbon_mode() + "   CLASS: " + spectrum.get_gravity_class()


def plot_spectra(spectra_batch):
    ## to visualize normalizations
    ## I want multiple pages of 4x2
    CA_XLIM = [3910, 3980]
    LINEW   = 0.5

    print("... generating continuum plots")
    print("\t saving as:   ", spectra_batch.io_params['output_name'])
    rows, columns = 8, 5

    pages = int(np.ceil(spectra_batch.length/(rows * columns)))

    pp = PdfPages('output/' + spectra_batch.io_params['output_name'] + '_spec.pdf')
    count = 0

    for i, spec in enumerate(spectra_batch.spectra_array):  ### loop through pages
        if i % rows == 0: ## if new page required
            fig, ax = plt.subplots(rows, columns, figsize=(8.5, 11), dpi=200)
            fig.subplots_adjust(hspace=0.5)



            [label.set_ylim([0.0, 1.2]) for label in np.concatenate(ax[:, 1:])]
            [label.set_yticks([0.0, 0.5,1.0, 1.2]) for label in np.concatenate(ax[:, 1:])]

            ## CaII
            [label.ticklabel_format(axis='both', useOffset=False) for label in ax[:, 2]]
            [label.set_xlim(CA_XLIM) for label in ax[:, 2]]
            [label.set_xticks([3915, 3930, 3945]) for label in ax[:, 2]]

            ## CH
            [label.set_xlim([4225, 4325]) for label in ax[:, 3]]
            [label.set_xticks(np.arange(4225, 4350, 25)) for label in ax[:, 3]]

            ## C2
            [label.set_xlim([4650, 4750]) for label in ax[:, 4]]
            [label.set_xticks(np.arange(4650, 4775,25)) for label in ax[:, 4]]

            [plt.setp(label.get_yticklabels(), visible=False) for label in ax[:, 0]]


            ## FILL BETWEENS


            [label.tick_params(direction='in', right=True, top=True) for label in np.concatenate(ax[:])]


        index = i % rows

        ##### MAIN PLOT SECTION

        ax[index, 0].set_yticks([0.0, max(spec.frame['flux'])])
        [label.set_xticks(np.linspace(min(spec.frame['wave']), max(spec.frame['wave']),5)) for label in ax[index,0:2]]

        ### Set title
        ax[index, 2].set_title(produce_title(spec), fontsize=10)


        ### Continuum Plot
        ax[index, 0].plot(spec.frame['wave'], spec.frame['flux'], linewidth=LINEW, color='black')
        ax[index, 0].plot(spec.frame['wave'], spec.frame['cont'], linewidth=LINEW)


        ### Normalization Plot
        ax[index, 1].axhline(1.00, linewidth=0.75, linestyle='--', color='red')
        ax[index, 1].plot(spec.frame['wave'], spec.frame['norm'], linewidth=LINEW, color='black')

        ### CaII Plot

        ax[index, 2].axhline(1.00, linewidth=0.75, linestyle='--', color='red')
        ax[index, 2].plot(spec.frame['wave'],spec.frame['norm'],
                                             linewidth=LINEW, color='black')

        ### CH Plot
        ax[index, 3].axhline(1.00, linewidth=0.75, linestyle='--', color='red')
        ax[index, 3].plot(spec.frame['wave'][spec.frame['wave'].between(4150, 4500, inclusive=True)],
                                             spec.frame['norm'][spec.frame['wave'].between(4150, 4500, inclusive=True)],
                                             linewidth=LINEW, color='black')

        ### C2 Plot
        ax[index, 4].axhline(1.00, linewidth=0.75, linestyle='--', color='red')
        ax[index, 4].plot(spec.frame['wave'][spec.frame['wave'].between(4650, 4850, inclusive=True)],
                                             spec.frame['norm'][spec.frame['wave'].between(4650, 4850, inclusive=True)],
                                             linewidth=LINEW, color='black')

        ###### SIGMA SHADING SECTION
        ############################
        synth_function = interp1d(spec.synth_spectrum['wave'], spec.synth_spectrum['norm'])
        ############################
        #CaII
        CA_WAVE = np.linspace(*spec.KP_bounds, 30)
        ax[index, 2].fill_between(CA_WAVE, synth_function(CA_WAVE) * (1. - spec.MCMC_COARSE['XI_CA'][0]),
                                           synth_function(CA_WAVE) * (1. + spec.MCMC_COARSE['XI_CA'][0]),
                                                      color='purple', alpha=0.25)

        [ax[index, 2].axvline(edge, linestyle='dotted', linewidth=0.75, alpha=0.8) for edge in CA_WAVE[[0,-1]]]

        #CH
        CH_WAVE = np.linspace(*list(spec.regions['CH']['wave'].iloc[[0,-1]]), 30)
        ax[index, 3].fill_between(CH_WAVE, synth_function(CH_WAVE) * (1. - spec.MCMC_COARSE['XI_CH'][0]),
                                           synth_function(CH_WAVE) * (1. + spec.MCMC_COARSE['XI_CH'][0]),
                                                      color='purple', alpha=0.25)


        #[ax[index, 3].axvline(edge, linestyle='dotted', linewidth=0.75, alpha=0.8) for edge in CA_WAVE[[0,-1]]]


        ###
        #if spec.get_carbon_mode()   == "CH":
        #    [label.plot(spec.synth_spectrum['wave'], spec.synth_spectrum['norm'], color='purple', linewidth=0.75, alpha=0.75) for label in ax[index, 1:-1]]


        #elif spec.get_carbon_mode() == "CH+C2":
        [label.plot(spec.synth_spectrum['wave'], spec.synth_spectrum['norm'],
                    color='purple', linewidth=LINEW, alpha=0.75) for label in ax[index, 1:]]

        ax[index,1].set_xlim([spec.frame['wave'].iloc[0], spec.frame['wave'].iloc[-1]])


        if (i + 1) % rows == 0 or (i + 1) == spectra_batch.length:

            pp.savefig(fig)

    plt.close()
    pp.close()

    return

##########
def plot_corner_array(spec_batch):

    pp = PdfPages('output/' + spec_batch.io_params['output_name'] + '_corner.pdf')

    fig_handle = []

    for item in spec_batch.spectra_array:
        fig_handle.append(plot_single_corner(item, spec_batch.io_params['output_name']))

    plt.close()
    [pp.savefig(fig) for fig in fig_handle]

    pp.close()

    return


def plot_single_corner(spectrum, io_path, burnin=0.25):

    ## for now sampler is sampler.chain
    ### There are three conditions, based on ndim
    ### get number of dimensions

    sampler = spectrum.MCMC_COARSE_sampler

    try:
        ndim = sampler.shape[2]
        iter = sampler.shape[1]

    except:
        ndim = sampler.chain.shape[2]
        iter = sampler.chain.shape[1]
        sampler = sampler.chain

    samples = sampler[:, int(burnin * iter):, :].reshape((-1, ndim))


    if ndim == 6:
        labels = [r'$T_{\rm eff}', '[Fe/H]', '[C/Fe]', r'S/N$_{\rm CaII}$', r'S/N$_{\rm CH}$', r'S/N$_{\rm C2}$']  #r'$\xi_{\rm CaII}$', r'$\xi_{\rm CH}$', r'$\xi_{\rm C2}$'
        #for i in range(3, ndim):
        #    samples[:, i] = np.divide(1., samples[:, i])

    elif ndim == 5:
        labels = [r'$T_{\rm eff}$', '[Fe/H]', '[C/Fe]', r'S/N$_{\rm CaII}$', r'S/N$_{\rm CH}$']
        #for i in range(3, ndim):
        #    samples[:, i] = np.divide(1., samples[:, i])

    elif ndim == 2:
        ### Fine parameters case
        labels = ['[Fe/H]', '[C/Fe]']



    fig = corner.corner(samples,
                        labels=labels,
                        color='black', hist_kwargs={'density': True})

    name = spectrum.get_name()
    fig.suptitle(name, fontsize=15)



    MEDIAN = np.median(samples, axis=0)

    #### MEDIAN is depreciated here, since I use a different estimate in kde_param
    value2 = [MCMC_interface.kde_param(row, x0 = x0)['result'] for row, x0 in zip(samples.T, MEDIAN)]
    kde_array = [MCMC_interface.kde_param(row, x0 = x0)['kde'] for row, x0 in zip(samples.T, MEDIAN)]

    std =    np.std(samples, axis=0)


    axes = np.array(fig.axes).reshape((ndim, ndim))

    ### This the parameter case.

    for yi in range(ndim):
        for xi in range(yi):
            ax = axes[yi, xi]
            ax.axvline(value2[xi], color="r")
            ax.axhline(value2[yi], color="r")
            ax.plot(value2[xi], value2[yi], "sr")


    for i in range(ndim):
        span = np.linspace(min(samples.T[i]), max(samples.T[i]), 30)
        axes[i,i].axvline(value2[i], color='r', alpha=0.75)
        axes[i,i].plot(span, kde_array[i].evaluate(span))

    [label.tick_params(direction='in', right=True, top=True) for label in axes.flatten()]

    plt.close()
    return fig











def plot_spectrum(spectrum):
    #### This is the main that I want to work with
    fig, ax = plt.subplots(1, 3, figsize=(10,3))

    ax[0].plot(spectrum.frame['wave'], spectrum.frame['flux'])
    ax[0].plot(spectrum.frame['wave'], spectrum.frame['cont'])

    ax[1].plot(spectrum.frame['wave'], spectrum.frame['norm'])

    return fig
################################################################################
################################################################################


def chi_plot(chi_frame, filename, alt = None, teff = None):
    ### chi_frame comes from synthetic_functions.interp_run
    fig, ax = plt.subplots(1,2, figsize=(10,3))
    #title = "KPNO21_2011  -  HE 0319 - 0215"
    title = filename + "  -  " + alt
    ax[0].plot(chi_frame['temp'], chi_frame['GI_D'], label='GI', color='blue')
    ax[0].plot(chi_frame['temp'], chi_frame['GII_D'], label='GII', color='green')
    ax[0].plot(chi_frame['temp'], chi_frame['GIII_D'], label='GIII', color='orange')

    ax[1].plot(chi_frame['temp'], chi_frame['GI_G'], label='GI', color='blue')
    ax[1].plot(chi_frame['temp'], chi_frame['GII_G'], label='GII', color='green')
    ax[1].plot(chi_frame['temp'], chi_frame['GIII_G'], label='GIII', color='orange')

    ax[0].set_title('Dwarf')
    ax[1].set_title('Giant')

    fig.suptitle(title)
    if teff != None:
        [label.axvline(teff, linestyle='--') for label in ax]

    #ax[1].set_ylim([0.0, 0.06])

    [label.set_ylabel(r"$\chi^2$") for label in ax]
    [label.set_xlabel(r"T$_{\rm eff}$", labelpad=-10) for label in ax]
    [label.tick_params(direction='in', top=True, right=True) for label in ax]


    ax[0].legend()

    plt.savefig("results/" + filename + ".pdf", format='pdf')



def GI_plot(spec_frame, arch_lib, filename):

    fig, ax = plt.subplots(2, 2, figsize=(10, 8))
    cmap = plt.cm.jet(np.linspace(0,1,20))

    ax[0,0].set_title("Dwarf")
    ax[0,1].set_title("Giant")


    ### Dwarf Left
    [ax[0, 0].plot(arch_lib['wave'], arch_lib['GI_D'][i], c=cmap[i], alpha=0.5, linewidth=0.75) for i in range(len(arch_lib['GI_D']))]
    [ax[1, 0].plot(arch_lib['wave'], arch_lib['GI_D'][i], c=cmap[i], alpha=0.5, linewidth=0.75) for i in range(len(arch_lib['GI_D']))]

    ### Giant Right
    [ax[0, 1].plot(arch_lib['wave'], arch_lib['GI_G'][i], c=cmap[i], alpha=0.5, linewidth=0.75) for i in range(len(arch_lib['GI_D']))]
    [ax[1, 1].plot(arch_lib['wave'], arch_lib['GI_G'][i], c=cmap[i], alpha=0.5, linewidth=0.75) for i in range(len(arch_lib['GI_D']))]


    [label.plot(spec_frame['wave'], spec_frame['norm'], color='black', linewidth=0.75, alpha=0.75) for label in ax[:,0]]
    [label.plot(spec_frame['wave'], spec_frame['norm'], color='black', linewidth=0.75, alpha=0.75) for label in ax[:,1]]

    [label.set_xlabel("Wavelength") for label in ax[1, :]]


    [label.set_xlim([3925, 3980]) for label in ax[0,:]]
    [label.set_xlim([4200, 4350]) for label in ax[1,:]]
    [label.set_ylim([0, 1.2]) for label in ax[0,:]]
    [label.set_ylim([0, 1.2]) for label in ax[1,:]]

    fig.suptitle("Group I : [Fe/H] = -2.5 A(C) = 7.9")

    plt.savefig("results/" + filename, format='pdf')


    return

def GII_plot(spec_frame, arch_lib, filename):

    fig, ax = plt.subplots(2, 2, figsize=(10, 8))
    cmap = plt.cm.jet(np.linspace(0,1,20))

    ax[0,0].set_title("Dwarf")
    ax[0,1].set_title("Giant")


    ### Dwarf Left
    [ax[0, 0].plot(arch_lib['wave'], arch_lib['GII_D'][i], c=cmap[i], alpha=0.5, linewidth=0.75) for i in range(len(arch_lib['GII_D']))]
    [ax[1, 0].plot(arch_lib['wave'], arch_lib['GII_D'][i], c=cmap[i], alpha=0.5, linewidth=0.75) for i in range(len(arch_lib['GII_D']))]

    ### Giant Right
    [ax[0, 1].plot(arch_lib['wave'], arch_lib['GII_G'][i], c=cmap[i], alpha=0.5, linewidth=0.75) for i in range(len(arch_lib['GII_D']))]
    [ax[1, 1].plot(arch_lib['wave'], arch_lib['GII_G'][i], c=cmap[i], alpha=0.5, linewidth=0.75) for i in range(len(arch_lib['GII_D']))]


    [label.plot(spec_frame['wave'], spec_frame['norm'], color='black', linewidth=0.75, alpha=0.75) for label in ax[:,0]]
    [label.plot(spec_frame['wave'], spec_frame['norm'], color='black', linewidth=0.75, alpha=0.75) for label in ax[:,1]]

    [label.set_xlabel("Wavelength") for label in ax[1, :]]


    [label.set_xlim([3925, 3980]) for label in ax[0,:]]
    [label.set_xlim([4200, 4350]) for label in ax[1,:]]

    [label.set_ylim([0, 1.2]) for label in ax[0,:]]
    [label.set_ylim([0, 1.2]) for label in ax[1,:]]

    fig.suptitle("Group II : [Fe/H] = -3.5  A(C) = 5.9")

    plt.savefig("results/" + filename, format='pdf')


    return

def GIII_plot(spec_frame, arch_lib, filename):
    fig, ax = plt.subplots(2, 2, figsize=(10, 8))
    cmap = plt.cm.jet(np.linspace(0,1,20))

    ax[0,0].set_title("Dwarf")
    ax[0,1].set_title("Giant")


    ### Dwarf Left
    [ax[0, 0].plot(arch_lib['wave'], arch_lib['GIII_D'][i], c=cmap[i], alpha=0.5, linewidth=0.75) for i in range(len(arch_lib['GIII_D']))]
    [ax[1, 0].plot(arch_lib['wave'], arch_lib['GIII_D'][i], c=cmap[i], alpha=0.5, linewidth=0.75) for i in range(len(arch_lib['GIII_D']))]

    ### Giant Right
    [ax[0, 1].plot(arch_lib['wave'], arch_lib['GIII_G'][i], c=cmap[i], alpha=0.5, linewidth=0.75) for i in range(len(arch_lib['GIII_D']))]
    [ax[1, 1].plot(arch_lib['wave'], arch_lib['GIII_G'][i], c=cmap[i], alpha=0.5, linewidth=0.75) for i in range(len(arch_lib['GIII_D']))]


    [label.plot(spec_frame['wave'], spec_frame['norm'], color='black', linewidth=0.75, alpha=0.75) for label in ax[:,0]]
    [label.plot(spec_frame['wave'], spec_frame['norm'], color='black', linewidth=0.75, alpha=0.75) for label in ax[:,1]]

    [label.set_xlabel("Wavelength") for label in ax[1, :]]


    [label.set_xlim([3925, 3980]) for label in ax[0,:]]
    [label.set_xlim([4200, 4350]) for label in ax[1,:]]

    [label.set_ylim([0, 1.2]) for label in ax[0,:]]
    [label.set_ylim([0, 1.2]) for label in ax[1,:]]


    fig.suptitle("Group III : [Fe/H] = -4.3  A(C) = 7.0")

    plt.savefig("results/" + filename, format='pdf')


    return

################################################################################

def plot_crit(frame_array, group_class):

    ## for the output from determine_crit_params

    AC_VALUES  = np.unique(frame_array['CARBON'])
    FEH_VALUES = np.unique(frame_array['FEH'])
    TEFF_VALUES = np.unique(frame_array['T'])
    xscale = 2.5
    yscale = 1.5
    print("Unique Carbon Values:  ", len(AC_VALUES))

    cmap = plt.cm.jet(np.linspace(0,1, len(TEFF_VALUES)))
    carbon = {key : value for key, value in zip(GROUP_STR, ["AC", 'AC', "CFE", 'CFE', 'AC', 'AC'])}

    ### now it will adjust fig
    columns = 4
    rows = int(np.ceil(len(AC_VALUES) / columns ))

    print(rows)
    fig = plt.figure(figsize=(rows * yscale, columns * xscale))
    handles = []
    for i  in range(len(AC_VALUES)):

        ax = fig.add_subplot(rows, columns, i+1)

        slice = frame_array[(frame_array['CARBON'] == AC_VALUES[i])]

        [ax.plot(slice[slice['T'] == VALUE]['FEH'], slice[slice['T'] == VALUE]['CHI'], color=cmap[i]) for i, VALUE in enumerate(TEFF_VALUES)]
        ax.set_title(carbon[group_class] + ':  %.2F' % AC_VALUES[i])
        handles.append(ax)

    fig.subplots_adjust(hspace=0.5)
    [label.tick_params(direction='in', top=True, right=True) for label in handles]


#########

def plot_crit_3D(frame_array, group_class):
    ## Should really do this in 3D anyway.
    ## for the output from determine_crit_params

    AC_VALUES  = np.unique(frame_array['CARBON'])
    FEH_VALUES = np.unique(frame_array['FEH'])
    TEFF_VALUES = np.unique(frame_array['T'])


    print("Unique Carbon Values:  ", len(AC_VALUES))

    cmap = plt.cm.jet(np.linspace(0,1, len(TEFF_VALUES)))
    carbon = {key : value for key, value in zip(GROUP_STR, ["AC", 'AC', "CFE", 'CFE', 'AC', 'AC'])}

    ### now it will adjust fig
    columns = 4
    rows = int(np.ceil(len(AC_VALUES) / columns ))

    print(rows)
    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    handles = []
    for i  in range(len(TEFF_VALUES)):



        slice = frame_array[(frame_array['T'] == TEFF_VALUES[i])]

        #ax.scatter(slice['FEH'], slice['CARBON'], slice['CHI'])
        ax.plot_trisurf(slice['FEH'], slice['CARBON'], -np.log(slice['CHI']), alpha=0.50, color=cmap[i])

        #[ax.plot(slice[slice['T'] == VALUE]['FEH'], slice[slice['T'] == VALUE]['CHI'], color=cmap[i]) for i, VALUE in enumerate(TEFF_VALUES)]
        #ax.set_title(carbon[group_class] + ':  %.2F' % AC_VALUES[i])
        handles.append(ax)
        ax.view_init(30, 25)
    #ax.set_zlim(ax.get_zlim()[::-1])
    #fig.subplots_adjust(hspace=0.5)

    ax.set_xlabel('[Fe/H]', fontsize=14)
    ax.set_ylabel(carbon[group_class], fontsize=14)
    ax.set_zlabel(r"$-\xi_{\omega}^2$", fontsize=14)

    [label.tick_params(direction='in', top=True, right=True) for label in handles]

    plt.show()


def plot_mcmc_sampler(SAMPLER, ndim, burnin,
                    suptitle, filename, acc_params, group):

    CARBON = {"GI" : r"A$(C)$",
              "GII": "[C/Fe]",
              "GIII": r"A$(C)$"}

    samples = SAMPLER.chain[:, 500:, :].reshape((-1, ndim))
    fig = corner.corner(samples,
                        labels=[r'$T_{\rm eff}$', '[Fe/H]', CARBON[group]],
                        color='black')



    fig.suptitle(suptitle, fontsize=15)
    value2 = np.median(samples, axis=0)
    std =    np.std(samples, axis=0)


    axes = np.array(fig.axes).reshape((ndim, ndim))

    for yi in range(ndim):
        for xi in range(yi):
            ax = axes[yi, xi]
            ax.axvline(value2[xi], color="r")
            ax.axhline(value2[yi], color="r")
            ax.plot(value2[xi], value2[yi], "sr")

    for i in range(ndim):
        axes[i,i].axvline(value2[i], color='r', alpha=0.75)

    ### For the real values
    if acc_params != None:
        for yi in range(ndim):
            for xi in range(yi):
                ax = axes[yi, xi]
                ax.axvline(acc_params[xi], color="g")
                ax.axhline(acc_params[yi], color="g")
                ax.plot(acc_params[xi], acc_params[yi], "sg")
        for i in range(ndim):
            axes[i,i].axvline(acc_params[i], color='g', alpha=0.75)


    #textstr = '\n'.join((
    #            r'$T_{\rm eff}=$%.0f K' % (value2[0], ),
    #            r'[Fe/H]=%.2f' % (value2[1], ),
    #            CARBON[group] + '=%.2f' % (value2[2], )))

    #props = dict(boxstyle='round', facecolor='white', alpha=0.5)

    #axes[0,1].text(0.05, 0.95, textstr, transform=axes[0,1].transAxes, fontsize=14,
    #    verticalalignment='top', bbox=props)


    return






#########################################
## Somehow code got deleted, which is crazy. So I'm rewriting..
##########################################


def plot_mcmc_samples(sampler, burnin = 0.25, params=None, suptitle=None, filename=None):
    ## for now sampler is sampler.chain
    ### There are three conditions, based on ndim
    ### get number of dimensions
    try:
        ndim = sampler.shape[2]
        iter = sampler.shape[1]

    except:
        ndim = sampler.chain.shape[2]
        iter = sampler.chain.shape[1]
        sampler = sampler.chain


    if ndim == 6:
        labels = ['Teff', '[Fe/H]', '[C/Fe]', 'sigmaCA', 'XI_CH', 'XI_C2']

    elif ndim == 5:
        labels = ['Teff', '[Fe/H]', '[C/Fe]', 'sigmaCA', 'XI_CH']

    elif ndim == 2:
        ### Fine parameters case
        labels = ['[Fe/H]', '[C/Fe]']


    samples = sampler[:, int(burnin * iter):, :].reshape((-1, ndim))


    fig = corner.corner(samples,
                        labels=labels,
                        color='black', hist_kwargs={'normed': True})


    if suptitle != None:
        fig.suptitle(suptitle, fontsize=15)


    MEDIAN = np.median(samples, axis=0)
    value2 = [MCMC_interface.kde_param(row, x0 = x0)['result'] for row, x0 in zip(samples.T, MEDIAN)]
    kde_array = [MCMC_interface.kde_param(row, x0 = x0)['kde'] for row, x0 in zip(samples.T, MEDIAN)]

    std =    np.std(samples, axis=0)


    axes = np.array(fig.axes).reshape((ndim, ndim))



    ### This the parameter case.
    '''
    for yi in range(ndim):
        for xi in range(yi):
            ax = axes[yi, xi]
            ax.axvline(value2[xi], color="r")
            ax.axhline(value2[yi], color="r")
            ax.plot(value2[xi], value2[yi], "sr")
    '''

    for i in range(ndim):
        span = np.linspace(min(samples.T[i]), max(samples.T[i]), 30)
        axes[i,i].axvline(value2[i], color='r', alpha=0.75)
        axes[i,i].plot(span, kde_array[i].evaluate(span))




    if filename != None:
        plt.savefig("results/corner_update/" + filename + "_corner.pdf", format='pdf')




def plot_spec(obs, interp,
              mcmc_args, params, sigma,
              name = None, filename = None):

    ##### This function needs to operate for both the CaII, CH, C2 or just CaII and CH case

    if len(params) == 6:
        plots = 3
    elif len(params) == 5:
        plots = 2


    fig, ax = plt.subplots(1, plots)

    [label.plot(obs['wave'], obs['norm']) for label in ax]



    if filename != None:
        plt.savefig('results/fits/' + filename + "_fit.pdf", format='pdf')


    plt.show()

    return
