import numpy as np
from matplotlib import pyplot as plt


def plot_spectrum(spectrum):
    fig, ax = plt.subplots(1, 3, figsize=(10, 3))

    ax[0].plot(spectrum.frame["wave"], spectrum.frame["flux"])
    ax[0].plot(spectrum.frame["wave"], spectrum.frame["cont"])

    ax[1].plot(spectrum.frame["wave"], spectrum.frame["norm"])

    return fig


def chi_plot(chi_frame, filename, alt=None, teff=None):
    ### chi_frame comes from synthetic_functions.interp_run
    fig, ax = plt.subplots(1, 2, figsize=(10, 3))
    # title = "KPNO21_2011  -  HE 0319 - 0215"
    title = filename + "  -  " + alt
    ax[0].plot(chi_frame["temp"], chi_frame["GI_D"], label="GI", color="blue")
    ax[0].plot(chi_frame["temp"], chi_frame["GII_D"], label="GII", color="green")
    ax[0].plot(chi_frame["temp"], chi_frame["GIII_D"], label="GIII", color="orange")

    ax[1].plot(chi_frame["temp"], chi_frame["GI_G"], label="GI", color="blue")
    ax[1].plot(chi_frame["temp"], chi_frame["GII_G"], label="GII", color="green")
    ax[1].plot(chi_frame["temp"], chi_frame["GIII_G"], label="GIII", color="orange")

    ax[0].set_title("Dwarf")
    ax[1].set_title("Giant")

    fig.suptitle(title)
    if teff is not None:
        [label.axvline(teff, linestyle="--") for label in ax]

    # ax[1].set_ylim([0.0, 0.06])

    [label.set_ylabel(r"$\chi^2$") for label in ax]
    [label.set_xlabel(r"T$_{\rm eff}$", labelpad=-10) for label in ax]
    [label.tick_params(direction="in", top=True, right=True) for label in ax]

    ax[0].legend()

    plt.savefig("results/" + filename + ".pdf", format="pdf")


def GI_plot(spec_frame, arch_lib, filename):
    fig, ax = plt.subplots(2, 2, figsize=(10, 8))
    cmap = plt.cm.jet(np.linspace(0, 1, 20))

    ax[0, 0].set_title("Dwarf")
    ax[0, 1].set_title("Giant")

    ### Dwarf Left
    [
        ax[0, 0].plot(arch_lib["wave"], arch_lib["GI_D"][i], c=cmap[i], alpha=0.5, linewidth=0.75)
        for i in range(len(arch_lib["GI_D"]))
    ]
    [
        ax[1, 0].plot(arch_lib["wave"], arch_lib["GI_D"][i], c=cmap[i], alpha=0.5, linewidth=0.75)
        for i in range(len(arch_lib["GI_D"]))
    ]

    ### Giant Right
    [
        ax[0, 1].plot(arch_lib["wave"], arch_lib["GI_G"][i], c=cmap[i], alpha=0.5, linewidth=0.75)
        for i in range(len(arch_lib["GI_D"]))
    ]
    [
        ax[1, 1].plot(arch_lib["wave"], arch_lib["GI_G"][i], c=cmap[i], alpha=0.5, linewidth=0.75)
        for i in range(len(arch_lib["GI_D"]))
    ]

    [
        label.plot(spec_frame["wave"], spec_frame["norm"], color="black", linewidth=0.75, alpha=0.75)
        for label in ax[:, 0]
    ]
    [
        label.plot(spec_frame["wave"], spec_frame["norm"], color="black", linewidth=0.75, alpha=0.75)
        for label in ax[:, 1]
    ]

    [label.set_xlabel("Wavelength") for label in ax[1, :]]

    [label.set_xlim([3925, 3980]) for label in ax[0, :]]
    [label.set_xlim([4200, 4350]) for label in ax[1, :]]
    [label.set_ylim([0, 1.2]) for label in ax[0, :]]
    [label.set_ylim([0, 1.2]) for label in ax[1, :]]

    fig.suptitle("Group I : [Fe/H] = -2.5 A(C) = 7.9")

    plt.savefig("results/" + filename, format="pdf")

    return


def GII_plot(spec_frame, arch_lib, filename):
    fig, ax = plt.subplots(2, 2, figsize=(10, 8))
    cmap = plt.cm.jet(np.linspace(0, 1, 20))

    ax[0, 0].set_title("Dwarf")
    ax[0, 1].set_title("Giant")

    ### Dwarf Left
    [
        ax[0, 0].plot(arch_lib["wave"], arch_lib["GII_D"][i], c=cmap[i], alpha=0.5, linewidth=0.75)
        for i in range(len(arch_lib["GII_D"]))
    ]
    [
        ax[1, 0].plot(arch_lib["wave"], arch_lib["GII_D"][i], c=cmap[i], alpha=0.5, linewidth=0.75)
        for i in range(len(arch_lib["GII_D"]))
    ]

    ### Giant Right
    [
        ax[0, 1].plot(arch_lib["wave"], arch_lib["GII_G"][i], c=cmap[i], alpha=0.5, linewidth=0.75)
        for i in range(len(arch_lib["GII_D"]))
    ]
    [
        ax[1, 1].plot(arch_lib["wave"], arch_lib["GII_G"][i], c=cmap[i], alpha=0.5, linewidth=0.75)
        for i in range(len(arch_lib["GII_D"]))
    ]

    [
        label.plot(spec_frame["wave"], spec_frame["norm"], color="black", linewidth=0.75, alpha=0.75)
        for label in ax[:, 0]
    ]
    [
        label.plot(spec_frame["wave"], spec_frame["norm"], color="black", linewidth=0.75, alpha=0.75)
        for label in ax[:, 1]
    ]

    [label.set_xlabel("Wavelength") for label in ax[1, :]]

    [label.set_xlim([3925, 3980]) for label in ax[0, :]]
    [label.set_xlim([4200, 4350]) for label in ax[1, :]]

    [label.set_ylim([0, 1.2]) for label in ax[0, :]]
    [label.set_ylim([0, 1.2]) for label in ax[1, :]]

    fig.suptitle("Group II : [Fe/H] = -3.5  A(C) = 5.9")

    plt.savefig("results/" + filename, format="pdf")

    return


def GIII_plot(spec_frame, arch_lib, filename):
    fig, ax = plt.subplots(2, 2, figsize=(10, 8))
    cmap = plt.cm.jet(np.linspace(0, 1, 20))

    ax[0, 0].set_title("Dwarf")
    ax[0, 1].set_title("Giant")

    ### Dwarf Left
    [
        ax[0, 0].plot(arch_lib["wave"], arch_lib["GIII_D"][i], c=cmap[i], alpha=0.5, linewidth=0.75)
        for i in range(len(arch_lib["GIII_D"]))
    ]
    [
        ax[1, 0].plot(arch_lib["wave"], arch_lib["GIII_D"][i], c=cmap[i], alpha=0.5, linewidth=0.75)
        for i in range(len(arch_lib["GIII_D"]))
    ]

    ### Giant Right
    [
        ax[0, 1].plot(arch_lib["wave"], arch_lib["GIII_G"][i], c=cmap[i], alpha=0.5, linewidth=0.75)
        for i in range(len(arch_lib["GIII_D"]))
    ]
    [
        ax[1, 1].plot(arch_lib["wave"], arch_lib["GIII_G"][i], c=cmap[i], alpha=0.5, linewidth=0.75)
        for i in range(len(arch_lib["GIII_D"]))
    ]

    [
        label.plot(spec_frame["wave"], spec_frame["norm"], color="black", linewidth=0.75, alpha=0.75)
        for label in ax[:, 0]
    ]
    [
        label.plot(spec_frame["wave"], spec_frame["norm"], color="black", linewidth=0.75, alpha=0.75)
        for label in ax[:, 1]
    ]

    [label.set_xlabel("Wavelength") for label in ax[1, :]]

    [label.set_xlim([3925, 3980]) for label in ax[0, :]]
    [label.set_xlim([4200, 4350]) for label in ax[1, :]]

    [label.set_ylim([0, 1.2]) for label in ax[0, :]]
    [label.set_ylim([0, 1.2]) for label in ax[1, :]]

    fig.suptitle("Group III : [Fe/H] = -4.3  A(C) = 7.0")

    plt.savefig("results/" + filename, format="pdf")

    return


def plot_crit(frame_array, group_class):
    ## for the output from determine_crit_params

    AC_VALUES = np.unique(frame_array["CARBON"])
    FEH_VALUES = np.unique(frame_array["FEH"])
    TEFF_VALUES = np.unique(frame_array["T"])
    xscale = 2.5
    yscale = 1.5
    print("Unique Carbon Values:  ", len(AC_VALUES))

    cmap = plt.cm.jet(np.linspace(0, 1, len(TEFF_VALUES)))
    carbon = {key: value for key, value in zip(GROUP_STR, ["AC", "AC", "CFE", "CFE", "AC", "AC"])}

    ### now it will adjust fig
    columns = 4
    rows = int(np.ceil(len(AC_VALUES) / columns))

    print(rows)
    fig = plt.figure(figsize=(rows * yscale, columns * xscale))
    handles = []
    for i in range(len(AC_VALUES)):
        ax = fig.add_subplot(rows, columns, i + 1)

        slice = frame_array[(frame_array["CARBON"] == AC_VALUES[i])]

        [
            ax.plot(slice[slice["T"] == VALUE]["FEH"], slice[slice["T"] == VALUE]["CHI"], color=cmap[i])
            for i, VALUE in enumerate(TEFF_VALUES)
        ]
        ax.set_title(carbon[group_class] + ":  %.2F" % AC_VALUES[i])
        handles.append(ax)

    fig.subplots_adjust(hspace=0.5)
    [label.tick_params(direction="in", top=True, right=True) for label in handles]


#########


def plot_crit_3D(frame_array, group_class):
    ## Should really do this in 3D anyway.
    ## for the output from determine_crit_params

    AC_VALUES = np.unique(frame_array["CARBON"])
    FEH_VALUES = np.unique(frame_array["FEH"])
    TEFF_VALUES = np.unique(frame_array["T"])

    print("Unique Carbon Values:  ", len(AC_VALUES))

    cmap = plt.cm.jet(np.linspace(0, 1, len(TEFF_VALUES)))
    carbon = {key: value for key, value in zip(GROUP_STR, ["AC", "AC", "CFE", "CFE", "AC", "AC"])}

    ### now it will adjust fig
    columns = 4
    rows = int(np.ceil(len(AC_VALUES) / columns))

    print(rows)
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    handles = []
    for i in range(len(TEFF_VALUES)):
        slice = frame_array[(frame_array["T"] == TEFF_VALUES[i])]

        # ax.scatter(slice['FEH'], slice['CARBON'], slice['CHI'])
        ax.plot_trisurf(slice["FEH"], slice["CARBON"], -np.log(slice["CHI"]), alpha=0.50, color=cmap[i])

        # [ax.plot(slice[slice['T'] == VALUE]['FEH'], slice[slice['T'] == VALUE]['CHI'], color=cmap[i]) for i, VALUE in enumerate(TEFF_VALUES)]
        # ax.set_title(carbon[group_class] + ':  %.2F' % AC_VALUES[i])
        handles.append(ax)
        ax.view_init(30, 25)
    # ax.set_zlim(ax.get_zlim()[::-1])
    # fig.subplots_adjust(hspace=0.5)

    ax.set_xlabel("[Fe/H]", fontsize=14)
    ax.set_ylabel(carbon[group_class], fontsize=14)
    ax.set_zlabel(r"$-\xi_{\omega}^2$", fontsize=14)

    [label.tick_params(direction="in", top=True, right=True) for label in handles]

    plt.show()

    # not used routine below, plot_mcmc_sampler


def plot_mcmc_sampler(SAMPLER, ndim, burnin, suptitle, filename, acc_params, group):
    CARBON = {"GI": r"A$(C)$", "GII": "[C/Fe]", "GIII": r"A$(C)$"}

    samples = SAMPLER.chain[:, 500:, :].reshape((-1, ndim))
    fig = corner.corner(samples, labels=[r"$T_{\rm eff}$", "[Fe/H]", CARBON[group]], color="black")

    fig.suptitle(suptitle, fontsize=15)
    value2 = np.median(samples, axis=0)
    std = np.std(samples, axis=0)

    axes = np.array(fig.axes).reshape((ndim, ndim))

    for yi in range(ndim):
        for xi in range(yi):
            ax = axes[yi, xi]
            ax.axvline(value2[xi], color="r")
            ax.axhline(value2[yi], color="r")
            ax.plot(value2[xi], value2[yi], "sr")

    for i in range(ndim):
        axes[i, i].axvline(value2[i], color="r", alpha=0.75)

    ### For the real values
    if acc_params != None:
        for yi in range(ndim):
            for xi in range(yi):
                ax = axes[yi, xi]
                ax.axvline(acc_params[xi], color="g")
                ax.axhline(acc_params[yi], color="g")
                ax.plot(acc_params[xi], acc_params[yi], "sg")
        for i in range(ndim):
            axes[i, i].axvline(acc_params[i], color="g", alpha=0.75)

    # textstr = '\n'.join((
    #            r'$T_{\rm eff}=$%.0f K' % (value2[0], ),
    #            r'[Fe/H]=%.2f' % (value2[1], ),
    #            CARBON[group] + '=%.2f' % (value2[2], )))

    # props = dict(boxstyle='round', facecolor='white', alpha=0.5)

    # axes[0,1].text(0.05, 0.95, textstr, transform=axes[0,1].transAxes, fontsize=14,
    #    verticalalignment='top', bbox=props)

    return


#########################################
## Somehow code got deleted, which is crazy. So I'm rewriting..
##########################################


# not used routine below,plot_mcmc_samples
def plot_mcmc_samples(sampler, burnin=0.25, params=None, suptitle=None, filename=None):
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
        labels = ["Teff", "[Fe/H]", "[C/Fe]", "sigmaCA", "XI_CH", "XI_C2"]

    elif ndim == 5:
        labels = ["Teff", "[Fe/H]", "[C/Fe]", "sigmaCA", "XI_CH"]

    elif ndim == 2:
        ### Fine parameters case
        labels = ["[Fe/H]", "[C/Fe]"]

    samples = sampler[:, int(burnin * iter) :, :].reshape((-1, ndim))

    fig = corner.corner(samples, labels=labels, color="black", hist_kwargs={"normed": True})

    if suptitle != None:
        fig.suptitle(suptitle, fontsize=14)

    MEDIAN = np.median(samples, axis=0)
    value2 = [kde_param(row, x0=x0)["result"] for row, x0 in zip(samples.T, MEDIAN)]
    kde_array = [kde_param(row, x0=x0)["kde"] for row, x0 in zip(samples.T, MEDIAN)]

    std = np.std(samples, axis=0)

    axes = np.array(fig.axes).reshape((ndim, ndim))

    ### This the parameter case.
    """
    for yi in range(ndim):
        for xi in range(yi):
            ax = axes[yi, xi]
            ax.axvline(value2[xi], color="r")
            ax.axhline(value2[yi], color="r")
            ax.plot(value2[xi], value2[yi], "sr")
    """

    for i in range(ndim):
        span = np.linspace(min(samples.T[i]), max(samples.T[i]), 30)
        axes[i, i].axvline(value2[i], color="r", alpha=0.75)
        axes[i, i].plot(span, kde_array[i].evaluate(span))

    if filename != None:
        plt.savefig("results/corner_update/" + filename + "_corner.pdf", format="pdf")


def plot_spec(obs, interp, mcmc_args, params, sigma, name=None, filename=None):
    ##### This function needs to operate for both the CaII, CH, C2 or just CaII and CH case

    if len(params) == 6:
        plots = 3
    elif len(params) == 5:
        plots = 2

    fig, ax = plt.subplots(1, plots)

    [label.plot(obs["wave"], obs["norm"]) for label in ax]

    if filename != None:
        plt.savefig("results/fits/" + filename + "_fit.pdf", format="pdf")

    plt.show()

    return
