import corner
import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from interface import config
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from scipy.interpolate import interp1d

from .MCMC_interface import kde_param

plt.ion()

plt.style.use("classic")
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["xtick.labelsize"] = 5.5
plt.rcParams["ytick.labelsize"] = 5.5
plt.rcParams["axes.linewidth"] = 0.5


def produce_title(spectrum) -> str:
    """
    Generate a formatted plot title string using spectrum metadata and MCMC results.

    This function retrieves relevant stellar parameters from the `spectrum` object,
    including effective temperature, surface gravity, metallicity, carbon abundance,
    carbon mode, gravity class, and radial velocity. It then formats these values into
    a human-readable title suitable for plot labeling.

    Parameters
    ----------
    spectrum : Spectrum
        An instance of the `Spectrum` class that contains stellar metadata,
        MCMC-derived parameters, and classification information.

    Returns
    -------
    str
        A formatted string representing the title for use in plots.
    """
    MCMC_DICT = spectrum.get_mcmc_dict(mode="BOTH")

    return (
        "#"
        + spectrum.get_sequence()
        + "  "
        + spectrum.get_starname()
        + "  "
        + "T$_{\\rm eff}$ : %.0F K log $g$: %.2F [Fe/H] : %.2F [C/Fe] : %.2F $A$(C) : %.2F"
        % (
            MCMC_DICT[0]["TEFF"][0],
            spectrum.logg,
            MCMC_DICT[1]["FEH"][0],
            MCMC_DICT[1]["CFE"][0],
            MCMC_DICT[1]["AC"][0],
        )
        + "   MODE:"
        + spectrum.get_carbon_mode()
        + " CLASS:"
        + spectrum.get_gravity_class()
        + "  RV: "
        + str(spectrum.get_rv())
        + " km/s"
        + " N_iter:"
        + str(spectrum.MCMC_iterations)
    )


def plot_spectra(spectra_batch) -> None:
    """
    Generate and save a multi-panel PDF of spectral plots for each spectrum in a batch.

    This function creates a multi-page PDF where each page displays several subplots
    showing the raw, normalized, and zoomed-in regions of each spectrum. For each spectrum,
    the following are plotted:
        - Raw flux with continuum overlay
        - Normalized spectrum
        - Zoomed-in views of Ca II, CH, and C₂ bands with error shading
        - Synthetic model overlay

    Parameters
    ----------
    spectra_batch : object
        An object containing:
            - spectra_array : list of individual spectrum-like objects
            - output_name : str, base name for the saved PDF file
            - length : int, total number of spectra in the batch

    Returns
    -------
    None
        The function saves a PDF file to disk and does not return any value.
    """
    CA_XLIM = [3910, 3980]
    LINEW = 0.3
    LINEW_zoom = 0.5

    print("... generating continuum plots")
    print("\t saving as:   ", spectra_batch.output_name)
    rows, columns = 8, 5

    # pages = int(np.ceil(spectra_batch.length / (rows * columns)))

    pp = PdfPages(spectra_batch.output_name + "_spec.pdf")
    # count = 0

    for i, spec in enumerate(spectra_batch.spectra_array):
        if i % rows == 0:
            fig, ax = plt.subplots(rows, columns, figsize=(8.5, 11), dpi=200)
            fig.subplots_adjust(hspace=0.7)

            [label.set_ylim([0.0, 1.2]) for label in np.concatenate(ax[:, 1:])]
            [label.set_yticks([0.0, 0.5, 1.0, 1.2]) for label in np.concatenate(ax[:, 1:])]

            # CaII
            [label.ticklabel_format(axis="both", useOffset=False) for label in ax[:, 2]]
            [label.set_xlim(CA_XLIM) for label in ax[:, 2]]
            [label.set_xticks([3915, 3930, 3945, 3960, 3975]) for label in ax[:, 2]]

            # CH
            [label.set_xlim([4220, 4325]) for label in ax[:, 3]]
            [label.set_xticks(np.arange(4225, 4350, 25)) for label in ax[:, 3]]

            # C2
            [label.set_xlim([4650, 4750]) for label in ax[:, 4]]
            [label.set_xticks(np.arange(4650, 4775, 25)) for label in ax[:, 4]]

            [plt.setp(label.get_yticklabels(), visible=False) for label in ax[:, 0]]

            [label.tick_params(direction="in", right=True, top=True) for label in np.concatenate(ax[:])]

        index = i % rows

        # Main Plot Section
        ax[index, 0].set_yticks([0.0, max(spec.frame["flux"])])

        [label.set_xticks(np.linspace(config.WAVE_BOUNDS[0], config.WAVE_BOUNDS[1], 7)) for label in ax[index, 0:2]]
        ang = u.Unit("Angstrom")
        [
            label.set_xlabel("wavelength ({:s})".format(ang.to_string(format="Latex")), labelpad=1, fontsize=6)
            for label in ax[index, 0:5]
        ]

        # Set title
        ax[index, 2].set_title(produce_title(spec), fontsize=7)

        # Continuum Plot
        ax[index, 0].plot(spec.frame["wave"], spec.frame["flux"], linewidth=LINEW, color="black", alpha=0.7)
        ax[index, 0].plot(
            spec.frame["wave"], spec.frame["cont"], linewidth=LINEW + 0.1, linestyle="-", color="teal", alpha=1
        )
        ax[index, 0].set_xticks(np.linspace(config.WAVE_BOUNDS[0], config.WAVE_BOUNDS[1], 7))

        # Normalization Plot
        ax[index, 1].axhline(1.00, linewidth=0.5, linestyle="dotted", color="teal", alpha=1)
        ax[index, 1].plot(spec.frame["wave"], spec.frame["norm"], linewidth=LINEW, color="black", alpha=0.7)
        ax[index, 1].set_xticks(np.linspace(config.WAVE_BOUNDS[0], config.WAVE_BOUNDS[1], 7))

        # CaII Plot

        ax[index, 2].axhline(1.00, linewidth=0.5, linestyle="dotted", color="teal", alpha=1)
        ax[index, 2].plot(spec.frame["wave"], spec.frame["norm"], linewidth=LINEW_zoom, color="black", alpha=0.7)

        # CH Plot
        ax[index, 3].axhline(1.00, linewidth=0.5, linestyle="dotted", color="teal", alpha=1)
        ax[index, 3].plot(
            spec.frame["wave"][spec.frame["wave"].between(4150, 4500, inclusive="both")],
            spec.frame["norm"][spec.frame["wave"].between(4150, 4500, inclusive="both")],
            linewidth=LINEW_zoom,
            color="black",
            alpha=0.7,
        )

        # C2 Plot
        ax[index, 4].axhline(1.00, linewidth=0.5, linestyle="dotted", color="teal", alpha=1)
        ax[index, 4].plot(
            spec.frame["wave"][spec.frame["wave"].between(4650, 4850, inclusive="both")],
            spec.frame["norm"][spec.frame["wave"].between(4650, 4850, inclusive="both")],
            linewidth=LINEW_zoom,
            color="black",
            alpha=0.7,
        )

        # Sigma Shading Section
        synth_function = interp1d(spec.synth_spectrum["wave"], spec.synth_spectrum["norm"])

        # CaII
        CA_WAVE = np.linspace(*spec.KP_bounds, 30)
        ax[index, 2].fill_between(
            CA_WAVE,
            synth_function(CA_WAVE) * (1.0 - spec.MCMC_COARSE["XI_CA"][0]),
            synth_function(CA_WAVE) * (1.0 + spec.MCMC_COARSE["XI_CA"][0]),
            color="palevioletred",
            alpha=0.5,
        )

        [ax[index, 2].axvspan(CA_WAVE[0], CA_WAVE[-1], color="black", alpha=0.25)]

        # CH
        CH_WAVE = np.linspace(*list(spec.regions["CH"]["wave"].iloc[[0, -1]]), 30)
        ax[index, 3].fill_between(
            CH_WAVE,
            synth_function(CH_WAVE) * (1.0 - spec.MCMC_COARSE["XI_CH"][0]),
            synth_function(CH_WAVE) * (1.0 + spec.MCMC_COARSE["XI_CH"][0]),
            color="palevioletred",
            alpha=0.5,
        )

        [
            label.plot(
                spec.synth_spectrum["wave"],
                spec.synth_spectrum["norm"],
                color="palevioletred",
                linewidth=LINEW,
                alpha=1,
            )
            for label in ax[index, 1:]
        ]

        ax[index, 1].set_xlim([spec.frame["wave"].iloc[0], spec.frame["wave"].iloc[-1]])

        if (i + 1) % rows == 0 or (i + 1) == spectra_batch.length:
            pp.savefig(fig)

    plt.close()
    pp.close()

    # return


def plot_mcmc_trace_array(spec_batch) -> None:
    """
    Generate and save MCMC trace plots for all spectra in a batch.

    This function loops over each spectrum in the batch and uses
    `plot_single_mcmc_trace()` to create a trace plot showing how each parameter
    evolves over MCMC steps. All plots are saved into a single PDF file named
    "<output_name>_mcmc_trace.pdf".

    Parameters
    ----------
    spec_batch : object
        An object with the following attributes:
        - spectra_array : list of spectrum-like objects
        - output_name : str, used as the base name for the output PDF

    Returns
    -------
    None
        The function writes a PDF to disk but returns nothing.
    """
    pp = PdfPages(spec_batch.output_name + "_mcmc_trace.pdf")

    fig_handle = []

    for item in spec_batch.spectra_array:
        fig_handle.append(plot_single_mcmc_trace(item))

    plt.close()
    [pp.savefig(fig) for fig in fig_handle]

    pp.close()

    # return


def plot_single_mcmc_trace(spectrum, n_thin: int = 1) -> plt.Figure:
    """
    Plot the MCMC trace (walkers over steps) for a single spectrum object.

    This function visualizes the evolution of MCMC chains for each parameter
    in the COARSE (or fine) sampler. It adapts the labels and number of plots
    based on the number of parameters used during sampling (`ndim`).

    Parameters
    ----------
    spectrum : object
        An object with the following attributes and methods:
        - MCMC_COARSE_sampler: the emcee sampler object
        - get_sequence(): returns an identifier string for the spectrum
        - get_filename(): returns the source filename
        - get_MCMC_iterations(): returns total number of iterations
        - mcmc_coarse_tau: estimated autocorrelation time
        - mcmc_coarse_acc_frac: mean acceptance fraction
        - mcmc_coarse_n_discard: suggested burn-in index

    n_thin : int, optional
        Thinning factor to reduce the number of MCMC samples plotted. Default is 1.

    Returns
    -------
    matplotlib.figure.Figure
        The matplotlib figure object containing the MCMC trace plots.
    """
    sampler = spectrum.MCMC_COARSE_sampler
    ndim = sampler.chain.shape[2]

    samples_all = sampler.get_chain(thin=n_thin)

    if ndim == 6:
        # COARSE run with CH+C2 mode
        labels = [r"$T_{\rm eff}$", "[Fe/H]", "[C/Fe]", r"$\xi_{\rm Ca II}$", r"$\xi_{\rm CH}$", r"$\xi_{\rm C_{2}}$"]

    elif ndim == 5:
        # COARSE run with CH mode
        labels = [r"$T_{\rm eff}$", "[Fe/H]", "[C/Fe]", r"$\xi_{\rm Ca II}$", r"$\xi_{\rm CH}$"]

    elif ndim == 2:
        labels = ["[Fe/H]", "[C/Fe]"]

    fig, axes = plt.subplots(ndim, 1, figsize=(6, 8), sharex=True)
    fig.suptitle("COARSE run trace plot: #" + spectrum.get_sequence() + "  " + spectrum.get_filename(), fontsize=10)

    for i in range(ndim):
        axes[i].plot(samples_all[:, :, i], "k", lw=0.3, alpha=0.2)
        axes[i].set_xlim(0, int(spectrum.get_MCMC_iterations() / n_thin))
        axes[i].set_ylabel(labels[i])
        axes[i].axvline(spectrum.mcmc_coarse_n_discard, ls="--", color="b", alpha=0.5)

    axes[ndim - 1].text(
        0.5, 0.5, "Autocorr_time= {:.3f}".format(spectrum.mcmc_coarse_tau), transform=axes[ndim - 1].transAxes
    )
    axes[ndim - 1].text(
        0.5,
        0.3,
        "Mean accept. frac. = {:.3f} ".format(spectrum.mcmc_coarse_acc_frac),
        transform=axes[ndim - 1].transAxes,
    )
    axes[ndim - 1].text(
        0.5,
        0.1,
        "Suggested burnin = {}".format(spectrum.mcmc_coarse_n_discard),
        transform=axes[ndim - 1].transAxes,
        color="b",
    )
    axes[ndim - 1].set_xlabel("step number")

    plt.close()
    return fig


def plot_corner_array(spec_batch) -> None:
    """
    Generate and save corner plots for all spectra in a batch.

    This function loops through a batch of spectra, calls `plot_single_corner`
    on each, and saves all the resulting corner plots into a single PDF file.

    Parameters
    ----------
    spec_batch : object
        An object containing:
        - spectra_array : list of spectrum-like objects
        - output_name : str, base name for the output PDF

    Returns
    -------
    None
        The function saves a PDF file named "<output_name>_corner.pdf" and does not return a value.
    """

    pp = PdfPages(spec_batch.output_name + "_corner.pdf")

    fig_handle = []

    for item in spec_batch.spectra_array:
        fig_handle.append(plot_single_corner(item, spec_batch.output_name))

    plt.close()
    [pp.savefig(fig) for fig in fig_handle]

    pp.close()

    # return


def plot_single_corner(spectrum, io_path: str, n_thin: int = 1) -> Figure:
    """
    Generate a corner plot for MCMC samples from a single spectrum object.

    This function retrieves MCMC samples from the spectrum, applies thinning and burn-in,
    estimates parameter values using KDE, and visualizes the joint distributions and
    1D marginals in a corner plot. It also overlays median values and KDEs, along with
    a LaTeX-formatted legend of key parameters.

    Parameters
    ----------
    spectrum : object
        An object with the following:
        - MCMC_COARSE_sampler : the emcee sampler object with `.get_chain()`
        - get_sequence() : returns spectrum sequence ID
        - get_starname() : returns star name
        - mcmc_coarse_n_discard : number of burn-in steps to discard
        - INPUT_CARBON_MODE : string indicating carbon mode (e.g., "CH" or "CH+C2")
        - get_output_row() : returns a DataFrame row containing derived parameters

    io_path : str
        Output path used in the plot title or for labeling.

    n_thin : int, optional
        Thinning factor for the MCMC chains. Default is 1 (no thinning).

    Returns
    -------
    matplotlib.figure.Figure
        The corner plot figure with parameter distributions and legend annotations.

    Notes
    -----
    Ensure your matplotlib setup supports LaTeX. If there are issues rendering LaTeX,
    you may need to install a LaTeX distribution and/or adjust your matplotlib settings.
    """

    sampler = spectrum.MCMC_COARSE_sampler

    ndim = sampler.chain.shape[2]
    # iter = sampler.chain.shape[1]

    samples = sampler.get_chain(discard=spectrum.mcmc_coarse_n_discard, thin=n_thin, flat=True)

    if ndim == 6:
        labels = [
            r"${\rm T}_{\rm eff}$",
            "[Fe/H]",
            "[C/Fe]",
            r"$\xi_{\rm CaII}$",
            r"$\xi_{\rm CH}$",
            r"$\xi_{\rm C_{2}}$",
        ]

    elif ndim == 5:
        labels = [r"${\rm T}_{\rm eff}$", "[Fe/H]", "[C/Fe]", r"$\xi_{\rm Ca II}$", r"$\xi_{\rm CH}$"]

    elif ndim == 2:
        labels = ["[Fe/H]", "[C/Fe]"]

    fig = corner.corner(samples, labels=labels, color="black", hist_kwargs={"density": True})

    fig.suptitle("#" + spectrum.get_sequence() + "  " + spectrum.get_starname(), fontsize=20)

    MEDIAN = np.median(samples, axis=0)

    value2 = [kde_param(row, x0=x0)["result"] for row, x0 in zip(samples.T, MEDIAN)]
    kde_array = [kde_param(row, x0=x0)["kde"] for row, x0 in zip(samples.T, MEDIAN)]

    # std = np.std(samples, axis=0)

    axes = np.array(fig.axes).reshape((ndim, ndim))

    for yi in range(ndim):
        for xi in range(yi):
            ax = axes[yi, xi]
            ax.axvline(value2[xi], color="r")
            ax.axhline(value2[yi], color="r")
            ax.plot(value2[xi], value2[yi], "sr")

    for i in range(ndim):
        span = np.linspace(min(samples.T[i]), max(samples.T[i]), 30)
        axes[i, i].axvline(value2[i], color="r", alpha=0.75)
        axes[i, i].plot(span, kde_array[i].evaluate(span), color="teal")

    [label.tick_params(direction="in", right=True, top=True) for label in axes.flatten()]

    output_row = spectrum.get_output_row()
    legend = output_row.loc[output_row["SEQUENCE"] == spectrum.get_sequence()]
    key_params = [
        ("T$_{eff}$(K)", "TEFF"),
        ("log $g$", "LOGG"),
        ("[Fe/H]", "FEH"),
        ("[C/Fe]", "CFE"),
        ("$A$(C)", "AC"),
        ("$\\xi_{\\rm Ca II}$", "XI_CA"),
        ("$\\xi_{\\rm CH}$", "XI_CH"),
    ]
    if spectrum.INPUT_CARBON_MODE == "CH+C2":
        key_params = key_params + [("$\\xi_{\\rm C_{2}}$", "XI_C2")]

    params_str = "\n".join(
        [
            f"{latex_key}"
            + f" = ${legend.iloc[0, legend.columns.get_loc(key)]} \\pm {legend.iloc[0, legend.columns.get_loc(key + '_ERR')]}$"
            for latex_key, key in key_params
            if key in legend and key + "_ERR" in legend
        ]
    )

    plt.figtext(0.7, 0.8, params_str, fontsize=15, ha="left", va="top")

    plt.close()
    return fig
