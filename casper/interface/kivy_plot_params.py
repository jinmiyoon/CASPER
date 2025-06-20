import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("module://kivy.garden.matplotlib.backend_kivy")
matplotlib.style.use("dark_background")

color_frame = {"GI": "blue", "GII": "green", "GIII": "orange"}


# not used
def build_spec_axis():
    """
    Set up a multi-panel figure for plotting spectral data.

    This function creates a matplotlib figure with three axes:
    - Top plot for the original spectrum and continuum.
    - Bottom plot for the normalized spectrum.
    - Middle bar for custom markers or labels.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The full figure object.
    axes : tuple
        A tuple of axes in the order (axTOP, axBOT, axBAR).
    lines : dict
        A dictionary of initialized line objects:
        - 'spec_line': line for the raw spectrum
        - 'cont_line': line for the continuum
        - 'norm_line': line for the normalized spectrum
    """

    fig = plt.figure()
    axTOP = fig.add_axes([0.1, 0.6, 0.8, 0.3])
    axBOT = fig.add_axes([0.1, 0.1, 0.8, 0.3])
    axBAR = fig.add_axes([0.1, 0.5, 0.8, 0.03])

    fig.suptitle("Spectrum", fontname="Times New Roman", fontsize=20)

    (spec_line,) = axTOP.plot([], [], color="white", linewidth=0.50)
    (cont_line,) = axTOP.plot([], [])
    (norm_line,) = axBOT.plot([], [], color="white", linewidth=0.50)

    axBOT.axhline(1.0, linestyle="--", color="red")

    [plt.setp(axis.get_yticklabels(), visible=False) for axis in [axTOP, axBAR]]
    plt.setp(axBAR.get_xticklabels(), visible=False)
    [axTOP.set_xlabel(r"$\lambda$ [$\AA$]", fontname="Times New Roman") for axis in [axTOP, axBOT]]

    [ax.tick_params(direction="in", top=True, right=True) for ax in [axTOP, axBOT]]
    axBAR.tick_params(length=0)

    return fig, (axTOP, axBOT, axBAR), {"spec_line": spec_line, "cont_line": cont_line, "norm_line": norm_line}


# not used
def build_class_axis():
    """
    Set up a figure with subplots for comparing chi-squared fits across stellar groups.

    This function creates a 3-row, 2-column grid of subplots. Each row represents a stellar classification group (GI, GII, GIII), with one column for dwarfs and one for giants. It also initializes empty line plots on each axis.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The full figure object.
    axes : list
        A list of the six subplot axes, ordered top-to-bottom, left-to-right.
    chi_lines : list
        A list of line objects for each subplot, one per group-class combination.
    """
    fig = plt.figure()
    fig.suptitle(r"Archetype $\chi^2$", fontname="Times New Roman", fontsize=20)
    chi_lines = []

    a1 = fig.add_subplot(3, 2, 1)
    a2 = fig.add_subplot(3, 2, 2, sharey=a1)
    a3 = fig.add_subplot(3, 2, 3)
    a4 = fig.add_subplot(3, 2, 4, sharey=a3)
    a5 = fig.add_subplot(3, 2, 5)
    a6 = fig.add_subplot(3, 2, 6, sharey=a5)

    axes = [a1, a2, a3, a4, a5, a6]

    [ax.tick_params(direction="in", top=True, right=True) for ax in axes]
    [
        ax.text(0.5, 0.85, GROUP, horizontalalignment="center", verticalalignment="center", transform=ax.transAxes)
        for GROUP, ax in zip(["GI", "GI", "GII", "GII", "GIII", "GIII"], axes)
    ]

    a1.set_title("Dwarf")
    a2.set_title("Giant")

    for cur, group in zip(axes, ["GI", "GI", "GII", "GII", "GIII", "GIII"]):
        (line,) = cur.plot([], [], color=color_frame[group], linewidth=2.0)

        chi_lines.append(line)

    return fig, axes, chi_lines


# not used
def build_custom_axis():
    """
    Create a simple, empty matplotlib figure with one axis.

    This function builds a figure with a single subplot,
    removes the x and y axis ticks, and adds a blank line object.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The full figure object.
    ax : matplotlib.axes.Axes
        The single subplot axis.
    line : list
        A list containing the empty Line2D object created on the axis.
    """

    fig = plt.figure()

    ax = fig.add_subplot(1, 1, 1)
    line = ax.plot([], [])
    ax.set_xticks([])
    ax.set_yticks([])

    return fig, ax, line
