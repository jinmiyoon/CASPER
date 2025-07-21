def ac(cfe: float, feh: float) -> float:
    """
    Compute the absolute carbon abundance (A(C)) from [C/Fe] and [Fe/H].

    This function adds the carbon-to-iron ratio ([C/Fe]), the metallicity ([Fe/H]),
    and the Asplund (2009) solar carbon abundance (8.43) to estimate the absolute carbon abundance A(C).

    Parameters
    ----------
    cfe : float
        Logarithmic carbon-to-iron abundance ratio [C/Fe].
    feh : float
        Metallicity, logarithmic iron-to-hydrogen abundance ratio [Fe/H].

    Returns
    -------
    float
        The absolute carbon abundance A(C).
    """
    return cfe + feh + 8.43


def cfe(ac: float, feh: float) -> float:
    """
    Calculate the carbon-to-iron abundance ratio ([C/Fe]) from absolute carbon abundance.

    This function subtracts the Asplund (2009) solar carbon abundance (8.43) and the metallicity ([Fe/H]) from the absolute carbon abundance (A(C)) to estimate [C/Fe].

    Parameters
    ----------
    ac : float
        Absolute carbon abundance A(C).
    feh : float
        Metallicity, logarithmic iron-to-hydrogen abundance ratio [Fe/H].

    Returns
    -------
    float
        The carbon-to-iron abundance ratio [C/Fe].
    """
    return ac - 8.43 - feh
