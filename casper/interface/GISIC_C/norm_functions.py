def in_molecular_band(wl: float, tol: float = 10) -> bool:
    """
    Check if a given wavelength falls within a known molecular absorption band.

    This function evaluates whether the input wavelength `wl` is within the
    range of any of the predefined molecular bands (e.g., G-band, C2 regions).

    Parameters
    ----------
    wl : float
        The wavelength value to check (in Angstroms).
    tol : float, optional
        Tolerance in Angstroms (currently unused), default is 10.

    Returns
    -------
    bool
        True if the wavelength falls within one of the defined bands,
        otherwise False.
    """

    bands = {"gband": [4250.0, 4318.0], "C2_N": [4100.0, 4220.0], "C2": [4550.0, 4750.0]}

    for band in bands:
        if (wl > bands[band][0]) & (wl < bands[band][1]):
            return True

    return False
