from casper.interface import config


def GBAND_vanilla(wave, flux, bounds=config.CH_BOUNDS):  # noqa: F821
    trim = flux[(wave > bounds[0]) & (wave < bounds[1])]

    return (1 - trim).sum()
