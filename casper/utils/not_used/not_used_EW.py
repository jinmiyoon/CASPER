from casper.interface import config


def GBAND_vanilla(wave, flux, bounds=config.CH_BOUNDS):  # noqa: F821
    trim = flux[(wave > bounds[0]) & (wave < bounds[1])]

    return (1 - trim).sum()


def CAII_K6_v(wave, flux):
    # 3930.7 - 3936.7

    trim = flux[(wave > 3930.7) & (wave < 3936.7)]
    return (1.0 - trim).sum()


def CAII_K12_v(wave, flux):
    # 3927.7 - 3939.7

    trim = flux[(wave > 3927.7) & (wave < 3939.7)]
    return (1.0 - trim).sum()


def CAII_K18_v(wave, flux):
    # 3924.7 - 3942.7

    trim = flux[(wave > 3927.7) & (wave < 3939.7)]
    return (1.0 - trim).sum()


def CAII_H(wave, flux):
    trim = flux[(wave > 3927.7) & (wave < 3939.7)]
