import numpy as np

# Wavelength range of interest for analysis (in Angstroms)
WAVE_BOUNDS = [3800.0, 5000.0]

# Synthetic wavelength array within the analysis bounds
SYNTH_WAVE = np.arange(WAVE_BOUNDS[0], WAVE_BOUNDS[1] + 1, 1)

# Full wavelength range used by the spectral interpolator
INTERPOLATOR_WAVE = np.arange(3000.0, 5001.0, 1)

# Get index range in INTERPOLATOR_WAVE that matches the analysis bounds (start and end wavelengths)
id_start_wave = np.where(INTERPOLATOR_WAVE == WAVE_BOUNDS[0])[0][0]
id_end_wave = np.where(INTERPOLATOR_WAVE == WAVE_BOUNDS[1])[0][0]


# Wavelength ranges for CH and Ca lines
CH_BOUNDS = [4222.0, 4322.0]
KP_BOUNDS = {"K6": [3930.7, 3936.7], "K12": [3927.7, 3939.7], "K18": [3924.7, 3942.7]}

# SIDEBANDS for SN and XI calculations
SIDEBANDS = {"CA": [[3884, 3923], [3995, 4045]], "CH": [[4000, 4080], [4440, 4500]], "C2": [[4500, 4600], [4760, 4820]]}

ARCHETYPE_PARAMS = {
    "HALO": {
        "GI": {"FEH": -2.5, "CFE": 1.97, "AC": 7.9},
        "GII": {"FEH": -3.5, "CFE": 0.97, "AC": 5.9},
        "GIII": {"FEH": -4.3, "CFE": 2.87, "AC": 7.0},
    },
    "UFD": {
        "GI": {"FEH": -1.5, "CFE": 1.07, "AC": 8.0},
        "GII": {"FEH": -3.0, "CFE": 0.87, "AC": 6.3},
        "GIII": {"FEH": -3.5, "CFE": 2.37, "AC": 7.3},
    },
}

SIGMA = np.linspace(15, 30, 10)
k, flux_min = 1, 70

band_check, boost, cahk = False, True, True

A_EBV = {"A_J": 0.709, "A_H": 0.449, "A_K": 0.302, "A_g": 3.303, "A_r": 2.285}
