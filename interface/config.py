
import numpy as np

## wavelength setting ##

# constraining the wavelength range of interest
WAVE_BOUNDS = [3800., 5000.]
SYNTH_WAVE = np.arange(WAVE_BOUNDS[0],WAVE_BOUNDS[1]+1,1)

# synthetic model wavelength range
INTERPOLATOR_WAVE = np.arange(3000., 5001., 1)
# need to define the index of wavelength for synthetic flux
id_start_wave =  np.where( INTERPOLATOR_WAVE == WAVE_BOUNDS[0])[0][0]


## wavelength ranges for CH and Ca lines ## 
CH_BOUNDS = [4222., 4322.]
KP_BOUNDS = {"K6"  : [3930.7, 3936.7],
            "K12" : [3927.7, 3939.7],
            "K18" : [3924.7, 3942.7]}



## Archetype Parameters ##
ARCHETYPE_PARAMS = {"HALO" : {'GI'  : {'FEH': -2.5, 'CFE': 1.97, 'AC' : 7.9},
                              'GII' : {'FEH': -3.5, 'CFE': 0.97, 'AC' : 5.9},
                              'GIII': {'FEH': -4.3, 'CFE': 2.87, 'AC' : 7.0}},

                    "UFD"  : {'GI'  : {'FEH': -1.5, 'CFE': 1.07, 'AC' : 8.0},
                              'GII' : {'FEH': -3.0, 'CFE': 0.87, 'AC' : 6.3},
                              'GIII': {'FEH': -3.5, 'CFE': 2.37, 'AC' : 7.3}}}

 
## Normalization parameters ##

SIGMA = np.linspace(15, 30, 10)
k,flux_min = 1, 70
#cahk = True
band_check, boost, cahk = False, True, True
#boost = True


## Extinction correction
A_EBV = {"A_J" : 0.709,
         "A_H" : 0.449,
         "A_K" : 0.302,
         "A_g" : 3.303,
         "A_r" : 2.285}
