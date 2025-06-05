###  CEMP Group Assignment and Stellar Parameter Estimation Routine (CASPER)
#### Main Developers: Devin D. Whitten, Jinmi Yoon
#### Email: devin.d.whitten@gmail.com, jinmi.yoon@gmail.com

This script package, CASPER, is designed to determine reliable stellar parameters (temperature, metallicity, surface gravity, and carbon abundance) of low/medium-resolution stellar spectra for cool Carbon-Enhanced Metal-Poor (CEMP) stars (Teff < 5000K). This package is under development for public use and thus needs more testings and refinements (Whitten, Yoon, et al. in prep). The description of the CASPER methodology can be found in Yoon, Whitten, et al. 2020 (The Astrophysical Journal, 894,7). The detailed documentation, along with the codes, will be available for public use in the near future.

### Python environment setup
- The required python packages can be found in [casper_requirement.yml](casper_requirement.yml).

- If you use `conda`, run this command to create an conda environment.

```shell
conda env create -f casper_requirement.yml
```


- If you want to create a lightweight python environment, you can use `micromamba`, which is fast alternative to conda, written in C++, that implements the same CLI interface. Follow this [instructions](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html) to install `micromamba. You can create the CASPER environment by running this command.

```shell
micromamba create --file casper_requirement.yml
``` 
To activate the environment, run in the Casper directory:
```shell
micromamba activate casper311
``` 
### How to run CASPER

First, you can set up your custom input parameters and spectra and the output directory and file name prefix in `interface/io_paths.py`.
You will need to create the `outputs` folder on the main directory where `main.py` is found.

CASPER uses .pkl files tracked with Git Large File Storage. Make sure to install the lfs.

```shell
brew install git-lfs
```
```shell
git lfs install
git lfs pull
``` 
To run CASPER, run this command on your terminal. 
```shell
python main.py
```

### Output files
Once you run CASPER, you will have several output files.

- `*_archetype_likelihood_table.txt`: a table of CEMP group archetype likelihood
- `*_corner.pdf` : a resulting corner plot
- `*_mcmc_trace.pdf`: a mcmc trace plot
- `*_out.csv` : a stellar parameter output file
- `*_snr.csv`: a CSV file of the SNRs of Ca K line, CH band, and C2 bands
- `*_spec.pdf`: a pdf file of the plots of spectral fits
- `*_spectra_output.csv`: an extracted output of observed and synthetic spectra
- `*_temp_cal_table.txt`: a table of temperature calibration


### Collaboration, Scientific Use
If you want to use this package for your scientific use and/or help to complete the development, please contact first both Devin Whitten (devin.d.whitten@gmail.com) and Jinmi Yoon (jinmi.yoon@gmail.com).

### Stellar Parameters Space for CASPER
[Fe/H] = [-4.5, -1.0], Teff = [4000, 5500] K, [C/Fe] = [-0.5. 4.5] , logg =[0.0, 5.5]

### Publications used this methodology :
- [Placco, ..., Whitten, et al., 2020, ApJ, 897, 78 ](https://ui.adsabs.harvard.edu/abs/2020ApJ...897...78P/abstract)
- [Yoon, Whitten, Beers, Lee, Masseron, and Placco, 2020, ApJ, 894, 7](https://ui.adsabs.harvard.edu/abs/2020ApJ...894....7Y/abstract)

![Logo](https://github.com/DevinWhitten/CASPER/blob/master/images/CASPER_logo.png)
![UMP Design](https://github.com/DevinWhitten/CCSLab/blob/master/images/UMP_Methodology_v3.png)
![Arch Design](https://github.com/DevinWhitten/CCSLab/blob/master/images/arch_dir_schem.png)



<img src="https://github.com/DevinWhitten/CCSLab/blob/master/images/continuum_animation.gif" width="80%"
style="display:block;margin: 0 auto;">
