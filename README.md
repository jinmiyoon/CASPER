## CASPER
### CEMP Group Assignment and Stellar Parameter Estimation Routine (CASPER)

This script package, CASPER, is designed to determine reliable stellar parameters (temperature, metallicity, surface gravity, and carbon abundance) of low/medium-resolution stellar spectra for cool Carbon-Enhanced Metal-Poor (CEMP) stars (Teff < 5000K). This package is under development for public use and thus needs more testings and refinements (Whitten, Yoon, et al. in prep). The description of the CASPER methodology can be found in Yoon, Whitten, et al. 2020 (The Astrophysical Journal, 894,7). The detailed documentation, along with the codes, will be available for public use in the near future.

### Installation
#### Required packages and versions
- See required packages found in the [pyproject.toml](pyproject.toml) or [caper311.yml](casper311.yml).

#### Python environment installation
You can use `conda` to create and activate the CASPER environment.
Change `env_name` below to your preferred name, run these commands on your terminal.

```shell
conda create -n env_name python=3.11
conda activate env_name
```

If you want to create a lightweight python environment, you can use `micromamba`, which is fast alternative to conda, written in C++, that implements the same CLI interface. Follow this [instructions](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html) to install `micromamba. You can create and activate the CASPER environment by running this command.

```shell
micromamba create -n env_name python=3.11
micromamba activate env_name
```

#### CASPER installation
##### Installation for users
The **casper** directory contains the python package itself, installable via pip. This will install the core dependencies defined in `pyproject.toml` fur running `casper`.

```shell
pip install .
```
##### Installation for developers
If you are interested in developing and contributing to **casper**, you should install this package with `-e`, it allows you to work on the package's source code and see changes reflected immediately without needing to reinstall.

```shell
pip install -e . # install editable mode
```
To install the optional dependencies for pytest or Sphinx autodoc, run the command below in addition to pip install in editable mode above.

```shell
pip install .[dev,test,docs] # install the dependencies of dev, test, docs
```
or
```shell
pip install .[all] # "all" includes the dependencies of dev, test, docs
```

### pre-commit for development

[pre-commit](https://pre-commit.com/) allows all collaborators push their commits compliant with the same set of lint and format rules in **pyproject.toml** by checking all files in the project at different stages of the git workflow. It runs commands specified in the **.pre-commit-config.yaml** config file and runs checks before committing or pushing, to catch errors that would have caused a build failure before they reach CI.

#### Install pre-commit
You will need to install `pre-commit` manually.
```bash
pip install pre-commit # if you haven't already installed the package
```

```bash
pre-commit install # install default hooks, `pre-commit`, `pre-push`, and `commit-msg`, as specified in the config file.
```

If this is your first time running, you should run the hooks against for all files and it will fix all files based on your setting.
```bash
pre-commit run --all-files
```
Finally, you will need to update `pre-commit` regularly by running
```bash
pre-commit autoupdate
```
For other configuration options and more detailed information, check out at the [pre-commit](https://pre-commit.com/) page.

### Setup

#### How to run CASPER

First, you can set up your custom input parameters and spectra and the output directory and file name prefix in [io_paths.py](interface/io_paths.py).

Then, you will need to pull the spectral library files from [Git Large File Storage (LFS)](https://git-lfs.com/).
If you don't already haven't installed `git-lfs`, run
```shell
brew install git-lfs
git lfs install
git lfs pull
```

To run CASPER, run this command on your terminal.

```shell
python main.py
```

#### Output files
Once you run CASPER, you will have several output files.

- `*_archetype_likelihood_table.txt`: a table of CEMP group archetype likelihood
- `*_corner.pdf` : a resulting corner plot
- `*_mcmc_trace.pdf`: a mcmc trace plot
- `*_out.csv` : a stellar parameter output file
- `*_snr.csv`: a CSV file of the SNRs of Ca K line, CH band, and C2 bands
- `*_spec.pdf`: a pdf file of the plots of spectral fits
- `*_spectra_output.csv`: an extracted output of observed and synthetic spectra
- `*_temp_cal_table.txt`: a table of temperature calibration

#### When testing with pytest
This project uses pytest for writing and running unit tests. Pytest is a lightweight testing framework that makes it easy to write simple test functions using plain Python and assert statements.

More information on pytest can be found at: https://docs.pytest.org/en/stable/

Install pytest via pip.
```shell
pip install pytest
```
Next, add a tests/ directory at the root of your project and place your test files there.

To run the pytest, cd to the `tests` folder and run the command below. This command will run all relevant tests.
```shell
pytest 
```
If you want to run a single pytest, cd to the directory where the test is found and run
```shell
pytest <your/file/path.py>
```

#### Building the Documentation with Sphinx
This project uses Sphinx (a documentation generator) to generate clean, readable documentation from Python docstrings. Sphinx supports both .rst and .md formats and can automatically extract and format documentation from your code using the extensions.

More information about Sphinx can be found at: https://www.sphinx-doc.org/en/master/usage/quickstart.html

Before setting up Sphinx, users must install the required documentation dependencies (`sphinx` ,`sphinx_rtd_theme`, `myst-parser`) via pip. These packages are installed when you install CASPER. If you only installed `[dev,test]` dependencies in `pyproject.toml`, run the following command to install `[doc]` dependencies.
```shell
pip install sphinx sphinx_rtd_theme myst-parser linkify-it-py
```
After dependencies are installed, run this command to initialize a new Sphinx documentation project in a folder called "docs".
1. It will prompt you for your project info (project name, author, version, etc.).
2. Creates a docs/ folder (if it doesn’t exist).
3. Generates starter config files inside docs/, including:
    - conf.py: your main configuration file
    - index.rst: the root page of your docs (serves as the welcome page)
    - makefile and make.bat: shortcuts to build docs on Linux/macOS or Windows
NOTE: Make sure to add any missing extensions and Sphinx configuration options to your conf.py if it differs from the repo's conf.py.
```shell
sphinx-quickstart docs
```
Next, this command auto-generates .rst files for your Python package so Sphinx can build documentation from your code. It finds all modules and sub-packages and generates .rst stub files (like casper.interface.rst, casper.utils.rst, etc.) in the docs/ folder.
Each .rst file includes .. automodule:: directives to tell Sphinx to pull in the docstrings from your code.
```shell
sphinx-apidoc -o docs casper #replace casper with project folder name if needed
```
Lastly, run this command at root level to build your Sphinx documentation as a static website (in HTML format). It will create a build/html folder in docs that stores the HTML code for the website.
```shell
sphinx-build -b html docs docs/_build/html
```

### Collaboration, Scientific Use
If you want to use this package for your scientific use and/or help to complete the development, please contact Jinmi Yoon (jinmi.yoon@gmail.com).

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
