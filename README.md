This repo is now a package. In order to install it we first need to fulfill the requirements.

# Installation of requirements
They can be installed by simply:
```pip install -r requirements.txt```

## Conflicting packages
### Corrfunc
Corrfunc can give errors if installed through pip. It might be needed to compile it from source:
```
module load gsl
git clone https://github.com/manodeep/Corrfunc.git
cd Corrfunc
make install
python -m pip install .
```
### halotools
Maybe halootols won't install through pip, you can install it using conda:
```
conda install -c astropy halotools
```

# Installation
After installing the requirements, we can install this package by:
```
pip install .
```

# Testing the installation

The test suite can be run by:
```
python -m unittest discover CoLoRe_corrf_analysis/tests/
```

Fitter tests are skipped by default, to run them (fails here might be ok):
```
RUN_FITTER_TESTS=Y python -m unittest discover CoLoRe_corrf_analysis/tests
```
