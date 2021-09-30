This repo is now a package. In order to install it we first need to fulfill the requirements.

They can be installed by simply:
```pip install -r requirements.txt```

Corrfunc can give errors if installed through pip. It might be needed to install it through:
```
git clone https://github.com/manodeep/Corrfunc.git
cd Corrfunc
make install
python -m pip install .
```

The test suite can be run by:
```
python -m unittest discover CoLoRe_corrf_analysis/tests/
```
