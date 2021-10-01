RUN_FITTER_TESTS=Y python -m coverage run --source . -m unittest discover CoLoRe_corrf_analysis/tests/
python -m coverage html --omit="*/tests*","*__init__.py","*hidden_*","setup.py"
