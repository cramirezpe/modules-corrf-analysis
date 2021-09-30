"""
    Tests for read_theory_to_xi. Run it using:
        coverage run --source . -m unittest discover CoLoRe_corrf_analysis/tests
        python -m coverage html --omit="*/tests*","*__init__.py","*hidden_*","setup.py"
"""

import os
import unittest
from unittest import skipIf
from pathlib import Path
from CoLoRe_corrf_analysis.cf_helper import CFComputations
import numpy as np
import shutil
from unittest.mock import PropertyMock, patch

class TestReadFiles(unittest.TestCase):
    def setUp(self):
        _current_dir = Path(__file__).parent
        self.test_files = _current_dir / 'test_files' / 'cf_helper' / '0'
        for _npole_file in (self.test_files / 'npole_files').glob('npole*'):
            shutil.copy(_npole_file, self.test_files)

        self.cfccomp = CFComputations(self.test_files, 1)

    def tearDown(self):
        for _npole_file in (self.test_files).glob('npole_*'):
            if _npole_file.is_file():
                os.remove(_npole_file.resolve())
    
    def test_file_copied(self):
        assert (self.test_files / 'npole_0.dat').is_file()

    @patch('CoLoRe_corrf_analysis.cf_helper.CFComputations.halotools_like_cf')
    def test_read_monopole(self, mock_func):
        pole_0 = self.cfccomp.compute_npole(0)
        target = np.loadtxt(self.test_files / 'npole_files' / 'npole_0.dat')
        np.testing.assert_equal(pole_0, target)
        assert not mock_func.called

    @patch('CoLoRe_corrf_analysis.cf_helper.CFComputations.halotools_like_cf')
    def test_read_quadrupole(self, mock_func):
        pole_2 = self.cfccomp.compute_npole(2)
        target = np.loadtxt(self.test_files / 'npole_files' / 'npole_2.dat')
        np.testing.assert_equal(pole_2, target)
        assert not mock_func.called

    def test_read_savg(self):
        savg = self.cfccomp.savg

        np.testing.assert_equal(savg, np.unique(( self.cfccomp.DD['smin'] + self.cfccomp.DD['smax'] ))/2)

class TestComputeFiles(unittest.TestCase):
    def setUp(self):
        _current_dir = Path(__file__).parent
        self.test_files = _current_dir / 'test_files' / 'cf_helper' / '0'
        for _npole_file in (self.test_files).glob('npole_*'):
            if _npole_file.is_file():
                os.remove(_npole_file.resolve())

        self.cfccomp = CFComputations(self.test_files, 1)

    def tearDown(self):
        for _npole_file in (self.test_files).glob('npole_*'):
            if _npole_file.is_file():
                os.remove(_npole_file.resolve())

    def test_read_monopole(self):
        cf = self.cfccomp.halotools_like_cf

        with patch('CoLoRe_corrf_analysis.cf_helper.CFComputations.halotools_like_cf', new_callable=PropertyMock) as mocked_cf:
            mocked_cf.return_value = cf
            pole_0 = self.cfccomp.compute_npole(0)
            assert mocked_cf.called

        target = np.loadtxt(self.test_files / 'npole_files' / 'npole_0.dat')
        np.testing.assert_equal(pole_0, target)

    def test_read_quadrupole(self):
        cf = self.cfccomp.halotools_like_cf

        with patch('CoLoRe_corrf_analysis.cf_helper.CFComputations.halotools_like_cf', new_callable=PropertyMock) as mocked_cf:
            mocked_cf.return_value = cf
            pole_2 = self.cfccomp.compute_npole(2)
            assert mocked_cf.called

        target = np.loadtxt(self.test_files / 'npole_files' / 'npole_2.dat')
        np.testing.assert_equal(pole_2, target)
    
class TestComputeFilesCross(unittest.TestCase):
    def setUp(self):
        _current_dir = Path(__file__).parent
        self.test_files = _current_dir / 'test_files' / 'cf_helper_cross' / '0'
        for _npole_file in (self.test_files).glob('npole_*'):
            if _npole_file.is_file():
                os.remove(_npole_file.resolve())

        self.cfccomp = CFComputations(self.test_files, 1)

    def tearDown(self):
        for _npole_file in (self.test_files).glob('npole_*'):
            if _npole_file.is_file():
                os.remove(_npole_file.resolve())

    def test_read_monopole(self):
        cf = self.cfccomp.halotools_like_cf

        with patch('CoLoRe_corrf_analysis.cf_helper.CFComputations.halotools_like_cf', new_callable=PropertyMock) as mocked_cf:
            mocked_cf.return_value = cf
            pole_0 = self.cfccomp.compute_npole(0)
            assert mocked_cf.called

        target = np.loadtxt(self.test_files / 'npole_files' / 'npole_0.dat')
        np.testing.assert_equal(pole_0, target)

    def test_read_quadrupole(self):
        cf = self.cfccomp.halotools_like_cf

        with patch('CoLoRe_corrf_analysis.cf_helper.CFComputations.halotools_like_cf', new_callable=PropertyMock) as mocked_cf:
            mocked_cf.return_value = cf
            pole_2 = self.cfccomp.compute_npole(2)
            assert mocked_cf.called

        target = np.loadtxt(self.test_files / 'npole_files' / 'npole_2.dat')
        np.testing.assert_equal(pole_2, target)
    