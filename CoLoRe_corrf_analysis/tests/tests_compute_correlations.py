import unittest
from unittest.mock import patch
from pathlib import Path
from shutil import rmtree
from types import SimpleNamespace

import numpy as np
from CoLoRe_corrf_analysis import compute_correlations

def mock_choice(a, size, p):
    _mask = [True for i in range(size)]
    for i in range(50):
        _mask[i] = False
    return _mask

class TestComputeCorrelationsAuto(unittest.TestCase):
    files_path = Path(__file__).parent / 'test_files' / 'correlations'
    catalogues = files_path.parent / 'catalogues'
    out_dir = files_path / 'auto' / 'output'

    args = SimpleNamespace(
        data=str((catalogues / 's4_rsd.fits').resolve()),
        randoms=str((catalogues / 's4_rsd_rand.fits').resolve()),
        data2=None, randoms2=None,
        out_dir=str((out_dir).resolve()),
        nthreads=8,
        mu_max=1, nmu_bins=4,
        min_bin=0.1, max_bin=200, n_bins = 7,
        norsd = False,
        zmin=0, zmax=10,
        drq_format=True,
        zmin_covd=0.8, zmax_covd=1.5, zstep_covd=0.01,
        random_downsampling=1, pixel_mask=None, nside=2,
        log_level='WARNING'
    ) 

    def setUp(self):
        try:
            self.out_dir.mkdir()
        except FileExistsError:
            rmtree(self.args.out_dir)
            self.out_dir.mkdir()
        
    def tearDown(self):
        if self.out_dir.is_dir():
            rmtree(self.out_dir)

    def test_autocorrelation(self):
        compute_correlations.main(self.args)

        DD = np.loadtxt(self.out_dir.parent / 'output' / '0_DD.dat')
        DR = np.loadtxt(self.out_dir.parent / 'output' / '0_DR.dat')
        RR = np.loadtxt(self.out_dir.parent / 'output' / '0_RR.dat')

        DD_target = np.loadtxt(self.out_dir.parent / 'target_values' / '0_DD.dat')
        DR_target = np.loadtxt(self.out_dir.parent / 'target_values' / '0_DR.dat')
        RR_target = np.loadtxt(self.out_dir.parent / 'target_values' / '0_RR.dat')

        np.testing.assert_equal((DD,DR,RR), (DD_target, DR_target, RR_target))        

class TestComputeCorrelationsCross(unittest.TestCase):
    files_path = Path(__file__).parent / 'test_files' / 'correlations'
    catalogues = files_path.parent / 'catalogues'
    out_dir = files_path / 'cross' / 'output'

    args = SimpleNamespace(
        data=str((catalogues / 's4_rsd.fits').resolve()),
        randoms=str((catalogues / 's4_rsd_rand.fits').resolve()),
        data2=str((catalogues / 's1_rsd.fits').resolve()),
        randoms2=None,
        out_dir=str((out_dir).resolve()),
        nthreads=8,
        mu_max=1, nmu_bins=4,
        min_bin=0.1, max_bin=200, n_bins = 7,
        norsd = False,
        zmin=0, zmax=10,
        drq_format=True,
        zmin_covd=0.8, zmax_covd=1.5, zstep_covd=0.01,
        random_downsampling=1, pixel_mask=None, nside=2,
        log_level='DEBUG'
    ) 

    def setUp(self):
        try:
            self.out_dir.mkdir()
        except FileExistsError:
            rmtree(self.args.out_dir)
            self.out_dir.mkdir()
        
    def tearDown(self):
        if self.out_dir.is_dir():
            rmtree(self.out_dir)

    def test_crosscorrelation(self):
        compute_correlations.main(self.args)

        DD = np.loadtxt(self.out_dir.parent / 'output' / '0_DD.dat')
        DR = np.loadtxt(self.out_dir.parent / 'output' / '0_DR.dat')
        RR = np.loadtxt(self.out_dir.parent / 'output' / '0_RR.dat')

        DD_target = np.loadtxt(self.out_dir.parent / 'target_values' / '0_DD.dat')
        DR_target = np.loadtxt(self.out_dir.parent / 'target_values' / '0_DR.dat')
        RR_target = np.loadtxt(self.out_dir.parent / 'target_values' / '0_RR.dat')

        np.testing.assert_equal((DD,DR,RR), (DD_target, DR_target, RR_target))   

    def test_crosscorrelation_2_randoms(self):
        self.args.randoms2 = str((self.catalogues / 's1_rsd_rand.fits').resolve())
        compute_correlations.main(self.args)

        DD = np.loadtxt(self.out_dir.parent / 'output' / '0_DD.dat')
        DR = np.loadtxt(self.out_dir.parent / 'output' / '0_DR.dat')
        RD = np.loadtxt(self.out_dir.parent / 'output' / '0_RD.dat')
        RR = np.loadtxt(self.out_dir.parent / 'output' / '0_RR.dat')

        DD_target = np.loadtxt(self.out_dir.parent / 'target_values_2_randoms' / '0_DD.dat')
        DR_target = np.loadtxt(self.out_dir.parent / 'target_values_2_randoms' / '0_DR.dat')
        RR_target = np.loadtxt(self.out_dir.parent / 'target_values_2_randoms' / '0_RR.dat')

        np.testing.assert_equal((DD,DR,RR), (DD_target, DR_target, RR_target))  

    def test_crosscorrelation_2(self):
        self.args.pixel_mask = [17,]
        compute_correlations.main(self.args)

        DD = np.loadtxt(self.out_dir.parent / 'output' / '0_DD.dat')
        DR = np.loadtxt(self.out_dir.parent / 'output' / '0_DR.dat')
        RR = np.loadtxt(self.out_dir.parent / 'output' / '0_RR.dat')

        DD_target = np.loadtxt(self.out_dir.parent / 'target_values' / '0_DD.dat')
        DR_target = np.loadtxt(self.out_dir.parent / 'target_values' / '0_DR.dat')
        RR_target = np.loadtxt(self.out_dir.parent / 'target_values' / '0_RR.dat')

        np.testing.assert_equal((DD,DR,RR), (DD_target, DR_target, RR_target))   

    @patch('numpy.random.choice', side_effect=mock_choice)
    def test_crosscorrelation_downsampling(self, mock_func):
        self.args.random_downsampling=0.9
        compute_correlations.main(self.args)

        DD = np.loadtxt(self.out_dir.parent / 'output' / '0_DD.dat')
        DR = np.loadtxt(self.out_dir.parent / 'output' / '0_DR.dat')
        RR = np.loadtxt(self.out_dir.parent / 'output' / '0_RR.dat')

        DD_target = np.loadtxt(self.out_dir.parent / 'target_values_downsampling' / '0_DD.dat')
        DR_target = np.loadtxt(self.out_dir.parent / 'target_values_downsampling' / '0_DR.dat')
        RR_target = np.loadtxt(self.out_dir.parent / 'target_values_downsampling' / '0_RR.dat')

        np.testing.assert_equal((DD,DR,RR), (DD_target, DR_target, RR_target))   