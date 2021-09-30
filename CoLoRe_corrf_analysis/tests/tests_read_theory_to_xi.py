"""
    Tests for read_theory_to_xi. Run it using:
        coverage run --source . -m unittest discover CoLoRe_corrf_analysis/tests
        python -m coverage html --omit="*/tests*","*__init__.py","*hidden_*","setup.py"
"""

import os
import unittest
from unittest import skipIf
from pathlib import Path
from CoLoRe_corrf_analysis.read_colore import ComputeModelsCoLoRe
import numpy as np


@skipIf('FAST_TEST' in os.environ, 'Skipping secondary functions')
class TestCommon(unittest.TestCase):
    def setUp(self):
        _current_dir = Path(__file__).parent
        self.sim_path = _current_dir / 'test_files' / 'colore_box'
        self.bias_filename = self.sim_path / 'bias.txt'
        self.nz_filename = self.sim_path / 'nz.txt'
        self.pk_filename = self.sim_path / 'pk.txt'

        self.theory = ComputeModelsCoLoRe(self.sim_path,
            source=2,
            nz_filename=self.nz_filename,
            bias_filename=self.bias_filename,
            pk_filename=self.pk_filename,
            smooth_factor=0.9,
            smooth_factor_cross=0.9,
            smooth_factor_rsd=0.9,
            smooth_factor_analysis=0,
            apply_lognormal=True)
    
    def tearDown(self):
        pass

    def test_param_cfg(self):
        paramcfg = self.theory.param_cfg
        self.assertEqual(paramcfg['field_par']['r_smooth'], 2)

    def test_smooth(self):
        smooth = self.theory.smooth_factor
        self.assertAlmostEqual(smooth, 0.9)

    def test_volume_between_zs(self):
        self.assertAlmostEqual(self.theory.get_volume_between_zs(0.6, 0.1).value, 44683901350.33176)
        self.assertAlmostEqual(self.theory.get_volume_between_zs(0.3).value, 7137310306.571973)
    
    def test_get_nz_histogram_from_Nz_file(self):
        hist = self.theory.get_nz_histogram_from_Nz_file(np.linspace(0.6,0.8,10))
        value = np.asarray([5.18415685, 5.15552966, 5.1180965 , 5.07282613, 5.02045853, 4.96201839, 4.89833224, 4.83019827, 4.75838343])
        np.testing.assert_almost_equal(hist, value)

    def test_get_nz_histogram_from_CoLoRe_box(self):
        hist = self.theory.get_nz_histogram_from_CoLoRe_box(np.linspace(1.38,1.41,10), rsd=False)
        values = np.asarray([ 33.3333333,  55.5555556,  33.3333333, 100.       ,  44.4444444, 33.3333333, 0, 0, 0])
        np.testing.assert_almost_equal(hist, values)

        hist = self.theory.get_nz_histogram_from_CoLoRe_box(np.linspace(1.38,1.41,10), rsd=True)
        values = np.asarray([32.1428571, 53.5714286, 75.       , 64.2857143, 64.2857143,
       10.7142857, 0, 0, 0])
        np.testing.assert_almost_equal(hist, values)

    def test_get_zeff(self):
        zeff = self.theory.get_zeff(zmin=0.5, zmax=0.6, method='Nz_file')
        self.assertAlmostEqual(zeff, 0.550096831361585)

    def test_get_zeff_2(self):
        zeff = self.theory.get_zeff(zmin=1.38, zmax=1.41, method='CoLoRe', rsd=False)
        self.assertAlmostEqual(zeff,  1.3904252199413487)

    def test_get_zeff_3(self):
        zeff = self.theory.get_zeff(zmin=1.38, zmax=1.41, method='CoLoRe', rsd=True)
        self.assertAlmostEqual(zeff, 1.3892245989304808)

    def test_z_bins_from_files(self):
        values = np.asarray([0. , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. , 1.1, 1.2, 1.3])
        np.testing.assert_almost_equal(self.theory.z_bins_from_files(), values)

    # def test_beta_from_growth(self):
    #     self.assertAlmostEqual(self.theory.beta_from_growth(0.8), 0.5523874685854183)
    
    def test_logarithmic_growth_rate(self):
        self.assertAlmostEqual(self.theory.logarithmic_growth_rate(0.8, read_file=False), 0.8315978564605921)

    def test_bias(self):
        self.assertAlmostEqual(self.theory.bias(0.4), 1.23549395)

    def test_get_a_eq(self):
        self.assertAlmostEqual(self.theory.get_a_eq(), 0.7541604105537322)

    def test_growth_factor(self):
        self.assertAlmostEqual(self.theory.growth_factor(0.3), 0.2966618242859617)

    def test_growth_factor_below_alim(self):
        alim = 0.01 * self.theory.get_a_eq()
        self.assertAlmostEqual(self.theory.growth_factor(0.8*alim), 0.8*alim)
   
class TestComputeModelsCoLoRe(unittest.TestCase):
    def setUp(self):
        _current_dir = Path(__file__).parent
        self.sim_path = _current_dir / 'test_files' / 'colore_box'
        self.bias_filename = self.sim_path / 'bias.txt'
        self.nz_filename = self.sim_path / 'nz.txt'
        self.pk_filename = self.sim_path / 'pk.txt'

        self.theory = ComputeModelsCoLoRe(self.sim_path,
            source=2,
            nz_filename=self.nz_filename,
            bias_filename=self.bias_filename,
            pk_filename=self.pk_filename,
            smooth_factor=0.9,
            smooth_factor_cross=0.9,
            smooth_factor_rsd=0.9,
            smooth_factor_analysis=0,
            apply_lognormal=True)

        self.theory_dm = ComputeModelsCoLoRe(self.sim_path,
            source=2,
            nz_filename=self.nz_filename,
            bias_filename=self.bias_filename,
            pk_filename=self.pk_filename,
            smooth_factor=0.9,
            smooth_factor_cross=0.9,
            smooth_factor_rsd=0.9,
            smooth_factor_analysis=0,
            apply_lognormal=True)
        
        self.theory_mm = ComputeModelsCoLoRe(self.sim_path,
            source=2,
            nz_filename=self.nz_filename,
            bias_filename=self.bias_filename,
            pk_filename=self.pk_filename,
            smooth_factor=0.9,
            smooth_factor_cross=0.9,
            smooth_factor_rsd=0.9,
            smooth_factor_analysis=0,
            apply_lognormal=True)

            
        self.theory_nolog = ComputeModelsCoLoRe(self.sim_path,
            source=2,
            nz_filename=self.nz_filename,
            bias_filename=self.bias_filename,
            pk_filename=self.pk_filename,
            smooth_factor=0.9,
            smooth_factor_cross=0.9,
            smooth_factor_rsd=0.9,
            smooth_factor_analysis=0,
            apply_lognormal=False)

        self.theory_smoothings = ComputeModelsCoLoRe(self.sim_path,
            source=2,
            nz_filename=self.nz_filename,
            bias_filename=self.bias_filename,
            pk_filename=self.pk_filename,
            smooth_factor=2,
            smooth_factor_cross=3,
            smooth_factor_rsd=3,
            smooth_factor_analysis=0,
            apply_lognormal=True)

        self.theory_with_analysis_smoothing = ComputeModelsCoLoRe(self.sim_path,
            source=2,
            nz_filename=self.nz_filename,
            bias_filename=self.bias_filename,
            pk_filename=self.pk_filename,
            smooth_factor=2,
            smooth_factor_cross=3,
            smooth_factor_rsd=1,
            smooth_factor_analysis=0.3,
            apply_lognormal=True)

    def tearDown(self):
        pass

    def test_read_pk(self):
        mean = np.mean(self.theory.input_pk)
        std = np.std(self.theory.input_pk)

        np.testing.assert_almost_equal([mean, std], [2466.1525969, 5985.2580126])

    def test_get_r(self):
        _ = self.theory.r

    def test_read_xi(self):
        mean = np.mean(self.theory.input_xi)
        std = np.std(self.theory.input_xi)
        np.testing.assert_almost_equal([mean, std], [317.0723792, 1190.9480197])

    def test_L_box(self):
        self.assertEqual(self.theory.L_box(), 5849.867290143846)

    def test_get_theory_pk_fixed_bias(self):
        _, _pk = self.theory.get_theory_pk(z=0.3, bias=3, lognormal=False)
        mean = np.mean(_pk)
        std = np.std(_pk)
        np.testing.assert_almost_equal([mean, std], [31778.7111751, 51096.7866222])
    
    def test_get_theory_pk_fixed_bias_2(self):
        _, _pk = self.theory.get_theory_pk(z=0.3, bias=3, lognormal=True)
        mean = np.mean(_pk)
        std = np.std(_pk)
        np.testing.assert_almost_equal([mean, std], [384406.2408837, 354083.5820729])

    def test_get_theory_pk_fixed_bias_3(self):
        _, _pk = self.theory_dm.get_theory_pk(z=0.3, bias=3, lognormal=False, tracer='dm')
        mean = np.mean(_pk)
        std = np.std(_pk)
        np.testing.assert_almost_equal([mean, std], [10592.903725 , 17032.2622074])

    def test_get_theory_pk_fixed_bias_4(self):
        _, _pk = self.theory_mm.get_theory_pk(z=0.3, bias=3, lognormal=False, tracer='mm')
        mean = np.mean(_pk)
        std = np.std(_pk)
        np.testing.assert_almost_equal([mean, std], [3530.9679083, 5677.4207358])

    def test_get_theory_pk(self):
        _, pk = self.theory.get_theory_pk(z=0.3, lognormal=False)
        mean = np.mean(pk)
        std = np.std(pk)
        np.testing.assert_almost_equal([mean, std], [4854.8548006, 7806.0900098])

    def test_get_theory_pk_2(self):
        _, pk = self.theory_smoothings.get_theory_pk(z=0.3, lognormal=False, smooth_factor=self.theory_smoothings.smooth_factor_rsd)
        mean = np.mean(pk)
        std = np.std(pk)

        np.testing.assert_almost_equal([mean, std], [4842.5123089, 7803.8232742])

    def test_get_npole_pk(self):
        pk = self.theory.get_npole_pk(0, 0.3, rsd=False)
        mean = np.mean(pk)
        std = np.std(pk)
        np.testing.assert_almost_equal([mean, std], [5661.9906253, 8366.7728458])

    def test_get_npole_pk_playing_with_smoothings(self):
        smooth_factor = self.theory.smooth_factor
        smooth_factor_rsd = self.theory.smooth_factor_rsd

        pk = self.theory_smoothings.get_npole_pk(0, 0.3, rsd=False, smooth_factor=smooth_factor, smooth_factor_rsd=smooth_factor_rsd)
        mean = np.mean(pk)
        std = np.std(pk)
        
        np.testing.assert_almost_equal([mean, std], [5661.9906253, 8366.7728458])

    def test_get_npole_pk_2(self):
        pk = self.theory.get_npole_pk(2, 0.3, rsd=False)
        mean = np.mean(pk)
        std = np.std(pk)
        self.assertAlmostEqual(mean, 0)
        self.assertAlmostEqual(std, 0)

    def test_get_npole_pk_3(self):
        pk = self.theory.get_npole_pk(0, 0.3, rsd=True)
        mean = np.mean(pk)
        std = np.std(pk)
        np.testing.assert_almost_equal([mean, std], [7832.7589607, 11850.477662])

    def test_get_npole_pk_4(self):
        pk = self.theory.get_npole_pk(2, 0.3, rsd=True)
        mean = np.mean(pk)
        std = np.std(pk)
        np.testing.assert_almost_equal([mean, std], [4614.2255734, 7419.1838131])

    def test_get_npole_pk_5(self):
        pk = self.theory_nolog.get_npole_pk(2, 0.3, rsd=True)
        mean = np.mean(pk)
        std = np.std(pk)
        np.testing.assert_almost_equal([mean, std], [4614.2255734, 7419.1838131])        

    def test_get_npole_pk_6(self):        
        pk = self.theory.get_npole_pk(4, 0.3, rsd=True)
        mean = np.mean(pk)
        std = np.std(pk)
        np.testing.assert_almost_equal([mean, std], [363.5852037, 584.6063256])

    def test_get_npole_pk_7(self):        
        pk = self.theory.get_npole_pk(0, 0.3, rsd=True, bias=0.4)
        mean = np.mean(pk)
        std = np.std(pk)
        np.testing.assert_almost_equal([mean, std], [1524.7644253, 2442.1429372])

    def test_get_npole_pk_7(self):        
        pk = self.theory_with_analysis_smoothing.get_npole_pk(0, 0.3, rsd=True, bias=0.4)
        mean = np.mean(pk)
        std = np.std(pk)
        np.testing.assert_almost_equal([mean, std], [1516.5875394, 2440.126665 ])

    def test_get_npole(self):
        xi = self.theory.get_npole(4, 0.3, rsd=True)
        mean = np.mean(xi)
        std = np.std(xi)
        np.testing.assert_almost_equal([mean, std], [0.0008141703529694578, 0.001857607536922607])    

    def test_mixing_smoothings(self):
        z = 0.4
        pk_l = self.theory_smoothings.get_theory_pk(z, bias=None, lognormal=True, smooth_factor=self.theory_smoothings.smooth_factor)[1]
        pk_s = self.theory_smoothings.get_theory_pk(z, bias=None, lognormal=False, smooth_factor=self.theory_smoothings.smooth_factor_rsd)[1]
              
        f = self.theory_smoothings.logarithmic_growth_rate(z, read_file=False)
        bias = self.theory_smoothings.bias(z)
        target = pk_l + (2*f/(3.0*bias) + (f/bias)**2/5.0)*pk_s

        value = self.theory_smoothings.get_npole_pk(0, z, True)
        np.testing.assert_almost_equal(value, target)

class TestLyaBox(unittest.TestCase):
    def setUp(self):
        _current_dir = Path(__file__).parent
        self.sim_path = _current_dir / 'test_files' / 'lyacolore_box'
        self.bias_filename = self.sim_path / 'bias.txt'
        self.nz_filename = self.sim_path / 'nz.txt'
        self.pk_filename = self.sim_path / 'pk.txt'

        self.theory = ComputeModelsCoLoRe(self.sim_path,
            source=1,
            bias_filename=self.bias_filename,
            pk_filename=self.pk_filename,
            nz_filename=self.nz_filename,
            smooth_factor=0.9,
            smooth_factor_cross=0.9,
            smooth_factor_rsd=0.9,
            smooth_factor_analysis=0,
            apply_lognormal=True)

        self.master_file = self.sim_path / 'master.fits'
    
    def tearDown(self):
        pass

    def test_get_nz_from_master(self):
        value = self.theory.get_nz_histogram_from_master_file(self.master_file, bins=np.linspace(2.2, 2.4, 8), rsd=True)
        target = np.array([5.0954654, 4.6778043, 3.8842482, 4.8866348, 5.3460621, 5.7637232, 5.3460621])
        np.testing.assert_almost_equal(value, target)

        value = self.theory.get_nz_histogram_from_master_file(self.master_file, bins=np.linspace(2.2, 2.4, 8), rsd=False)
        target = np.array([5.2021403, 4.7859691, 4.1200951, 4.9108205, 5.3269917, 5.3686088, 5.2853746])
        np.testing.assert_almost_equal(value, target)

    def test_get_zeff(self):
        value = self.theory.get_zeff(zmin=2.1, zmax=2.4, method='master_file', master_file=self.master_file)
        target = 2.233852727789108
        np.testing.assert_almost_equal(value, target)

    def test_logarithmic_growth_rate(self):
        value = self.theory.logarithmic_growth_rate(2.3)
        self.assertEqual(value, 0.9680556767606249)

    # def test_beta_from_file(self):
    #     value = self.theory.beta_from_file(2.5)
    #     self.assertEqual(value, 0.23565962090703166)

    def test_combine_z_npoles_pk(self):
        pk = self.theory.combine_z_npoles(0, [2.3, 2.4, 2.8], rsd=False, mode='pk', method='master_file', master_file=self.master_file)
        mean = np.mean(pk)
        std = np.std(pk)
        np.testing.assert_almost_equal([mean, std], [15203.0492363, 19841.0608076])

    def test_combine_z_npoles_xi(self):
        xi = self.theory.combine_z_npoles(0, [2.3, 2.4, 2.8], rsd=False, mode='xi', method='master_file', master_file=self.master_file)
        mean = np.mean(xi)
        std = np.std(xi)
        np.testing.assert_almost_equal([mean, std], [10.4900322,  9.7391197])