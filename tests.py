"""
    Tests for read_theory_to_xi. Run it using:
        coverage run --source . -m unittest tests.py
        python -m coverage html --omit="*/tests*","*__init__.py","*hidden_*","setup.py"
"""

import unittest
from unittest import skipIf
from pathlib import Path
from read_theory_to_xi import ReadXiCoLoReFromPk
import numpy as np
import os

@skipIf('FAST_TEST' in os.environ, 'Skipping secondary functions')
class TestCommon(unittest.TestCase):
    def setUp(self):
        self.sim_path = Path("/global/cscratch1/sd/damonge/CoLoRe_sims/sim1000")
        self.bias_filename = Path('/global/u2/c/cramirez/Codes/CoLoRe/CoLoRe_LyA_v3/examples/LSST/BzBlue.txt')
        self.nz_filename = Path('/global/u2/c/cramirez/Codes/CoLoRe/CoLoRe_LyA_v3/examples/LSST/NzBlue.txt')
        self.pk_filename = Path('/global/u2/c/cramirez/Codes/CoLoRe/CoLoRe_LyA_v3/examples/simple/Pk_CAMB_test.dat')

        self.theory = ReadXiCoLoReFromPk(self.sim_path,
            source=2,
            nz_filename=self.nz_filename,
            tracer='dd',
            bias_filename=self.bias_filename,
            pk_filename=self.pk_filename,
            apply_lognormal=True)
    
    def tearDown(self):
        pass

    def test_param_cfg(self):
        paramcfg = self.theory.param_cfg
        self.assertEqual(paramcfg['field_par']['r_smooth'], 2)

    def test_smooth(self):
        smooth = self.theory.smooth_factor
        self.assertAlmostEqual(smooth, 4.611918222528009)

    def test_volume_between_zs(self):
        self.assertAlmostEqual(self.theory.get_volume_between_zs(0.6, 0.1).value, 44683901350.33176)
        self.assertAlmostEqual(self.theory.get_volume_between_zs(0.3).value, 7137310306.571973)
    
    def test_get_nz_histogram_from_Nz_file(self):
        hist = self.theory.get_nz_histogram_from_Nz_file(np.linspace(0.6,0.8,10))
        value = np.asarray([5.18415685, 5.15552966, 5.1180965 , 5.07282613, 5.02045853, 4.96201839, 4.89833224, 4.83019827, 4.75838343])
        np.testing.assert_almost_equal(hist, value)

    def test_get_nz_histogram_from_CoLoRe_box(self):
        hist = self.theory.get_nz_histogram_from_CoLoRe_box(np.linspace(0.6,0.8,10), rsd=False)
        values = np.asarray([4.619804  , 4.79760715, 4.88211416, 4.9855852 , 5.05555574, 5.06905524, 5.16379215, 5.22820073, 5.19828564])
        np.testing.assert_almost_equal(hist, values)

        hist = self.theory.get_nz_histogram_from_CoLoRe_box(np.linspace(0.6,0.8,10), rsd=True)
        values = np.asarray([4.60962804, 4.80880161, 4.87342244, 4.97886478, 5.05726929,
       5.03794825, 5.15734554, 5.25790419, 5.21881586])
        np.testing.assert_almost_equal(hist, values)

    def test_get_zeff(self):
        zeff = self.theory.get_zeff(zmin=0.5, zmax=0.6, method='Nz_file')
        self.assertAlmostEqual(zeff, 0.550096831361585)

    def test_get_zeff_2(self):
        zeff = self.theory.get_zeff(zmin=0.5, zmax=0.6, method='CoLoRe', rsd=False)
        self.assertAlmostEqual(zeff,  0.5533117963181481)

    def test_get_zeff_3(self):
        zeff = self.theory.get_zeff(zmin=0.5, zmax=0.6, method='CoLoRe', rsd=True)
        self.assertAlmostEqual(zeff, 0.5533579649219648)

    def test_z_bins_from_files(self):
        values = np.asarray([0. , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. , 1.1, 1.2, 1.3])
        np.testing.assert_almost_equal(self.theory.z_bins_from_files(), values)

    def test_beta_from_growth(self):
        self.assertAlmostEqual(self.theory.beta_from_growth(0.8), 0.5523874685854183)
    
    def test_velocity_growth_factor(self):
        self.assertAlmostEqual(self.theory.velocity_growth_factor(0.8, read_file=False), 0.8315978564605921)

    def test_bias(self):
        self.assertAlmostEqual(self.theory.bias(0.4), 1.23549395)

    def test_get_a_eq(self):
        self.assertAlmostEqual(self.theory.get_a_eq(), 0.7541604105537322)

    def test_growth_factor(self):
        self.assertAlmostEqual(self.theory.growth_factor(0.3), 0.2966618242859617)

    def test_growth_factor_below_alim(self):
        alim = 0.01 * self.theory.get_a_eq()
        self.assertAlmostEqual(self.theory.growth_factor(0.8*alim), 0.8*alim)

class TestReadXiCoLoReFromPk(unittest.TestCase):
    def setUp(self):
        self.sim_path = Path("/global/cscratch1/sd/damonge/CoLoRe_sims/sim1000")
        self.bias_filename = Path('/global/u2/c/cramirez/Codes/CoLoRe/CoLoRe_LyA_v3/examples/LSST/BzBlue.txt')
        self.nz_filename = Path('/global/u2/c/cramirez/Codes/CoLoRe/CoLoRe_LyA_v3/examples/LSST/NzBlue.txt')
        self.pk_filename = Path('/global/u2/c/cramirez/Codes/CoLoRe/CoLoRe_LyA_v3/examples/simple/Pk_CAMB_test.dat')

        self.theory = ReadXiCoLoReFromPk(self.sim_path,
            source=2,
            nz_filename=self.nz_filename,
            tracer='dd',
            bias_filename=self.bias_filename,
            pk_filename=self.pk_filename,
            apply_lognormal=True)

        self.theory_dm = ReadXiCoLoReFromPk(self.sim_path,
            source=2,
            nz_filename=self.nz_filename,
            tracer='dm',
            bias_filename=self.bias_filename,
            pk_filename=self.pk_filename,
            apply_lognormal=True)
        
        self.theory_mm = ReadXiCoLoReFromPk(self.sim_path,
            source=2,
            nz_filename=self.nz_filename,
            tracer='mm',
            bias_filename=self.bias_filename,
            pk_filename=self.pk_filename,
            apply_lognormal=True)

            
        self.theory_nolog = ReadXiCoLoReFromPk(self.sim_path,
            source=2,
            nz_filename=self.nz_filename,
            tracer='dd',
            bias_filename=self.bias_filename,
            pk_filename=self.pk_filename,
            apply_lognormal=False)
    
    def tearDown(self):
        pass

    def test_read_pk(self):
        mean = np.mean(self.theory.pk0)
        self.assertAlmostEqual(mean, 2466.1518595321545)
        
        std = np.std(self.theory.pk0)
        self.assertAlmostEqual(std, 5985.256239698585)

    def test_get_r(self):
        _ = self.theory.r

    def test_read_xi(self):
        mean = np.mean(self.theory.xi0)
        self.assertAlmostEqual(mean,  317.0726045133148)
        
        std = np.std(self.theory.xi0)
        self.assertAlmostEqual(std, 1190.9502887614178)

    def test_L_box(self):
        self.assertEqual(self.theory.L_box(), 5849.867290143846)

    def test_get_theory_pk_fixed_bias(self):
        _, _pk = self.theory.get_theory_pk(z=0.3, bias=3, lognormal=False)
        mean = np.mean(_pk)
        std = np.std(_pk)
        self.assertAlmostEqual(mean, 31778.6998344434)
        self.assertAlmostEqual(std, 51096.77284683284)

    def test_get_theory_pk_fixed_bias_2(self):
        _, _pk = self.theory.get_theory_pk(z=0.3, bias=3, lognormal=True)
        mean = np.mean(_pk)
        std = np.std(_pk)
        self.assertAlmostEqual(mean, 384403.77828437777)
        self.assertAlmostEqual(std, 354082.411167533)

    def test_get_theory_pk_fixed_bias_3(self):
        _, _pk = self.theory_dm.get_theory_pk(z=0.3, bias=3, lognormal=False)
        mean = np.mean(_pk)
        std = np.std(_pk)
        self.assertAlmostEqual(mean, 10592.899944814464)
        self.assertAlmostEqual(std, 17032.25761561095)

    def test_get_theory_pk_fixed_bias_4(self):
        _, _pk = self.theory_mm.get_theory_pk(z=0.3, bias=3, lognormal=False)
        mean = np.mean(_pk)
        std = np.std(_pk)
        self.assertAlmostEqual(mean, 3530.9666482714892)
        self.assertAlmostEqual(std, 5677.419205203649)

    def test_get_theory_pk(self):
        _, pk = self.theory.get_theory_pk(z=0.3, lognormal=False)
        mean = np.mean(pk)
        std = np.std(pk)
        self.assertAlmostEqual(mean, 4854.853068072107)
        self.assertAlmostEqual(std, 7806.087905306993)

    def test_get_npole_pk(self):
        pk = self.theory.get_npole_pk(0, 0.3, rsd=False)
        mean = np.mean(pk)
        std = np.std(pk)
        self.assertAlmostEqual(mean, 5662.23092510887)
        self.assertAlmostEqual(std, 8366.678491541392)

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
        self.assertAlmostEqual(mean, 7832.998485753411)
        self.assertAlmostEqual(std, 11850.370762557883)

    def test_get_npole_pk_4(self):
        pk = self.theory.get_npole_pk(2, 0.3, rsd=True)
        mean = np.mean(pk)
        std = np.std(pk)
        self.assertAlmostEqual(mean, 4614.22392676962)
        self.assertAlmostEqual(std, 7419.181812939575)

    def test_get_npole_pk_5(self):
        pk = self.theory_nolog.get_npole_pk(2, 0.3, rsd=True)
        mean = np.mean(pk)
        std = np.std(pk)
        self.assertAlmostEqual(mean, 4614.22392676962)
        self.assertAlmostEqual(std, 7419.181812939575)        

    def test_get_npole_pk_6(self):        
        pk = self.theory.get_npole_pk(4, 0.3, rsd=True)
        mean = np.mean(pk)
        std = np.std(pk)
        self.assertAlmostEqual(mean, 363.5850739740466)
        self.assertAlmostEqual(std, 584.6061680350734)

    def test_get_npole_pk_7(self):        
        pk = self.theory.get_npole_pk(0, 0.3, rsd=True, bias=0.4)
        mean = np.mean(pk)
        std = np.std(pk)
        self.assertAlmostEqual(mean, 1524.7921471717505)
        self.assertAlmostEqual(std, 2442.1272198962065)

    def test_get_npole(self):
        xi = self.theory.get_npole(4, 0.3, rsd=True)
        mean = np.mean(xi)
        std = np.std(xi)
        self.assertAlmostEqual(mean, 0.0008141703529694578)
        self.assertAlmostEqual(std, 0.001857607536922607)      

    def test_init(self):
        _ = ReadXiCoLoReFromPk(self.sim_path,
            source=2,
            nz_filename=self.nz_filename,
            tracer='dd',
            bias_filename=self.bias_filename,
            smooth_factor=2,
            smooth_factor_rsd=3,
            apply_lognormal=True)


class TestLyaBox(unittest.TestCase):
    def setUp(self):
        self.sim_path = Path("/global/project/projectdirs/desi/users/cramirez/lya_mock_2LPT_runs/CoLoRe/CoLoRe_lognormal/CoLoRe_seed0_4096")

        self.theory = ReadXiCoLoReFromPk(self.sim_path,
            source=1,
            tracer='dd',
            apply_lognormal=True)

        self.master_file = Path('/global/project/projectdirs/desi/users/cramirez/lya_mock_2LPT_11_runs/LyaCoLoRe/LyaCoLoRe_lognormal/LyaCoLoRe_seed0_4096/master.fits')
    
    def tearDown(self):
        pass

    def test_get_nz_from_master(self):
        value = self.theory.get_nz_histogram_from_master_file(self.master_file, bins=np.linspace(2.2, 2.4, 8), rsd=True)
        target = np.array([5.49711132, 5.34705626, 5.17662751, 5.01765572, 4.85396635,
       4.65546546, 4.45211739])
        np.testing.assert_almost_equal(value, target)

        value = self.theory.get_nz_histogram_from_master_file(self.master_file, bins=np.linspace(2.2, 2.4, 8), rsd=False)
        target = np.array([5.48844434, 5.34888828, 5.18037364, 5.01554038, 4.84591095,
       4.66256701, 4.4582754 ])
        np.testing.assert_almost_equal(value, target)

    def test_get_zeff(self):
        value = self.theory.get_zeff(zmin=2.1, zmax=2.4, method='master_file', master_file=self.master_file)
        target = 2.234216481167806
        np.testing.assert_almost_equal(value, target)

    def test_velocity_growth_factor(self):
        value = self.theory.velocity_growth_factor(2.3)
        self.assertEqual(value, 0.9680556767606249)

    def test_beta_from_file(self):
        value = self.theory.beta_from_file(2.5)
        self.assertEqual(value, 0.23565962090703166)

    def test_combine_z_npoles_pk(self):
        pk = self.theory.combine_z_npoles(0, [2.3, 2.4, 2.8], rsd=False, mode='pk', method='master_file', master_file=self.master_file)
        mean = np.mean(pk)
        std = np.std(pk)
        np.testing.assert_almost_equal(mean, 15170.067024363809)
        np.testing.assert_almost_equal(std, 19804.55115202347)

    def test_combine_z_npoles_xi(self):
        xi = self.theory.combine_z_npoles(0, [2.3, 2.4, 2.8], rsd=False, mode='xi', method='master_file', master_file=self.master_file)
        mean = np.mean(xi)
        std = np.std(xi)
        np.testing.assert_almost_equal(mean, 10.439991963256283)
        np.testing.assert_almost_equal(std, 9.691521839160348)