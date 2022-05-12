"""
    Tests for fitter module.
"""

from CoLoRe_corrf_analysis.fitter import Fitter
from CoLoRe_corrf_analysis.cf_helper import CFComputations
from CoLoRe_corrf_analysis.read_colore import ComputeModelsCoLoRe

import unittest
from unittest import skipUnless
import numpy as np
import os
from pathlib import Path


@skipUnless("RUN_FITTER_TESTS" in os.environ, "Only run when activated in environment")
class TestAuto(unittest.TestCase):
    def setUp(self):
        _current_dir = Path(__file__).parent
        self.data_path = _current_dir / "test_files" / "cf_helper"
        self.sim_path = _current_dir / "test_files" / "colore_box"
        self.bias_filename = self.sim_path / "bias.txt"
        self.nz_filename = self.sim_path / "nz.txt"
        self.pk_filename = self.sim_path / "pk.txt"

        boxes = [CFComputations(self.data_path / str(i), 1) for i in range(3)]

        self.theory = ComputeModelsCoLoRe(
            self.sim_path,
            source=2,
            nz_filename=self.nz_filename,
            bias_filename=self.bias_filename,
            pk_filename=self.pk_filename,
            smooth_factor=0.9,
            smooth_factor_cross=0.9,
            smooth_factor_rsd=0.9,
            smooth_factor_analysis=0,
            apply_lognormal=True,
        )

        self.fitter = Fitter(
            boxes=boxes,
            z=0.2,
            poles=[0, 2],
            theory=self.theory,
            rsd=True,
            bias0=3,
            smooth_factor0=1,
            smooth_factor_rsd0=1,
            smooth_factor_cross0=1,
            rmin={0: 10, 2: 40},
            rmax={0: 200, 2: 200},
        )

    def tearDown(self):
        for _npole_file in (self.data_path / "0").glob("npole_*"):
            if _npole_file.is_file():
                os.remove(_npole_file.resolve())

    def test_fitter(self):
        self.fitter.run_fit(free_params=["bias", "smooth_factor"])
        params = self.fitter.out.params

        values = [
            params["bias"].value,
            params["bias"].stderr,
            params["smooth_factor"].value,
            params["bias"].vary,
            params["smooth_factor_rsd"].vary,
            params["smooth_factor"].vary,
        ]
        target = [1.0948031, 0.0530377, 38.7995783, True, False, True]

        np.testing.assert_almost_equal(values, target, decimal=3)


@skipUnless("RUN_FITTER_TESTS" in os.environ, "Only run when activated in environment")
class TestCross(unittest.TestCase):
    def setUp(self):
        _current_dir = Path(__file__).parent
        self.data_path = _current_dir / "test_files" / "cf_helper_cross"
        self.sim_path = _current_dir / "test_files" / "colore_box"
        self.bias_filename = self.sim_path / "bias.txt"
        self.nz_filename = self.sim_path / "nz.txt"
        self.pk_filename = self.sim_path / "pk.txt"

        boxes = [CFComputations(self.data_path / str(i), 1) for i in range(3)]
        self.theory = ComputeModelsCoLoRe(
            self.sim_path,
            source=2,
            nz_filename=self.nz_filename,
            bias_filename=self.bias_filename,
            pk_filename=self.pk_filename,
            smooth_factor=0.9,
            smooth_factor_cross=0.9,
            smooth_factor_rsd=0.9,
            smooth_factor_analysis=0,
            apply_lognormal=True,
        )

        self.fitter = Fitter(
            boxes=boxes,
            z=0.2,
            poles=[0, 2],
            theory=self.theory,
            rsd=True,
            rsd2=False,
            bias0=3,
            bias20=3,
            smooth_factor0=1,
            smooth_factor_rsd0=1,
            smooth_factor_cross0=1,
            rmin={0: 10, 2: 40},
            rmax={0: 200, 2: 200},
        )

    def tearDown(self):
        for _npole_file in (self.data_path / "0").glob("npole_*"):
            if _npole_file.is_file():
                os.remove(_npole_file.resolve())

    def test_fitter(self):
        self.fitter.run_fit(free_params=["bias", "bias2"])
        params = self.fitter.out.params

        values = [
            params["bias"].value,
            params["bias"].stderr,
            params["bias2"].value,
            params["bias"].vary,
            params["smooth_factor_rsd"].vary,
            params["bias2"].vary,
        ]
        target = [0.0101518, 0.0088402, 0.8899939, True, False, True]

        np.testing.assert_almost_equal(values, target, decimal=3)
