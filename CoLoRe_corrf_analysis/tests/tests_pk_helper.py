"""
    Tests for pk_helper. Run it using:
        coverage run --source . -m unittest discover CoLoRe_corrf_analysis/tests
        python -m coverage html --omit="*/tests*","*__init__.py","*hidden_*","setup.py"
"""

import os
import unittest
from pathlib import Path

import numpy as np

from CoLoRe_corrf_analysis.pk_helper import PKComputations


class TestComputePk(unittest.TestCase):
    def setUp(self):
        _current_dir = Path(__file__).parent
        self.test_files = _current_dir / "test_files" / "colore_box_snapshot"
        for _npole_file in self.test_files.glob("pk_data*.dat"):
            if _npole_file.is_file():
                os.remove(_npole_file.resolve())

        self.pkcomp = PKComputations(self.test_files, 1, self.test_files / "param.cfg", rsd=False)
        self.pkcomp_rsd = PKComputations(self.test_files, 1, self.test_files / "param.cfg", rsd=True)

    def tearDown(self):
        for _npole_file in self.test_files.glob("pk_data*.dat"):
            if _npole_file.is_file():
                os.remove(_npole_file.resolve())


    def test_compute_pk_rebin(self):
        pkcomp = PKComputations(self.test_files, 1, self.test_files / "param.cfg", rsd=False, pk_n_bins=10)

        for pole in 0, 2:
            pk = pkcomp.compute_npole(pole)

            target = np.loadtxt(
                self.test_files
                / "npole_files"
                / f"pk_data_src1_{pole}_nbins_10.dat"
            )
            np.testing.assert_equal(pk, target)

    def test_compute_pk_extra_fields(self):
        for field, name in ((0, "gaussian"), (-1, "nonlinear"), (-2, "eta")):
            pkcomp = PKComputations(self.test_files, field, self.test_files / "param.cfg") 

            for pole in 0,2:
                pk = pkcomp.compute_npole(pole)

                target = np.loadtxt(
                    self.test_files
                    / "npole_files"
                    / f"pk_data_{name}_{pole}.dat"
                )
                np.testing.assert_equal(pk, target)

                pkcomp2 = PKComputations(
                    self.test_files, field, self.test_files / "param.cfg"
                )

                pk2 = pkcomp2.compute_npole(pole)

                np.testing.assert_equal(pk, pk2)

    def test_compute_pk(self):
        for pole in 0, 2:
            pk = self.pkcomp.compute_npole(pole)

            target = np.loadtxt(
                self.test_files
                / "npole_files"
                / f"pk_data_src1_{pole}.dat"
            )
            np.testing.assert_equal(pk, target)

            pkcomp2 = PKComputations(
                self.test_files, 1, self.test_files / "param.cfg"
            )

            pk2 = pkcomp2.compute_npole(pole)

            np.testing.assert_equal(pk, pk2)

    def test_compute_pk_rsd(self):
        for pole in 0, 2:
            pk = self.pkcomp_rsd.compute_npole(pole)

            target = np.loadtxt(
                self.test_files
                / "npole_files"
                / f"pk_data_src1_rsd_{pole}.dat"
            )
            np.testing.assert_equal(pk, target)

            pkcomp2 = PKComputations(
                self.test_files, 1, self.test_files / "param.cfg", rsd=True,
            )

            pk2 = pkcomp2.compute_npole(pole)

            np.testing.assert_equal(pk, pk2)