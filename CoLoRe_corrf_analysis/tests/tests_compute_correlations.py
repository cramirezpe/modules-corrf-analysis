import json
import unittest
from pathlib import Path
from shutil import rmtree
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from astropy.io import fits
from CoLoRe_corrf_analysis import compute_correlations


def mock_choice(a, size, p):
    _mask = [True for i in range(size)]
    for i in range(50):
        _mask[i] = False
    return _mask


def mock_random_random(length):
    return np.linspace(0, 1, length)


def mock_random_poisson(_lambda, length):
    return _lambda * np.ones(length, dtype=int)


def mock_random_int(min, max):
    return min


class TestComputeCorrelationsAuto(unittest.TestCase):
    files_path = Path(__file__).parent / "test_files" / "correlations"
    catalogues = files_path.parent / "catalogues"
    out_dir = files_path / "auto" / "output"

    args = SimpleNamespace(
        data=[Path(i) for i in catalogues.resolve().glob("s4_rsd.fits")],
        data_norsd=True,
        randoms=[Path(i) for i in catalogues.resolve().glob("s4_rsd_rand.fits")],
        data_format="zcat",
        data2_format="zcat",
        data2=None,
        randoms2=None,
        generate_randoms2=False,
        store_generated_rands=True,
        randoms_from_nz_file=None,
        data2_norsd=False,
        out_dir=Path((out_dir).resolve()),
        nthreads=8,
        mu_max=1,
		velocity_boost=1,
        nmu_bins=4,
        min_bin=0.1,
        max_bin=200,
        n_bins=7,
        norsd=False,
        random_positions_method="pixel",
        zmin=0,
        zmax=10,
        zmin_covd=0.8,
        zmax_covd=1.5,
        zstep_covd=0.01,
        randoms_factor=1,
        randoms_downsampling=1,
        data_downsampling=1,
        pixel_mask=None,
        nside=2,
        log_level="DEBUG",
        compute_npoles=None,
        reverse_RSD=False,
        reverse_RSD2=False,
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

        DD = np.loadtxt(self.out_dir.parent / "output" / "DD.dat")
        DR = np.loadtxt(self.out_dir.parent / "output" / "DR.dat")
        RR = np.loadtxt(self.out_dir.parent / "output" / "RR.dat")

        DD_target = np.loadtxt(self.out_dir.parent / "target_values" / "0_DD.dat")
        DR_target = np.loadtxt(self.out_dir.parent / "target_values" / "0_DR.dat")
        RR_target = np.loadtxt(self.out_dir.parent / "target_values" / "0_RR.dat")

        np.testing.assert_equal((DD, DR, RR), (DD_target, DR_target, RR_target))

        with open(self.out_dir / "sizes.json") as json_file:
            sizes = json.load(json_file)

        for field, value in zip(("Data", "Randoms"), (22123, 1032)):
            self.assertEqual(sizes[field], value)

    @patch("numpy.random.choice", side_effect=mock_choice)
    @patch("numpy.random.poisson", side_effect=mock_random_poisson)
    @patch("numpy.random.randint", side_effect=mock_random_int)
    @patch("numpy.random.random", side_effect=mock_random_random)
    def test_autocorrelation_make_randoms(
        self, mock_func, mock_func_2, mock_func_3, mock_func_4
    ):
        self.args.randoms = None
        self.args.pixel_mask = [17]
        compute_correlations.main(self.args)

        DD = np.loadtxt(self.out_dir.parent / "output" / "DD.dat")
        DR = np.loadtxt(self.out_dir.parent / "output" / "DR.dat")
        RR = np.loadtxt(self.out_dir.parent / "output" / "RR.dat")

        DD_target = np.loadtxt(self.out_dir.parent / "target_values_rand" / "DD.dat")
        DR_target = np.loadtxt(self.out_dir.parent / "target_values_rand" / "DR.dat")
        RR_target = np.loadtxt(self.out_dir.parent / "target_values_rand" / "RR.dat")

        randoms = self.out_dir.parent / "output" / "Randoms.fits"
        randoms_target = self.out_dir.parent / "target_values_rand" / "Randoms.fits"

        f_randoms = fits.open(randoms)
        z = f_randoms[1].data["Z"]
        RA = f_randoms[1].data["RA"]
        f_randoms.close()
        f_target = fits.open(randoms_target)
        z_targ = f_target[1].data["Z"]
        RA_targ = f_target[1].data["RA"]
        np.testing.assert_equal(z, z_targ)
        np.testing.assert_equal(RA, RA_targ)

        np.testing.assert_equal(DD, DD_target)
        np.testing.assert_equal(DR, DR_target)
        np.testing.assert_equal(RR, RR_target)


class TestComputeCorrelationsCross(unittest.TestCase):
    files_path = Path(__file__).parent / "test_files" / "correlations"
    catalogues = files_path.parent / "catalogues"
    out_dir = files_path / "cross" / "output"

    args = SimpleNamespace(
        data=[Path(i) for i in catalogues.resolve().glob("s4_rsd.fits")],
        data_norsd=False,
        randoms=[Path(i) for i in catalogues.resolve().glob("s4_rsd_rand.fits")],
        data2=[Path(i) for i in catalogues.resolve().glob("s1_rsd.fits")],
        data2_norsd=False,
        randoms_from_nz_file=None,
        data_format="zcat",
        data2_format="zcat",
        randoms2=None,
        generate_randoms2=False,
        out_dir=Path((out_dir).resolve()),
        nthreads=8,
        random_positions_method="pixel",
        mu_max=1,
		velocity_boost=1,
        nmu_bins=4,
        min_bin=0.1,
        max_bin=200,
        n_bins=7,
        norsd=False,
        zmin=0,
        zmax=10,
        zmin_covd=0.8,
        zmax_covd=1.5,
        zstep_covd=0.01,
        randoms_downsampling=1,
        data_downsampling=1,
        pixel_mask=None,
        nside=2,
        log_level="DEBUG",
        compute_npoles=None,
        reverse_RSD=False,
        reverse_RSD2=False,
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

        DD = np.loadtxt(self.out_dir.parent / "output" / "DD.dat")
        DR = np.loadtxt(self.out_dir.parent / "output" / "DR.dat")
        RR = np.loadtxt(self.out_dir.parent / "output" / "RR.dat")

        DD_target = np.loadtxt(self.out_dir.parent / "target_values" / "0_DD.dat")
        DR_target = np.loadtxt(self.out_dir.parent / "target_values" / "0_DR.dat")
        RR_target = np.loadtxt(self.out_dir.parent / "target_values" / "0_RR.dat")

        np.testing.assert_equal((DD, DR, RR), (DD_target, DR_target, RR_target))

    def test_crosscorrelation_2_randoms(self):
        self.args.randoms2 = [str((self.catalogues / "s1_rsd_rand.fits").resolve())]
        compute_correlations.main(self.args)

        DD = np.loadtxt(self.out_dir.parent / "output" / "DD.dat")
        DR = np.loadtxt(self.out_dir.parent / "output" / "DR.dat")
        RD = np.loadtxt(self.out_dir.parent / "output" / "RD.dat")
        RR = np.loadtxt(self.out_dir.parent / "output" / "RR.dat")

        DD_target = np.loadtxt(
            self.out_dir.parent / "target_values_2_randoms" / "0_DD.dat"
        )
        DR_target = np.loadtxt(
            self.out_dir.parent / "target_values_2_randoms" / "0_DR.dat"
        )
        RR_target = np.loadtxt(
            self.out_dir.parent / "target_values_2_randoms" / "0_RR.dat"
        )

        np.testing.assert_equal((DD, DR, RR), (DD_target, DR_target, RR_target))

    def test_crosscorrelation_2(self):
        self.args.pixel_mask = [
            17,
        ]
        compute_correlations.main(self.args)

        DD = np.loadtxt(self.out_dir.parent / "output" / "DD.dat")
        DR = np.loadtxt(self.out_dir.parent / "output" / "DR.dat")
        RR = np.loadtxt(self.out_dir.parent / "output" / "RR.dat")

        DD_target = np.loadtxt(self.out_dir.parent / "target_values" / "0_DD.dat")
        DR_target = np.loadtxt(self.out_dir.parent / "target_values" / "0_DR.dat")
        RR_target = np.loadtxt(self.out_dir.parent / "target_values" / "0_RR.dat")

        np.testing.assert_equal((DD, DR, RR), (DD_target, DR_target, RR_target))

    @patch("numpy.random.choice", side_effect=mock_choice)
    def test_crosscorrelation_downsampling(self, mock_func):
        self.args.randoms_downsampling = 0.9
        self.args.data_downsampling = 0.9
        compute_correlations.main(self.args)

        DD = np.loadtxt(self.out_dir.parent / "output" / "DD.dat")
        DR = np.loadtxt(self.out_dir.parent / "output" / "DR.dat")
        RR = np.loadtxt(self.out_dir.parent / "output" / "RR.dat")

        DD_target = np.loadtxt(
            self.out_dir.parent / "target_values_downsampling" / "0_DD.dat"
        )
        DR_target = np.loadtxt(
            self.out_dir.parent / "target_values_downsampling" / "0_DR.dat"
        )
        RR_target = np.loadtxt(
            self.out_dir.parent / "target_values_downsampling" / "0_RR.dat"
        )

        np.testing.assert_equal((DD, DR, RR), (DD_target, DR_target, RR_target))


class TestComputeRandoms(unittest.TestCase):
    @patch("numpy.random.random", side_effect=mock_random_random)
    def test_random_redshfits_from_data(self, random_mock):
        cat = Path(__file__).parent / "test_files" / "catalogues" / "s4_rsd.fits"
        data = compute_correlations.FieldData(
            cat=[cat], label="data", file_type="zcat", rsd=True
        )
        data.prepare_data(zmin=0, zmax=10, downsampling=1, pixel_mask=None, nside=2)

        rand = compute_correlations.FieldData(cat=None, label=None, file_type=None)
        rand.define_data_from_size(1000)
        rand.generate_random_redshifts_from_data(data)

        target = np.loadtxt(
            Path(__file__).parent
            / "test_files"
            / "randoms"
            / "target_values"
            / "target_zs_from_data.dat"
        )

        np.testing.assert_equal(rand.data["Z"], target)

    @patch("numpy.random.random", side_effect=mock_random_random)
    def test_random_redshifts_from_nz_file(self, random_mock):
        rand = compute_correlations.FieldData(cat=None, label=None, file_type=None)
        nz_file = Path(__file__).parent / "test_files" / "randoms" / "NzRed.txt"

        rand.generate_random_redshifts_from_file(nz_file)

        np.savetxt(
            nz_file.parent / "target_values" / "target_zs.dat", rand.data["Z"][:1000]
        )
        target = np.loadtxt(nz_file.parent / "target_values" / "target_zs.dat")

        np.testing.assert_equal(rand.data["Z"][:1000], target)

    @patch("numpy.random.random", side_effect=mock_random_random)
    def test_random_redshifts_from_nz_file_2(self, random_mock):
        rand = compute_correlations.FieldData(cat=None, label=None, file_type=None)
        nz_file = Path(__file__).parent / "test_files" / "randoms" / "NzRed.txt"

        rand.generate_random_redshifts_from_file(nz_file, zmin=0.5, zmax=0.6)

        np.savetxt(
            nz_file.parent / "target_values" / "target_zs_cut.dat",
            rand.data["Z"][:1000],
        )
        target = np.loadtxt(nz_file.parent / "target_values" / "target_zs_cut.dat")

        np.testing.assert_equal(rand.data["Z"][:1000], target)

    @patch("numpy.random.random", side_effect=mock_random_random)
    def test_random_positions(self, random_mock):
        rand = compute_correlations.FieldData(cat=None, label=None, file_type=None)

        rand.define_data_from_size(1000)
        rand.generate_random_positions(pixel_mask=[12], nside=2)

        target_folder = (
            Path(__file__).parent / "test_files" / "randoms" / "target_values"
        )

        ra_target = np.loadtxt(target_folder / "target_ra.dat")
        dec_target = np.loadtxt(target_folder / "target_dec.dat")

        np.testing.assert_equal(
            (rand.data["RA"], rand.data["DEC"]), (ra_target, dec_target)
        )

    @patch("numpy.random.random", side_effect=mock_random_random)
    def test_random_positions_2(self, random_mock):
        rand = compute_correlations.FieldData(cat=None, label=None, file_type=None)

        rand.define_data_from_size(1000)
        rand.generate_random_positions()

        target_folder = (
            Path(__file__).parent / "test_files" / "randoms" / "target_values"
        )

        ra_target = np.loadtxt(target_folder / "target_ra2.dat")
        dec_target = np.loadtxt(target_folder / "target_dec2.dat")

        np.testing.assert_equal(
            (rand.data["RA"], rand.data["DEC"]), (ra_target, dec_target)
        )


class TestComputeCorrelationsReadCoLoRe(unittest.TestCase):
    files_path = Path(__file__).parent / "test_files" / "correlations"
    colore_box = files_path.parent / "colore_box"
    catalogues = files_path.parent / "catalogues"
    out_dir = files_path / "CoLoRe_read" / "output"

    args = SimpleNamespace(
        data=[Path(i) for i in colore_box.resolve().glob("out_srcs_s2*")],
        data_norsd=False,
        randoms=[Path(i) for i in catalogues.resolve().glob("s1_rsd.fits")],
        data2=[Path(i) for i in colore_box.resolve().glob("out_srcs_s2*")],
        data2_norsd=True,
        data_format="CoLoRe",
        data2_format="CoLoRe",
        randoms2=None,
        generate_randoms2=False,
        randoms_from_nz_file=None,
        out_dir=Path((out_dir).resolve()),
        nthreads=8,
        random_positions_method="pixel",
        mu_max=1,
		velocity_boost=1,
        nmu_bins=4,
        min_bin=0.1,
        max_bin=200,
        n_bins=7,
        norsd=False,
        zmin=0,
        zmax=10,
        randoms_factor=1,
        zmin_covd=0.8,
        zmax_covd=1.5,
        zstep_covd=0.01,
        randoms_downsampling=1,
        data_downsampling=1,
        pixel_mask=None,
        nside=2,
        log_level="DEBUG",
        compute_npoles=None,
        reverse_RSD=False,
        reverse_RSD2=False,
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

    def test_crosscorrelation_CoLoRe_read(self):
        compute_correlations.main(self.args)

        DD = np.loadtxt(self.out_dir.parent / "output" / "DD.dat")
        DR = np.loadtxt(self.out_dir.parent / "output" / "DR.dat")
        RR = np.loadtxt(self.out_dir.parent / "output" / "RR.dat")

        DD_target = np.loadtxt(self.out_dir.parent / "target_values" / "DD.dat")
        DR_target = np.loadtxt(self.out_dir.parent / "target_values" / "DR.dat")
        RR_target = np.loadtxt(self.out_dir.parent / "target_values" / "RR.dat")


        DD = np.savetxt(self.out_dir.parent / "output" / "DD.dat", DD )
        DR = np.savetxt(self.out_dir.parent / "output" / "DR.dat", DR )
        RR = np.savetxt(self.out_dir.parent / "output" / "RR.dat", RR )

        DD_target = np.savetxt(self.out_dir.parent / "target_values" / "DD.dat", DD_target )
        DR_target = np.savetxt(self.out_dir.parent / "target_values" / "DR.dat", DR_target )
        RR_target = np.savetxt(self.out_dir.parent / "target_values" / "RR.dat", RR_target )


        np.testing.assert_equal((DD, DR, RR), (DD_target, DR_target, RR_target))
