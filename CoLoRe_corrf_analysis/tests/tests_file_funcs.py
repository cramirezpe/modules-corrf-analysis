from pathlib import Path
import unittest
import shutil
import numpy as np

from CoLoRe_corrf_analysis.file_funcs import FileFuncs


class TestFileFuncs(unittest.TestCase):
    files_path = Path(__file__).parent / "test_files" / "filefunc"

    def setUp(self):
        self.s1_s4_full = FileFuncs.get_full_path(
            self.files_path / "s1_s4",
            rsd=True,
            rmin=0.1,
            rmax=200,
            zmin=0.5,
            zmax=0.7,
            nside=2,
            N_bins=41,
            rsd2=False,
        )
        self.s1_full = FileFuncs.get_full_path(
            self.files_path / "s1",
            rsd=True,
            rmin=0.1,
            rmax=200,
            zmin=0.5,
            zmax=0.7,
            nside=2,
            N_bins=41,
            rsd2=None,
        )

    def tearDown(self):
        pass

    def test_get_full_path(self):
        assert (self.s1_s4_full / "1000" / "0").is_dir()

    def test_get_full_path_2(self):
        assert (self.s1_full / "1000" / "0").is_dir()

    def test_get_availabel_count_files(self):
        counts = FileFuncs.get_available_count_files(self.s1_full / "1000" / "1")

        assert sorted(counts) == ["DR", "RD", "RR"]

    def test_get_available_pixels(self):
        available_pixels = FileFuncs.get_available_pixels(self.s1_full)

        assert 45 == len(available_pixels)
        for pixel in available_pixels:
            assert pixel.name not in ("1", "17", "18")

    def test_mix_sims(self):
        sims = FileFuncs.mix_sims(self.s1_s4_full)
        xi = np.asarray([sim.DR[0] for sim in sims])
        np.testing.assert_equal((xi["npairs"].mean(axis=0)), 97322.70833333333)

        pixels = [3, 4, 5, 6]
        sims = FileFuncs.mix_sims(self.s1_s4_full, pixels=pixels)
        _ = sims[0].DD["savg"]


class TestCopyConts(unittest.TestCase):
    files_path = Path(__file__).parent / "test_files" / "filefunc"

    def setUp(self):
        self.s1_s4_full = FileFuncs.get_full_path(
            self.files_path / "s1_s4",
            rsd=True,
            rmin=0.1,
            rmax=200,
            zmin=0.5,
            zmax=0.7,
            nside=2,
            N_bins=41,
            rsd2=False,
        )
        self.s1_full = FileFuncs.get_full_path(
            self.files_path / "s1",
            rsd=True,
            rmin=0.1,
            rmax=200,
            zmin=0.5,
            zmax=0.7,
            nside=2,
            N_bins=41,
            rsd2=None,
        )

    def tearDown(self):
        shutil.copy(
            self.s1_full / "1000" / "0" / "backupfordidi.dat",
            self.s1_full / "1000" / "0" / "0_DD.dat",
        )

        (self.s1_full / "1000" / "1" / "DD.dat").unlink(missing_ok=True)
        (self.s1_full / "1000" / "1" / "origin_DD.txt").unlink(missing_ok=True)

    def test_copy_counts_wont_copy_if_exists(self):
        with self.assertRaises(ValueError) as cm:
            FileFuncs.copy_counts_file(
                in_path=self.s1_s4_full / "1000" / "3",
                out_path=self.s1_full / "1000" / "3",
                in_counts="DD",
            )

        assert cm.exception.args[0] == "File already exists"
        assert cm.exception.args[1] == self.s1_full / "1000" / "3" / "DD.dat"

    def test_copy_counts_error_when_npole_exists(self):
        with self.assertRaises(ValueError) as cm:
            FileFuncs.copy_counts_file(
                in_path=self.s1_s4_full / "1000" / "0",
                out_path=self.s1_full / "1000" / "0",
                in_counts="DD",
            )

        assert cm.exception.args[0] == "Computed npole in output path. Aborting copy..."
        assert cm.exception.args[1][:-5] == str(self.s1_full / "1000" / "0" / "npole_")

    def test_succesful_copy(self):
        FileFuncs.copy_counts_file(
            in_path=self.s1_s4_full / "1000" / "1",
            out_path=self.s1_full / "1000" / "1",
            in_counts="DD",
        )

        origin = np.loadtxt(self.s1_s4_full / "1000" / "1" / "DD.dat")
        target = np.loadtxt(self.s1_full / "1000" / "1" / "DD.dat")

        np.testing.assert_equal(origin, target)

        with open(self.s1_full / "1000" / "1" / "origin_DD.txt") as f:
            line = f.readline()

        assert line == str((self.s1_s4_full / "1000" / "1" / "DD.dat").resolve())
