import json
import logging
import numpy as np
from pathlib import Path
import warnings

from functools import cached_property
from Corrfunc.utils import convert_3d_counts_to_cf
from halotools.mock_observables import tpcf_multipole

logger = logging.getLogger(__name__)


class CFComputations:
    dtypes = {
        "names": ("smin", "smax", "savg", "mu_max", "npairs", "weightavg"),
        "formats": ("<f8", "<f8", "<f8", "<f8", "<f8", "<f8"),
    }

    def __init__(self, results_path, label=""):
        """Class to handle results from corrfunc and compute multipoles.

        Args:
            results_path (Path): Path to the results from corrfunc.
            label (str, optional): Label the results object. (Default: '').
        """
        self.results_path = results_path

        try:
            with open(self.results_path / "sizes.json") as json_file:
                self.sizes = json.load(json_file)
        except OSError:
            warnings.warn(
                "Trying to read sizes json file failed. This may happen when running analysis on previous corrfunc runs. Setting all datas/randoms to the same size."
            )
            self.sizes = dict(Data=1, Randoms=1)

        if "Data2" not in self.sizes:
            self.sizes["Data2"] = self.sizes["Data"]
        if "Randoms2" not in self.sizes:
            self.sizes["Randoms2"] = self.sizes["Randoms"]

        self.label = label

    def __str__(self):  # pragma: no cover
        return self.label

    @property
    def DD(self):
        return np.loadtxt(self.results_path / "DD.dat", dtype=self.dtypes)

    @property
    def DR(self):
        return np.loadtxt(self.results_path / "DR.dat", dtype=self.dtypes)

    @property
    def RR(self):
        return np.loadtxt(self.results_path / "RR.dat", dtype=self.dtypes)

    @property
    def RD(self):
        try:
            return np.loadtxt(self.results_path / "RD.dat", dtype=self.dtypes)
        except OSError:  # pragma: no cover
            np.savetxt(self.results_path / "RD.dat", self.DR)
            return self.RD

    @property
    def r(self):
        """Return savg"""
        return self.savg

    @property
    def savg(self):
        """Read savg (separations in r for the data"""
        try:
            return np.loadtxt(self.results_path / "savg.dat")
        except OSError:
            return np.unique((self.DD["smin"] + self.DD["smax"])) / 2

    @property
    def mubins(self):
        try:
            return np.loadtxt(self.results_path / "mubins.dat")
        except OSError:
            return np.concatenate([[0], np.unique(self.DD["mu_max"])])

    @cached_property
    def cf(self):
        return convert_3d_counts_to_cf(
            self.sizes["Data"],
            self.sizes["Data2"],
            self.sizes["Randoms"],
            self.sizes["Randoms2"],
            self.DD,
            self.DR,
            self.RD,
            self.RR,
        )

    @cached_property
    def halotools_like_cf(self):
        """Transform results from self.cf into something halootols could read"""
        cf_array = []
        for smin in np.unique(self.DD["smin"]):
            cf_array.append(self.cf[self.DD["smin"] == smin])
        return cf_array

    def estimate_cubic_randoms(self):
        """Estimate cubic uniform randoms and write them to file for DR, RD and RR"""
        smin = self.DD["smin"]
        smax = self.DD["smax"]

        N1 = self.sizes["Data"]
        N2 = self.sizes["Data2"]
        NR1 = N1
        NR2 = N2

        V_box = self.sizes["Box"]**3

        V_shell = 4 * np.pi / 3 * (smax**3 - smin**3)

        prefactor = V_shell / len(np.unique(self.DD["mu_max"])) / V_box

        RR_factor = (NR1 * NR2)
        DR_factor = (N1*NR2)
        RD_factor = (N2*NR1)

        for name, factor in zip(["RR", "DR", "RD"], [RR_factor, DR_factor, RD_factor]):
            pairs = np.array(
                list(
                    zip(smin, smax, self.DD["savg"], self.DD["mu_max"], prefactor*factor, self.DD["weightavg"])
                ),
                dtype=self.DD.dtype,
            )
            np.savetxt(self.results_path / f"{name}.dat", pairs)

    def compute_npole(self, n):
        """Compute multipoles by using halotools utility. Multipoles will be saved to file in order to speed up analysis (if possible).

        Args:
            n (int): Multipole to return.

        Returns:
            multipole as 1D array of length len(self.savg)
        """
        try:
            npole = np.loadtxt(self.results_path / f"npole_{n}.dat")
        except OSError:
            npole = tpcf_multipole(self.halotools_like_cf, self.mubins, order=n)
            try:
                np.savetxt(self.results_path / f"npole_{n}.dat", npole)
            except PermissionError:  # pragma: no cover
                logger.info("Unable to save npole to file due to permission error")

        return npole

    def remove_computed_npoles(self, poles=[0, 2, 4]):  # pragma: no cover
        """Remove computed multipoles from files. This might be needed in order to avoid loading saved multipoles with method self.compute_npole()"""
        for pole in poles:
            file = self.results_path / f"npole_{pole}.dat"
            file.unlink(missing_ok=True)


class CFComputationsAbacus:
    """Simple class to read correlation measurements from Abacus results."""

    def __init__(self, path=None):
        self.path = path
        self.npoles = dict()

        try:
            r, _0, _2, _4 = np.loadtxt(path, unpack=True)
            self.npoles[0] = _0
            self.npoles[2] = _2
            self.npoles[4] = _4
        except ValueError:
            r, _0 = np.loadtxt(path, unpack=True)
            self.npoles[0] = _0
            self.npoles[2] = np.zeros_like(_0)
            self.npoles[4] = np.zeros_like(_0)

        self.r = r
        self.savg = r

    def compute_npole(self, n):
        return self.npoles[n]


def main():  # pragma: no cover
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--search-path", type=Path, help="Path to recursively search for boxes"
    )

    parser.add_argument(
        "--multipoles", nargs="+", type=int, help="multipoles to process"
    )

    args = parser.parse_args()
    for root, dirs, files in os.walk(args.search_path.resolve()):
        if ("DD.dat" in files) or ("0_DD.dat" in files):
            print(root, "has a box")
            cfccomp = CFComputations(Path(root), 1)
            for pole in args.multipoles:
                print("computing pole", pole)
                cfccomp.compute_npole(pole)


if __name__ == "__main__":  # pragma: no cover
    main()
