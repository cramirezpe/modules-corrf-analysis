from __future__ import annotations

import logging

import fitsio
import libconf
from pathlib import Path
import numpy as np
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from typing import Optional, TypeVar, Tuple

logger = logging.getLogger(__name__)


def cat2delta(
    cat: np.ndarray, lbox: float, ngrid: int, w_rsd: bool = False
) -> np.ndarray:
    # Gets overdensity grid from a catalog
    ix = np.floor((cat["X"] + cat["DX_RSD"] * w_rsd) * ngrid / lbox).astype(int)
    iy = np.floor(cat["Y"] * ngrid / lbox).astype(int)
    iz = np.floor(cat["Z"] * ngrid / lbox).astype(int)
    ix[ix >= ngrid] -= ngrid
    ix[ix < 0] += ngrid
    iy[iy >= ngrid] -= ngrid
    iy[iy < 0] += ngrid
    iz[iz >= ngrid] -= ngrid
    iz[iz < 0] += ngrid
    ind = ix + ngrid * (iy + ngrid * iz)
    n = np.bincount(ind, minlength=ngrid**3)
    nmean = len(cat) / ngrid**3
    delt = n / nmean - 1
    return delt.reshape([ngrid, ngrid, ngrid])


class PKComputations:
    ngrid: int
    lbox: float

    def __init__(
        self,
        box_dir: Path | str,
        source: int = 1,
        param_cfg: Optional[Path | str] = None,
        rsd: bool = False,
        label: str = "",
    ):
        """Class to handle results from CoLoRe in snapshot. Computing PK.

        Args:
            box_dir (str or Path): Path to CoLoRe output files.
            source (int, optional): Source to read from CoLoRe output. (Default: 1).
                Use 0 for linear density field and -1 for non-linear density field.
            param_cfg (str or Path, optional): Path to param_cfg file. (Default: None).
            rsd (bool, optional): Whether to use RSD. (Default: False).
            label (str, optional): Label the results object. (Default: '').
        """
        self.box_dir = Path(box_dir)

        self.param_cfg_file: Optional[Path] = None
        if (self.box_dir.parent / "param.cfg").is_file() and param_cfg is None:
            self.param_cfg_file = self.box_dir.parent / "param.cfg"
        elif param_cfg is not None:
            self.param_cfg_file = Path(param_cfg)

        if self.param_cfg_file is not None and self.param_cfg_file.is_file():
            with open(self.param_cfg_file) as f:
                self.param_cfg = libconf.load(f)
        else:
            raise FileNotFoundError(
                "Couldn't determine param_cfg: ", self.param_cfg_file
            )

        self.source = source
        self.rsd = rsd

        if self.source < 1 and self.rsd:
            raise ValueError("Can't use RSD for non-tracer density field.")

        self.label = label

        self.ngrid = self.param_cfg["field_par"]["n_grid"]
        self.lbox = self.param_cfg["global"]["l_box"]

    def read_grid(self) -> np.ndarray:
        if self.source == 0:
            prefix = "out_dens_gaussian"
        else:
            prefix = "out_dens_nonlinear"

        with open(self.box_dir / f"{prefix}_0.dat", "rb") as f:
            nfiles, size_float = np.fromfile(f, dtype=np.int32, count=2)
            lbox = np.fromfile(f, dtype=np.float64, count=1)[0]
            ngrid = np.fromfile(f, dtype=np.int32, count=1)[0]

        f_type: type
        if size_float == 4:
            f_type = np.float32
        else:
            f_type = np.float64

        grid_out = np.zeros([ngrid, ngrid, ngrid])

        for ifil in np.arange(nfiles):
            with open(self.box_dir / f"{prefix}_{ifil}.dat", "rb") as f:
                nf, sz = np.fromfile(f, dtype=np.int32, count=2)
                _ = np.fromfile(f, dtype=np.float64, count=1)
                ng, nz_here, iz0_here = np.fromfile(f, dtype=np.int32, count=3)
                for iz in np.arange(nz_here):
                    d = np.fromfile(f, dtype=f_type, count=ng * ng).reshape([ng, ng])
                    grid_out[iz0_here + iz, :, :] = d

        self.ngrid = ngrid
        self.lbox = lbox
        return np.array(grid_out)  # return dens

    def __str__(self) -> str:
        return self.label

    @property
    def k(self) -> np.ndarray:
        kb = np.linspace(0, self.ngrid * np.pi / self.lbox, self.ngrid // 2 + 1)
        return 0.5 * (kb[1:] + kb[:-1])[1:]

    def compute_npole(self, n: int) -> np.ndarray:
        """Compute multipoles by reading the grid files. Multipoles will be saved to file in order to speed up analysis (if possible).

        Args:
            n (int): Multipole to return.

        Returns:
            multipole as 1D array of length len(self.k)
        """
        if self.source == 0:
            name = "gaussian"
        elif self.source == -1:
            name = "nonlinear"
        else:
            name = f"src{self.source}"
            if self.rsd:
                name += "_rsd"

        file = self.box_dir / f"pk_data_{name}_{n}.dat"

        if file.is_file():
            try:
                npole = np.loadtxt(file)
                return npole
            except OSError:
                pass

        # Computes power spectrum from an overdensity grid
        if self.source < 1:
            cat = None
            d = self.read_grid()
        else:
            with fitsio.FITS(self.box_dir / f"out_srcs_s{self.source}_0.fits") as hdul:
                cat = hdul[1].read()
                d = cat2delta(cat, self.lbox, self.ngrid, w_rsd=self.rsd)

        # FFT
        dk = np.fft.rfftn(d) * (self.lbox / self.ngrid) ** 3

        # k sampling
        kfull = np.fft.fftfreq(self.ngrid, d=self.lbox / self.ngrid) * 2 * np.pi
        khalf = np.fft.rfftfreq(self.ngrid, d=self.lbox / self.ngrid) * 2 * np.pi

        # P(k) esti[ma]tor
        ks = np.sqrt(
            kfull[:, None, None] ** 2
            + kfull[None, :, None] ** 2
            + khalf[None, None, :] ** 2
        )
        if n == 0:
            wmu = 1
        else:
            mu = khalf[None, None, :] / (ks + 1e-7)  # Avoid division by zero
            if n == 2:
                wmu = 0.5 * (3 * mu**2 - 1)
            elif n == 4:
                wmu = 0.125 * (35 * mu**4 - 30 * mu**2 + 3)
        sm, kb = np.histogram(
            ks.flatten(),
            bins=self.ngrid // 2,
            range=(0, self.ngrid * np.pi / self.lbox),
            weights=(np.real(dk * np.conjugate(dk)) * wmu).flatten(),
        )
        ncell, _ = np.histogram(
            ks.flatten(),
            bins=self.ngrid // 2,
            range=(0, self.ngrid * np.pi / self.lbox),
        )
        npole = (2 * n + 1) * sm / (ncell * self.lbox**3)
        km = 0.5 * (kb[1:] + kb[:-1])

        npole = npole[1:]

        # Remove shot noise if needed
        if (cat is not None) and (n == 0):
            sn = self.lbox**3 / len(cat)
            npole -= sn

        try:
            np.savetxt(file, npole)
        except PermissionError:
            logger.info("Unable to save npole to file due to permission error.")

        return npole

class PKComputationsAbacus:
    """Simple class to read Power spectra from Abacus results."""
    def __init__(self, path: Optional[Path] = None):
        self.path = path
        self.npoles = dict()

        try:
            kcen, kmin, kmax, kavg, nmod, _0, _2, _4 = np.loadtxt(path, unpack=True)
            self.npoles[0] = _0
            self.npoles[2] = _2
            self.npoles[4] = _4
        except ValueError:
            kcen, kmin, kmax, kavg, nmod, _0 = np.loadtxt(path, unpack=True)
            self.npoles[0] = _0
            self.npoles[2] = np.zeros_like(_0)
            self.npoles[4] = np.zeros_like(_0)

        self.k = kavg

    def compute_npole(self, n: int):
        return self.npoles[n]
    


# class PKComputationsCustom(PKComputations):
#     """Allows to generate sources by giving a CoLoRe box. It also allows to include threshold."""

#     def __init__(
#         self,
#         threshold: float = -1,
#         bias: float = 0,
#         ndens: float = 0.0005,
#         bias_model: int = 2,
#         *args,
#         **kwargs,
#     ):
#         super().__init__(args, **kwargs)
#         self.threshold = threshold
#         self.bias = bias
#         self.ndens = ndens
#         self.bias_model = bias_model

#     def compute_npole(self, n: int) -> np.ndarray:
#         """Compute multipoles by reading the grid files. Multipoles will be saved to file in order to speed up analysis (if possible).

#         Args:
#             n (int): Multipole to return.

#         Returns:
#             multipole as 1D array of length len(self.k)
#         """
#         name = f"threshold_{self.threshold}_bias_{self.bias}_ndens_{self.ndens}_bias_model_{self.bias_model}"

#         file = self.box_dir / f"pk_data_{name}_{n}.dat"

#         if file.is_file():
#             try:
#                 npole = np.loadtxt(file)
#                 return npole
#             except OSError:
#                 pass

#         dg = self.read_grid()


#         dg[(dg <= -1) | (dg < self.threshold)] = 0

#         if self.bias_model == 2:
#             msk = dg < 0
#             dg[msk] = np.exp(dg[msk]*self.bias / (1 + dg[msk]))
#             dg[~msk] = 1 + dg[~msk]*self.bias
#         elif self.bias_model == 3:
#             dg = 1 + self.bias * dg
#             dg[dg < 0]= 0
#         else:
#             dg = (1 + dg)**self.bias
            
#         dens_norm = self.ngrid**3 / np.sum(dg)

#         # Add sources by poisson sampling
#         cell_vol = (self.lbox / self.ngrid) ** 3

#         lambda_ = self.ndens*cell_vol*dg*dens_norm

#         rg = np.random.Generator(np.random.MT19937(seed=42))
#         nsources = rg.poisson(lambda_)

#         # Computes power spectrum from an overdensity grid
        
