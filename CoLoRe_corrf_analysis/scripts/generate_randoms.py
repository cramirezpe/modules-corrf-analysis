import argparse
from pathlib import Path

import numpy as np
from CoLoRe_corrf_analysis.compute_correlations import FieldData
import logging
import sys

logger = logging.getLogger(__name__)


def getArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--from-nz-file",
        type=Path,
        required=False,
        help="Compute randoms from given dndz file",
    )
    parser.add_argument(
        "--from-cat",
        type=Path,
        nargs='+',
        required=False,
        help="Compute randoms from given drq cat",
    )
    parser.add_argument(
        "--from-colore",
        type=Path,
        nargs='+',
        required=False,
        help="Compute randoms from given CoLoRe box",
    )
    parser.add_argument(
        "--out-cat",
        type=Path,
        required=Path,
        help="Output filename for random catalogue",
    )
    parser.add_argument("--zmin", type=float, default=0)
    parser.add_argument("--zmax", type=float, default=1.5)
    parser.add_argument(
        "--downsampling", type=float, default=1, help="Downsampling to apply"
    )
    parser.add_argument(
        "--pixel-mask", type=int, required=False, nargs="+", help="Pixel mask"
    )
    parser.add_argument(
        "--nside", type=int, default=2, help="nside to use for pixel mask"
    )

    parser.add_argument(
        "--log-level",
        default="WARNING",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"],
    )

    args = parser.parse_args()
    return args


def main(args=None):
    if args is None:  # pragma: no cover
        args = getArgs()

    level = logging.getLevelName(args.log_level)
    logging.basicConfig(
        stream=sys.stdout,
        level=level,
        format="%(levelname)s:%(name)s:%(funcName)s:%(message)s",
    )

    if args.from_nz_file != None:
        from scipy.integrate import quad
        from scipy.interpolate import interp1d

        z, nz = np.loadtxt(args.from_nz_file, unpack=True)
        nz_interp = interp1d(z, nz)
        NRAND = 41252.96 * quad(nz_interp, args.zmin, args.zmax)[0]
        logger.info(f"Number of randoms to generate {NRAND}")
        NRAND *= args.downsampling
        NRAND = int(NRAND)
        logger.info(f"Number of randoms to generate after downsampling {NRAND}")

        rand = FieldData(None, "Randoms", file_type=None)
        rand.define_data_from_size(NRAND)
        rand.generate_random_redshifts_from_file(
            args.from_nz_file, zmin=args.zmin, zmax=args.zmax
        )

    if args.from_cat != None:
        rand = FieldData(args.from_cat, "Randoms", file_type="zcat")
        rand.prepare_data(
            args.zmin, args.zmax, args.downsampling, args.pixel_mask, args.nside
        )
    elif args.from_colore != None:
        rand = FieldData(args.from_colore, "Randoms", file_type="CoLoRe")
        rand.prepare_data(
            args.zmin, args.zmax, args.downsampling, args.pixel_mask, args.nside
        )

    rand.generate_random_positions(pixel_mask=args.pixel_mask, nside=args.nside)

    if ".fits" not in args.out_cat.name:
        rand.store_data_in_cat(args.out_cat / ".fits")
    else:
        rand.store_data_in_cat(args.out_cat)

    return


if __name__ == "__main__":
    main()
