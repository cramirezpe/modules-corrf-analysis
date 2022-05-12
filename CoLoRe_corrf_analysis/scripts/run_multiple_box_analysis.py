import argparse
from available_sims import search_corrs
import logging

logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getlogger("mcfit").setLevel(logging.CRITICAL)
from module_files_plots import *
import sys
import logging
import ast


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check-boxes",
        action="store_true",
        help="Show computed boxes",
    )

    parser.add_argument(
        "--path-to-boxes",
        type=Path,
        default=Path(
            "/global/project/projectdirs/desi/users/cramirez/CoLoRe_analysis/NbodyKit/multibias2_0.5_4//hanyu_david_box/binned_analysis_with_error"
        ),
        help="Path to corrfunc results (default: current boxes)",
    )

    parser.add_argument(
        "--path-to-theory-box",
        type=Path,
        default=Path("/global/cscratch1/sd/damonge/CoLoRe_sims/sim1000"),
        help="Path to CoLoRe output to read theory. (default: David sim 1000)",
    )

    parser.add_argument(
        "--boxes",
        nargs="+",
        type=int,
        default=None,
        help="Boxes to use 1000, 1001... (default: use all available)",
    )

    parser.add_argument(
        "--source",
        type=int,
        default=2,
        help="CoLoRe source number (for the used objects) (default: 2)",
    )

    parser.add_argument(
        "--bias-filename",
        type=Path,
        default=Path(
            "/global/u2/c/cramirez/Codes/CoLoRe/CoLoRe_LyA_v3/examples/LSST/BzBlue.txt"
        ),
    )

    parser.add_argument(
        "--nz-filename",
        type=Path,
        default=Path(
            "/global/u2/c/cramirez/Codes/CoLoRe/CoLoRe_LyA_v3/examples/LSST/NzBlue.txt"
        ),
    )

    parser.add_argument(
        "--pk-filename",
        type=Path,
        default=Path(
            "/global/u2/c/cramirez/Codes/CoLoRe/CoLoRe_LyA_v3/examples/simple/Pk_CAMB_test.dat"
        ),
    )

    parser.add_argument(
        "--rmin",
        type=float,
        default=0.1,
        help="Minimum separation in the computation of correlations (default: 0.1)",
    )

    parser.add_argument(
        "--rmax",
        type=float,
        default=200,
        help="Maximum separation in the computation of correlations (default: 200)",
    )

    parser.add_argument(
        "--zmin",
        type=float,
        default=0.7,
        help="Minimum redshift when masking the catalog (default: 0.7)",
    )

    parser.add_argument(
        "--zmax",
        type=float,
        default=0.9,
        help="Maximum redshift when masking the catalog (default: 0.9)",
    )

    parser.add_argument(
        "--rsd",
        action="store_true",
        help="Use rsd catalogues (if available)",
    )

    parser.add_argument(
        "--nside",
        type=int,
        default=2,
        help="nside of the pixelization (used to compute errorbars through dispersion of measurements) (default: 2)",
    )

    parser.add_argument(
        "--pixels",
        type=int,
        nargs="+",
        default=None,
        help="Healpix pixels to select (default: use all available)",
    )

    parser.add_argument(
        "--multipoles",
        nargs="+",
        type=int,
        default=[0],
        help="Multipoles to plot (default: [0]",
    )

    parser.add_argument(
        "--zeff",
        type=float,
        default=None,
        help="zeff to use for theory (defaults to computing it from catalog, ¡Slowest part of the code!, try to run it only once)",
    )

    parser.add_argument(
        "--plot-titles",
        type=str,
        nargs="+",
        default=None,
        help="Titles for plots (one plot per multipole)",
    )

    parser.add_argument(
        "--plot-data-args",
        type=ast.literal_eval,
        default=dict(),
        help="Arguments for data plot (using literal eval)",
    )

    parser.add_argument(
        "--plot-theory-args",
        type=ast.literal_eval,
        default=dict(label="Lognormal theory + smoothing"),
        help="Arguments for theory plot (using literal eval)",
    )

    parser.add_argument(
        "--pk-smooth",
        type=float,
        default=6,
        help="Smooth factor to use for input_pk theory method",
    )

    parser.add_argument(
        "--pk-skip-lognormal",
        action="store_true",
        help="Do not apply lognormal transformation after getting Xi from Pk",
    )

    parser.add_argument(
        "--apply-lognormal-multipole",
        action="store_true",
        help="Apply lognormal transformation to the final correlation obtained (after computing multipole)",
    )

    parser.add_argument(
        "--log-level",
        default=None,
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"],
    )

    args = parser.parse_args()

    if args.log_level != None:
        level = logging.getLevelName(args.log_level)
        logging.basicConfig(
            stream=sys.stdout,
            level=level,
            format="%(levelname)s:%(name)s:%(funcName)s:%(message)s",
        )

    if args.check_boxes:
        logger.info("Checking boxes")
        print("Failed paths:")
        sort_keys = ["nside", "rsd", "rmax", "zmin"]
        table = search_corrs(args.path_to_boxes, sort_keys=sort_keys)
        print("Succesful paths:")
        print(table)
        return

    fig, ax = plt.subplots()
    args.ax = ax

    do_plotting(args)


def do_plotting(args):
    bias_filename = args.bias_filename if args.bias_filename != "None" else None
    nz_filename = args.nz_filename if args.nz_filename != "None" else None
    pk_filename = args.pk_filename if args.pk_filename != "None" else None

    logger.info("Defining theory class")
    logger.info("Using input pk method")
    theory = ReadXiCoLoReFromPk(
        args.path_to_theory_box,
        source=args.source,
        nz_filename=nz_filename,
        bias_filename=bias_filename,
        pk_filename=pk_filename,
        smooth_factor=args.pk_smooth,
        apply_lognormal=not args.pk_skip_lognormal,
    )

    logger.info("Collecting corrfunc results")
    boxes = FileFuncs.mix_sims(
        FileFuncs.get_full_path(
            args.path_to_boxes,
            args.rsd,
            args.rmin,
            args.rmax,
            args.zmin,
            args.zmax,
            args.nside,
        ),
        boxes=args.boxes,
        pixels=args.pixels,
    )
    logger.info(f"Number of pixels available: {len(boxes)}")

    logger.info("Setting zeff")
    if args.zeff is None:
        zeff = theory.get_zeff(args.zmin, args.zmax)
    else:
        zeff = args.zeff
    logger.info(f"zeff:\t{zeff}")

    logger.info("Starting plot section")

    # Set titles for the different plots
    if args.plot_titles is None:
        titles = [f"Correlation npole {i}" for i in args.multipoles]
    elif len(args.plot_titles) != len(args.multipoles):
        logger.warning("number of titles != number of multipoles. Using default titles")
        titles = [f"Correlation npole {i}" for i in args.multipoles]
    else:
        titles = args.plot_titles

    data_args = {
        **dict(label=r"$z \in ({},{})$".format(args.zmin, args.zmax)),
        **args.plot_data_args,
    }
    for pole, title in zip(args.multipoles, titles):
        if args.ax is None:
            fig, ax = plt.subplots()
        Plots.plot_theory(
            pole,
            z=round(zeff, 1),
            theory=theory,
            ax=ax,
            plot_args=args.plot_theory_args,
            rsd=args.rsd,
            apply_lognormal=args.apply_lognormal_multipole,
        )
        Plots.plot_data(pole, boxes=boxes, ax=ax, plot_args=data_args, rsd=args.rsd)
        ax.set_title(title)
        ax.legend()
        plt.show()


if __name__ == "__main__":
    main()
