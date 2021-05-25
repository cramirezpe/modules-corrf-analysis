import argparse
from available_sims import search_corrs
from module_files_plots import *
import sys
import logging
logger = logging.getLogger(__name__)

plot_title = 'Monopole'
plot_args_data = dict(label='data')
plot_args_theory = dict(label='theory')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check-boxes',
        action='store_true',
        help='Show computed boxes',
    )

    parser.add_argument('--path-to-boxes',
        type=Path,
        default=Path('/global/cscratch1/sd/cramirez/NBodyKit/hanyu_david_box/binned_analysis_with_error'),
        help='Path to corrfunc results (default: current boxes)',
    )

    parser.add_argument('--path-to-theory-box', 
        type=Path,
        default=Path('/global/cscratch1/sd/damonge/CoLoRe_sims/sim1000'),
        help='Path to CoLoRe output to read theory. (default: David sim 1000)'
    )

    parser.add_argument('--source',
        type=int,
        default=2,
        help='CoLoRe source number (for the used objects) (default: 2)',
    )

    parser.add_argument('--bias-filename',
        type=Path,
        default=Path('/global/u2/c/cramirez/Codes/CoLoRe/CoLoRe_LyA_v3/examples/LSST/BzBlue.txt'),
    )

    parser.add_argument('--nz-filename',
        type=Path,
        default=Path('/global/u2/c/cramirez/Codes/CoLoRe/CoLoRe_LyA_v3/examples/LSST/NzBlue.txt'),
    )

    # parser.add_argument('--pk-filename',
    #     type=Path,
    #     default=Path('/global/u2/c/cramirez/Codes/CoLoRe/CoLoRe_LyA_v3/examples/simple/Pk_CAMB_test.dat'),
    # )

    parser.add_argument('--rmin',
        type=float,
        default=0.1,
        help='Minimum separation in the computation of correlations (default: 0.1)',
    )

    parser.add_argument('--rmax',
        type=float,
        default=200,
        help='Maximum separation in the computation of correlations (default: 200)',
    )

    parser.add_argument('--zmin',
        type=float,
        default=0.7,
        help='Minimum redshift when masking the catalog (default: 0.7)',
    )

    parser.add_argument('--zmax',
        type=float,
        default=0.9,
        help='Maximum redshift when masking the catalog (default: 0.9)',
    )

    parser.add_argument('--rsd',
        action='store_true',
        help='Use rsd catalogues (if available)',
    )

    parser.add_argument('--nside',
        type=int,
        default=2,
        help='nside of the pixelization (used to compute errorbars through dispersion of measurements) (default: 2)',
    )

    parser.add_argument('--multipoles',
        nargs= '+',
        type=int,
        default=[0],
        help='Multipoles to plot (default: [0]',
    )

    parser.add_argument('--zeff',
        type=float,
        default=None,
        help='zeff to use for theory (defaults to computing it from catalog, ¡Slowest part of the code!, try to run it only once)',
    )

    parser.add_argument('--log-level', default=None, choices=['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'NOTSET'])

    args = parser.parse_args()

    if args.log_level != None:
        level = logging.getLevelName(args.log_level)
        logging.basicConfig(stream=sys.stdout, level=level, format='%(levelname)s:%(name)s:%(funcName)s:%(message)s')
        logging.getLogger('matplotlib').setLevel(logging.WARNING)

    if args.check_boxes:
        logger.info('Checking boxes') 
        print('Failed paths:')
        sort_keys = ['nside', 'rsd', 'rmax', 'zmin']
        table = search_corrs(args.path_to_boxes, sort_keys=sort_keys)
        print('Succesful paths:')
        print(table)
        return

    do_plotting(args)

def do_plotting(args):
    bias_filename = args.bias_filename if args.bias_filename != "None" else None
    nz_filename = args.nz_filename if args.nz_filename != "None" else None
    # pk_filename = args.pk_filename if args.pk_filename != "None" else None

    logger.info('Defining theory class')
    theory = ReadXiCoLoRe(args.path_to_theory_box, 
        source=args.source,
        nz_filename=nz_filename, 
        tracer='dd',
        bias_filename=bias_filename
    )

    logger.info('Collecting corrfunc results')
    boxes = FileFuncs.mix_sims(
        FileFuncs.get_full_path(args.path_to_boxes,
            args.rsd, 
            args.rmin, args.rmax,
            args.zmin, args.zmax, 
            args.nside
        )
    )

    logger.info('Setting zeff')
    if args.zeff is None:
        zeff = theory.get_zeff(z=np.arange(args.zmin, args.zmax, 0.1))
    else:
        zeff = args.zeff
    logger.info(f'zeff:\t{zeff}')

    logger.info('Starting plot section')
    for pole in args.multipoles:
        fig, ax = plt.subplots()
        Plots.plot_theory(pole, z=round(zeff, 1), theory=theory, ax=ax, plot_args=plot_args_theory, rsd=args.rsd)
        Plots.plot_data(pole, boxes=boxes, ax=ax, plot_args=plot_args_data, rsd=args.rsd)
        ax.set_title(plot_title)
        ax.legend()
        plt.show()



if __name__ == '__main__':
    main()