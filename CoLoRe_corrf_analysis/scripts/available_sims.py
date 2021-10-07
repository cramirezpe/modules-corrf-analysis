from pathlib import Path
import argparse
from CoLoRe_corrf_analysis.file_funcs import FileFuncs

def main():
    parser = argparse.ArgumentParser(description='Search for available correlations')

    parser.add_argument('--path', type=Path, default='.', help='Path to search on')
    parser.add_argument('--sort', type=str, nargs='+', default=None, help='Sorting by then by.. options: nside, rsd, rmin, rmax, zmin, zmax, N')
    parser.add_argument('--reverse', type=str, nargs='+', default=None, help='Reverse previous sorting. All of them if True, each one if list of bools')
    parser.add_argument('--show-incompleted', action='store_true', help='Print path to sub-boxes that are not completed.')
    args = parser.parse_args()

    if args.sort is not None:
        for x in args.sort:
            assert x in ('nside', 'rsd', 'rmin', 'rmax', 'N_bins', 'zmin', 'zmax', 'N')

        if args.reverse is not None:
            reverse = []
            for rev in args.reverse:
                reverse.append(str2bool(rev))

            if len(reverse) == 1 and len(args.sort) != 1:
                value = reverse[0][:]
                reverse = [value for i in args.sort]
            else:
                if len(reverse) != len(args.sort):
                    raise ValueError('Length of sort list different from reverse list')
        else:
            reverse = args.reverse
    else:
        reverse = None

    table = FileFuncs.get_available_runs(args.path, args.sort, reverse, args.show_incompleted)
    print(table)

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':
    main()