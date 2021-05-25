from pathlib import Path
import argparse
from tabulate import tabulate

def main():
    parser = argparse.ArgumentParser(description='Search for available correlations')

    parser.add_argument('--path', type=Path, default='.', help='Path to search on')
    parser.add_argument('--sort', type=str, nargs='+', default=None, help='Sorting by then by.. options: nside, rsd, rmin, rmax, zmin, zmax, N')
    parser.add_argument('--reverse', type=str, nargs='+', default=None, help='Reverse previous sorting. All of them if True, each one if list of bools')
    args = parser.parse_args()

    if args.sort is not None:
        for x in args.sort:
            assert x in ('nside', 'rsd', 'rmin', 'rmax', 'zmin', 'zmax', 'N')

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

    table = search_corrs(args.path, args.sort, reverse)
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

def search_corrs(path, sort_keys=None, reverse=None):
    t_header = ["nside", "rsd", "rmin", "rmax", "zmin", "zmax", "N"]
    t_rows = []
    nsides_path = path.glob('nside*')
    for nside_path in nsides_path:
        nside = nside_path.name[6:]
        for rsd_path in nside_path.iterdir():
            rsd = True if rsd_path.name == 'rsd' else False
            for range_path in rsd_path.iterdir():
                rmin = range_path.name.split('_')[0]
                rmax = range_path.name.split('_')[1]
                for zbin_path in range_path.iterdir():
                    zmin = zbin_path.name.split('_')[0]
                    zmax = zbin_path.name.split('_')[1]
                    sum_ = 0
                    for box in zbin_path.iterdir():
                        for sim_path in box.iterdir():
                            if (sim_path / '0_DD.dat').is_file():
                                sum_ += 1
                            else:
                                print(sim_path.resolve())
                    # print(f'nside: {nside_path.name[6:]}\t{rsd_path.name}\t{range_path.name}\t\t{zbin_path.name}\t\t{sims}')
                    t_rows.append((nside, rsd, rmin, rmax, zmin, zmax, sum_))

    if sort_keys is not None:
        if reverse is None: 
            reverse = [False for i in sort_keys]
        assert len(reverse) == len(sort_keys)

        for key, rev in zip(reversed(sort_keys),reversed(reverse)):
            t_rows.sort(key=lambda x: float(x[t_header.index(key)]), reverse=rev)

    # if sort_keys is not None:
    #     indices = []
    #     for key in sort_keys:
    #         indices.append( t_header.index(key))

    #     t_rows.sort(key=lambda x: [x[i] for i in indices])
    # t_rows.sort(key=lambda x: x[4])
    return tabulate(t_rows, t_header, tablefmt='pretty')

if __name__ == '__main__':
    main()