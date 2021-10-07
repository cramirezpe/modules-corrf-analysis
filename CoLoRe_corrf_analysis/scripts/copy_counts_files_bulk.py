from pathlib import Path
import argparse
from CoLoRe_corrf_analysis.file_funcs import FileFuncs

def main():
    parser = argparse.ArgumentParser(description='Copy counts files from one analysis folder to another')
    parser.add_argument('--in-analysis', type=Path)
    parser.add_argument('--out-analysis', type=Path)

    parser.add_argument('--counts-to-copy', type=str, nargs='+', choices=['DD', 'DR', 'RD', 'RR'], help='Counts to copy.')
    parser.add_argument('--counts-to-copy-as', required=False, type=str, nargs='+', choices=['DD', 'DR', 'RD', 'RR'],
    help='Se the output copy as a different count type. This is needed to translate from DR to RD.')
    
    parser.add_argument('--nside', type=int)
    parser.add_argument('--in-out-rsd', type=str, nargs=2)
    parser.add_argument('--in-out-rsd2', type=str, default=[None, None], nargs=2)

    parser.add_argument('--in-out-zmin', type=float, nargs=2, default=[0.5, 0.5])
    parser.add_argument('--in-out-zmax', type=float, nargs=2, default=[0.7, 0.7])

    parser.add_argument('--in-out-rmin', type=float, nargs=2, default=[0.1, 0.1])
    parser.add_argument('--in-out-rmax', type=float, nargs=2, default=[200,200])
    parser.add_argument('--in-out-Nbins', type=int, nargs=2, default=[41, 41])

    parser.add_argument('--in-boxes', type=int, nargs='+', help='Input boxes to use')
    parser.add_argument('--out-boxes', type=int, nargs='+', help='Output boxes to copy files on')

    args = parser.parse_args()

    if len(args.in_boxes) != len(args.out_boxes):
        raise ValueError('In and out boxes should have the same length')

    if args.counts_to_copy_as == None:
        args.counts_to_copy_as = args.counts_to_copy
    elif len(args.counts_to_copy_as) != len(args.counts_to_copy):
        raise ValueError('counts-to-copy and counts-to-copy-as should have the same size.')

    in_rsd = str2bool(args.in_out_rsd[0])
    out_rsd = str2bool(args.in_out_rsd[1])
    in_rsd2 = str2bool(args.in_out_rsd2[0])
    out_rsd2 = str2bool(args.in_out_rsd2[1])

    copy_counts_from_analysis(in_analysis=args.in_analysis, 
        out_analysis=args.out_analysis, 
        counts_to_copy=args.counts_to_copy, counts_to_copy_as=args.counts_to_copy_as, nside=args.nside, 
        in_rsd=in_rsd, in_rsd2=in_rsd2, in_zmin=args.in_out_zmin[0], in_zmax=args.in_out_zmax[0], in_rmin=args.in_out_rmin[0], in_rmax=args.in_out_rmax[0], in_Nbins=args.in_out_Nbins[0],
        out_rsd=out_rsd, out_rsd2=out_rsd2, out_zmin=args.in_out_zmin[1], out_zmax=args.in_out_zmax[1], out_rmin=args.in_out_rmin[1], out_rmax=args.in_out_rmax[1], out_Nbins=args.in_out_Nbins[1], 
        in_boxes=args.in_boxes, out_boxes=args.out_boxes
    )

def copy_counts_from_analysis(in_analysis, out_analysis, counts_to_copy, counts_to_copy_as, nside, in_rsd, in_rsd2, in_zmin, in_zmax, in_rmin, in_rmax, in_Nbins, out_rsd, out_rsd2, out_zmin, out_zmax, out_rmin, out_rmax, out_Nbins, in_boxes, out_boxes):
    full_in_path = FileFuncs.get_full_path(in_analysis, in_rsd, in_rmin, in_rmax, in_zmin, in_zmax, nside, in_Nbins, in_rsd2)
    if not full_in_path.is_dir():
        raise FileNotFoundError('Input analysis does not exist', full_in_path.resolve())
    
    full_out_path = FileFuncs.get_full_path(out_analysis, out_rsd, out_rmin, out_rmax, out_zmin, out_zmax, nside, out_Nbins, out_rsd2)
    full_out_path.mkdir(parents=True, exist_ok=True)

    in_boxes = [full_in_path / str(in_box) for in_box in in_boxes]
    out_boxes = [full_out_path / str(out_box) for out_box in out_boxes]

    for in_box, out_box in zip(in_boxes, out_boxes):
        for pixel in in_box.iterdir():
            (full_out_path / out_box.name / pixel.name).mkdir(parents=True, exist_ok=True)
            for in_counts, out_counts in zip(counts_to_copy, counts_to_copy_as):
                FileFuncs.copy_counts_file(in_path=pixel,
                    out_path=full_out_path / out_box.name / pixel.name, 
                    in_counts=in_counts,
                    out_counts=out_counts
                )

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v == None:
        return None
    if v.lower() in ('none'):
        return None
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean or None value expected.')

if __name__ == '__main__':
    main()