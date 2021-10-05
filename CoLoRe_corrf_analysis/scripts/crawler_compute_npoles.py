import os
import argparse
from pathlib import Path
from CoLoRe_corrf_analysis.cf_helper import CFComputations

def main():
    parser = argparse.ArgumentParser(description='Crawler to compute npoles')

    parser.add_argument('--path', type=str, default='.', help='Path to search on')
    parser.add_argument('--npoles', nargs='+', type=int, help='Npoles to compute'), 
    args = parser.parse_args()

    for subdir, dirs, files in os.walk(args.path):
        if 'DD.dat' in files or '0_DD.dat' in files:
            print('Computing for dir', subdir)
            CFComp = CFComputations(Path(subdir),  N_data_rand_ratio=1)
            for pole in args.npoles:
                print(f'\tComputing npole {pole}')
                _ = CFComp.compute_npole(pole)

if __name__ == '__main__':
    main()