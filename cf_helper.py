import numpy as np
from pathlib import Path

from functools import cached_property
from Corrfunc.utils import convert_3d_counts_to_cf
from halotools.mock_observables import tpcf_multipole

class CFComputations:
    dtypes = {
    'names': ('smin', 'smax', 'savg', 'mu_max', 'npairs', 'weightavg'),
    'formats': ('<f8', '<f8', '<f8', '<f8', '<f8', '<f8')
    }

    def __init__(self, results_path, N_data_rand_ratio, label=''):
        self.results_path = results_path
        self.N_data = 1
        self.N_rand = 1/N_data_rand_ratio
        self.label = label
        
    def __str__(self): # pragma: no cover
        return self.label

    @property
    def DD(self):
        try:
            return np.loadtxt(self.results_path / 'DD.dat', dtype=self.dtypes)
        except OSError:
            return np.loadtxt(self.results_path / '0_DD.dat', dtype=self.dtypes)

    @property
    def DR(self):
        try:
            return np.loadtxt(self.results_path / 'DR.dat', dtype=self.dtypes)
        except OSError:
            return np.loadtxt(self.results_path / '0_DR.dat', dtype=self.dtypes)

    @property
    def RR(self):
        try:
            return np.loadtxt(self.results_path / 'RR.dat', dtype=self.dtypes)
        except OSError:
            return np.loadtxt(self.results_path / '0_RR.dat', dtype=self.dtypes)

    @property
    def savg(self):
        try:
            return np.loadtxt(self.results_path / 'savg.dat')
        except OSError:
            return np.unique(( self.DD['smin'] + self.DD['smax'] ))/2  
 
    @property
    def mubins(self):
        try:
            return np.loadtxt(self.results_path / 'mubins.dat')
        except OSError:
            return np.concatenate([[0], np.unique(self.DD['mu_max'])])
        
    @cached_property
    def cf(self):
        self.cf = convert_3d_counts_to_cf(self.N_data, self.N_data, self.N_rand, self.N_rand, 
                                          self.DD, self.DR, self.DR, self.RR)
        return self.cf
        
    @cached_property
    def halotools_like_cf(self):
        cf_array = []
        for smin in np.unique(self.DD['smin']):
            cf_array.append( self.cf[self.DD['smin'] == smin])
        return cf_array

    def compute_npole(self, n):
        try:
            npole = np.loadtxt(self.results_path / f'npole_{n}.dat')
        except OSError:
            npole = tpcf_multipole(self.halotools_like_cf, self.mubins, order=n)
            np.savetxt(self.results_path / f'npole_{n}.dat', npole)
        
        return npole

def main(): # pragma: no cover
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument('--search-path', 
        type=Path,
        help='Path to recursively search for boxes'
    )

    parser.add_argument('--multipoles',
        nargs='+',
        type=int,
        help='multipoles to process'
    )

    args = parser.parse_args()
    for root, dirs, files in os.walk(args.search_path.resolve()):
        if ('DD.dat' in files) or ('0_DD.dat' in files):
            print(root, 'has a box')
            cfccomp = CFComputations(Path(root), 1)
            for pole in args.multipoles:
                print('computing pole', pole)
                cfccomp.compute_npole(pole)

if __name__ == '__main__': # pragma: no cover
    main()