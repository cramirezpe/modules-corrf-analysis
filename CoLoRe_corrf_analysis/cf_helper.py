import json
import logging
import numpy as np
from pathlib import Path
import warnings

from functools import cached_property
from Corrfunc.utils import convert_3d_counts_to_cf
from halotools.mock_observables import tpcf_multipole

logger=logging.getLogger(__name__)
class CFComputations:
    dtypes = {
    'names': ('smin', 'smax', 'savg', 'mu_max', 'npairs', 'weightavg'),
    'formats': ('<f8', '<f8', '<f8', '<f8', '<f8', '<f8')
    }

    def __init__(self, results_path, label=''):
        '''Class to handle results from corrfunc and compute multipoles.
        
        Args:
            results_path (Path): Path to the results from corrfunc.
            label (str, optional): Label the results object. (Default: '').
            '''
        self.results_path = results_path

        try:
            with open(self.results_path / 'sizes.json') as json_file:
                self.sizes = json.load(json_file)
        except OSError:
            warnings.warn('Trying to read sizes json file failed. This may happen when running analysis on previous corrfunc runs. Setting all datas/randoms to the same size.')
            self.sizes = dict(Data=1, Randoms=1)

        if 'Data2' not in self.sizes:
            self.sizes['Data2'] = self.sizes['Data']
        if 'Randoms2' not in self.sizes:
            self.sizes['Randoms2'] = self.sizes['Randoms']
            
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
    def RD(self):
        try:
            try:
                return np.loadtxt(self.results_path / 'RD.dat', dtype=self.dtypes)
            except OSError:
                return np.loadtxt(self.results_path / '0_RD.dat', dtype=self.dtypes)
        except OSError:  # pragma: no cover
            np.savetxt(self.results_path / 'RD.dat', self.DR)
            return self.RD

    @property
    def savg(self):
        ''' Read savg (separations in r for the data'''
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
        return convert_3d_counts_to_cf(self.sizes['Data'], self.sizes['Data2'], self.sizes['Randoms'], self.sizes['Randoms2'], 
                                          self.DD, self.DR, self.RD, self.RR)

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
            try:
                np.savetxt(self.results_path / f'npole_{n}.dat', npole)
            except PermissionError: # pragma: no cover
                logger.info('Unable to save npole to file due to permission error')
        
        return npole

    def remove_computed_npoles(self, poles=[0,2,4]):
        for pole in poles:
            file = self.results_path / f'npole_{pole}.dat'
            file.unlink(missing_ok=True)

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