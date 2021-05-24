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
        
        try:
            self.DD = np.loadtxt(results_path / 'DD.dat', dtype=self.dtypes)
            self.DR = np.loadtxt(results_path / 'DR.dat', dtype=self.dtypes)
            self.RR = np.loadtxt(results_path / 'RR.dat', dtype=self.dtypes)
        except OSError:
            self.DD = np.loadtxt(results_path / '0_DD.dat', dtype=self.dtypes)
            self.DR = np.loadtxt(results_path / '0_DR.dat', dtype=self.dtypes)
            self.RR = np.loadtxt(results_path / '0_RR.dat', dtype=self.dtypes)
        
        self.savg = np.unique(( self.DD['smin'] + self.DD['smax'] ))/2     
        
        self.mubins = np.concatenate([[0], np.unique(self.DD['mu_max'])])
        self.mumean = (self.mubins[1:] + self.mubins[:-1])/2
        
    def __str__(self):
        return self.label
        
    @cached_property
    def cf(self):
        self.cf = convert_3d_counts_to_cf(self.N_data, self.N_data, self.N_rand, self.N_rand, 
                                          self.DD, self.DR, self.DR, self.RR)
        return self.cf
        
    @cached_property
    def monopole(self):
        savg = ( self.DD['smin'] + self.DD['smax'] )/2
        cf_monopole = []
        for s in np.unique(savg):
            cf_monopole.append( self.cf[savg == s].sum() )
        
        self.cf_monopole = cf_monopole
        return self.cf_monopole
    
    @cached_property
    def halotools_like_cf(self):
        cf_array = []
        for smin in np.unique(self.DD['smin']):
            cf_array.append( self.cf[self.DD['smin'] == smin])
        return cf_array

    def compute_npole(self, n):
        return tpcf_multipole(self.halotools_like_cf, self.mubins, order=n)
