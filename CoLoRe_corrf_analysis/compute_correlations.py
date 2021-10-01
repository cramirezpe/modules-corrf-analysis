#!/usr/bin/env python
'''
    Script to compute the correlation of input master.fits catalogues using master_randoms.fits as randoms
'''

import numpy as np
from astropy.io import fits
import healpy as hp
from Corrfunc.mocks.DDsmu_mocks import DDsmu_mocks
from scipy import interpolate
import scipy.integrate as integrate
import glob
import argparse
import logging
from pathlib import Path
import sys
import os

version='0.12'
logger = logging.getLogger(__name__)

def getArgs(): # pragma: no cover
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",       
        type=str,   
        required=True, 
        help='Input glob pattern to data files')

    parser.add_argument("--data-format",
        type=str,
        choices=['CoLoRe', 'zcat'])

    parser.add_argument("--data-norsd",
        action='store_true',
        help='If CoLoRe format selected: read noRSD redshifts for data')

    parser.add_argument("--data2",
        type=str,
        required=False,
        help='Input glob patterns to data2 files')

    parser.add_argument("--data2-format",
        type=str,
        choices=['CoLoRe', 'zcat'])

    parser.add_argument("--data2-norsd",
        action='store_true',
        help='If CoLoRe format selected: read noRSD redshifts for data2')
    
    parser.add_argument("--randoms",    
        type=str,   
        required=True, 
        help='Input glob pattern to randoms files')

    parser.add_argument("--randoms2",
        type=str,
        required=False,
        help='Input glob patterns to randoms2 files')

    parser.add_argument("--out-dir",
        type=str,
        required=True,
        help='Output dir')

    parser.add_argument("--nthreads",   
        type=int,   
        default=8)

    parser.add_argument('--mu-max',     
        type=float, 
        default=1.0)

    parser.add_argument('--nmu-bins',   
        type=int,
        default=100)   

    parser.add_argument('--min-bin',    
        type=float,
        default=0.1)

    parser.add_argument('--max-bin',    
        type=float,
        default=200)

    parser.add_argument('--n-bins',     
        type=int,
        default=41)

    parser.add_argument('--zmin',
        type=float,
        default=0)

    parser.add_argument('--zmax',
        type=float,
        default=10)

    parser.add_argument('--zmin-covd',
        type=float,
        default=1,
        help='Zmin for covd interpolation')
    
    parser.add_argument('--zmax-covd',
        type=float, 
        default=5,
        help='zmax for covd interpolation')
        
    parser.add_argument('--zstep-covd',
        type=float,
        default=0.01,
        help='Bins for covd interpolation')

    parser.add_argument('--random-downsampling',
        type=float,
        default=1,
        help='Random downsampling to apply before computing')

    parser.add_argument('--pixel-mask',
        nargs='+',
        type=int, 
        default=None,
        help='Pixels to include. Defaults to all sky')

    parser.add_argument('--nside',
        default=2,
        type=int,
        help='nside for the pixel mask')

    parser.add_argument('--log-level', default='WARNING', choices=['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'NOTSET'])

    args = parser.parse_args()
    return args

class FieldData:
    def __init__(self, path, label, file_type, rsd=False):
        self.path   = path
        self.label  = label
        self.file_type = file_type
        if file_type == 'zcat':
            self.rsd = False
        else:
            self.rsd = rsd

    def __str__(self):
        return self.label

    @property
    def zfield(self):
        if self.file_type == 'zcat':
            return 'Z'
        elif self.file_type == 'CoLoRe':
            return 'Z_COSMO'

    def get_cats(self):
        self.cat = sorted(glob.glob(self.path))

    def open_fits(self, imock):
        self.fits = fits.open(self.cat[imock])

    def define_data(self):
        _cat_length = 0
        for i in range(len(self.cat)):
            self.open_fits(i)
            _cat_length += len(self.fits[1].data)
        self.data = np.empty(_cat_length,dtype=[('RA','f8'),('DEC','f8'),('Z','f8'),('Weight','f8')])

    def fill_data(self):
        ''' Fill the data arrays from the input source.

        Args:
            zfield (str, optional): Field name for redshift. (Default: Z).
        '''
        _index = 0
        for i in range(len(self.cat)):
            self.open_fits(i)
            _file_size = len(self.fits[1].data)
            self.data['RA'][_index:_index+_file_size]  = self.fits[1].data['RA']
            self.data['DEC'][_index:_index+_file_size] = self.fits[1].data['DEC']
            self.data['Z'][_index:_index+_file_size]   = self.fits[1].data[self.zfield]
            if self.rsd:
                self.data['Z'][_index:_index+_file_size] += self.fits[1].data['DZ_RSD']
            _index += _file_size

    def apply_downsampling(self, downsampling): # pragma: no cover
        _mask = np.random.choice(a=[True, False], size=len(self.data), p=[downsampling, 1-downsampling])
        self.data = self.data[_mask]

    def apply_pixel_mask(self, pixel_mask, nside=2):
        _pixels = hp.ang2pix(nside, self.data['RA'], self.data['DEC'], lonlat=True)
        _mask   = np.in1d(_pixels, pixel_mask)
        self.data = self.data[_mask]

    def apply_redshift_mask(self, zmin, zmax):
        _mask = self.data['Z'] > zmin
        _mask &= self.data['Z'] < zmax
        self.data = self.data[_mask]

    def compute_cov(self, interpolator):
        self.cov = interpolator(self.data['Z'])

def main(args=None):
    if args is None: # pragma: no cover
        args = getArgs()

    print('version', version)
    level = logging.getLevelName(args.log_level)
    logging.basicConfig(stream=sys.stdout, level=level, format='%(levelname)s:%(name)s:%(funcName)s:%(message)s')
    
    if logging.root.level <= logging.DEBUG: # pragma: no cover
        import time
        time_start = time.time()

    logger.debug("\n".join([f'{key}:\t{args.__dict__[key]}' for key in args.__dict__.keys()]))

    if (args.randoms2!=None) and (args.data2==None): # pragma: no cover
        raise ValueError('If two randoms are provided, two datasets are required.')

    data = FieldData(args.data, 'Data', file_type=args.data_format, rsd=not(args.data_norsd))
    rand = FieldData(args.randoms, 'Randoms', file_type='zcat')
    field_objects = [data, rand]

    if args.data2 != None:
        data2 = FieldData(args.data2, 'Data2', file_type=args.data2_format, rsd=not(args.data2_norsd))
        field_objects.append(data2)
        if args.randoms2 != None:
            rand2 = FieldData(args.randoms2, 'Randoms2', file_type='zcat')
            field_objects.append(rand2)

    for obj in field_objects:
        obj.get_cats()
        logger.debug('{} files:{}'.format(obj.label, "\n\t".join(obj.cat)))

    z=np.arange(args.zmin_covd,args.zmax_covd,args.zstep_covd)
    covd=[]
    for i in range(len(z)):
        covd.append(cov(z[i]))
    covd=np.array(covd)
    f = interpolate.interp1d(z, covd)

    Path(args.out_dir).mkdir(exist_ok=True)

    info_file = Path(args.out_dir + '/README')
    text=''
    for i in range(len(data.cat)):
        text+="\n".join('{}\n\t{}'.format(i, "\n\t".join([obj.cat[i] for obj in field_objects])))
    info_file.write_text(text)

    logger.info('Reading files:\n\t{}'.format("\n\t".join([obj.cat[i] for obj in field_objects])))
    for obj in field_objects:
        obj.define_data()
        obj.fill_data()
        logger.debug(f'Length of {obj.label} cat: {len(obj.data)}')

    if args.random_downsampling != 1: # pragma: no cover
        logger.debug(f'Applying downsampling: {args.random_downsampling}')
        for obj in field_objects:
            obj.apply_downsampling(args.random_downsampling)
            logger.debug(f'Length of {obj.label} after downsampling: {len(obj.data)}')

    for obj in field_objects:
        obj.apply_redshift_mask(args.zmin, args.zmax)
        logger.debug(f'Length of {obj.label} after redshift mask: {len(obj.data)}')

    if args.pixel_mask is not None:
        logger.debug(f'Applying pixel mask')
        for obj in field_objects:
            obj.apply_pixel_mask(args.pixel_mask, args.nside)
            logger.debug(f'Length of {obj.label} after pixel mask: {len(obj.data)}')

    for obj in field_objects:
        obj.compute_cov(f)

    bins2p=np.linspace(args.min_bin, args.max_bin, args.n_bins)

    logger.info('Starting computation for current file:')
    if logging.root.level <= logging.DEBUG: # pragma: no cover
        start_computation = time.time()

    if args.data2 == None:
        data2 = data
    if args.randoms2 == None:
        rand2 = rand


    logger.info('Computing DD...')
    DD = DDsmu_mocks(autocorr=data==data2, 
        cosmology=2, nthreads=args.nthreads, mu_max=args.mu_max, nmu_bins=args.nmu_bins, binfile=bins2p, RA1=data.data['RA'], DEC1=data.data['DEC'], CZ1=data.cov, 
        RA2=data2.data['RA'], DEC2=data2.data['DEC'], CZ2=data2.cov, 
        is_comoving_dist=True, verbose=True)

    logger.info('Computing DR...')
    DR = DDsmu_mocks(autocorr=0, 
        cosmology=2, nthreads=args.nthreads, mu_max=args.mu_max, nmu_bins=args.nmu_bins, binfile=bins2p, RA1=data.data['RA'], DEC1=data.data['DEC'], CZ1=data.cov, 
        RA2=rand2.data['RA'],DEC2=rand2.data['DEC'], CZ2=rand2.cov, 
        is_comoving_dist=True, verbose=True)

    logger.info('Computing RR...')
    RR = DDsmu_mocks(autocorr=rand==rand2, 
        cosmology=2, nthreads=args.nthreads, mu_max=args.mu_max, nmu_bins=args.nmu_bins, binfile=bins2p, RA1=rand.data['RA'], DEC1=rand.data['DEC'], CZ1=rand.cov,
        RA2=rand2.data['RA'], DEC2=rand2.data['DEC'], CZ2=rand2.cov,
        is_comoving_dist=True, verbose=True)

    if data2!=data:
        logger.info('Computing RD...')
        RD = DDsmu_mocks(autocorr=0, 
            cosmology=2, nthreads=args.nthreads, mu_max=args.mu_max, nmu_bins=args.nmu_bins, binfile=bins2p,
            RA1=rand.data['RA'], DEC1=rand.data['DEC'], CZ1=rand.cov,
            RA2=data2.data['RA'], DEC2=data2.data['DEC'], CZ2=data2.cov,
            is_comoving_dist=True, verbose=True)
        np.savetxt(args.out_dir + f'/RD.dat', RD)
    np.savetxt(args.out_dir + f'/DD.dat', DD)
    np.savetxt(args.out_dir + f'/DR.dat', DR)
    np.savetxt(args.out_dir + f'/RR.dat', RR)
    
    if logging.root.level <= logging.DEBUG: # pragma: no cover
        logger.debug(f'Relative ellapsed time: {time.time() - start_computation}')

def hhz(z):
    om=0.3147
    return 1/np.sqrt(om*(1+z)**3+(1-om))

def cov(z):
    return 2998*integrate.quad(hhz,0,z)[0]
   

if __name__ == '__main__': # pragma: no cover
    main()