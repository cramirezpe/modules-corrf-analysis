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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",       
        type=str,   
        required=True, 
        help='Input glob pattern to data files')

    parser.add_argument("--data2",
        type=str,
        required=False,
        help='Input glob patterns to data2 files')
    
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

    parser.add_argument('--norsd',
        action='store_true')

    parser.add_argument('--zmin',
        type=float,
        default=0)

    parser.add_argument('--zmax',
        type=float,
        default=10)

    parser.add_argument('--drq-format',
        action='store_true',
        help='Read input data as drq catalog')

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

    print('verson', version)
    args = parser.parse_args()
    level = logging.getLevelName(args.log_level)
    logging.basicConfig(stream=sys.stdout, level=level, format='%(levelname)s:%(name)s:%(funcName)s:%(message)s')
    
    if logging.root.level <= logging.DEBUG:
        import time
        time_start = time.time()

    logger.debug("\n".join([f'{key}:\t{args.__dict__[key]}' for key in args.__dict__.keys()]))

    if (args.randoms2!=None) and (args.data2==None):
        raise ValueError('If two randoms are provided, two datasets are required.')

    datacat = sorted(glob.glob(args.data))
    if args.data2 != None:
        datacat2 = sorted(glob.glob(args.data2))
        if args.randoms2 != None:
            randcat2 = sorted(glob.glob(args.randoms2))
    randcat = sorted(glob.glob(args.randoms))
    

    logger.debug('Data files:{}'.format("\n\t".join(datacat)))
    if args.data2 != None:
        logger.debug('Data2 files:{}'.format("\n\t".join(datacat2)))
        if args.randoms2 != None:
            logger.debug('Randoms2 files:{}'.format("\n\t".join(randcat2)))
    logger.debug('Random files:{}'.format("\n\t".join(randcat)))

    z=np.arange(args.zmin_covd,args.zmax_covd,args.zstep_covd)
    covd=[]
    for i in range(len(z)):
        covd.append(cov(z[i]))
    covd=np.array(covd)
    f = interpolate.interp1d(z, covd)

    Path(args.out_dir).mkdir(exist_ok=True)

    info_file = Path(args.out_dir + '/README')
    if args.data2 != None:
        if args.randoms2 != None:
            info_file.write_text("\n".join(f"{i}\n\t{datacat[i]}\n\t{randcat[i]}\n\t{datacat2[i]}\n\t{randcat2[i]}\n" for i in range(len(datacat))))
        else:
            info_file.write_text("\n".join(f"{i}\n\t{datacat[i]}\n\t{randcat[i]}\n\t{datacat2[i]}\n" for i in range(len(datacat))))
    else:
        info_file.write_text(
        "\n".join(f"{i}\n\t{datacat[i]}\n\t{randcat[i]}\n" for i in range(len(datacat)))
        )

    for imock in range(len(datacat)):
        if args.data2 != None:
            if args.randoms2 != None:
                logger.info(f'Reading file:\n\t{datacat[imock]}\n\t{datacat2[imock]}\n\t{randcat[imock]}\n\t{randcat2[imock]}') 
            else:
                logger.info(f'Reading file:\n\t{datacat[imock]}\n\t{datacat2[imock]}\n\t{randcat[imock]}') 
        else:
            logger.info(f'Reading file:\n\t{datacat[imock]}\n\t{randcat[imock]}') 
        real_fits=fits.open(datacat[imock])
        rand_fits=fits.open(randcat[imock])

        data = np.empty(len(real_fits[1].data),dtype=[('RA','f8'),('DEC','f8'),('Z','f8'),('Weight','f8')])
        rand = np.empty(len(rand_fits[1].data),dtype=[('RA','f8'),('DEC','f8'),('Z','f8'),('Weight','f8')])

        if args.drq_format:
            zfield = 'Z'
        elif args.norsd:
            zfield = 'Z_QSO_NO_RSD'
        else:
            zfield = 'Z_QSO_RSD'

        data['RA']  = real_fits[1].data['RA']
        data['DEC'] = real_fits[1].data['DEC']
        data['Z']   = real_fits[1].data[zfield]
        rand['RA']  = rand_fits[1].data['RA']
        rand['DEC'] = rand_fits[1].data['DEC']
        rand['Z']   = rand_fits[1].data[zfield]

        logger.debug(f'Length of data cat: {len(data)}')
        logger.debug(f'Length of rand cat: {len(rand)}')

        if args.data2 != None:
            logger.info('Reading second data cat')
            real2_fits=fits.open(datacat2[imock])
            data2=np.empty(len(real2_fits[1].data),dtype=[('RA','f8'),('DEC','f8'),('Z','f8'),('Weight','f8')])
            data2['RA']     = real2_fits[1].data['RA']
            data2['DEC']    = real2_fits[1].data['DEC']
            data2['Z']      = real2_fits[1].data[zfield]
            logger.debug(f'Length of data2 cat: {len(data2)}')

        if args.random_downsampling != 1:
            logger.debug(f'Applying downsampling: {args.random_downsampling}')
            downsampling = args.random_downsampling
            data_mask = np.random.choice(a=[True, False], size=len(data), p=[downsampling, 1-downsampling])
            rand_mask = np.random.choice(a=[True, False], size=len(rand), p=[downsampling, 1-downsampling])

            data = data[data_mask]
            rand = rand[rand_mask]

            logger.debug(f'Length of data after downsampling: {len(data)}')
            logger.debug(f'Length of rand after downsampling: {len(rand)}')

            if args.data2 != None:
                data2_mask = np.random.choice(a=[True, False], size=len(data2), p=[downsampling, 1-downsampling])
                data2 = data2[data2_mask]
                logger.debug(f'Length of data2 after downsampling: {len(data2)}')
    
        if args.pixel_mask is not None:
            logger.debug(f'Applying pixel mask')
            data_pixs = hp.ang2pix(args.nside, data['RA'], data['DEC'], lonlat=True)
            data_mask = np.in1d(data_pixs, args.pixel_mask)

            rand_pixs = hp.ang2pix(args.nside, rand['RA'], rand['DEC'], lonlat=True)
            rand_mask = np.in1d(rand_pixs, args.pixel_mask)

            data = data[data_mask]
            rand = rand[rand_mask]

            logger.debug(f'Length of data after pixel mask: {len(data)}')
            logger.debug(f'Length of rand after pixel mask: {len(rand)}')

            if args.data2 != None:
                data2_pixs = hp.ang2pix(args.nside, data2['RA'], data2['DEC'], lonlat=True)
                data2_mask = np.in1d(data2_pixs, args.pixel_mask)

                data2 = data2[data2_mask]
                logger.debug(f'Length of data2 after pixel mask: {len(data2)}')
            
        data_mask =  data['Z'] < args.zmax
        data_mask &= data['Z'] > args.zmin

        rand_mask =  rand['Z'] < args.zmax
        rand_mask &= rand['Z'] > args.zmin

        data = data[data_mask]
        rand = rand[rand_mask]

        logger.debug(f'Length of data after z mask: {len(data)}')
        logger.debug(f'Length of rand after z mask: {len(rand)}')

        if args.data2 != None:
            data2_mask = data2['Z']  < args.zmax
            data2_mask &= data2['Z'] > args.zmin
            data2=data2[data2_mask]
            logger.debug(f'Length of data2 after z mask: {len(data2)}')

        cov_real=f(data['Z'])
        cov_rand=f(rand['Z'])
        if args.data2 != None:
            cov_real2=f(data2['Z'])

        bins2p=np.linspace(args.min_bin, args.max_bin, args.n_bins)

        logger.info('Starting computation for current file:')
        if logging.root.level <= logging.DEBUG:
            start_computation = time.time()

        if args.data2 == None:
            data2 = data
            cov_real2 = cov_real
        logger.info('Computing DD...')
        if args.data2 != None:
            autocorr=0
        else:
            autocorr=1
        DD = DDsmu_mocks(autocorr=autocorr,cosmology=2,nthreads=args.nthreads,mu_max=args.mu_max,nmu_bins=args.nmu_bins,binfile=bins2p,RA1=data['RA'],DEC1=data['DEC'],CZ1=cov_real,RA2=data2['RA'], DEC2=data2['DEC'],CZ2=cov_real2, is_comoving_dist=True,verbose=True)
        logger.info('Computing DR...')
        DR = DDsmu_mocks(autocorr=0,cosmology=2,nthreads=args.nthreads,mu_max=args.mu_max,nmu_bins=args.nmu_bins,binfile=bins2p,RA1=data['RA'],DEC1=data['DEC'],CZ1=cov_real,RA2=rand['RA'],DEC2=rand['DEC'],CZ2=cov_rand,is_comoving_dist=True,verbose=True)
        logger.info('Computing RR...')
        RR = DDsmu_mocks(autocorr=1,cosmology=2,nthreads=args.nthreads,mu_max=args.mu_max,nmu_bins=args.nmu_bins,binfile=bins2p,RA1=rand['RA'],DEC1=rand['DEC'],CZ1=cov_rand,is_comoving_dist=True,verbose=True)
        if args.data2 != None:
            logger.info('Computing RD...')
            RD = DDsmu_mocks(autocorr=0,cosmology=2,nthreads=args.nthreads,mu_max=args.mu_max,nmu_bins=args.nmu_bins,binfile=bins2p,RA1=rand['RA'],DEC1=rand['DEC'],CZ1=cov_rand,RA2=data2['RA'],DEC2=data2['DEC'],CZ2=cov_real2,is_comoving_dist=True,verbose=True)
            np.savetxt(args.out_dir + f'/{imock}_RD.dat', RD)
        np.savetxt(args.out_dir + f'/{imock}_DD.dat', DD)
        np.savetxt(args.out_dir + f'/{imock}_DR.dat', DR)
        np.savetxt(args.out_dir + f'/{imock}_RR.dat', RR)
        
        logger.debug(f'Relative ellapsed time: {time.time() - start_computation}')


def hhz(z):
    om=0.3147
    return 1/np.sqrt(om*(1+z)**3+(1-om))

def cov(z):
    return 2998*integrate.quad(hhz,0,z)[0]
   

if __name__ == '__main__':
    main()