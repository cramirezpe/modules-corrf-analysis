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
import argparse
import logging
from pathlib import Path
import sys

version='0.12'
logger = logging.getLogger(__name__)

def getArgs(): # pragma: no cover
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",       
        nargs='+',
        type=Path, 
        required=True, 
        help='Input data files')

    parser.add_argument("--data-format",
        type=str,
        choices=['CoLoRe', 'zcat'],
        default='zcat')

    parser.add_argument("--data-norsd",
        action='store_true',
        help='If CoLoRe format selected: read noRSD redshifts for data')

    parser.add_argument("--data2",
        nargs='+',
        type=Path, 
        required=False,
        help='Input data2 files')

    parser.add_argument("--data2-format",
        type=str,
        choices=['CoLoRe', 'zcat'],
        default='zcat')

    parser.add_argument("--compute-npoles",
        nargs='*',
        type=int,
        default=[0, 2, 4],
        help='Compute npoles from output counts'
        )
        
    parser.add_argument("--data2-norsd",
        action='store_true',
        help='If CoLoRe format selected: read noRSD redshifts for data2')
    
    parser.add_argument("--randoms",    
        nargs='+',
        type=Path, 
        required=False, 
        help='Input random files. If not provided they will be generated')

    parser.add_argument("--randoms2",
        nargs='+',
        type=Path,
        required=False,
        help='Input randoms2 files')

    parser.add_argument("--generate-randoms2",
        action='store_true',
        help='Generate randoms2. Default: (Use the ones form randoms1)')

    parser.add_argument("--randoms-from-nz-file",
        type=Path,
        required=False,
        help="Compute randoms from dndz file provided as Path")

    parser.add_argument("--store-generated-rands",
        action='store_true',
        help='Store generated randoms in the output dir')

    parser.add_argument("--out-dir",
        type=Path,
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
    def __init__(self, cat, label, file_type, rsd=False):
        self.cat   = cat
        self.label  = label
        self.file_type = file_type
        if file_type == 'zcat':
            self.rsd = False
        else:
            self.rsd = rsd

    def __str__(self): # pragma: no cover
        return self.label

    @property
    def zfield(self):
        if self.file_type == 'zcat':
            return 'Z'
        elif self.file_type == 'CoLoRe':
            return 'Z_COSMO'

    def open_fits(self, imock):
        self.fits = fits.open(self.cat[imock])

    def define_data_from_fits(self):
        _cat_length = 0
        for i in range(len(self.cat)):
            self.open_fits(i)
            _cat_length += len(self.fits[1].data)
        self.data = np.empty(_cat_length,dtype=[('RA','f8'),('DEC','f8'),('Z','f8'),('Weight','f8')])

    def define_data_from_size(self, N):
        self.data = np.empty(N,dtype=[('RA','f8'),('DEC','f8'),('Z','f8'),('Weight','f8')])

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

    def generate_random_redshifts_from_file(self, file, zmin=None, zmax=None):
        '''
            Generate random redshift values from an input dndz filename through the process:
                p(z<zi) = N(z_i)/N(zmax) ->
                z(p) = N_inv(N(zmax) p)
            where p in (0,1) and N the cumulative distribution.
        '''
        from scipy.interpolate import interp1d

        logger.info(f'Generating random catalog from file: {str(file)}')
        in_z, in_nz = np.loadtxt(file, unpack=True)

        zmin = zmin if zmin != None else in_z[0]
        zmax = zmax if zmax != None else in_z[-1]

        # Cumulative distribution
        in_Nz = np.cumsum(in_nz)
        N = interp1d(in_z, in_Nz)
        N_inv = interp1d(in_Nz, in_z)

        NRAND = len(self.data)

        logger.debug('Generatin random number')
        ran = np.random.random(NRAND)
        self.data['Z'] = N_inv( ran*(N(zmax)-N(zmin)) + N(zmin) )

    def generate_random_redshifts_from_data(self, data):
        from scipy.interpolate import interp1d

        logger.info(f'Generating random catalog from {data.label} to {self.label}')  
        logger.info('Sorting redshifts')

        z_sort = np.sort(data.data['Z'])
        p = np.linspace(0, 1, len(z_sort), endpoint=True) #endpoint set to true will cause a biased estimator... but I chose it anyway to avoid invalid values later on.

        z_gen = interp1d(p, z_sort)

        NRAND = len(self.data)
        logger.info('Interpolating redshift')
        ran1 = np.random.random(NRAND)
        self.data['Z'] = z_gen(ran1)

    def generate_random_positions(self, pixel_mask=None, nside=None):
        logger.info(f'Computing random positions for field {self.label}')
        logger.critical('¡Random positions generator gives bad randoms! Use it at your own risk.')
        NRAND = len(self.data)
        if pixel_mask == None:
            ran1 = np.random.random(int(NRAND))
            ran2 = np.random.random(int(NRAND))

            self.data['RA'] = np.degrees(np.pi*2*ran1)
            self.data['DEC'] = np.degrees(np.arcsin(2.*(ran2-0.5)))

            return

        _lambda = NRAND / len(pixel_mask)
        randoms_per_pixel = np.random.poisson(_lambda, len(pixel_mask))
        extra_objs = randoms_per_pixel.sum() - NRAND
        if extra_objs > 0: # pragma: no cover
            for i in range(np.abs(extra_objs)):
                randoms_per_pixel[np.random.randint(0, len(pixel_mask))] -= 1
        elif extra_objs < 0: # pragma: no cover
            for i in range(np.abs(extra_objs)):
                randoms_per_pixel[np.random.randint(0, len(pixel_mask))] += 1
        
        pix_area = hp.pixelfunc.nside2pixarea(nside)   
        _index=0

        for pixel, N in zip(pixel_mask, randoms_per_pixel):
            logger.info(f'Computing random positions for pixel {pixel}')
            pixel_center = hp.pix2ang(nside=nside, ipix=pixel)
            corners = hp.rotator.vec2dir(
                hp.boundaries(nside, [pixel], step=1000)[0]
            )
            th_range = (min(corners[0]), max(corners[0]))
            ph_range = (min(corners[1]), max(corners[1]))

            range_size = np.abs((ph_range[1]-ph_range[0]) * (np.cos(th_range[0])-np.cos(th_range[1])))

            THs = []
            PHs = []
            valid_fraction = pix_area / range_size
            while len(THs) < N:
                randoms_left = N - len(PHs)
                logger.debug(f'Randoms left: {randoms_left}')
                ran1 = np.random.random(int(randoms_left/valid_fraction))
                ran2 = np.random.random(int(randoms_left/valid_fraction))

                new_PHs = ran1*(ph_range[1]-ph_range[0]) + ph_range[0]
                new_THs = np.arccos( np.cos(th_range[0]) - ran2*( np.cos(th_range[0])-np.cos(th_range[1]) ) )
                new_pixels = hp.pixelfunc.ang2pix(nside, new_THs, new_PHs)
                mask = new_pixels == pixel
                THs = np.append(THs, new_THs[mask])
                PHs = np.append(PHs, new_PHs[mask])        

            self.data['RA'][_index:_index+int(N)] = np.degrees(PHs[:int(N)])
            self.data['DEC'][_index:_index+int(N)] = 90  - np.degrees(THs[:int(N)])
            _index+=N

    def store_data_in_cat(self, filename):
        logger.info(f'Writting catalogue {self.label} into {filename}')
        values = (self.data['RA'], self.data['DEC'], self.data['Z'])
        labels = ('RA', 'DEC', 'Z')
        dtypes = ('D', 'D', 'D')

        cols = []
        for value, label, dtype in zip(values, labels, dtypes):
            cols.append(fits.Column(name=label, format=dtype, array=value))

        logger.debug('Defining FITS headers')
        p_hdr = fits.Header()
        t_hdr = fits.Header()

        logger.debug('Defining hdulist')
        hdulist = fits.HDUList()

        hdulist.append(fits.PrimaryHDU(header=p_hdr))
        hdulist.append(fits.BinTableHDU.from_columns(cols, header=t_hdr))

        try:
            hdulist.writeto(filename, overwrite=False)
        except OSError: # pragma: no cover
            logger.warning('Unable to write catalog to filename path.')
        
        hdulist.close() 

    def prepare_data(self, zmin, zmax, downsampling, pixel_mask, nside):
        logger.info('\nReading files:\n\t{}'.format("\n\t".join([str(cat) for cat in self.cat])))
        self.define_data_from_fits()
        self.fill_data()
        logger.debug(f'Length of {self.label} cat: {len(self.data)}')

        if downsampling != 1:
            logger.debug(f'Applying downsampling: {downsampling}')
            self.apply_downsampling(downsampling)
            logger.debug(f'Length of {self.label} after downsampling: {len(self.data)}')

        self.apply_redshift_mask(zmin, zmax)
        logger.debug(f'Length of {self.label} after redshift mask: {len(self.data)}')

        if pixel_mask != None:
            logger.debug(f'Applying pixel mask')
            self.apply_pixel_mask(pixel_mask, nside)
            logger.debug(f'Length of {self.label} after pixel mask: {len(self.data)}')

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

    z=np.arange(args.zmin_covd,args.zmax_covd,args.zstep_covd)
    covd=[]
    for i in range(len(z)):
        covd.append(cov(z[i]))
    covd=np.array(covd)
    f = interpolate.interp1d(z, covd)

    Path(args.out_dir).mkdir(exist_ok=True)

    data = FieldData(args.data, 'Data', file_type=args.data_format, rsd=not(args.data_norsd))
    data.prepare_data(args.zmin, args.zmax, args.random_downsampling, args.pixel_mask, args.nside)
    data.compute_cov(f)

    rand = FieldData(args.randoms, 'Randoms', file_type='zcat')
    if args.randoms != None:
        rand.prepare_data(args.zmin, args.zmax, args.random_downsampling, args.pixel_mask, args.nside)
    else:
        rand.define_data_from_size(len(data.data))
        if args.randoms_from_nz_file != None: # pragma: no cover
            rand.generate_random_redshifts_from_file(args.randoms_from_nz_file, zmin=args.zmin, zmax=args.zmax)
        else:
            rand.generate_random_redshifts_from_data(data)
        rand.generate_random_positions(pixel_mask=args.pixel_mask, nside=args.nside)
        rand.cat = []
        if args.store_generated_rands:
            rand.store_data_in_cat(Path(args.out_dir) / (rand.label + '.fits'))
    rand.compute_cov(f)

    if args.data2 != None:
        data2 = FieldData(args.data2, 'Data2', file_type=args.data2_format, rsd=not(args.data2_norsd))
        data2.prepare_data(args.zmin, args.zmax, args.random_downsampling, args.pixel_mask, args.nside)
        data2.compute_cov(f)
        if args.randoms2 != None:
            rand2 = FieldData(args.randoms2, 'Randoms2', file_type='zcat')
            rand2.prepare_data(args.zmin, args.zmax, args.random_downsampling, args.pixel_mask, args.nside)
            rand2.compute_cov(f)
        elif args.generate_randoms2: # pragma: no cover
            rand2 = FieldData(args.randoms2, 'Randoms2', file_type='zcat')
            rand2.define_data_from_size(len(data2.data))
            if args.randoms_from_nz_file != None:
                rand2.generate_random_redshifts_from_file(args.randoms_from_nz_file, zmin=args.zmin, zmax=args.zmax)
            else:
                rand2.generate_random_redshifts_from_data(data2)
            rand2.generate_random_positions(pixel_mask=args.pixel_mask, nside=args.nside)
            rand2.cat = []
            if args.store_generated_rands:
                rand2.store_data_in_cat(Path(args.out_dir) / (rand2.label + '.fits'))
            rand2.compute_cov(f)
        else:
            rand2 = rand
    else:
        data2 = data
        rand2 = rand

    bins2p=np.linspace(args.min_bin, args.max_bin, args.n_bins)

    logger.info('Starting computation for current file:')
    if logging.root.level <= logging.DEBUG: # pragma: no cover
        start_computation = time.time()

    info_file = Path(args.out_dir / 'README')
    text=''
    for obj in [data, data2, rand, rand2]:
        text+="\n{}\n\t{}".format(obj.label, "\n\t".join([str(cat) for cat in obj.cat]))

    info_file.write_text(text)

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
        np.savetxt(args.out_dir / f'RD.dat', RD)
    np.savetxt(args.out_dir / f'DD.dat', DD)
    np.savetxt(args.out_dir / f'DR.dat', DR)
    np.savetxt(args.out_dir / f'RR.dat', RR)
    
    if logging.root.level <= logging.DEBUG: # pragma: no cover
        logger.debug(f'Relative ellapsed time: {time.time() - start_computation}')

    if args.compute_npoles != None: # pragma: no cover
        logger.info(f'Computing npoles:')
        from CoLoRe_corrf_analysis.cf_helper import CFComputations
        CFComp = CFComputations(args.out_dir,  N_data_rand_ratio=1)
        for pole in args.compute_npoles:
            logger.info(f'\tnpole {pole}')
            _ = CFComp.compute_npole(pole)

def hhz(z):
    om=0.3147
    return 1/np.sqrt(om*(1+z)**3+(1-om))

def cov(z):
    return 2998*integrate.quad(hhz,0,z)[0]

if __name__ == '__main__': # pragma: no cover
    main()