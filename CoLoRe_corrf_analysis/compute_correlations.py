#!/usr/bin/env python
'''
    Script to compute the correlation of input master.fits catalogues using master_randoms.fits as randoms
'''

import argparse
import logging
import sys
from pathlib import Path

import healpy as hp
import numpy as np
import scipy.integrate as integrate
from astropy.io import fits
from Corrfunc.mocks.DDsmu_mocks import DDsmu_mocks
from scipy import interpolate
import json

from CoLoRe_corrf_analysis.cf_helper import CFComputations
from CoLoRe_corrf_analysis.file_funcs import FileFuncs

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
        help='Generate randoms2. Default: (Use the ones from randoms1)')

    parser.add_argument("--randoms-from-nz-file",
        type=Path,
        required=False,
        help="Compute randoms from dndz file provided as Path. The file will be read as dN/dzdOmega in deg^-2. The number of randoms will be determined by this value and --randoms-factor.")

    parser.add_argument("--store-generated-rands",
        action='store_true',
        help='Store generated randoms in the output dir')

    parser.add_argument('--randoms-factor',
        default=1,
        help='Modify the quantity of randoms by this factor.')

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

    parser.add_argument('--data-downsampling',
        type=float,
        default=1,
        help='Downsampling to be applied to data')

    parser.add_argument('--randoms-downsampling',
        type=float, 
        default=1,
        help='Downsampling to be applied to randoms')

    parser.add_argument('--pixel-mask',
        nargs='+',
        type=int, 
        default=None,
        help='Pixels to include. Defaults to all sky')

    parser.add_argument('--nside',
        default=2,
        type=int,
        help='nside for the pixel mask')

    parser.add_argument('--reverse-RSD',
        action='store_true',
        help='Reverse the effect of RSD')

    parser.add_argument('--reverse-RSD2',
        action='store_true',
        help='Reverse the effect of RSD for field 2')

    parser.add_argument('--log-level', default='WARNING', choices=['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'NOTSET'])

    args = parser.parse_args()
    return args

class FieldData:
    ''' FieldData class used to store data fields (Data and Randoms)
    
    It is used to read from input files, compute randoms, apply masks...
    '''
    def __init__(self, cat, label, file_type, rsd=False, reverse_RSD=False):
        '''
        Args:
            cat (array of str or Path): Input data to read, they can be CoLoRe files or
                drq catalogues (defined in file_type).
            label (str): Label to identify the class.
            file_type (str): Input type. Options:
                - "CoLoRe": CoLoRe srcs output files.
                - "zcat": fits file with the fields "Z", "RA" and "DEC" in the first header.
            rsd (bool, optional): Whether to include RSD in data (only for CoLoRe). (Default: False).
            reverse_RSD (bool, optional): Whether to reverse the effect of RSD (only for CoLoRe).
                (Default: False)
        '''
        self.cat   = cat
        self.label  = label
        self.file_type = file_type
        if file_type == 'zcat':
            self.rsd = False
        else:
            self.rsd = rsd
        self.reverse_RSD = reverse_RSD

    def __str__(self): # pragma: no cover
        return self.label

    @property
    def zfield(self):
        ''' The zfield to read from input data is different depending on the file_type.'''
        if self.file_type == 'zcat':
            return 'Z'
        elif self.file_type == 'CoLoRe':
            return 'Z_COSMO'

    def open_fits(self, imock):
        self.fits = fits.open(self.cat[imock])

    def define_data_from_fits(self):
        ''' Method to define the data structure by reading input fits files.'''
        _cat_length = 0
        for i in range(len(self.cat)):
            self.open_fits(i)
            _cat_length += len(self.fits[1].data)
        self.data = np.empty(_cat_length,dtype=[('RA','f8'),('DEC','f8'),('Z','f8'),('Weight','f8')])

    def define_data_from_size(self, N):
        ''' Method to define the data structure by its length.'''
        self.data = np.empty(N,dtype=[('RA','f8'),('DEC','f8'),('Z','f8'),('Weight','f8')])

    def fill_data(self):
        ''' Fill the data arrays from the input source.'''
        _index = 0
        for i in range(len(self.cat)):
            self.open_fits(i)
            _file_size = len(self.fits[1].data)
            self.data['RA'][_index:_index+_file_size]  = self.fits[1].data['RA']
            self.data['DEC'][_index:_index+_file_size] = self.fits[1].data['DEC']
            self.data['Z'][_index:_index+_file_size]   = self.fits[1].data[self.zfield]
            if self.rsd:
                if self.reverse_RSD:
                    self.data['Z'][_index:_index+_file_size] -= self.fits[1].data['DZ_RSD']
                else:
                    self.data['Z'][_index:_index+_file_size] += self.fits[1].data['DZ_RSD']
            
            _index += _file_size

    def apply_downsampling(self, downsampling): # pragma: no cover
        '''Method to apply downsampling to data.
        
        Args:
            downsampling (float): downsampling to apply (1 to keep all data)''' 
        _mask = np.random.choice(a=[True, False], size=len(self.data), p=[downsampling, 1-downsampling])
        self.data = self.data[_mask]

    def apply_pixel_mask(self, pixel_mask, nside=2):
        ''' Method to apply a given pixel mask to data.
        
        Args:
            pixel_mask (array of int): valid pixels for the mask.
            nside (int, optional): nside of the pixelization. (Default: 2).'''
        _pixels = hp.ang2pix(nside, self.data['RA'], self.data['DEC'], lonlat=True)
        _mask   = np.in1d(_pixels, pixel_mask)
        self.data = self.data[_mask]

    def apply_redshift_mask(self, zmin, zmax):
        '''Method to apply a redshfit mask to data.
        
        Args:
            zmin (float): Min. redshift for valid objects.
            zmax (float): Max. redshift for valid objects.''' 
        _mask = self.data['Z'] > zmin
        _mask &= self.data['Z'] < zmax
        self.data = self.data[_mask]

    def compute_cov(self, interpolator):
        ''' Compute comoving distance to all objects''' 
        self.cov = interpolator(self.data['Z'])

    def generate_random_redshifts_from_file(self, file, zmin=None, zmax=None, factor=1, pixel_mask=None, nside=2):
        '''
            Generate random redshift values from an input dndz filename through the process:
                p(z<zi) = N(z_i)/N(zmax) ->
                z(p) = N_inv(N(zmax) p)
            where p in (0,1) and N the cumulative distribution.

            Args:
                file (str or Path): Input filem, will be read as dN/dzdOmega in deg^-2.
                factor (float, optional): Factor to increase or decrease the number of randoms generated. (Default: 1)
                pixel_mask (array of int, optional): Pixel mask in order to get the correct sky area. (Default: all_sky)
                nside (int, optional): nside of the pixel_mask pixelization.
        '''
        from scipy.interpolate import interp1d
        from scipy.integrate import quad

        logger.info(f'Generating random catalog from file: {str(file)}')
        in_z, in_nz = np.loadtxt(file, unpack=True)

        zmin = zmin if zmin != None else in_z[0]
        zmax = zmax if zmax != None else in_z[-1]

        # Cumulative distribution
        in_Nz = np.cumsum(in_nz)
        n = interp1d(in_z, in_nz)
        N = interp1d(in_z, in_Nz)
        N_inv = interp1d(in_Nz, in_z)

        pixarea = hp.pixelfunc.nside2pixarea(nside, degrees=True)
        pixels = len(pixel_mask) if pixel_mask is not None else 48
        area = pixarea*pixels
        NRAND = int(quad(n, zmin, zmax)[0]*area)
        self.define_data_from_size(NRAND)

        logger.debug('Generating random numbers')
        ran = np.random.random(NRAND)
        self.data['Z'] = N_inv( ran*(N(zmax)-N(zmin)) + N(zmin) )

    def generate_random_redshifts_from_data(self, data, factor=1):
        ''' Generate random redshifts by reading the redshift distribution from other data object.
        
        Args:
            data (FieldData): data to use for reading the redshift distribution.
            factor (float, optional): Factor to increase or decrease the number of randoms generated. (Default: 1)'''
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
        ''' Generate random positions in the sky. Using the usual method (reversing distribution function).
            If a pixel mask is given, the code will generate randoms for each pixel independently by previously
            poisson sampling the number of randoms that should be set in each pixel. 

            Args:
                pixel_mask (array of int, optional): Pixels to include. (Default: all_sky)
                nside (int, optional): nside of the pixel_mask pixelization.
        '''
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

    def store_data_in_cat(self, filename): # pragma: no cover
        ''' Save data into a fits file.
        
            Args:
                filename (str or Path)'''
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
        ''' Method to prepare the data by reading it, applying masks and downsamplings
        
            Args:
                zmin (float):
                zmax (float):
                downsampling (float):
                pixel_mask (array of int):
                nside (int):
        '''
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

    # Logic for which fields I need to incorporate
    available_counts = FileFuncs.get_available_count_files(args.out_dir)
    to_compute = set()
    if args.data2 != None: # Cross-correlation
        if 'DD' not in available_counts:
            to_compute.add('D1')
            to_compute.add('D2')

        if 'RR' not in available_counts:
            to_compute.add('R1')
            to_compute.add('R2')

        if 'RD' not in available_counts:
            to_compute.add('R1')
            to_compute.add('D2')

        if 'DR' not in available_counts:
            to_compute.add('R2')
            to_compute.add('D1')
    else: # Auto-correlation
        to_compute.add('D1')
        if ('RR' not in available_counts) or ('DR' not in available_counts):
            to_compute.add('R1')

    if (args.out_dir / 'sizes.json').is_file():
        with open(args.out_dir / 'sizes.json') as json_file:
            sizes = json.load(json_file)
    else:
        sizes = dict()

    data_to_use = set()
    if 'D1' in to_compute:
        data = FieldData(args.data, 'Data', file_type=args.data_format, rsd=not(args.data_norsd), reverse_RSD=args.reverse_RSD)
        data.prepare_data(args.zmin, args.zmax, args.data_downsampling, args.pixel_mask, args.nside)
        data.compute_cov(f)
        data_to_use.add(data)

    if 'D2' in to_compute or args.generate_randoms2: # @ fix this. I could get randoms from dndz
        data2 = FieldData(args.data2, 'Data2', file_type=args.data2_format, rsd=not(args.data2_norsd), reverse_RSD=args.reverse_RSD2)
        data2.prepare_data(args.zmin, args.zmax, args.data_downsampling, args.pixel_mask, args.nside)
        data2.compute_cov(f)
        data_to_use.add(data2)
    else:
        data2 = data

    if 'R1' in to_compute or ('R2' in to_compute and args.randoms == None and not args.generate_randoms2): 
        rand = FieldData(args.randoms, 'Randoms', file_type='zcat')
        if args.randoms != None:
            rand.prepare_data(args.zmin, args.zmax, args.randoms_downsampling, args.pixel_mask, args.nside)
        else:
            if args.randoms_from_nz_file != None: # pragma: no cover
                rand.generate_random_redshifts_from_file(args.randoms_from_nz_file, zmin=args.zmin, zmax=args.zmax, factor=args.randoms_factor, pixel_mask=args.pixel_mask, nside=args.nside)
            else:
                rand.define_data_from_size(len(data.data))
                rand.generate_random_redshifts_from_data(data, factor=args.randoms_factor)
            rand.generate_random_positions(pixel_mask=args.pixel_mask, nside=args.nside)
            rand.cat = []
            if args.store_generated_rands:
                rand.store_data_in_cat(Path(args.out_dir) / (rand.label + '.fits'))
        rand.compute_cov(f)
        data_to_use.add(rand)
        
    if 'R2' in to_compute:
        if args.randoms2 != None:
            rand2 = FieldData(args.randoms2, 'Randoms2', file_type='zcat')
            rand2.prepare_data(args.zmin, args.zmax, args.randoms_downsampling, args.pixel_mask, args.nside)
            rand2.compute_cov(f)
        elif args.generate_randoms2: # pragma: no cover
            rand2 = FieldData(args.randoms2, 'Randoms2', file_type='zcat')
            if args.randoms_from_nz_file != None:
                rand2.generate_random_redshifts_from_file(args.randoms_from_nz_file, zmin=args.zmin, zmax=args.zmax)
            else:
                rand2.define_data_from_size(len(data2.data))
                rand2.generate_random_redshifts_from_data(data2, factor=args.randoms_factor)
            rand2.generate_random_positions(pixel_mask=args.pixel_mask, nside=args.nside)
            rand2.cat = []
            if args.store_generated_rands:
                rand2.store_data_in_cat(Path(args.out_dir) / (rand2.label + '.fits'))
            rand2.compute_cov(f)
        else:
            rand2 = rand
            data_to_use.add(rand2)
    elif 'R1' in to_compute:
        rand2 = rand

    bins2p=np.linspace(args.min_bin, args.max_bin, args.n_bins)

    logger.info('Starting computation for current file:')
    if logging.root.level <= logging.DEBUG: # pragma: no cover
        start_computation = time.time()

    info_file = Path(args.out_dir / 'README')
    text=''
    for obj in data_to_use:
        text+="\n{}\n\t{}".format(obj.label, "\n\t".join([str(cat) for cat in obj.cat]))
        sizes[obj.label] = len(obj.data)

    info_file.write_text(text)

    with open(args.out_dir / 'sizes.json', 'w') as json_file:
        json.dump(sizes, json_file)

    if 'DD' not in available_counts:
        logger.info('Computing DD...')
        DD = DDsmu_mocks(autocorr=data==data2, 
            cosmology=2, nthreads=args.nthreads, mu_max=args.mu_max, nmu_bins=args.nmu_bins, binfile=bins2p, RA1=data.data['RA'], DEC1=data.data['DEC'], CZ1=data.cov, 
            RA2=data2.data['RA'], DEC2=data2.data['DEC'], CZ2=data2.cov, 
            is_comoving_dist=True, verbose=True)
        np.savetxt(args.out_dir / f'DD.dat', DD)

    if 'DR' not in available_counts:
        logger.info('Computing DR...')
        DR = DDsmu_mocks(autocorr=0, 
            cosmology=2, nthreads=args.nthreads, mu_max=args.mu_max, nmu_bins=args.nmu_bins, binfile=bins2p, RA1=data.data['RA'], DEC1=data.data['DEC'], CZ1=data.cov, 
            RA2=rand2.data['RA'],DEC2=rand2.data['DEC'], CZ2=rand2.cov, 
            is_comoving_dist=True, verbose=True)
        np.savetxt(args.out_dir / f'DR.dat', DR)

    if 'RR' not in available_counts:
        logger.info('Computing RR...')
        RR = DDsmu_mocks(autocorr=rand==rand2, 
            cosmology=2, nthreads=args.nthreads, mu_max=args.mu_max, nmu_bins=args.nmu_bins, binfile=bins2p, RA1=rand.data['RA'], DEC1=rand.data['DEC'], CZ1=rand.cov,
            RA2=rand2.data['RA'], DEC2=rand2.data['DEC'], CZ2=rand2.cov,
            is_comoving_dist=True, verbose=True)
        np.savetxt(args.out_dir / f'RR.dat', RR)

    if 'RD' not in available_counts and data2!=data:
        logger.info('Computing RD...')
        RD = DDsmu_mocks(autocorr=0, 
            cosmology=2, nthreads=args.nthreads, mu_max=args.mu_max, nmu_bins=args.nmu_bins, binfile=bins2p,
            RA1=rand.data['RA'], DEC1=rand.data['DEC'], CZ1=rand.cov,
            RA2=data2.data['RA'], DEC2=data2.data['DEC'], CZ2=data2.cov,
            is_comoving_dist=True, verbose=True)
        np.savetxt(args.out_dir / f'RD.dat', RD)

    if logging.root.level <= logging.DEBUG: # pragma: no cover
        logger.debug(f'Relative ellapsed time: {time.time() - start_computation}')

    np.savetxt(args.out_dir / 'N_data.dat', [len(i.data) for i in (data, data2)])
    np.savetxt(args.out_dir / 'N_rand.dat', [len(i.data) for i in (rand, rand2)])

    if args.compute_npoles != None: # pragma: no cover
        logger.info(f'Computing npoles:')
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