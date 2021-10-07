import logging
from functools import cached_property
from pathlib import Path

import astropy
from astropy.io import fits
import libconf
import numpy as np
from astropy import cosmology
from scipy.interpolate import InterpolatedUnivariateSpline, interp1d
from mcfit import xi2P, P2xi

logger = logging.getLogger(__name__)

class ReadCoLoRe:
    _default_h = 0.7
    _default_Om0 = 0.3
    _default_Ode0 = 0.7
    _default_Ob0 = 0.05
    _default_cosmo = astropy.cosmology.LambdaCDM(_default_h*100, Om0=_default_Om0, Ode0=_default_Ode0, Ob0=_default_Ob0)

    def __init__(self, box_path, source, bias_filename=None, nz_filename=None, param_cfg_filename=None, zmin=None, zmax=None):
        ''' Class to read CoLoRe sims and compute relevant statistics.

        Args:
            box_path (str or Path): Path to CoLoRe output files.
            source (int): Soruce to read from CoLoRe output.
            bias_filename (str or Path, optional): bias filename for computations with bias. (Default: Searches for bias_filename in param_cfg).
            nz_filename (str or Path, optional): Nz filename. (Default: Searches for nz_filename in the param_cfg).
            param_cfg_filename (str or Path, optional): Path to param_cfg file. Used to search for nz_filename and bias_filename if they are not fixed. (Default: globbing .cfg in the box path).
            zmin (float, optional): zmin to consider when dealing with nz distribution. Defaults to zmin in paramcfg if it exists, 0 if not.
            zmax (float, optional): zmax to consider when dealing with nz distribution. Defaults to zmax in paramcfg if it exists, 1000 if not.
        '''

        self.box_path = Path(box_path)
        self.source = source

        self.param_cfg_filename = param_cfg_filename

        if bias_filename is None: # pragma: no cover
            try:
                self.bias_filename = self.param_cfg[f'srcs{self.source}']['bias_filename']
            except Exception: 
                print('Failed reading bias from param.cfg')
        else:
            self.bias_filename = bias_filename

        if nz_filename is None: # pragma: no cover
            try:
                self.nz_filename = self.param_cfg[f'srcs{self.source}']['nz_filename']
            except Exception: 
                print('Failed reading nz from param.cfg')
        else:
            self.nz_filename = nz_filename

        if zmin is None: # pragma: no cover
            if self.param_cfg is None:
                self.zmin = 0
            else:
                self.zmin = self.param_cfg['global']['z_min']
        else: # pragma: no cover
            self.zmin = zmin
        if zmax is None: # pragma: no cover
            if self.param_cfg is None:
                self.zmax = 1000
            else:
                self.zmax = self.param_cfg['global']['z_max']
        else: # pragma: no cover
            self.zmax = zmax
            
    @cached_property
    def param_cfg(self):
        ''' Get param_cfg values in libconf format.

        Returns:
            Libconf version of param_cfg values
        '''
        if self.param_cfg_filename is None:
            self.param_cfg_filename = next( Path(self.box_path).glob('*.cfg') )
        
        with open(self.param_cfg_filename) as f:
            param_cfg = libconf.load(f)

        return param_cfg

    @cached_property
    def cosmo(self): # pragma: no cover
        ''' Returns cosmo parameters from param_cfg + cosmo parameters computed as in CoLoRe.

        Returns:
            dict of cosmo parameters.
        ''' 
        cosmo = dict(
            omega_M = float( self.param_cfg['cosmo_par']['omega_M'] ),
            omega_L = float( self.param_cfg['cosmo_par']['omega_L'] ),
            omega_B = float( self.param_cfg['cosmo_par']['omega_B'] ),
            h       = float( self.param_cfg['cosmo_par']['h'] ), 
            w       = float( self.param_cfg['cosmo_par']['w'] ),
            ns      = float( self.param_cfg['cosmo_par']['ns'] ),
            sigma_8 = float( self.param_cfg['cosmo_par']['sigma_8'] ),
            T_CMB   = float( 2.275 )
        )

        cosmo['omega_K'] = 1 - cosmo['omega_M'] - cosmo['omega_L']

        # Compute ksign
        if np.abs(cosmo['omega_K']) < 1e-6:
            ksign = 0
        elif cosmo['omega_K'] > 0:
            ksign = -1
        else:
            ksign = 1
        cosmo['ksign'] = ksign

        # constantw & DE
        if np.abs( cosmo['w']+1 < 1e-6 ):
            cosmo['constantw'] = 1
            cosmo['normalDE'] = 1
        else:
            cosmo['constantw'] = 1
            cosmo['normalDE'] = 0

        return cosmo

    def get_volume_between_zs(self, z2, z1=0):
        ''' Get comoving volume of the spherical shell between two redshifts using the self.cosmo parameters for a LCDM universe.
        
        Args:
            z2 (float): Second redshift of the spherical shell.
            z1 (float, optional): First redshift of the spherical shell. (Default: defaults to 0)
        
        Returns:
            Comoving horizon in Mpc/h
        '''
        cosmo = astropy.cosmology.LambdaCDM(self.cosmo['h']*100, Om0=self.cosmo['omega_M'], Ode0=self.cosmo['omega_L'], Ob0=self.cosmo['omega_B'])
        
        if z1==0:
            return cosmo.comoving_volume(z2)

        return cosmo.comoving_volume(max(z1,z2)) - cosmo.comoving_volume(min(z1,z2))

    def L_box(self):
        '''Get box size
        
        Returns:
            Box size in Mpc/h
        ''' 
        cosmo = astropy.cosmology.LambdaCDM(self.cosmo['h']*100,
                                   Om0=self.cosmo['omega_M'],
                                   Ode0=self.cosmo['omega_L'], 
                                   Ob0=self.cosmo['omega_B'])
        r_max = cosmo.comoving_distance(float( self.param_cfg['global']['z_max'] ))*self.cosmo['h']
        return 2*r_max.value*(1+2/self.param_cfg['field_par']['n_grid']) 

    def get_nz_histogram_from_Nz_file(self, bins):
        ''' Get nz histogram from input Nz file
        
        Args:
            bins (array): Bin edges for the histogram.

        Returns:
            1D array of counts for the histogram.
        '''
        z, nz = np.loadtxt(self.nz_filename, unpack=True)
        nz_interpolation = interp1d(z, nz)

        bins_centers = 0.5*(bins[1:] + bins[:-1])
        new_nz = nz_interpolation(bins_centers)

        new_nz[bins_centers < self.zmin] = 0
        new_nz[bins_centers > self.zmax] = 0

        db = np.array(np.diff(bins), float)
        return new_nz/(new_nz*db).sum()

    def get_nz_histogram_from_CoLoRe_box(self, bins, rsd):
        ''' Get nz histogram from CoLoRe files:

        Args:
            bins (array): Bin edges of the histogram.
            rsd (bool): Whether to use RSD redshifts or not.

        Returns:
            1D array of counts for the histogram.
        '''

        fits_file = next( self.box_path.glob(f'out_srcs_s{self.source}*') )
        with astropy.io.fits.open(fits_file) as hdul:
            z = hdul[1].data['Z_COSMO']
            if rsd:
                z += hdul[1].data['DZ_RSD']
        
        z = z[z<self.zmax]
        z = z[z>self.zmin]

        Nz, _ = np.histogram(z, bins=bins, density=True)
        
        return Nz

    @staticmethod
    def get_nz_histogram_from_master_file(master_file, bins, rsd):
        ''' Get nz histogram from LyaCoLoRe master file
        
        Args:
            master_file (path): Path to master file.
            bins (array): Bin edges of the histogram.
            rsd (bool): Whether to use RSD redshifts or not.

        Returns:
            1D array of counts for the histogram.
        '''
        master_hdul = astropy.io.fits.open(master_file)
        if rsd:
            zcat = master_hdul[1].data['Z_QSO_RSD']
        else:
            zcat = master_hdul[1].data['Z_QSO_NO_RSD']

        Nz, _ = np.histogram(zcat, bins=bins, density=True)

        return Nz

    def get_zeff(self, zmin, zmax, nbins=100, method='Nz_file', master_file=None, rsd=True):
        ''' Get effective redsfhit for the correlation by weighting the density of pairs (~ N**2) over the redshift range given.
        
        Args:
            zmin (double): Min redshift of the range
            zmax (double): Max redshift of the range
            nbins (double, optional): Number of bins in which to divide the range before integration.
            method (str, optional): Method to extract the Nz histogram. Options:
                CoLoRe: Reads the CoLoRe output files.
                master_file: Reads the LyaCoLoRe master file provided in master_file
                Nz_file: Reads the Nz file provided in param.cfg (DEFAULT)
            master_file (str or Path, optional): Path to the master catalog of objects. Only for 'master_file' method, and mandatory for it. 
            rsd (bool, optional): Whether to use RSD; not used for "Nz_file" method. (Default: True)
        '''
        logger.debug('Defining bins')

        bins = np.linspace(zmin, zmax, nbins)
        bin_width = bins[1] - bins[0]
        bin_centers = (bins[1:] + bins[:-1]) / 2

        logger.debug('Generating Nz')
        if method == 'CoLoRe':
            Nz = self.get_nz_histogram_from_CoLoRe_box(bins, rsd)
        elif method == 'master_file':
            if master_file is None: # pragma: no cover
                raise ValueError('master_file needs to be defined to use master_file method')
            else:
                Nz = self.get_nz_histogram_from_master_file(master_file, bins, rsd)
        elif method == 'Nz_file':
            Nz = self.get_nz_histogram_from_Nz_file(bins)
        else: # pragma: no cover
            raise ValueError('Method for combining Nz should be in ("CoLoRe", "master_file", "Nz_file")')

        norm_Nz2 = 1/(bin_width*(Nz**2).sum())
        Nz2 = norm_Nz2 * Nz**2

        zs = np.array([zi*Nzi for zi, Nzi in zip(bin_centers, Nz2)])

        return zs.sum(axis=0) * bin_width

    def z_bins_from_files(self):
        '''Get zbins from CoLoRe prediction ouptut files
        
        Returns:
            1D array with redshifts used.
        ''' 
        return np.sort(np.array( [float(x.name[18:23]) for x in self.box_path.glob(f'out_pk_srcs_pop{self.source-1}*')] ))
        # return np.fromiter( map(lambda x: float(x.name[18:23]), self.box_path.glob('out_pk*')), np.float)

    @cached_property
    def bias(self):
        ''' Get bias interpolation object from the input bias file.

        Returns:
            Interp1d object for bias.
        '''
        bias_z, bias_bz = np.loadtxt(self.bias_filename, unpack=True)

        return interp1d(bias_z, bias_bz, fill_value='extrapolate')

    def get_a_eq(self): # pragma: no cover
        ''' Computes and returns a_eq as computed in CoLoRe code:

        Returns:
            float aeq.
        '''
        aeqk = 1
        aeqL = 1

        if self.cosmo['ksign'] != 0:
            aeqK = self.cosmo['omega_M']/np.abs(self.cosmo['omega_K'])
        
        if self.cosmo['omega_L'] != 0:
            if self.cosmo['normalDE'] == 1:
                aeqL = (self.cosmo['omega_M']/self.cosmo['omega_L'])**0.333
            else:
                aeqL = (self.cosmo['omega_M']/self.cosmo['omega_L'])**(-1/(3*self.cosmo['w']))

        return min(aeqk, aeqL)

    def growth_factor(self, a):
        '''Get unnormalized growth factor for a single scale factor (as CoLoRe).

        Args:
            a (float): Scale factor

        Returns:
            Unnormalized growth factor as a float.
        '''
        a_eq = self.get_a_eq()

        alim = 0.01 * a_eq
        int0 = 0.4 * np.sqrt(alim**5/(self.cosmo['omega_M']**3))
        if a <= alim:
            return a
        else:
            relerrt = 1e-4

        # Definning dum as in CoLoRe
        def dum(a): # pragma: no cover
            if self.cosmo['normalDE']:
                dum = np.sqrt(a/(self.cosmo['omega_M']+self.cosmo['omega_L']*a**3+self.cosmo['omega_K']*a))
            elif self.cosmo['constantw']:
                dum = np.sqrt(a/(self.cosmo['omega_M']+self.cosmo['omega_L']*a**(-3*self.cosmo['w'])+self.cosmo['omega_K']*a))
            else:
                raise ValueError('constantw != 1')
                
            return dum

        x = np.linspace(alim, a, 1000)
        y = [dum(i)**3 for i in x]
        integral = simple_integral(x, y)

        return (int0 + integral)*2.5*self.cosmo['omega_M']/(a*dum(a))

    def logarithmic_growth_rate(self, z, read_file=True):
        ''' Compute logarithmic growth rate f = dlogD/da(z):
        
        Args:
            z (float): Redshift to evaluate f
            read_file (bool): If True, read growth factor from CoLoRe fits file; If False, compute f from param.cfg cosmology (Default: True)

        Returns:
            logarithmic growth rate as float.
        '''
        if read_file:
            dlogDdz = self.dlogDdz

            f =  -dlogDdz(z)*(1+z)
        else:
            a = 1/(1+z)
            D = self.growth_factor(a)

            if self.cosmo['normalDE'] == 1:
                coeff = 0 
                apow = a**3
            else: # pragma: no cover
                coeff = 1 + self.cosmo['w']
                apow = a**(-3*self.cosmo['w'])

            f = 0.5 * (5*self.cosmo['omega_M']*a/D - (3*self.cosmo['omega_M'] + 3*coeff*self.cosmo['omega_L']*apow + 2*self.cosmo['omega_K']*a))/(self.cosmo['omega_M']+self.cosmo['omega_L']*apow+self.cosmo['omega_K']*a)
        return f
    
    @cached_property
    def dlogDdz(self):
        ''' Read dlogDdz from file and cretes an interpolation function.

        Returns:
            interpolation object for dlogDdz(z)
        '''
        hdul = astropy.io.fits.open(self.box_path / f'out_srcs_s{self.source}_0.fits')
        z = hdul[4].data['Z']
        D = hdul[4].data['D']

        func = InterpolatedUnivariateSpline(z, np.log(D), k=1)
        dlogDdz = func.derivative()
        return dlogDdz
       
class ComputeModelsCoLoRe(ReadCoLoRe):
    def __init__(self, box_path, source, bias_filename=None, nz_filename=None, pk_filename=None, param_cfg_filename=None, zmin=None, zmax=None, smooth_factor=1.1, smooth_factor_rsd=1.0, smooth_factor_cross=2.1, smooth_factor_analysis=0.35, 
    analysis_bin_size=5, apply_lognormal=True):
        super().__init__(box_path=box_path,
                        source=source,
                        bias_filename=bias_filename,
                        nz_filename=nz_filename,
                        param_cfg_filename=param_cfg_filename,
                        zmin=zmin,
                        zmax=zmax)
        ''' Class to compute CoLoRe theoretical models with and without RSD.

        Args (from ReadCoLoRe):
            box_path (str or Path): Path to CoLoRe output files.
            source (int): Soruce to read from CoLoRe output.
            bias_filename (str or Path, optional): bias filename for computations with bias. (Default: Searches for bias_filename in param_cfg).
            nz_filename (str or Path, optional): Nz filename. (Default: Searches for nz_filename in the param_cfg).
            param_cfg_filename (str or Path, optional): Path to param_cfg file. Used to search for nz_filename and bias_filename if they are not fixed. (Default: globbing .cfg in the box path).
            zmin (float, optional): zmin to consider when dealing with nz distribution. Defaults to zmin in paramcfg if it exists, 0 if not.
            zmax (float, optional): zmax to consider when dealing with nz distribution. Defaults to zmax in paramcfg if it exists, 1000 if not.

        Args:
            smooth_factor (float, optional): Smoothing prefactor for the lognormalized field dd (<delta_LN delta_LN>), as the 1.1 in "double rsm2_gg=par->r2_smooth+1.1*pow(par->l_box/par->n_grid,2)/12.". (Default: 1.1).
            smooth_factor_rsd (float, optional): Smoothing prefactor for the matter matter field. <delta_L delta_L>. (Default: 1.0).
            smooth_factor_cross (float, optional): Smoothing prefactor for the matter galaxy (dm) field. <delta_LN delta_L>. (Default: 2.1).
            smooth_factor_analysis (float, optional): Smoothing prefactor for the analysis smoothing: smooth_factor_analysis*(analysis_bin_size/2). (Default: 0.35).
            analysis_bin_size (float, optional): Bin size to use when applying smoothing factor from analysis: smooth_factor_analysis*(analysis_bin_size/2. (Default: 5).
            apply_lognormal (bool, optional): Whether to use lognormalized fields (where it applies). (Default: True).
        '''
        
        if pk_filename is None: # pragma: no cover
            try:
                self.pk_filename = self.param_cfg['global']['pk_filename']
            except Exception:
                logger.warning('Failed reading pk filename from param.cfg')
        else:
            self.pk_filename = pk_filename

        self.r_smooth = self.param_cfg['field_par']['r_smooth']
        self.n_grid   = self.param_cfg['field_par']['n_grid']
        self._smooth_factor = smooth_factor
        self._smooth_factor_rsd = smooth_factor_rsd
        self._smooth_factor_cross = smooth_factor_cross
        self.analysis_smoothing = (analysis_bin_size/2)**2*smooth_factor_analysis

        self.apply_lognormal = apply_lognormal

    # Here I use this complicated way of putting smoothing factors 
    # because I want to avoid modifying them by using class_object.smooth_factor=2.3
    # I had old code modifying this and I don't want it to work as this anymore.
    @property
    def smooth_factor(self):
        return self._smooth_factor

    @property
    def smooth_factor_rsd(self):
        return self._smooth_factor_rsd

    @property
    def smooth_factor_cross(self):
        return self._smooth_factor_cross
    
    @property
    def input_pk(self):
        ''' log-spaced input power spectrum. Log spacing is desirable when applying FFTlog through mcfit.
        
        Returns:
            2D array (k, pk) for the log-spaced input power spectrum.
         '''
        k, pk = np.loadtxt(self.pk_filename, unpack=True)

        _min = min(np.log10(k))
        _max = max(np.log10(k))

        k_logspaced = np.logspace(_min, _max, len(k))
        pk_logspaced = interp1d(k, pk, fill_value='extrapolate')(k_logspaced)
    
        return k_logspaced, pk_logspaced

    @property
    def input_xi(self):
        '''log-spaced input correlation function. This is just the input power spectrum FFTlogged.
        
        Returns:
            2D array (r, xi) for the log-spaced input correlation function.
        ''' 
        k, pk = self.input_pk

        r, xi = P2xi(k)(pk)
        return r, xi

    @cached_property
    def r(self):
        ''' Log-spaced separations for the correlation function used during the analysis.
        
        Returns:
            1D array for the separations r
        '''
        return self.input_xi[0]

    @cached_property
    def k(self):
        '''Log-spaced separations for the power spectrum during the analysis.
        
        Returns:
            1D array for th separations of k
        ''' 
        return self.input_pk[0]

    def get_theory_pk(self, z, bias=None, bias2=None, lognormal=False, smooth_factor=None, tracer='dd'):
        ''' Get theoretical power spectrum by smoothing, evolving, biasing and lognormalizing the input power spectrum.

        Args:
            z (float): Redshift for the power spectrum.
            bias (float, optional): Value for the bias. (Default: Read it from the input bias).
            bias2 (float, optional): When cross-correlating fields, bias for the second field. (Default: same as bias -> auto-correlation).
            lognormal (bool, optional): Whether to apply lognormal transformation. (Default: False).
            smooth_factor (float, optional): Smoothing prefactor to apply for the power spectrum. (Default: using dd smoothing factor.)
            tracer (str, optional): Tracer to compute the power spectrum. Options:
                dd (DEFAULT): galaxy-galaxy like. It will bias the power spectrum twice (bias*bias2)
                dm: galaxy-matter like. It will bias the power spectrum once (bias) 
                mm: matter-matter like. Unbiased power spectrum.

        Returns:
            2D array (k, pk) for the power spectrum.
        '''
        k, pk = self.input_pk
        
        if smooth_factor is None:
            smooth_factor = self.smooth_factor

        if bias is None:
            bias = self.bias(z)
        
        if bias2 == None:
            bias2 = bias
        elif tracer != 'dd':  # pragma: no cover
            raise ValueError('Mixed biases only valid whith dd tracer')

        smoothing = self.r_smooth**2 + smooth_factor*(self.L_box()/self.n_grid)**2/12 + self.analysis_smoothing

        pk *= np.exp(-smoothing*k**2)
        
        if tracer == 'dd':
            bias_factor = bias*bias2
        elif tracer == 'dm':
            bias_factor = bias
        elif tracer == 'mm':
            bias_factor = 1
        else: # pragma: no cover
            raise ValueError('Invalid tracer value', tracer)

        growth_factor_factor = self.growth_factor(1/(1+z))/self.growth_factor(1)
        growth_factor_factor **=2

        evolved_pk = growth_factor_factor*pk*bias_factor

        if lognormal:
            _r, _xi = P2xi(k)(evolved_pk)
            _xi = from_xi_g_to_xi_ln(_xi)
            _k, evolved_pk = xi2P(_r)(_xi)
            np.testing.assert_almost_equal(k, _k)

        return k, evolved_pk

    def get_theory(self, z, bias=None, bias2=None, lognormal=False, smooth_factor=None, tracer='dd'): # pragma: no cover
        ''' Get theoretical correlation function by smoothing, evolving, biasing and lognormalizing the input power spectrum.

        Args:
            z (float): Redshift for the power spectrum.
            bias (float, optional): Value for the bias. (Default: Read it from the input bias).
            bias2 (float, optional): When cross-correlating fields, bias for the second field. (Default: same as bias -> auto-correlation).
            lognormal (bool, optional): Whether to apply lognormal transformation. (Default: False).
            smooth_factor (float, optional): Smoothing prefactor to apply for the power spectrum. (Default: using dd smoothing factor.)
            tracer (str, optional): Tracer to compute the power spectrum. Options:
                dd (DEFAULT): galaxy-galaxy like. It will bias the power spectrum twice (bias*bias2)
                dm: galaxy-matter like. It will bias the power spectrum once (bias) 
                mm: matter-matter like. Unbiased power spectrum.

        Returns:
            2D array (r, xi) for the power spectrum.
        '''
        pkl = self.get_theory_pk(z=z, bias=bias, bias2=bias2, lognormal=lognormal, smooth_factor=smooth_factor, tracer=tracer)
        
        _, xil = P2xi(self.k, l=0)(pkl)
        
        return xil

    def get_npole_pk(self, n, z, rsd=True, bias=None, rsd2=None, bias2=None, smooth_factor=None, smooth_factor_rsd=None, smooth_factor_cross=None):
        ''' Get the theoretical multipole power spectrum according to the model.

        Args:
            n (int): Multipole to compute.
            z (float): Redshfit to evaluate the theory on.
            rsd (bool, optional): Whether to include RSD. (Default: True).
            rsd2 (bool, optional): Only for cross-correlations. Whether to include RSD for the second field. (Default: Same as rsd -> autocorrelation).
            bias (float, optional): Force a value for bias. (Default: Compute bias from input bias file).
            bias2 (float, optional): Only for cross-correlations. Fix a value of bias for the second field. (Default: Same as bias -> autocorrelation)
            smooth_factor (float, optional): Smoothing prefactor for the lognormalized field dd (<delta_LN delta_LN>), as the 1.1 in "double rsm2_gg=par->r2_smooth+1.1*pow(par->l_box/par->n_grid,2)/12.". (Default: Value used in __init__ method).
            smooth_factor_rsd (float, optional): Smoothing prefactor for the matter matter field. <delta_L delta_L>. (Default: Value used in __init__ method).
            smooth_factor_cross (float, optional): Smoothing prefactor for the matter galaxy (dm) field. <delta_LN delta_L>. (Default: Value used in __init__ method).

        Returns:
            1D array (p_l(k)) for the multipole. If k is needed use self.k
        '''
        smooth_factor = smooth_factor if smooth_factor is not None else self.smooth_factor
        smooth_factor_rsd = smooth_factor_rsd if smooth_factor_rsd is not None else self.smooth_factor_rsd
        smooth_factor_cross = smooth_factor_cross if smooth_factor_cross is not None else self.smooth_factor_cross
        
        if bias is None:
            logger.debug('Bias is None, computing it from input file')
            bias = self.bias(z)
        if bias2 is None:
            bias2 = bias

        if rsd2 is None:
            rsd2 = rsd

        if n not in (0, 2, 4):  # pragma: no cover
            raise ValueError('multipole should be in (0, 2, 4', n)

        if self.apply_lognormal:
            logger.debug('Compute lognormalized input pk')
            pk_l = self.get_theory_pk(z, bias=bias, bias2=bias2, lognormal=True, smooth_factor=smooth_factor, tracer='dd')[1]
        else:
            pk_l = self.get_theory_pk(z, bias=bias, bias2=bias2, smooth_factor=smooth_factor, tracer='dd')[1]

        if (not rsd and not rsd2):
            logger.debug('No RSD')
            if n==0:
                logger.debug('Returning no-RSD monopole')
                return pk_l
            else:
                logger.debug(f'Returning no-RSD n-pole: {n}')
                return np.zeros_like(pk_l)


        logger.debug('RSD')
        try:
            f = self.logarithmic_growth_rate(z, read_file=True)
        except IndexError:
            logger.debug('Getting growth factor from CoLoRe files failed, computing it...')
            f = self.logarithmic_growth_rate(z, read_file=False)
        
        pk_rsd          = self.get_theory_pk(z, bias=None,  smooth_factor=smooth_factor_rsd, tracer='mm')[1]
        pk_cross_bias1  = self.get_theory_pk(z, bias=bias,  smooth_factor=smooth_factor_cross, tracer='dm')[1]
        pk_cross_bias2  = self.get_theory_pk(z, bias=bias2, smooth_factor=smooth_factor_cross, tracer='dm')[1]

        if n==0:
            logger.debug('Returning monopole')
            return pk_l + (f**2/5)*rsd*rsd2*pk_rsd + (f/3)*(rsd2*pk_cross_bias1 + rsd*pk_cross_bias2)
        if n==2:
            logger.debug('Returning quadrupole')
            return (4*f**2/7)*rsd*rsd2*pk_rsd + (2*f/3)*(rsd2*pk_cross_bias1 + rsd*pk_cross_bias2)
        if n==4:
            logger.debug('Returning hexadecapole')
            return (8*f**2/35)*rsd*rsd2*pk_rsd


    def get_npole(self, n, z, rsd=True, bias=None, rsd2=None, bias2=None, smooth_factor=None, smooth_factor_rsd=None, smooth_factor_cross=None):
        ''' Get the theoretical multipole correlation according to the model. 

        Args:
            n (int): Multipole to compute.
            z (float): Redshfit to evaluate the theory on.
            rsd (bool, optional): Whether to include RSD. (Default: True).
            rsd2 (bool, optional): Only for cross-correlations. Whether to include RSD for the second field. (Default: Same as rsd -> autocorrelation).
            bias (float, optional): Force a value for bias. (Default: Compute bias from input bias file).
            bias2 (float, optional): Only for cross-correlations. Fix a value of bias for the second field. (Default: Same as bias -> autocorrelation)
            smooth_factor (float, optional): Smoothing prefactor for the lognormalized field dd (<delta_LN delta_LN>), as the 1.1 in "double rsm2_gg=par->r2_smooth+1.1*pow(par->l_box/par->n_grid,2)/12.". (Default: Value used in __init__ method).
            smooth_factor_rsd (float, optional): Smoothing prefactor for the matter matter field. <delta_L delta_L>. (Default: Value used in __init__ method).
            smooth_factor_cross (float, optional): Smoothing prefactor for the matter galaxy (dm) field. <delta_LN delta_L>. (Default: Value used in __init__ method).

        Returns:
            1D array (xi_0(k)) for the multipole. If r is needed use self.r
        '''
        k = self.input_pk[0]

        pkl = self.get_npole_pk(n=n, z=z, rsd=rsd, bias=bias, rsd2=rsd2, bias2=bias2, smooth_factor=smooth_factor, smooth_factor_rsd=smooth_factor_rsd, smooth_factor_cross=smooth_factor_cross)

        _r, xil = P2xi(k, l=n)(pkl)
        
        return xil

    def combine_z_npoles(self, n, zbins, rsd, mode='pk', bias=None, method='Nz_file', master_file=None):
        ''' Get a prediction combining multipoles in different redshift bins.
        Args:
            n (int): Multipole to compute.
            zbins (array of doubles): Redshift bins to use.
            rsd (bool): Whether to use RSD or not.
            bias (1D array, optional): Force a value for bias for each redshfit bin (Default: get it from bias filename).
            mode (str, optional): Whether to combine correlation function (xi) or power spectra (pk). (Default: pk).
            method (str, optional): Method to extract the Nz histogram. Options:
                CoLoRe: Reads the CoLoRe output files. 
                master_file: Reads the LyaCoLoRe master file provided in master_file
                Nz_file: Reads the Nz file provided in param.cfg. (DEFAULT)
            master_file (str or Path, optional): Path to the master catalog of objects. Only for 'master_file' method, and mandatory for it. 
        '''
        logger.info(f'Combining redshifts.')

        if bias is None:
            bias = [None for i in range(len(zbins)-1)]

        z_effs = [self.get_zeff(zmin, zmax, method=method, master_file=master_file, rsd=rsd) for zmin,zmax in zip(zbins[:-1], zbins[1:])]
        z_widths = np.diff(zbins)

        logger.debug('Generating Nz distributions')
        if method == 'CoLoRe': # pragma: no cover
            Nz = self.get_nz_histogram_from_CoLoRe_box(zbins, rsd)
        elif method == 'master_file':
            if master_file is None: # pragma: no cover
                raise ValueError('master_file needs to be defined to use master_file method')
            else:
                Nz = self.get_nz_histogram_from_master_file(master_file, zbins, rsd)
        elif method == 'Nz_file': # pragma: no cover
            Nz = self.get_nz_histogram_from_Nz_file(zbins)
        else:  # pragma: no cover
            raise ValueError('Method for combining Nz should be in ("CoLoRe", "master_file", "Nz_file")')

        norm_Nz2 = 1/(z_widths*(Nz**2).sum())
        Nz2 = norm_Nz2 * Nz**2

        logger.debug('Getting npoles for each of the redshift bins')
        if mode == 'pk':
            pks = np.array([self.get_npole_pk(n=n, z=zi, rsd=rsd, bias=biasi)*Nzi*z_width for zi,Nzi,biasi,z_width in zip(z_effs, Nz2, bias,z_widths)])
        
            pk = pks.sum(axis=0)

            return pk
        elif mode == 'xi':
            xis = np.array([self.get_npole(n=n, z=zi, rsd=rsd, bias=biasi)*Nzi*z_width for zi,Nzi,biasi,z_width in zip(z_effs, Nz2, bias,z_widths)])
        
            xi = xis.sum(axis=0)

            return xi
        else: # pragma: no cover
            raise ValueError('Mode not in ("xi", "pk")')

def simple_integral(x, y):
    integrand = InterpolatedUnivariateSpline(x, y, k=1)
    return integrand.integral(x[0], x[-1])

def from_xi_g_to_xi_ln(xi):
    logger.debug('lognormalizing')
    return np.exp(xi) - 1