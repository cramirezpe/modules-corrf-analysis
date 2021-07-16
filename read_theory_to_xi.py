import logging
from functools import cached_property
from pathlib import Path

import astropy
from astropy.io import fits
import camb
import libconf
import numpy as np
from astropy import cosmology
from camb import initialpower, model
from scipy.interpolate import InterpolatedUnivariateSpline, interp1d
from mcfit import xi2P, P2xi

logger = logging.getLogger(__name__)

class ReadTheoryCoLoRe:
    _default_h = 0.7
    _default_Om0 = 0.3
    _default_Ode0 = 0.7
    _default_Ob0 = 0.05
    _default_cosmo = astropy.cosmology.LambdaCDM(_default_h*100, Om0=_default_Om0, Ode0=_default_Ode0, Ob0=_default_Ob0)

    def __init__(self, box_path, source, tracer='dd', bias_filename=None, nz_filename=None, param_cfg_filename=None, zmin=None, zmax=None):
        '''
        Tool to get theoretical values from a CoLoRe sim

        Args:
            tracer (str, optional): which tracer to use for computations (options are dd dm mm). Defaults to dd
            bias_filename (str or Path, optional): bias filename for computations with bias. Default: Searches for bias_filename in the param_cfg
            nz_filename (str or Path, optional): Nz filename. Default: Searches for nz_filename in the param_cfg
            param_cfg_filename (str or Path, optional): Path to param_cfg file. Used to search for nz_filename and bias_filename if they are not fixed. Default: globbing .cfg in the box path
            zmin (float, optional): zmin to consider when dealing with nz distribution. Defaults to zmin in paramcfg if it exists, 0 if not.
            zmax (float, optional): zmax to consider when dealing with nz distribution. Defaults to zmax in paramcfg if it exists, 1000 if not.
        '''

        if tracer not in ('dd', 'dm', 'mm'): # pragma: no cover
            raise ValueError('Tracer should be in "dd", "dm", "mm"')
        self.tracer = tracer
        self.box_path = Path(box_path)
        self.source = source

        self.param_cfg_filename = param_cfg_filename

        if bias_filename is None:
            try:
                self.bias_filename = self.param_cfg[f'srcs{self.source}']['bias_filename']
            except Exception: # pragma: no cover
                print('Failed reading bias from param.cfg')
        else:
            self.bias_filename = bias_filename

        if nz_filename is None:
            try:
                self.nz_filename = self.param_cfg[f'srcs{self.source}']['nz_filename']
            except Exception: # pragma: no cover
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
        if self.param_cfg_filename is None:
            self.param_cfg_filename = next( Path(self.box_path).glob('*.cfg') )
        
        with open(self.param_cfg_filename) as f:
            param_cfg = libconf.load(f)

        return param_cfg

    @cached_property
    def cosmo(self): # pragma: no cover
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
        cosmo = self._default_cosmo
        
        if z1==0:
            return cosmo.comoving_volume(z2)

        return cosmo.comoving_volume(max(z1,z2)) - cosmo.comoving_volume(min(z1,z2))

    def get_nz_histogram_from_Nz_file(self, bins):
        z, nz = np.loadtxt(self.nz_filename, unpack=True)
        nz_interpolation = interp1d(z, nz)

        bins_centers = 0.5*(bins[1:] + bins[:-1])
        new_nz = nz_interpolation(bins_centers)

        new_nz[bins_centers < self.zmin] = 0
        new_nz[bins_centers > self.zmax] = 0

        db = np.array(np.diff(bins), float)
        return new_nz/(new_nz*db).sum()

    def get_nz_histogram_from_CoLoRe_box(self, bins, rsd):
        '''
        Get nz histogram from CoLoRe files:

        Args:
            bins: Bin edges of the histogram
            rsd: Whether to use RSD redshifts or not
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
        master_hdul = astropy.io.fits.open(master_file)
        if rsd:
            zcat = master_hdul[1].data['Z_QSO_RSD']
        else:
            zcat = master_hdul[1].data['Z_QSO_NO_RSD']

        Nz, _ = np.histogram(zcat, bins=bins, density=True)

        return Nz

    def get_zeff(self, zmin, zmax, nbins=100, method='Nz_file', master_file=None, rsd=True):
        '''
        Get effective redsfhit for the correlation by weighting the density of pairs over the redshift range given.
        Args:
            zmin (double): Min redshift of the range
            zmax (double): Max redshift of the range
            nbins (double, optional): Number of bins in which to divide the range before integration.
            method (str, optional): Method to extract the Nz histogram. Options:
                CoLoRe: Reads the CoLoRe output files.
                master_file: Reads the LyaCoLoRe master file provided in master_file
                Nz_file: Reads the Nz file provided in param.cfg (DEFAULT)
            master_file (str or Path): Path to the master catalog of objects
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

    def combine_z_npoles(self, n, zbins, rsd, mode='pk', bias=None, method='Nz_file', master_file=None):
        '''
        Get a prediction combining multipoles in different redshift bins.
        Args:
            n (int): Multipole to compute.
            zbins (array of doubles): Redshift bins to use.
            rsd (bool): Whether to use RSD or not.
            bias (float, optional): Force a value for bias for each redshfit bin (Default: get it from bias filename).
            mode (str, optional): Whether to combine correlation function (xi) or power spectra (pk)
            method (str, optional): Method to extract the Nz histogram. Options:
                CoLoRe: Reads the CoLoRe output files. 
                master_file: Reads the LyaCoLoRe master file provided in master_file
                Nz_file: Reads the Nz file provided in param.cfg. (DEFAULT)
            master_file(Path or str): Path to the LyaCoLoRe master file if master_file mode is used.
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

    def z_bins_from_files(self):
        return np.sort(np.array( [float(x.name[18:23]) for x in self.box_path.glob(f'out_pk_srcs_pop{self.source-1}*')] ))
        # return np.fromiter( map(lambda x: float(x.name[18:23]), self.box_path.glob('out_pk*')), np.float)

    def beta_from_growth(self, z):
        '''
        Get beta(z) computing growth factor using CoLoRe param.cfg cosmology
        '''
        f = self.velocity_growth_factor(z, read_file=False)

        return f/self.bias(z)

    def velocity_growth_factor(self, z, read_file=True):
        '''
        velocity growth factor f = dlogD/da(z):
        
        Args:
            z (float): Redshift to evaluate f
            read_file (bool): If True, read growth factor from CoLoRe fits file; If False, compute f from param.cfg cosmology
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


    def beta_from_file(self, z_out):
        '''
        Extract beta(z) from file, outputing it for the z_out values given
        '''
        f = self.velocity_growth_factor(z_out, read_file=True)
        return f/self.bias(z_out)

    @cached_property
    def bias(self):
        '''
        Get bias interpolation object 
        '''
        bias_z, bias_bz = np.loadtxt(self.bias_filename, unpack=True)

        return interp1d(bias_z, bias_bz, fill_value='extrapolate')

    def get_a_eq(self): # pragma: no cover
        ''' 
        Computes and returns a_eq as computed in CoLoRe code:
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
        '''Get unnormalized growth factor for a single scale factor

        Args:
            a (float): Scale factor
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

    @cached_property
    def dlnDdlna(self): # pragma: no cover
        '''
        Read dlogDloga from file and creates an interpolation function 
        '''
        hdul = astropy.io.fits.open(self.box_path / f'out_srcs_s{self.source}_0.fits')
        z = hdul[4].data['Z']
        D = hdul[4].data['D']
        a = 1/(1+z)

    @cached_property
    def dlogDdz(self):
        '''
        Read dlogDdz from file and cretes an interpolation function 
        '''
        hdul = astropy.io.fits.open(self.box_path / f'out_srcs_s{self.source}_0.fits')
        z = hdul[4].data['Z']
        D = hdul[4].data['D']

        func = InterpolatedUnivariateSpline(z, np.log(D), k=1)
        dlogDdz = func.derivative()
        return dlogDdz
       
class ReadXiCoLoReFromPk(ReadTheoryCoLoRe):
    def __init__(self, box_path, source, tracer='dd', bias_filename=None, nz_filename=None, pk_filename=None, param_cfg_filename=None, zmin=None, zmax=None, smooth_factor=None, smooth_factor_rsd=None, apply_lognormal=True):
        super().__init__(box_path=box_path,
                        source=source,
                        tracer=tracer,
                        bias_filename=bias_filename,
                        nz_filename=nz_filename,
                        param_cfg_filename=param_cfg_filename,
                        zmin=zmin,
                        zmax=zmax)
        
        if pk_filename is None:
            try:
                self.pk_filename = self.param_cfg['global']['pk_filename']
            except Exception: # pragma: no cover
                logger.warning('Failed reading pk filename from param.cfg')
        else:
            self.pk_filename = pk_filename

        if smooth_factor is None:
            r_smooth = self.param_cfg['field_par']['r_smooth']
            n_grid   = self.param_cfg['field_par']['n_grid']
            self._smooth_factor = r_smooth**2 + 0.9*(self.L_box()/n_grid)**2/12
        else:
            self._smooth_factor = smooth_factor

        if smooth_factor_rsd is None:
            self._smooth_factor_rsd = self.smooth_factor
        else:
            self._smooth_factor_rsd = smooth_factor_rsd

        self.apply_lognormal = apply_lognormal

    @property
    def smooth_factor(self):
        return self._smooth_factor

    @property
    def smooth_factor_rsd(self):
        return self._smooth_factor_rsd
    
    @property
    def pk0(self):
        k, pk = np.loadtxt(self.pk_filename, unpack=True)
    
        return k, pk

    @property
    def xi0(self):
        k, pk = self.pk0

        r, xi = P2xi(k)(pk)
        return r, xi

    @cached_property
    def r(self):
        return self.xi0[0]

    # def get_theory(self, z, bias=None, lognormal=False): # pragma: no cover
    #     r, xi = self.xi0
        
    #     if self.tracer == 'dd':
    #         bias_factor = self.bias(z)**2 if bias is None else bias**2
    #     elif self.tracer == 'dm':
    #         bias_factor = self.bias(z) if bias is None else bias
    #     else:
    #         bias_factor = 1


    #     growth_factor_factor = self.growth_factor(1/(1+z))/self.growth_factor(1)
    #     growth_factor_factor **=2

    #     evolved_xi = growth_factor_factor*xi

    #     if lognormal:
    #         evolved_xi = from_xi_g_to_xi_ln(evolved_xi)

    #     return r, evolved_xi*bias_factor

    def get_theory_pk(self, z, bias=None, lognormal=False, smooth=None):
        k, pk = self.pk0

        if smooth is None:
            pk *= np.exp(-self.smooth_factor*k**2)
        else:
            pk *= np.exp(-smooth*k**2)
        
        if self.tracer == 'dd':
            bias_factor = self.bias(z)**2 if bias is None else bias**2
        elif self.tracer == 'dm':
            bias_factor = self.bias(z) if bias is None else bias
        else:
            bias_factor = 1


        growth_factor_factor = self.growth_factor(1/(1+z))/self.growth_factor(1)
        growth_factor_factor **=2

        evolved_pk = growth_factor_factor*pk*bias_factor

        if lognormal:
            _r, _xi = P2xi(k)(evolved_pk)
            _xi = from_xi_g_to_xi_ln(_xi)
            k, evolved_pk = xi2P(_r)(_xi)

        return k, evolved_pk

    def get_npole_pk(self, n, z, rsd=True, bias=None, smooth=None, smooth_rsd=None):
        '''
        Compute the npole for the correspondent z_bin

        Args:
            n (int): multipole
            z (float): redshift value for theory.
            rsd (bool, optional): whether to include redshift space distortions. (default: True).
            bias (float, optional): force a value of bias. (default: compute the correspondent value of bias for the redshift given.')
        '''
        smooth = smooth if smooth is not None else self.smooth_factor
        smooth_rsd = smooth_rsd if smooth_rsd is not None else self.smooth_factor_rsd
        
        if self.apply_lognormal:
            logger.debug('Compute lognormalized input pk')
            pk_l = self.get_theory_pk(z, bias=bias, lognormal=True, smooth=smooth)[1]
        else:
            pk_l = self.get_theory_pk(z, bias=bias, smooth=smooth)[1]

        if not rsd:
            logger.debug('No rsd')
            if n == 0:
                logger.debug('monopole selected, only output pk')
                return pk_l
            else:
                return np.zeros_like(pk_l)
        else:
            logger.debug('rsd')
            if bias is None:
                logger.debug('Bias is none, computing it to get beta')
                try:
                    beta = self.beta_from_file(z)
                except IndexError:
                    logger.warning('Getting growth factor from CoLoRe files failed, computing growth factor...')
                    beta = self.beta_from_growth(z)
            else:
                try: 
                    f = self.velocity_growth_factor(z, read_file=True)
                except IndexError:
                    logger.warning('Getting growth factor from CoLoRe files failed, computing growth factor...')
                    f = self.velocity_growth_factor(z, read_file=False)
                beta = f/bias
            
            logger.debug(f'beta value: {beta}')

            pk_rsd_smooth = self.get_theory_pk(z, bias=bias, smooth=smooth_rsd)[1]
            if np.isinf(beta) and (bias == 0): # pragma: no cover
                logger.debug('Bias is zero. Computing theory with only clustering from RSD')
                pk = self.get_theory_pk(z, bias=1, smooth=smooth_rsd)[1] # pk used should be the matter one, bias 1
                if n == 0:
                    return (f**2/5)*pk_rsd_smooth
                if n == 2:
                    return (4*f**2/7.0)*pk_rsd_smooth
                if n == 4:
                    return 8*beta**2/35 * pk_rsd_smooth
                else:
                    raise ValueError('n not in 0 2 4')

            if n == 0:
                logger.debug('returning monopole')
                return pk_l + (2*beta/3.0 + beta**2/5.0)*pk_rsd_smooth
            if n == 2: 
                logger.debug('returning quadrupole')
                return (4*beta/3.0 + 4*beta**2/7.0)*pk_rsd_smooth
            if n == 4:
                logger.debug('returning n=4')
                return 8*beta**2/35 * pk_rsd_smooth
            else: # pragma: no cover
                raise ValueError('n not in 0 2 4')

    def get_npole(self, n, z, rsd=True, bias=None):
        '''
        Compute the npole for the correspondent z_bin

        Args:
            n (int): multipole
            z (float): redshift value for theory.
            rsd (bool, optional): whether to include redshift space distortions. (default: True).
            bias (float, optional): force a value of bias. (default: compute the correspondent value of bias for the redshift given.')

        Returns:
            1-D array. Correlation for the given multipole
        '''
        k = self.pk0[0]

        pkl = self.get_npole_pk(n=n, z=z, rsd=rsd, bias=bias)

        _r, xil = P2xi(k, l=n)(pkl)
        # if n == 0:
        #     def jl(x):
        #         return np.sin(x)/x
        # elif n == 2:
        #     def jl(x): #negative added because later we will apply i^l
        #         return -(3/x**3 - 1/x)*np.sin(x) + 3*np.cos(x)/x**2
        # else:
        #     raise ValueError('n not in 0 2')

        # xil = []
        # for r in self.r:
        #     fk = k**2*pkl*jl(k*r)/(2*np.pi**2)
        #     integrand = InterpolatedUnivariateSpline(k, fk, k=1)
        #     xil.append(integrand.integral(k[0], k[-1]))
        
        return xil

    def L_box(self):
        cosmo = astropy.cosmology.LambdaCDM(self.cosmo['h']*100,
                                   Om0=self.cosmo['omega_M'],
                                   Ode0=self.cosmo['omega_L'], 
                                   Ob0=self.cosmo['omega_B'])
        r_max = cosmo.comoving_distance(float( self.param_cfg['global']['z_max'] ))*self.cosmo['h']
        return 2*r_max.value*(1+2/self.param_cfg['field_par']['n_grid']) 

def simple_integral(x, y):
    integrand = InterpolatedUnivariateSpline(x, y, k=1)
    return integrand.integral(x[0], x[-1])

def from_xi_g_to_xi_ln(xi):
    # return np.log(1 + xi)
    logger.debug('lognormalizing')
    return np.exp(xi) - 1