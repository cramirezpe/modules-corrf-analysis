import logging
from functools import cached_property
from pathlib import Path

import astropy
import camb
import libconf
import numpy as np
from astropy import cosmology
from camb import initialpower, model
from scipy.interpolate import InterpolatedUnivariateSpline, interp1d
from mcfit import xi2P, P2xi

logger = logging.getLogger(__name__)

def get_theory(z, source, box_path):
    '''Get theory from pk_srcs files into numpy arrays'''
    ks, pdd, pdm, pmm =  np.loadtxt(box_path / f'out_pk_srcs_pop{source-1}_z{z:.3f}.txt', unpack=True)
    return ks, pdd, pdm, pmm

r_values = np.array([  2.59875,   7.59625,  12.59375,  17.59125,  22.58875,  27.58625,
        32.58375,  37.58125,  42.57875,  47.57625,  52.57375,  57.57125,
        62.56875,  67.56625,  72.56375,  77.56125,  82.55875,  87.55625,
        92.55375,  97.55125, 102.54875, 107.54625, 112.54375, 117.54125,
       122.53875, 127.53625, 132.53375, 137.53125, 142.52875, 147.52625,
       152.52375, 157.52125, 162.51875, 167.51625, 172.51375, 177.51125,
       182.50875, 187.50625, 192.50375, 197.50125])

def from_pk_to_correlation(k, pk, r_values=r_values):
    result = []
    for r in r_values:
        values = k**2*pk*np.sin(k*r)/(k*r*2*np.pi**2)
        integrand = InterpolatedUnivariateSpline(k, values, k=1)
        result.append(integrand.integral(k[0], k[-1]))
    return result

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

        if tracer not in ('dd', 'dm', 'mm'):
            raise ValueError('Tracer should be in "dd", "dm", "mm"')
        self.tracer = tracer
        self.box_path = Path(box_path)
        self.source = source

        self.param_cfg_filename = param_cfg_filename

        if bias_filename is None:
            try:
                self.bias_filename = self.param_cfg[f'srcs{self.source}']['bias_filename']
            except Exception:
                print('Failed reading bias from param.cfg')
        else:
            self.bias_filename = bias_filename

        if nz_filename is None:
            try:
                self.nz_filename = self.param_cfg[f'srcs{self.source}']['nz_filename']
            except Exception:
                print('Failed reading nz from param.cfg')
        else:
            self.nz_filename = nz_filename

        if zmin is None:
            if self.param_cfg is None:
                self.zmin = 0
            else:
                self.zmin = self.param_cfg['global']['z_min']
        else:
            self.zmin = zmin
        if zmax is None:
            if self.param_cfg is None:
                self.zmax = 1000
            else:
                self.zmax = self.param_cfg['global']['z_max']
        else:
            self.zmax = zmax
            
    @cached_property
    def param_cfg(self):
        if self.param_cfg_filename is None:
            self.param_cfg_filename = next( Path(self.box_path).glob('*.cfg') )
        
        with open(self.param_cfg_filename) as f:
            param_cfg = libconf.load(f)

        return param_cfg

    def get_parameter_from_cfg(self, parameter):
        print('deprecated method get_parameter_from_cfg')
        if self.param_cfg is None:
            if self.param_cfg_filename is None:
                self.param_cfg_filename = next( Path(self.box_path).glob('*.cfg') )
            
            with open(self.param_cfg_filename) as f:
                self.param_cfg = libconf.load(f)

        return self.param_cfg[f'srcs{self.source}'][parameter]

    @cached_property
    def cosmo(self):
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

    def combine_z(self, z=None):
        '''
        Args: 
            z (array, optional): Array of redshift bins to combine. Default to all the redshift bins available for the box.
        '''
        if z is None:
            z = list( map(lambda x: float(x.name[18:23]), self.box_path.glob('out_pk*')) )

        z = np.sort(z)

        total_volume = self.get_volume_between_zs(z[-1])
        bin_width = z[1] - z[0]

        pk = 0  
        for zi in z:
            zmin = zi
            zmax = zi + bin_width
            
            volume = self.get_volume_between_zs(zmax, zmin)

            ks, pki = self.get_theory(zi)
            pk += pki*volume
        
        pk /= total_volume
        return pk

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

    def get_zeff(self, z=None, method='CoLoRe', master_file=None, rsd=True):
        '''
        Get effective redsfhit for the correlation by weighting the density of pairs over all redshift
        Args:
            z (array, optional): Array of redshift bins to combine. Default to all the redshift bins available for the box.
            method (str, optional): Method to extract the Nz histogram. Options:
                CoLoRe: Reads the CoLoRe output files. (DEFAULT)
                master_file: Reads the LyaCoLoRe master file provided in master_file
                Nz_file: Reads the Nz file provided in param.cfg
            master_file (str or Path): Path to the master catalog of objects
        '''
        logger.debug('Defining bins')

        if z is None:
            z = self.z_bins_from_files()
        if len(z) == 1:
            return z

        bin_width = z[1] - z[0]
        bins = np.concatenate((z,[z[-1]+bin_width]))

        logger.debug('Generating Nz')
        if method == 'CoLoRe':
            Nz = self.get_nz_histogram_from_CoLoRe_box(bins, rsd)
        elif method == 'master_file':
            if master_file is None:
                raise ValueError('master_file needs to be defined to use master_file method')
            else:
                Nz = self.get_nz_histogram_from_master_file(master_file, bins, rsd)
        elif method == 'Nz_file':
            Nz = self.get_nz_histogram_from_Nz_file(bins)
        else: 
            raise ValueError('Method for combining Nz should be in ("CoLoRe", "master_file", "Nz_file")')

        norm_Nz2 = 1/(bin_width*(Nz**2).sum())
        Nz2 = norm_Nz2 * Nz**2

        zs = np.array([zi*Nzi for zi, Nzi in zip(z, Nz2)])

        return zs.sum(axis=0) * bin_width

    def combine_z_using_Nz(self, rsd, n_pole, z=None, bias=None, method='CoLoRe', master_file=None):
        '''
        Combine multiple redshift bins using the distribution of objects on redshift as weight. Nz computed from catalog.
        Args:
            master_file (str or Path): Path to the master catalog of objects
            rsd (bool): Whether to read Z data with/without rsd
            n_pole (int): npole to compute
            z (array, optional): Array of redshift bins to combine. Default to all the redshift bins available for the box.
            method (str, optional): Method to extract the Nz histogram. Options:
                CoLoRe: Reads the CoLoRe output files. (DEFAULT)
                master_file: Reads the LyaCoLoRe master file provided in master_file
                Nz_file: Reads the Nz file provided in param.cfg
        '''
        logger.info(f'Combining redshifts.')
        if z is None:
            z = self.z_bins_from_files()
        if len(z) == 1:
            return self.get_npole(n=n_pole, z=z, rsd=rsd)

        if bias is None:
            bias = [None for i in z]

        bin_width = z[1] - z[0]
        bins = np.concatenate((z,[z[-1]+bin_width]))

        logger.debug('Generating Nz')
        if method == 'CoLoRe':
            Nz = self.get_nz_histogram_from_CoLoRe_box(bins, rsd)
        elif method == 'master_file':
            if master_file is None:
                raise ValueError('master_file needs to be defined to use master_file method')
            else:
                Nz = self.get_nz_histogram_from_master_file(master_file, bins, rsd)
        elif method == 'Nz_file':
            Nz = self.get_nz_histogram_from_Nz_file(bins)
        else: 
            raise ValueError('Method for combining Nz should be in ("CoLoRe", "master_file", "Nz_file")')

        norm_Nz2 = 1/(bin_width*(Nz**2).sum())
        Nz2 = norm_Nz2 * Nz**2

        logger.debug('Retreiving npoles for each of the redshift bins')
        pks = np.array([self.get_npole(n=n_pole, z=zi, rsd=rsd, bias=biasi)*Nzi for zi,Nzi,biasi in zip(z, Nz2, bias)])
        

        pk = pks.sum(axis=0) * bin_width

        return pk

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
            else:
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

    def get_a_eq(self):
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
        def dum(a):
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
    def dlnDdlna(self):
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

class ReadPkCoLoRe(ReadTheoryCoLoRe):         
    def get_theory(self, z, bias=None):
        ks, pdd, pdm, pmm = np.loadtxt(self.box_path / f'out_pk_srcs_pop{self.source-1}_z{z:.3f}.txt', unpack=True)
        if self.tracer == 'dd':
            if bias is None:
                return ks, pdd
            else:
                return ks, pmm*bias**2
        if self.tracer == 'dm':
            if bias is None:
                return ks, pdm
            else:
                return ks, pmm*bias
        if self.tracer == 'mm':
            return ks, pmm

    @cached_property
    def k(self):
        return self.get_theory(0)[0]

    def get_npole(self, n, rsd=True, z=None, beta=None):
        raise ValueError('get npole not supported for pk')


class ReadXiCoLoRe(ReadTheoryCoLoRe):
    def get_theory(self, z, bias=None):
        logger.debug(f'Getting theory for z: {z} and bias: {bias}')
        ks, pdd, pdm, pmm = np.loadtxt(self.box_path / f'out_xi_srcs_pop{self.source-1}_z{z:.3f}.txt', unpack=True)
        if self.tracer == 'dd':
            if bias is None:
                return ks, pdd
            else:
                return ks, pmm*bias**2
        if self.tracer == 'dm':
            if bias is None:
                return ks, pdm
            else:
                return ks, pmm*bias
        if self.tracer == 'mm':
            return ks, pmm

    @cached_property
    def r(self):
        return self.get_theory(0)[0]

    def get_npole(self, n, z, rsd=True, bias=None):
        '''
        Compute the npole from the correspondend z_bin

        Args:
            n (int): multipole
            z (float, optional): redshift value for theory.
            bias (float, optional): force a value of bias. If not set the value of bias will be extracted from bias_file and from Pk.
        '''
        xi = self.get_theory(z, bias=bias)[1]

        if not rsd:
            if n == 0:
                return xi
            else:
                return np.zeros_like(xi)
        else:
            if bias is None:
                try:
                    beta = self.beta_from_file(z)
                except IndexError:
                    logger.warning('Getting beta from CoLoRe files failed, computing growth factor...')
                    beta = self.beta_from_growth(z)
            else:
                try:
                    f = self.velocity_growth_factor(z, read_file=True)
                except IndexError:
                    f = self.velocity_growth_factor(z, read_file=False)
                beta = f/bias

            if np.isinf(beta) and (bias == 0):
                xi = self.get_theory(z, bias=1)[1] # xi used should be the matter one, bias 1
                if n == 0:
                    return (f**2/5)*xi
                if n == 2:
                    xibar = self.get_xibar(xi)
                    return (4*f**2/7.0)*(xi - xibar)
                if n == 4:
                    xibar = self.get_xibar(xi)
                    xibarbar = self.get_xibarbar(xi)
                    return 8*beta**2/35 * (xi + 5*xibar/2.0 - 7*xibarbar/2.0)
                else:
                    raise ValueError('n not in 0 2 4')

            if n == 0:
                return (1 + 2*beta/3.0 + beta**2/5.0)*xi
            if n == 2:
                xibar = self.get_xibar(xi)
                return (4*beta/3.0 + 4*beta**2/7.0)*(xi - xibar)
            if n == 4:
                xibar = self.get_xibar(xi)
                xibarbar = self.get_xibarbar(xi)
                return 8*beta**2/35 * (xi + 5*xibar/2.0 - 7*xibarbar/2.0)
            else:
                raise ValueError('n not in 0 2 4')

    def get_xibar(self, xi):
        integrals = []
        r = np.concatenate(([0],self.r)) #adding an initial value of 0, not well studied!!! WARNING!
        xi = np.concatenate(([0], xi))
        for i in range(1, len(r)):
            i+=1 #To start with an array of 2 and end with an array like self.r
            integrals.append( simple_integral(r[:i],  xi[:i]*r[:i]**2))
        return 3*np.array(integrals)/self.r**3

    def get_xibarbar(self, xi):
        integrals = []
        r = np.concatenate(([0], self.r))
        xi = np.concatenate(([0], xi))
        for i in range(1, len(r)):
            i += 1 
            integrals.append( simple_integral(r[:i], xi[:i]*r[:i]**4))
        return 5*np.array(integrals)/self.r**5
        
class ReadXiCoLoReFromPk(ReadXiCoLoRe):
    def __init__(self, box_path, source, tracer='dd', bias_filename=None, nz_filename=None, pk_filename=None, param_cfg_filename=None, zmin=None, zmax=None, smooth_factor=None, apply_lognormal=True):
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
            except Exception:
                logger.warning('Failed reading pk filename from param.cfg')
        else:
            self.pk_filename = pk_filename

        self.smooth_factor = smooth_factor
        self.apply_lognormal = apply_lognormal
    
    @cached_property
    def pk0(self):
        k, pk = np.loadtxt(self.pk_filename, unpack=True)

        if self.smooth_factor is None:
            r_smooth = self.param_cfg['field_par']['r_smooth']
            n_grid   = self.param_cfg['field_par']['n_grid']
            self.smooth_factor = r_smooth + (self.L_box()/n_grid)**2/12
    
        return k, pk*np.exp(-self.smooth_factor*k**2)

    @cached_property
    def xi0(self):
        k, pk = self.pk0

        r, xi = P2xi(k)(pk)
        return r, xi


    @cached_property
    def r(self):
        return self.xi0[0]

    def get_theory(self, z, bias=None, lognormal=False):
        r, xi = self.xi0
        
        if self.tracer == 'dd':
            bias_factor = self.bias(z)**2 if bias is None else bias**2
        elif self.tracer == 'dm':
            bias_factor = self.bias(z) if bias is None else bias
        else:
            bias_factor = 1


        growth_factor_factor = self.growth_factor(1/(1+z))/self.growth_factor(1)
        growth_factor_factor **=2

        evolved_xi = growth_factor_factor*xi

        if lognormal:
            evolved_xi = from_xi_g_to_xi_ln(evolved_xi)

        return r, evolved_xi*bias_factor

    def get_theory_pk(self, z, bias=None, lognormal=False):
        k, pk = self.pk0
        
        if self.tracer == 'dd':
            bias_factor = self.bias(z)**2 if bias is None else bias**2
        elif self.tracer == 'dm':
            bias_factor = self.bias(z) if bias is None else bias
        else:
            bias_factor = 1


        growth_factor_factor = self.growth_factor(1/(1+z))/self.growth_factor(1)
        growth_factor_factor **=2

        evolved_pk = growth_factor_factor*pk

        if lognormal:
            _r, _xi = P2xi(k)(evolved_pk)
            _xi = from_xi_g_to_xi_ln(_xi)
            k, evolved_pk = xi2P(_r)(_xi)

        return k, evolved_pk*bias_factor

    def get_npole_pk(self, n, z, rsd=True, bias=None):
        '''
        Compute the npole for the correspondent z_bin

        Args:
            n (int): multipole
            z (float): redshift value for theory.
            rsd (bool, optional): whether to include redshift space distortions. (default: True).
            bias (float, optional): force a value of bias. (default: compute the correspondent value of bias for the redshift given.')
        '''
        pk = self.get_theory_pk(z, bias=bias)[1]
        if self.apply_lognormal:
            logger.debug('Compute lognormalized input pk')
            pk_l = self.get_theory_pk(z, bias=bias, lognormal=True)[1]
        else:
            pk_l = pk

        if not rsd:
            logger.debug('No rsd')
            if n == 0:
                logger.debug('monopole selected, only output pk')
                return pk_l
            else:
                return np.zeros_like(pk)
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

            if np.isinf(beta) and (bias == 0):
                logger.debug('Bias is zero. Computing theory with only clustering from RSD')
                pk = self.get_theory_pk(z, bias=1)[1] # pk used should be the matter one, bias 1
                if n == 0:
                    return (f**2/5)*pk
                if n == 2:
                    return (4*f**2/7.0)*pk
                if n == 4:
                    return 8*beta**2/35 * pk
                else:
                    raise ValueError('n not in 0 2 4')

            if n == 0:
                logger.debug('returning monopole')
                return pk_l + (2*beta/3.0 + beta**2/5.0)*pk
            if n == 2: 
                logger.debug('returning quadrupole')
                return (4*beta/3.0 + 4*beta**2/7.0)*pk
            if n == 4:
                logger.debug('returning n=4')
                return 8*beta**2/35 * pk
            else:
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

class CAMBCorrelation(ReadXiCoLoRe):
    def __init__(self, box_path, source, bias_filename=None, nz_filename=None, param_cfg_filename=None, output_save_location=None, zmin=None, zmax=None):
        logger.info(f'Initializing CAMBCorrelation instance for box {box_path}')
        super().__init__(box_path=box_path,
                        source=source,
                        tracer='dd',
                        bias_filename=bias_filename,
                        nz_filename=nz_filename,
                        param_cfg_filename=param_cfg_filename,
                        zmin=zmin,
                        zmax=zmax)

        self.output_save_location = output_save_location
        
        if logging.root.level <= logging.DEBUG:
            logger.debug('Cosmological parameters:')
            for key in self.cosmo.keys():
                logger.debug(f'\t{key}\t{self.cosmo[key]}')

        h = self.cosmo['h']
        self.pars = camb.CAMBparams()
        if False:
            self.pars.set_cosmology(H0=67.31, ombh2=0.022, omch2=0.122)
            self.pars.InitPower.set_params(ns=0.965)
        else:
            self.pars.set_cosmology(H0=h*100, 
                ombh2=self.cosmo['omega_B']*h**2, 
                omch2=(self.cosmo['omega_M'] - self.cosmo['omega_B'])*h**2,
                # omk=self.cosmo['omega_K'],
                # TCMB=self.cosmo['T_CMB'],
                )
            self.pars.InitPower.set_params(ns=self.cosmo['ns'])

            # default_AS = self.pars.InitPower.ScalarPowerAmp[0]
            # self.pars.set_matter_power(redshifts=[0.0], kmax=100.0)
            # default_results = camb.get_results(self.pars)
            # default_sigma_8 = default_results.get_sigma8()[0]
            # new_As = default_AS*((sigma_8/default_sigma_8)**2)
            # self.pars.InitPower.set_params(As=new_As)

    @cached_property
    def r(self):
        return super().get_theory(0)[0]

    def get_theory(self, z, bias=None):
        if self.output_save_location is None:
            _, xi = self.get_theory_CAMB(z)
        else:
            try:
                if bias is None:
                    xi = np.load( Path(self.output_save_location / f'{z:.3f}.npy') )
                else:
                    xi = np.load( Path(self.output_save_location / f'{z:.3f}_{bias:.3f}.npy'))
                logger.info(f'Obtaining theoretical values from file. z = {z}')
            except FileNotFoundError:
                logger.info(f'File not found, computing new theoretical values. z = {z}. b = {bias}')
                _, xi = self.get_theory_CAMB(z, bias=bias)
                np.save( Path(self.output_save_location / f'{z:.3f}.npy'), xi)

        return np.array((self.r, xi))

    def get_theory_CAMB(self, z, return_pk=False, bias=None):
        '''
            Get theory predicted by CAMB (correlation)

            Args:
                z (double): Redshift of the prediction
                return_pk (bool, optional): Return power spectrum. Default: False
        '''
        logger.debug('Setting matter power spectrum')
        self.pars.set_matter_power(redshifts=[z], kmax=50)

        #Linear spectra
        logger.debug('Getting results from parameters')
        self.pars.NonLinear = model.NonLinear_none
        self.results = camb.get_results(self.pars)
        
        logger.debug(f'Getting matter power spectrum for redshift {z}')
        kh, z, pk = self.results.get_matter_power_spectrum(minkh=1e-4, maxkh=10, npoints = 9999)

        if bias is None:
            bias = self.bias(z)
        pk *= bias**2

        logger.debug('Converting power spectrum into correlation')
        if return_pk:
            return np.array((kh, pk[0,:]))
        else:
            return np.array((self.r, from_pk_to_correlation(kh, pk[0,:], self.r)))

    def clean_results(self):
        if self.output_save_location is None:
            raise ValueError('No output save location defined')

        for file in self.output_save_location.glob('*.npy'):
            file.unlink(missing_ok=True)


def simple_integral(x, y):
    integrand = InterpolatedUnivariateSpline(x, y, k=1)
    return integrand.integral(x[0], x[-1])

def from_xi_g_to_xi_ln(xi):
    # return np.log(1 + xi)
    logger.debug('lognormalizing')
    return np.exp(xi) - 1