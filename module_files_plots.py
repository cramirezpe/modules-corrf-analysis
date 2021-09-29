from pathlib import Path
import os

from cf_helper import CFComputations
from read_theory_to_xi import ComputeModelsCoLoRe

import numpy as np
from functools import cached_property
from itertools import combinations
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from lmfit import minimize as lmfitminimize
from lmfit import Parameters
import matplotlib.pyplot as plt
from tabulate import tabulate

import logging
logger = logging.getLogger(__name__)

class FileFuncs:
    @staticmethod
    def get_full_path(basedir, rsd=True, rmin=0.1, rmax=200, zmin=0.7, zmax=0.9, nside=2, N_bins=41):
        '''Method to get the full path of a auto (not cross) correlation already existing.
        
        Args:
            rsd (bool, optional): Whether to use RSD or not. (Default: True).
            rmin (float, optional): Min. separation in the correlation output. (Default: 0.1).
            rmax (float, optional): Max. separation in the correlation output. (Default: 200).
            zmin (float, optional): Min. redshift for the correlation. (Default: 0.7).
            zmax (float, optional): Max. redshift for the correlation. (Default: 0.9).
            nside (float, optional): nside for the separation of the sky in pixels. Used to compute errorbars. (Default: 2).
            N_bins (int, optional):  Number of bins for r. (Default: 41).

        Returns:
            Path to the results for each box (multiple boxes can be combined).
        ''' 
        rsd = 'rsd' if rsd else 'norsd'
        return Path(basedir) / f'nside_{nside}' / rsd / f'{rmin}_{rmax}_{N_bins}' / f'{zmin}_{zmax}' 

    @staticmethod
    def get_available_pixels(path, boxes=None):
        '''Method to search pixels with results inside of a given auto or cross correlation path
        
        Args:
            boxes (array, optional): Array of the boxes we wan to include. (Default: Use all boxes available).

        Returns:
            1D array of Paths, pointing to each of the pixels for which there are correlations computed.
        '''
        available_pixels = []

        if boxes is None:
            boxes = [x.name for x in path.iterdir()]

        for box in boxes:
            _boxdir = path / str(box)
            for _subdir in _boxdir.iterdir():
                if (_subdir / '0_DD.dat').is_file() or (_subdir / 'DD.dat').is_file():
                    available_pixels.append(_subdir)
        return available_pixels

    @classmethod
    def mix_sims(cls, path, boxes=None, pixels=None, data_rand_ratio=1, cross_correlation=False):
        '''Method to create a CFComputations object for each of the available pixels in one or more boxes.
        
        Args:
            path (Path): Path to the boxes. It can be obtained for auto-correlations using cls.get_full_path.
            boxes (array, optional): Array of the boxes we want to include. (Default: All boxes available).
            pixels (array, optional): Array of pixels we want to include. (Default: All available pixels).
            data_rand_ratio (float, optional): Ratio data/randoms. (Default: 1).
            cross_correlation (bool, optional): Whether we are working with a cross-correlation. Needed to read RD correctly. (Default: False). 

        Returns:
            1D array of CFComputations objects. 
        '''
        boxes = [x.name for x in path.iterdir()] if boxes is None else boxes

        if pixels is None:
            paths = cls.get_available_pixels(path, boxes=boxes)
        else:
            paths = []
            for box in boxes:
                for pixel in pixels:
                    paths.append( path / str(box) / str(pixel) )

        output = []
        for _boxpath in paths:
            output.append( CFComputations(_boxpath, N_data_rand_ratio=data_rand_ratio, cross_correlation=cross_correlation) )
        return output


class Plots:
    @classmethod
    def plot_best_fit(cls, fitter, pole, ax=None, plot_args=dict()):
        '''
        fitter (module_files_plots.Fitter object): Fitter object storing the fit we want to plot results of.
        pole (int): Multipole we want to plot.
        ax (matplotlib.axes._subplots.AxesSubplot, optional): Axis to use. (Default: Create a new axis).
        plot_args (dict, optional): Extra arguments for the plotting (e.g: c='C0'). (Default: dict()).
        '''

        fitted_region = (fitter.rmin[pole], fitter.rmax[pole])
        cls.plot_theory(pole=pole, z=fitter.z, theory=fitter.theory,
         ax=ax, bias=fitter.out.params['bias'].value, 
                         smooth_factor=fitter.out.params['smooth_factor'].value, 
                         smooth_factor_rsd=fitter.out.params['smooth_factor_rsd'].value, 
                         smooth_factor_cross=fitter.out.params['smooth_factor_cross'].value, 
                         fitted_region=fitted_region, plot_args=plot_args)
           
    @staticmethod
    def plot_theory(pole, z, theory, ax=None, plot_args=dict(), bias=None, bias2=None, smooth_factor=None, smooth_factor_rsd=None, smooth_factor_cross=None, rsd=True, rsd2=None, fitted_region=(0,301)):
        ''' Plot a given model in a given axis.

        Args:
            z (float): Redshift to use for the model.
            theory (read_theory_to_xi.ComputeModelsCoLoRe object, optional): Theory object used for the theoretical model. (Default: Theory from the fitter will be used).
            ax (matplotlib.axes._subplots.AxesSubplot, optional): Axis to use. (Default: Create a new axis).
            plot_args (dict, optional): Extra arguments for the plotting (e.g: c='C0'). (Default: dict()).
            bias (float, optional): Value for the bias. (Default: Read it from the input bias).
            bias2 (float, optional): When cross-correlating fields, bias for the second field. 
            smooth_factor (float, optional): Smoothing prefactor for the lognormalized field dd (<delta_LN delta_LN>), as the 1.1 in "double rsm2_gg=par->r2_smooth+1.1*pow(par->l_box/par->n_grid,2)/12.". (Default: Value used in __init__ method).
            smooth_factor_rsd (float, optional): Smoothing prefactor for the matter matter field. <delta_L delta_L>. (Default: Value used in __init__ method).
            smooth_factor_cross (float, optional): Smoothing prefactor for the matter galaxy (dm) field. <delta_LN delta_L>. (Default: Value used in __init__ method).
            rsd (bool, optional): Whether to include RSD. (Default: True).
            rsd2 (bool, optional): Only for cross-correlations. Whether to include RSD for the second field. (Default: Same as rsd -> autocorrelation).
            fitted_region (tuple, optional): Mark fitted region (rfit_min, rfit_max) with solid line.
        '''
        if ax is None:
            fig, ax = plt.subplots()
        plot_args = { **dict(c='C1'), **plot_args }
        
        # if bias is None:
        #     bias = theory.bias(z)

        xi_th = np.asarray(theory.get_npole(n=pole, z=z, bias=bias, bias2=bias2, rsd=rsd, rsd2=rsd2, smooth_factor=smooth_factor, smooth_factor_rsd=smooth_factor_rsd, smooth_factor_cross=smooth_factor_cross))

        msk = theory.r < 301
        msk_fitted = theory.r > fitted_region[0]
        msk_fitted &= theory.r < fitted_region[1]

        dashed_plot_args = dict(plot_args)
        dashed_plot_args['label'] = None
        dashed_plot_args['ls'] = '--'

        ax.plot(theory.r[msk], (theory.r[msk])**2*xi_th[msk], **dashed_plot_args)
        ax.plot(theory.r[msk_fitted], theory.r[msk_fitted]**2*xi_th[msk_fitted], **plot_args)
        ax.set_xlabel(r'$r \, [{\rm Mpc/h}]$')
        ax.set_ylabel(r'$r^2 \xi(r)$')

    @staticmethod
    def get_xi(pole, boxes):
        ''' Get Xi and errorbars for a given set of data boxes
        
        Args:
            boxes (1D array of cf_helper.CFComputations objects): Boxes with the data information.

        Returns:
            2D array (xi, xierr) with correlation and correlation error.
        '''
        xis = np.array( [box.compute_npole(pole, ) for box in boxes] ) 
        xi = xis.mean(axis=0)
        xierr = xis.std(axis=0, ddof=1)/np.sqrt(len(boxes))
        return xi, xierr

    @classmethod
    def plot_data(cls, pole, boxes, ax=None, plot_args=dict(), delta_r=0):
        '''Plot data for the given boxes in an axis.
        
        Args:
            pole (int): Multipole to plot.
            boxes (1D array of cf_helper.CFComputations objects): Boxes with the data information.
            ax (matplotlib.axes._subplots.AxesSubplot, optional): Axis to use. (Default: Create a new axis).
            plot_args (dict, optional): Extra arguments for the plotting (e.g: c='C0'). (Default: dict()).
            delta_r (float, optional): Displace theory by delta_r (for multiplotting). (Default: 0).
        ''' 
        if ax is None:
            fig, ax = plt.subplots()
        
        plot_args = { **dict(fmt='.', c='C0'), **plot_args}
        xi, xierr = cls.get_xi(pole, boxes)

        box = boxes[0]
        if delta_r != 0:
            delta_r = delta_r*np.diff(box.savg)
            delta_r = np.append(delta_r, delta_r[-1])
        ax.errorbar(box.savg+delta_r, box.savg**2*xi, box.savg**2*xierr, **plot_args)      
        
        ax.set_xlabel(r'$r \, [{\rm Mpc/h}]$')
        ax.set_ylabel(r'$r^2 \xi(r)$')

def from_xi_g_to_xi_ln(xi):
    # return np.log(1 + xi)
    return np.exp(xi) - 1

class Fitter:
    def __init__(self, boxes, z, poles, theory, rsd, bias0=None, smooth_factor0=None, smooth_factor_rsd0=None, smooth_factor_cross0=None, rmin=None, rmax=None):
        '''Fitter class used to fit bias (only for auto-correlations) and smooth_factors
        
        Args:
            boxes (1D array of cf_helper.CFComputations objects): Boxes with the data information.
            z (float): Redshift to use for the model.
            poles (array of int): Multipoles to use for the fitter.
            theory (read_theory_to_xi.ComputeModelsCoLoRe object, optional): Theory object used for the theoretical model.
            rsd (bool): Whether to use RSD or not.
            bias0 (float, optional): Initial value for bias. (Default: Read it from input bias).
            smooth_factor0 (float, optional): Initial value for smooth_factor. (Default: Use the one from ComputeModelsCoLoRe.__init__).
            smooth_factor_rsd0 (float, optional): Initial value for smooth_factor_rsd. (Default: Use the one from ComputeModelsCoLoRe.__init__).
            smooth_factor_cross0 (float, optional): Initial value for smooth_factor_cross. (Default: Use the one from ComputeModelsCoLoRe.__init__).
            rmin (dict, optional): Min r for the fits. (Default: {0:10, 2:40}, 10 for the monopole, 40 for the quadrupole).
            rmax (dict, optional): Max r for the fits. (Default: {0:200, 2:200}, 200 both for monopole and quadrupole).
        '''
        self.boxes  = boxes
        self.z      = z
        self.poles  = poles
        
        self.rsd    = rsd
        self.theory = theory

        if bias0 is None:
            self.bias0  = theory.bias(z)
        else:
            self.bias0 = bias0

        if smooth_factor0 is None:
            self.smooth_factor0 = theory.smooth_factor
        else:
            self.smooth_factor0 = smooth_factor0

        if smooth_factor_rsd0 is None:
            self.smooth_factor_rsd0 = theory.smooth_factor_rsd
        else:
            self.smooth_factor_rsd0 = smooth_factor_rsd0

        if smooth_factor_cross0 is None:
            self.smooth_factor_cross0 = theory.smooth_factor_cross
        else:
            self.smooth_factor_cross0 = smooth_factor_cross0

        self.r = self.boxes[0].savg
        self.rmin = rmin if rmin is not None else {0:10, 2:40}
        self.rmax = rmax if rmax is not None else {0:200, 2:200}

        self.masks = dict()
        for _pole in poles:
            self.masks[_pole] = (self.r > self.rmin[_pole]) & (self.r < self.rmax[_pole])

    @cached_property
    def xis(self):
        '''Method to combine data from all the different boxes.
        
        Returns: 
            2D array with the correlation for each box.
        '''
        xis = dict()
        for pole in self.poles:
            xis[pole] = np.array( [box.compute_npole(pole) for box in self.boxes] )
        return xis

    @cached_property
    def data(self):
        '''Method to obtain the correlation mean for all the different boxes.
        
        Returns:
            1D array with the correlation.
        '''
        data_ = np.array([])
        for _pole in self.poles:
            data_ = np.append(data_, self.xis[_pole].mean(axis=0)[self.masks[_pole]])
        return data_

    @cached_property
    def err(self):
        '''Method to obtain the correlation error for all the different boxes.
        
        Returns:
            1D array with the error.
        '''
        err_ = np.array([])
        for _pole in self.poles:
            err_ = np.append(err_, self.xis[_pole].std(axis=0, ddof=1)[self.masks[_pole]]/len(self.boxes))
        return err_

    def model(self, bias, smooth_factor, smooth_factor_rsd, smooth_factor_cross, pole):
        '''Method to get an interpolation object with a given model.
        
        Args:
            bias (float): Bias to use for the model.
            smooth_factor (float, optional): Smoothing prefactor for the lognormalized field dd (<delta_LN delta_LN>), as the 1.1 in "double rsm2_gg=par->r2_smooth+1.1*pow(par->l_box/par->n_grid,2)/12.".
            smooth_factor_rsd (float, optional): Smoothing prefactor for the matter matter field. <delta_L delta_L>.
            smooth_factor_cross (float, optional): Smoothing prefactor for the matter galaxy (dm) field. <delta_LN delta_L>.
            pole (int): Multipole to compute.
            
        Returns:
            interp1d object with a the model.
        '''
        xi = self.theory.get_npole(n=pole, z=self.z, bias=bias, rsd=self.rsd, smooth_factor=smooth_factor, smooth_factor_rsd=smooth_factor_rsd, smooth_factor_cross=smooth_factor_cross)
        try:
            model_xi = interp1d(self.theory.xi0[0], xi)
        except AttributeError:
            model_xi = interp1d(self.theory.r, xi)
        return model_xi(self.r)

    def residual(self, params):
        '''Compute residual for a given parameters.
        
        Args:
            params (Parameters): Parameters to compute the model with.
            
        Returns:
            The residual float.
        ''' 
        _model = np.array([])
        for _pole in self.poles:
            _model = np.append(_model, self.model(params['bias'], params['smooth_factor'], params['smooth_factor_rsd'], params['smooth_factor_cross'], _pole)[self.masks[_pole]])

        return (self.data -_model) / self.err

    def run_fit(self, free_params):
        '''
            Run the fit with a certain number of free parameters. Initial guess given during the initialization of the class.

            Args:
                free_params (list of str): List with the fields to set free (bias, smooth_factor and smooth_factor_rsd are the options).

            Returns:
                lmfit minimize output. Which is also stored in self.out.
        '''
        assert isinstance(free_params, list) # I need a certain order in the free_params list for this method to work

        defaults = dict(
            bias = self.bias0,
            smooth_factor = self.smooth_factor0,
            smooth_factor_rsd = self.smooth_factor_rsd0,
            smooth_factor_cross = self.smooth_factor_cross0
        )

        for i in free_params:
            assert i in ('bias', 'smooth_factor', 'smooth_factor_rsd', 'smooth_factor_cross') 

        params = Parameters()
        params.add('bias', value=defaults['bias'], min=0, vary='bias' in free_params)
        params.add('smooth_factor', value=defaults['smooth_factor'], min=0, vary='smooth_factor' in free_params)
        params.add('smooth_factor_rsd', value=defaults['smooth_factor_rsd'], min=0, vary='smooth_factor_rsd' in free_params)
        params.add('smooth_factor_cross', value=defaults['smooth_factor_cross'], min=0, vary='smooth_factor_cross' in free_params)

        self.out = lmfitminimize(self.residual, params)
        return self.out

    def pars_tab(self):
        '''Get a tabulate table with the results of the fit.'''
        headers = ['name', 'value', 'stderr', 'stderror(%)', 'init value', 'min', 'max', 'vary']

        rows = []
        for parname in self.out.params:
            par = self.out.params[parname]
            row = []
            row.append(par.name)
            row.append(round(float(par.value),3))
            if par.stderr is None:
                row.append('')
                row.append('')
            elif par.value == 0:
                row.append(round(float(par.stderr), 3))
                row.append(0)
            else:
                row.append(round(float(par.stderr),3))
                row.append(round(float(par.stderr/par.value)*100,3))
            row.append(par.init_value)
            row.append(par.min)
            row.append(par.max)
            row.append(par.vary)
            rows.append(row)

        return tabulate(rows, headers=headers, tablefmt='github', numalign='decimal', stralign='left')

    def corrs_tab(self):
        '''Get a tabulate table with the correlations within parameters of the fit.'''
        headers = ['name', 'name', 'corr']
        rows = []
        
        _vars = self.out.var_names
        _var_nums = [i for i in range(len(_vars))]

        for pair in combinations(_var_nums, 2):
            i, j = pair
            try:
                correlation = self.out.covar[i][j]
                correlation /= np.sqrt(self.out.covar[i][i])
                correlation /= np.sqrt(self.out.covar[j][j])

                rows.append([_vars[i], _vars[j], correlation])
            except AttributeError:
                rows.append([_vars[i], _vars[j], None])
        
        return tabulate(rows, headers=headers, tablefmt='github', numalign='decimal', stralign='left')
       
    def nu(self):
        nu_ = len(self.data)
        return nu_