from pathlib import Path
import os

from cf_helper import CFComputations
from read_theory_to_xi import ReadXiCoLoReFromPk

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
        rsd = 'rsd' if rsd else 'norsd'
        return Path(basedir) / f'nside_{nside}' / rsd / f'{rmin}_{rmax}_{N_bins}' / f'{zmin}_{zmax}' 

    @staticmethod
    def get_available_pixels(path, boxes=None):
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
    def mix_sims(cls, path, boxes=None, pixels=None, data_rand_ratio=1):
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
            output.append( CFComputations(_boxpath, N_data_rand_ratio=data_rand_ratio) )
        return output


class Plots:
    @staticmethod
    def plot_results_of_fit(fit, boxes, pole, title='', theory=None, ax=None, delta_r=0):
        logger.info('Better to use plot_theory with bias value defined')
        if ax is None:
            fig, ax = plt.subplots()

        box = boxes[0]
        z = fit.z
        bias = fit.results.x[0]
        if theory is None:
            theory = fit.theory

        # fig, axs = plt.subplots(ncols=2)
        
        theory_r = theory.r
        f = fit.f
        beta = f/bias
        
        xi_th = theory.get_npole(n=pole, z=z, beta=beta)
        ax.plot(theory_r, theory_r**2*xi_th, label=f'bias {bias}')
        xi_th = interp1d(theory_r, xi_th)(box.savg)*box.savg**2
        
        xis = np.array( [box.compute_npole(pole)*box.savg**2 for box in boxes] )
        xi = np.mean(xis, axis=0)
        xierr = xis.std(axis=0, ddof=1)
        xierr /= np.sqrt(len(xierr))
        
        ax.errorbar(box.savg+delta_r, xi, xierr, fmt='.')
        # axs[1].errorbar(box.savg, xi/xi_th-1, xierr/xi_th, fmt='.')
        # axs[1].set_ylim(-0.5, 0.5)
        # axs[1].hlines(0, -1, 200, colors='k', lw=1)
        
        ax.set_ylabel(r'$r^2 \xi(r)$')
        # axs[1].set_ylabel(r'$\xi(r)/\xi_{th}(r)-1$')
        ax.set_title(title)
        # axs[1].set_title(title + ' ratio')
        ax.set_xlabel(r'$r [Mpc/h]$')

        ax.set_xlim(-5, 200)
        # ax.legend()

    @staticmethod
    def plot_theory(pole, z, theory, ax=None, plot_args=dict(), bias=None, smooth_factor=None, smooth_factor_rsd=None, smooth_factor_cross=None, rsd=True, apply_lognormal=False, fitted_region=(0,301)):
        if ax is None:
            fig, ax = plt.subplots()
        plot_args = { **dict(c='C1'), **plot_args }
        
        # if bias is None:
        #     bias = theory.bias(z)

        xi_th = np.asarray(theory.get_npole(n=pole, z=z, bias=bias, rsd=rsd, smooth_factor=smooth_factor, smooth_factor_rsd=smooth_factor_rsd, smooth_factor_cross=smooth_factor_cross))
        if apply_lognormal:
            xi_th = np.asarray(from_xi_g_to_xi_ln(xi_th))

        msk = theory.r < 301
        msk_fitted = theory.r > fitted_region[0]
        msk_fitted &= theory.r < fitted_region[1]

        dashed_plot_args = dict(plot_args)
        dashed_plot_args['label'] = None
        dashed_plot_args['ls'] = '--'
        
        ax.plot(theory.r[msk], (theory.r[msk])**2*xi_th[msk], **dashed_plot_args)
        ax.plot(theory.r[msk_fitted], theory.r[msk_fitted]**2*xi_th[msk_fitted], **plot_args)
        ax.set_xlabel(r'$r [Mpc/h]$')
        ax.set_ylabel(r'$r^2 \xi(r)$')

    @staticmethod
    def plot_theory_mix_z(pole, z_array, theory, ax=None, plot_args=dict(), bias=None, rsd=True, apply_lognormal=False):
        if ax is None:
            fig, ax = plt.subplots()
        plot_args = { **dict(c='C1'), **plot_args }

        xi_th = theory.combine_z_using_Nz(rsd=rsd, n_pole=pole, z=z_array, bias=bias)
        if apply_lognormal:
            xi_th = from_xi_g_to_xi_ln(xi_th)
        try:
            ax.plot(theory.pk0[0], theory.pk0[0]**2*xi_th, **plot_args) #compatibility with theories from pk
        except AttributeError:
            ax.plot(theory.r, theory.r**2*xi_th, **plot_args)
        ax.set_xlabel(r'$r [Mpc/h]$')
        ax.set_ylabel(r'$r^2 \xi(r)$')

    @staticmethod
    def get_xi(pole, boxes):
        xis = np.array( [box.compute_npole(pole, ) for box in boxes] ) 
        xi = xis.mean(axis=0)
        xierr = xis.std(axis=0, ddof=1)/np.sqrt(len(boxes))
        return xi, xierr

    @classmethod
    def plot_data(cls, pole, boxes, ax=None, plot_args=dict(), delta_r=0):
        if ax is None:
            fig, ax = plt.subplots()
        
        plot_args = { **dict(fmt='.', c='C0'), **plot_args}
        xi, xierr = cls.get_xi(pole, boxes)

        box = boxes[0]
        if delta_r != 0:
            delta_r = delta_r*np.diff(box.savg)
            delta_r = np.append(delta_r, delta_r[-1])
        ax.errorbar(box.savg+delta_r, box.savg**2*xi, box.savg**2*xierr, **plot_args)      
        
        ax.set_xlabel(r'$r [Mpc/h]$')
        ax.set_ylabel(r'$r^2 \xi(r)$')

def from_xi_g_to_xi_ln(xi):
    # return np.log(1 + xi)
    return np.exp(xi) - 1

class Fitter:
    def __init__(self, boxes, z, poles, theory, rsd, bias0=None, smooth_factor0=None, smooth_factor_rsd0=None, smooth_factor_cross0=None, rmin=50, rmax=150):
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

        r = self.boxes[0].savg
        mask = r > rmin
        mask &= r < rmax
        self.r = r[mask]
        self.mask = mask

    @cached_property
    def xis(self):
        xis = dict()
        for pole in self.poles:
            xis[pole] = np.array( [box.compute_npole(pole) for box in self.boxes] )
        return xis

    @cached_property
    def data(self):
        data_ = np.array([])
        for _pole in self.poles:
            data_ = np.append(data_, self.xis[_pole].mean(axis=0)[self.mask])
        return data_

    @cached_property
    def err(self):
        err_ = np.array([])
        for _pole in self.poles:
            err_ = np.append(err_, self.xis[_pole].std(axis=0, ddof=1)[self.mask]/len(self.boxes))
        return err_

    def model(self, bias, smooth_factor, smooth_factor_rsd, smooth_factor_cross, pole):
        xi = self.theory.get_npole(n=pole, z=self.z, bias=bias, rsd=self.rsd, smooth_factor=smooth_factor, smooth_factor_rsd=smooth_factor_rsd, smooth_factor_cross=smooth_factor_cross)
        try:
            model_xi = interp1d(self.theory.xi0[0], xi)
        except AttributeError:
            model_xi = interp1d(self.theory.r, xi)
        return model_xi(self.r)

    def chi_i(self, pole, bias, smooth_factor, smooth_factor_rsd, smooth_factor_cross):
        model_= self.model(bias, smooth_factor, smooth_factor_rsd, smooth_factor_cross, pole)
        
        if len(self.boxes) == 1:
            return (self.data[pole]-model_)
        else:
            return (self.data[pole]-model_)/self.err[pole]

    def residual(self, params):
        _model = np.array([])
        for _pole in self.poles:
            _model = np.append(_model, self.model(params['bias'], params['smooth_factor'], params['smooth_factor_rsd'], params['smooth_factor_cross'], _pole))

        return (self.data -_model) / self.err

    def run_fit(self, free_params):
        '''
            Run the fit with a certain number of free parameters. Initial guess given during the initialization of the class.

            Args:
                free_params (list of str): List with the fields to set free (bias, smooth_factor and smooth_factor_rsd are the options).

            Returns:
                Run lmfit minimize and store output in self.out
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

    def corr_coeff(self):
        _vars = self.out.var_names
        _var_nums = [i for i in range(len(_vars))]

        print('Correlation coefficient:\n----------------------')
        for pair in combinations(_var_nums, 2):
            i, j = pair
            try:
                correlation = self.out.covar[i][j]
                correlation /= np.sqrt(self.out.covar[i][i])
                correlation /= np.sqrt(self.out.covar[j][j])

                print(f'{_vars[i]}\t{_vars[j]}\t{correlation}')
            except AttributeError:
                return
       
    def nu(self):
        nu_ = len(self.data)
        return nu_