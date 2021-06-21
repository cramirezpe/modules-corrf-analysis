from pathlib import Path
import os

from cf_helper import CFComputations
from read_theory_to_xi import ReadXiCoLoRe, ReadPkCoLoRe, CAMBCorrelation, ReadXiCoLoReFromPk

import numpy as np
from functools import cached_property
from scipy.interpolate import interp1d
from scipy.optimize import minimize
import matplotlib.pyplot as plt

import logging
logger = logging.getLogger(__name__)

class FileFuncs:
    @staticmethod
    def get_full_path(basedir, rsd=True, rmin=0.1, rmax=200, zmin=0.7, zmax=0.9, nside=2):
        rsd = 'rsd' if rsd else 'norsd'
        return Path(basedir) / f'nside_{nside}' / rsd / f'{rmin}_{rmax}' / f'{zmin}_{zmax}' 

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

class fit_bias:
    def __init__(self, boxes, z, poles, theory, bias0=3.5):
        assert isinstance(list(poles), list)

        self.boxes = boxes
        self.z = z
        self.poles = poles
        self.bias0 = bias0
        self.theory = theory

        r = self.boxes[0].savg
        mask = r > 50
        mask &= r < 150
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
        data_ = dict()
        for pole in self.poles:
            data_[pole] = self.xis[pole].mean(axis=0)[self.mask]
        return data_

    @cached_property
    def err(self):
        err_ = dict()
        for pole in self.poles:
            err_[pole] = self.xis[pole].std(axis=0, ddof=1)[self.mask]/len(self.boxes)
        return err_

    @cached_property
    def f(self):
        try:
            return self.theory.velocity_growth_factor(z=self.z, read_file=True)
        except IndexError:
            return self.theory.velocity_growth_factor(z=self.z, read_file=False)

    def model(self, bias , pole):
        beta = self.f/bias
        xi = self.theory.get_npole(n=pole, z=self.z, beta=beta)
        try:
            model_xi = interp1d(self.theory.pk0[0], xi)
        except AttributeError:
            model_xi = interp1d(self.theory.r, xi)
        return model_xi(self.r)

    def chi_i(self, pole, bias):
        model_= self.model(bias, pole)
        
        if len(self.boxes) == 1:
            return (self.data[pole]-model_)
        else:
            return (self.data[pole]-model_)/self.err[pole]

    def chi2(self, bias):
        chi2_ = 0
        for pole in self.poles:
            x = self.chi_i(pole, bias)
            chi2_ += (x**2).sum()
        return chi2_

    def run_fit(self):
        self.results = minimize(self.chi2, x0=self.bias0)
                                
    def nu(self):
        nu_ = 0
        for pole in self.poles:
            nu_ += len(self.data[pole])
        return nu_

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
    def plot_theory(pole, z, theory, ax=None, plot_args=dict(), bias=None, rsd=True, apply_lognormal=False):
        if ax is None:
            fig, ax = plt.subplots()
        plot_args = { **dict(c='C1'), **plot_args }
        
        # if bias is None:
        #     bias = theory.bias(z)

        xi_th = np.asarray(theory.get_npole(n=pole, z=z, bias=bias, rsd=rsd))
        if apply_lognormal:
            xi_th = np.asarray(from_xi_g_to_xi_ln(xi_th))

        msk = theory.r < 301
        ax.plot(theory.r[msk], (theory.r[msk])**2*xi_th[msk], **plot_args)
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
    def plot_data(pole, boxes, ax=None, plot_args=dict(), rsd=True, delta_r=0):
        if ax is None:
            fig, ax = plt.subplots()
        
        plot_args = { **dict(fmt='.', c='C0'), **plot_args}
        xis = np.array( [box.compute_npole(pole, ) for box in boxes] )  # @maybe I should include more parameters here.       
        xi = xis.mean(axis=0)
        xierr = xis.std(axis=0, ddof=1)/np.sqrt(len(boxes))

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