import numpy as np
import matplotlib.pyplot as plt

import logging
logger = logging.getLogger(__name__)

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

        bias=fitter.out.params['bias'].value
        if fitter.cross:
            bias2 = fitter.out.params['bias2'].value
        else:
            bias2 = bias
        cls.plot_theory(pole=pole, z=fitter.z, theory=fitter.theory,
         ax=ax, bias=bias, bias2=bias2,
                        rsd=fitter.rsd, rsd2=fitter.rsd2,
                         smooth_factor=fitter.out.params['smooth_factor'].value, 
                         smooth_factor_rsd=fitter.out.params['smooth_factor_rsd'].value, 
                         smooth_factor_cross=fitter.out.params['smooth_factor_cross'].value, 
                         fitted_region=fitted_region, plot_args=plot_args,
                         reverse_rsd=fitter.reverse_rsd, reverse_rsd2=fitter.reverse_rsd2)
           
    @staticmethod
    def plot_theory(pole, z, theory, ax=None, plot_args=dict(), bias=None, bias2=None, smooth_factor=None, smooth_factor_rsd=None, smooth_factor_cross=None, rsd=True, rsd2=None, fitted_region=(0,301), reverse_rsd=False, reverse_rsd2=False):
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
            reverse_rsd (bool, optional): Reverse redshift (rsd terms will be negative). (Default: False)
            reverse_rsd2 (bool, optional): Reverse redshift for second field in cross-correlations (rsd terms negative). (Default: False)
        '''
        if ax is None:
            fig, ax = plt.subplots()
        plot_args = { **dict(c='C1'), **plot_args }
        
        # if bias is None:
        #     bias = theory.bias(z)

        xi_th = np.asarray(theory.get_npole(n=pole, z=z, bias=bias, bias2=bias2, rsd=rsd, rsd2=rsd2, smooth_factor=smooth_factor, smooth_factor_rsd=smooth_factor_rsd, smooth_factor_cross=smooth_factor_cross, reverse_rsd=reverse_rsd, reverse_rsd2=reverse_rsd2))

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