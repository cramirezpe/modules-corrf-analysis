import numpy as np
import matplotlib.pyplot as plt

import logging

logger = logging.getLogger(__name__)


class Plots:
    @classmethod
    def plot_best_fit(cls, fitter, pole, ax=None, plot_args=dict(), no_labels=False):
        """
        fitter (module_files_plots.Fitter object): Fitter object storing the fit we want to plot results of.
        pole (int): Multipole we want to plot.
        ax (matplotlib.axes._subplots.AxesSubplot, optional): Axis to use. (Default: Create a new axis).
        plot_args (dict, optional): Extra arguments for the plotting (e.g: c='C0'). (Default: dict()).
        """

        fitted_region = (fitter.rmin[pole], fitter.rmax[pole])
        shown_region = (fitter.r[0], fitter.r[-1])

        bias = fitter.out.params["bias"].value
        if fitter.cross:
            bias2 = fitter.out.params["bias2"].value
        else:
            bias2 = bias
        cls.plot_theory(
            pole=pole,
            z=fitter.out.params["z"].value,
            theory=fitter.theory,
            ax=ax,
            bias=bias,
            bias2=bias2,
            rsd=fitter.rsd,
            rsd2=fitter.rsd2,
            smooth_factor=fitter.out.params["smooth_factor"].value,
            smooth_factor_rsd=fitter.out.params["smooth_factor_rsd"].value,
            smooth_factor_cross=fitter.out.params["smooth_factor_cross"].value,
            scale_factor=fitter.out.params["scale_factor"].value,
            fitted_region=fitted_region,
            shown_region=shown_region,
            plot_args=plot_args,
            reverse_rsd=fitter.reverse_rsd,
            reverse_rsd2=fitter.reverse_rsd2,
            no_labels=no_labels,
        )

    @classmethod
    def plot_best_fit_pk(cls, fitter, pole, ax=None, plot_args=dict(), no_labels=False, kmin=None, kmax=None):
        """
        fitter (module_files_plots.Fitter object): Fitter object storing the fit we want to plot results of.
        pole (int): Multipole we want to plot.
        ax (matplotlib.axes._subplots.AxesSubplot, optional): Axis to use. (Default: Create a new axis).
        plot_args (dict, optional): Extra arguments for the plotting (e.g: c='C0'). (Default: dict()).
        """

        fitted_region = (fitter.kmin.get(pole, None), fitter.kmax.get(pole, None))
        shown_region = (
            fitter.k[0] if kmin is None else kmin, 
            fitter.k[-1] if kmax is None else kmax,
        )

        bias = fitter.out.params["bias"].value
        if fitter.cross:
            bias2 = fitter.out.params["bias2"].value
        else:
            bias2 = bias
        cls.plot_theory_pk(
            pole=pole,
            z=fitter.out.params["z"].value,
            theory=fitter.theory,
            ax=ax,
            bias=bias,
            bias2=bias2,
            rsd=fitter.rsd,
            rsd2=fitter.rsd2,
            smooth_factor=fitter.out.params["smooth_factor"].value,
            smooth_factor_rsd=fitter.out.params["smooth_factor_rsd"].value,
            smooth_factor_cross=fitter.out.params["smooth_factor_cross"].value,
            scale_factor=fitter.out.params["scale_factor"].value,
            fitted_region=fitted_region,
            shown_region=shown_region,
            plot_args=plot_args,
            reverse_rsd=fitter.reverse_rsd,
            reverse_rsd2=fitter.reverse_rsd2,
            no_labels=no_labels,
        )

    @staticmethod
    def plot_theory(
        pole,
        z,
        theory,
        ax=None,
        plot_args=dict(),
        bias=None,
        bias2=None,
        smooth_factor=None,
        smooth_factor_rsd=None,
        smooth_factor_cross=None,
        scale_factor=1,
        rsd=True,
        rsd2=None,
        fitted_region=(0, 301),
        shown_region=(0, 301),
        reverse_rsd=False,
        reverse_rsd2=False,
        no_labels=False,
    ):
        """Plot a given model in a given axis.

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
        """
        if ax is None:
            fig, ax = plt.subplots()
        plot_args = {**dict(c="C1"), **plot_args}

        # if bias is None:
        #     bias = theory.bias(z)

        xi_th = np.asarray(
            scale_factor
            * theory.get_npole(
                n=pole,
                z=z,
                bias=bias,
                bias2=bias2,
                rsd=rsd,
                rsd2=rsd2,
                smooth_factor=smooth_factor,
                smooth_factor_rsd=smooth_factor_rsd,
                smooth_factor_cross=smooth_factor_cross,
                reverse_rsd=reverse_rsd,
                reverse_rsd2=reverse_rsd2,
            )
        )

        msk = (theory.r < shown_region[1]) & (theory.r > shown_region[0])
        msk_fitted = msk & (theory.r > fitted_region[0])
        msk_fitted &= theory.r < fitted_region[1]

        dashed_plot_args = dict(plot_args)
        dashed_plot_args["label"] = None
        dashed_plot_args["ls"] = "--"

        ax.plot(theory.r[msk], (theory.r[msk]) ** 2 * xi_th[msk], **dashed_plot_args)
        ax.plot(
            theory.r[msk_fitted],
            theory.r[msk_fitted] ** 2 * xi_th[msk_fitted],
            **plot_args
        )
        if not no_labels:
            ax.set_xlabel(r"$r \, [{\rm Mpc/h}]$")
            ax.set_ylabel(r"$r^2 \xi(r)$")

        return theory.r[msk], theory.r[msk] ** 2 * xi_th[msk]

    @staticmethod
    def plot_theory_pk(
        pole,
        z,
        theory,
        ax=None,
        plot_args=dict(),
        bias=None,
        bias2=None,
        smooth_factor=None,
        smooth_factor_rsd=None,
        smooth_factor_cross=None,
        scale_factor=1,
        rsd=True,
        rsd2=None,
        fitted_region=(None, None),
        shown_region=(0.01, 2),
        reverse_rsd=False,
        reverse_rsd2=False,
        no_labels=False,
    ):
        if ax is None:
            fig, ax = plt.subplots()
        plot_args = {**dict(c="C1"), **plot_args}

        # if bias is None:
        #     bias = theory.bias(z)

        pk_th = np.asarray(
            scale_factor
            * theory.get_npole_pk(
                n=pole,
                z=z,
                bias=bias,
                bias2=bias2,
                rsd=rsd,
                rsd2=rsd2,
                smooth_factor=smooth_factor,
                smooth_factor_rsd=smooth_factor_rsd,
                smooth_factor_cross=smooth_factor_cross,
                reverse_rsd=reverse_rsd,
                reverse_rsd2=reverse_rsd2,
            )
        )

        msk = (theory.k < shown_region[1]) & (theory.k > shown_region[0])
        
        if fitted_region[0] is None or fitted_region[1] is None:
            msk_fitted = np.zeros_like(msk, dtype=bool)
        else:
            msk_fitted = msk & (theory.k > fitted_region[0])
            msk_fitted &= theory.k < fitted_region[1]
        
        dashed_plot_args = dict(plot_args)
        dashed_plot_args["label"] = None
        dashed_plot_args["ls"] = "--"

        ax.plot(theory.k[msk], pk_th[msk], **dashed_plot_args)
        ax.plot(
            theory.k[msk_fitted],
            pk_th[msk_fitted],
            **plot_args
        )
        if not no_labels:
            ax.set_xlabel(r"$k \, [{\rm h/Mpc}]$")
            ax.set_ylabel(r"$P(k)$")

        ax.set_xscale('log')
        ax.set_yscale('log')

        return theory.k[msk], pk_th[msk]

    @staticmethod
    def get_xi(pole, boxes, jacknife=False):
        """Get Xi and errorbars for a given set of data boxes

        Args:
            boxes (1D array of cf_helper.CFComputations objects): Boxes with the data information.

        Returns:
            2D array (xi, xierr) with correlation and correlation error.
        """
        xis = np.array(
            [
                box.compute_npole(
                    pole,
                )
                for box in boxes
            ]
        )

        if not jacknife:
            xi = xis.mean(axis=0)
            xierr = xis.std(axis=0, ddof=1) / np.sqrt(len(boxes))
        else:
            jacknife_xis = []
            for i in range(len(xis)):
                reduced_xis = np.delete(xis, i, axis=0)
                jacknife_xis.append(reduced_xis.mean(axis=0))
            jacknife_xis = np.asarray(jacknife_xis)
            xi = jacknife_xis.mean(axis=0)
            xierr = (len(boxes) - 1) * ((jacknife_xis - xi) ** 2).mean(axis=0)
            xierr = np.sqrt(xierr)

        # xierr = xis.std(axis=0) # This is the same error that David uses.
        return xi, xierr

    @classmethod
    def plot_data(
        cls,
        pole,
        boxes,
        ax=None,
        plot_args=dict(),
        delta_r=0,
        shaded_errors=False,
        shaded_errors_args=dict(facecolor="#AAAAAA"),
        no_labels=False,
        error_rescaling=1,
        jacknife=False,
    ):
        """Plot data for the given boxes in an axis.

        Args:
            pole (int): Multipole to plot.
            boxes (1D array of cf_helper.CFComputations objects): Boxes with the data information.
            ax (matplotlib.axes._subplots.AxesSubplot, optional): Axis to use. (Default: Create a new axis).
            plot_args (dict, optional): Extra arguments for the plotting (e.g: c='C0'). (Default: dict()).
            delta_r (float, optional): Displace theory by delta_r (for multiplotting). (Default: 0).
        """
        if ax is None:
            fig, ax = plt.subplots()

        xi, xierr = cls.get_xi(pole, boxes, jacknife=jacknife)
        xierr *= error_rescaling

        box = boxes[0]
        if delta_r != 0:
            delta_r = delta_r * np.diff(box.savg)
            delta_r = np.append(delta_r, delta_r[-1])
        if shaded_errors:
            if delta_r != 0:
                raise ValueError("delta_r incompatible with shaded erros")
            else:
                ax.fill_between(
                    box.savg,
                    box.savg ** 2 * (xi + xierr),
                    box.savg ** 2 * (xi - xierr),
                    **shaded_errors_args
                )
        else:
            plot_args = {**dict(fmt=".", c="C0"), **plot_args}
            ax.errorbar(
                box.savg + delta_r,
                box.savg ** 2 * xi,
                box.savg ** 2 * xierr,
                **plot_args
            )

        if not no_labels:
            ax.set_xlabel(r"$r \, [{\rm Mpc/h}]$")
            ax.set_ylabel(r"$r^2 \xi(r)$")
