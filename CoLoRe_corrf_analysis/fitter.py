from CoLoRe_corrf_analysis.read_colore import ComputeModelsCoLoRe

from functools import cached_property
from itertools import combinations
from scipy.interpolate import interp1d
from lmfit import minimize as lmfitminimize
from lmfit import Parameters
from tabulate import tabulate
import numpy as np

import logging

logger = logging.getLogger(__name__)


class FitterBase:
    def __init__(
        self,
        boxes,
        z,
        poles,
        theory,
        rsd,
        rsd2=None,
        rmin=None,
        rmax=None,
        reverse_rsd=False,
        reverse_rsd2=False,
    ):
        """Fitter class used to fit bias and smooth_factors

        Args:
            boxes (1D array of cf_helper.CFComputations objects): Boxes with the data information.
            z (float): Redshift to use for the model.
            poles (array of int): Multipoles to use for the fitter.
            theory (read_theory_to_xi.ComputeModelsCoLoRe object, optional): self.theory object used for the theoretical model.
            rsd (bool): Whether to use RSD or not.
            rsd2 (bool): For cross-correlations, use RSD for second field. (Default: None -> auto-correlation)
            rmin (dict, optional): Min r for the fits. (Default: {0:10, 2:40}, 10 for the monopole, 40 for the quadrupole).
            rmax (dict, optional): Max r for the fits. (Default: {0:200, 2:200}, 200 both for monopole and quadrupole).
            reverse_rsd (bool, optional): Reverse redshift (rsd terms will be negative). (Default: False)
            reverse_rsd2 (bool, optional): Reverse redshift for second field in cross-correlations (rsd terms negative). (Default: False)
        """
        self.boxes = boxes
        self.z0 = z
        self.poles = poles

        self.rsd = rsd
        self.rsd2 = rsd2
        self.reverse_rsd = reverse_rsd
        self.reverse_rsd2 = reverse_rsd2

        if self.rsd2 != None:
            logger.info("Second RSD provided. Cross-correlation mode enabled")
            self.cross = True
        else:
            self.cross = False

        self.theory = theory

        self.data_dict = None
        self.err_dict = None

    def data(self, pole):
        """Method to obtain the data for all the different boxes.

        Returns:
            1D array with data for given Npole
        """
        if self.data_dict is None:
            self.data_dict = dict()
            for _pole in self.poles:
                self.data_dict[_pole] = np.array(
                    [box.compute_npole(_pole) for box in self.boxes]
                ).mean(axis=0)

        return self.data_dict[pole]

    def err(self, pole):
        """Method to obtain the correlation error for all the different boxes.

        Returns:
            1D array with the error.
        """
        if self.err_dict is None:
            self.err_dict = dict()
            for _pole in self.poles:
                if len(self.boxes) == 1:
                    self.err_dict[_pole] = np.ones_like(self.data(_pole))
                else:
                    self.err_dict[_pole] = np.array(
                        [box.compute_npole(_pole) for box in self.boxes]
                    ).std(axis=0, ddof=1) / len(
                        self.boxes
                    )
        
        return self.err_dict[pole]

    def model(self, pole, params: dict):
        y = self.get_th_npole(
            n=pole,
            **{key: params[key] for key in params if key != "scale_factor"}, 
            rsd=self.rsd,
            rsd2=self.rsd2,
            reverse_rsd=self.reverse_rsd,
            reverse_rsd2=self.reverse_rsd2,
        )*params.get('scale_factor', 1)

        model = interp1d(self.th_x, y)
        return model(self.x)

    def best_model(self, pole):
        return self.model(
            pole=pole, 
            params={key: value.value for key, value in self.out.params.items()}
        )

    def get_residual(self):
        _data = np.array([])
        _err = np.array([])
        for _pole in self.poles:
            _data = np.append(_data, self.data(_pole)[self.masks[_pole]])
            _err = np.append(_err, self.err(_pole)[self.masks[_pole]])

        def residual(params):
            """Compute residual for a given parameters.

            Args:
                params (Parameters): Parameters to compute the model with.

            Returns:
                The residual float.
            """
            _model = np.array([])
            for _pole in self.poles:
                _model = np.append(
                    _model,
                    self.model(_pole, params)[self.masks[_pole]],
                )

            return (_data - _model) / _err

        return residual

    @property
    def default_parameters(self):
        return {
            "z": {
                "value": self.z0,
                "min": 0,
                "vary": False,
            },
            "smooth_factor": {
                "value": self.theory.smooth_factor,
                "min": 0,
                "vary": False,
            },
            "smooth_factor_rsd": {
                "value": self.theory.smooth_factor_rsd,
                "min": 0,
                "vary": False,
            },
            "smooth_factor_cross": {
                "value": self.theory.smooth_factor_cross,
                "min": 0,
                "vary": False,
            },
            "bias": {
                "value": self.theory.bias(self.z0),
                "min": 0,
                "vary": False,
            },
            "bias2": {
                "value": None if self.cross else self.theory.bias(self.z0),
                "min": 0,
                "max": 40,
                "vary": False,
            },
            "scale_factor": {
                "value": 1,
                "min": 0,
                "vary": False,
            },
        }

    def run_fit(self, nan_policy: str="raise", **fit_params):
        """
        Run the fit with a certain number of free parameters. Initial guess given during the initialization of the class.

        Args:
            nan_policy: Policy to handle nans (use omit to ignore them). See
                https://lmfit.github.io/lmfit-py/faq.html#i-get-errors-from-nan-in-my-fit-what-can-i-do
            **fit_params (Dict): Parameters to send to the fit. The format is a dict where each 
            key corresponds to a kwarg to be sent to lmfit.Parameter init. See method .default_parameters
            for reference.

        Returns:
            lmfit minimize output. Which is also stored in self.out.
        """
        params = Parameters()

        for key, defaults in self.default_parameters.items():
            if key == "bias2" and not self.cross:
                continue
            
            values = {**defaults, **fit_params.get(key, dict())}
            values["name"] = key
            params.add(**values)

        for key, values in fit_params.items():
            if key not in self.default_parameters.keys():
                values["name"] = key
                params.add(**values)

        residual = self.get_residual()

        self.out = lmfitminimize(residual, params, nan_policy=nan_policy)
        return self.out

    def pars_tab(self):  # pragma: no cover
        """Get a tabulate table with the results of the fit."""
        headers = [
            "name",
            "value",
            "stderr",
            "stderror(%)",
            "init value",
            "min",
            "max",
            "vary",
        ]

        rows = []
        for parname in self.out.params:
            par = self.out.params[parname]
            row = []
            row.append(par.name)
            row.append(round(float(par.value), 3))
            if par.stderr is None:
                row.append("")
                row.append("")
            elif par.value == 0:
                row.append(round(float(par.stderr), 3))
                row.append(0)
            else:
                row.append(round(float(par.stderr), 3))
                row.append(round(float(par.stderr / par.value) * 100, 3))
            row.append(par.init_value)
            row.append(par.min)
            row.append(par.max)
            row.append(par.vary)
            rows.append(row)

        return tabulate(
            rows,
            headers=headers,
            tablefmt="github",
            numalign="decimal",
            stralign="left",
        )

    def corrs_tab(self):  # pragma: no cover
        """Get a tabulate table with the correlations within parameters of the fit."""
        headers = ["name", "name", "corr"]
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

        return tabulate(
            rows,
            headers=headers,
            tablefmt="github",
            numalign="decimal",
            stralign="left",
        )


class Fitter(FitterBase):
    def __init__(self, *args, rmin=None, rmax=None, **kwargs):
        super().__init__(*args, **kwargs)

        self.r = self.x = self.boxes[0].savg
        self.rmin = rmin if rmin is not None else {0: 10, 2: 40}
        self.rmax = rmax if rmax is not None else {0: 200, 2: 200}

        self.masks = dict()
        for _pole in self.poles:
            self.masks[_pole] = (self.r > self.rmin[_pole]) & (
                self.r < self.rmax[_pole]
            )

        self.get_th_npole = self.theory.get_npole
        self.th_x = self.theory.r


class FitterPK(FitterBase):
    def __init__(self, *args, kmin=None, kmax=None, **kwargs):
        super().__init__(*args, **kwargs)

        self.k = self.x = self.boxes[0].k
        self.kmin = kmin if kmin is not None else {0: 0.01, 2: 0.01}
        self.kmax = kmax if kmax is not None else {0: 0.2, 2: 0.2}

        self.masks = dict()
        for _pole in self.poles:
            self.masks[_pole] = (self.k > self.kmin[_pole]) & (
                self.k < self.kmax[_pole]
            )

        self.get_th_npole = self.theory.get_npole_pk
        self.th_x = self.theory.k
