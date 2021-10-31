"""
    Document to store all the terms and to easily obtain data from them to use in other places.

    Only works at NERSC.    
""" 

from CoLoRe_corrf_analysis.plot_methods import Plots
from CoLoRe_corrf_analysis.file_funcs import FileFuncs
from CoLoRe_corrf_analysis.fitter import Fitter
from CoLoRe_corrf_analysis.read_colore import ComputeModelsCoLoRe

from pathlib import Path
from functools import cached_property
import matplotlib.pyplot as plt
import numpy as np

class Term:
    def __init__(self, basedir, label, source1, source2, rsd1, rsd2, rsd1_rev, rsd2_rev, bias1=None, bias2=None):
        self.basedir = Path(basedir)
        
        self.label = label
        
        self.source1 = source1
        self.source2 = source2
        
        self.rsd1 = rsd1
        self.rsd2 = rsd2
        
        self.rsd1_rev = rsd1_rev
        self.rsd2_rev = rsd2_rev
        
        self.bias1 = bias1
        self.bias2 = bias2

        self.datadict = dict()
        
    def get_boxes(self, rmin=0.1, rmax=200, N_bins=41, zmin=0.5, zmax=0.7, nside=2, force_path=None):
        self.zmin=zmin
        self.zmax=zmax
        rsd1_rev_str = 'n' if self.rsd1_rev else ""
        rsd2_rev_str = 'n' if self.rsd2_rev else ""
        
        if force_path != None:
            boxes_path = force_path
        elif self.source2 == None:
            boxes_path = self.basedir / f's{self.source1}{rsd1_rev_str}'
        else:
            boxes_path = self.basedir / f's{self.source1}{rsd1_rev_str}_s{self.source2}{rsd2_rev_str}'
            
        self.boxes = FileFuncs.mix_sims(
            FileFuncs.get_full_path(boxes_path,
                                    rsd=self.rsd1, rsd2=self.rsd2,
                                    rmin=rmin, rmax=rmax, N_bins=N_bins,
                                    zmin=zmin, zmax=zmax,
                                    nside=2)
        )
        
    def get_theory(self, **kwargs):
        self.theory = ComputeModelsCoLoRe(**kwargs)

    def data(self, pole):
        if pole in self.datadict.keys():
            _xi, _xierr = self.datadict[pole]
        else:
            _xi, _xierr = Plots.get_xi(pole=pole, boxes=self.boxes)
            self.datadict[pole] = (_xi, _xierr)
        return _xi, _xierr

    def xi_per_pixel(self, pole):
        xis = np.array( [box.compute_npole(pole, ) for box in self.boxes] )
        return xis
    
    def model(self, pole):
        _xi = self.theory.get_npole(n=pole, z=self.z, rsd=self.rsd1, bias=self.bias1, rsd2=self.rsd2, bias2=self.bias2, reverse_rsd=self.rsd1_rev, reverse_rsd2=self.rsd2_rev)
        return _xi

    def best_fit(self, pole, fitter):
        bias = fitter.out.params['bias']
        if self.rsd2 != None:
            bias2 = fitter.out.params['bias2']
        else: 
            bias2 = None
        _xi = self.theory.get_npole(n=pole, z=self.z, rsd=self.rsd1, bias=bias, rsd2=self.rsd2, bias2=bias2, reverse_rsd=self.rsd1_rev, reverse_rsd2=self.rsd2_rev)
        return _xi
        
    
    @cached_property
    def rdata(self):
        return self.boxes[0].savg
    
    @cached_property
    def z(self):
        return self.theory.get_zeff(self.zmin, self.zmax)
        
    @cached_property
    def rmodel(self):
        return self.theory.r
        
    def __str__(self):
        return str(self.label)

class MixedTerm(Term):
    def __init__(self, positive, negative, label=None):
        if type(positive) == Term:
            positive = [positive]
        else:
            positive = list(positive)

        if type(negative) == Term:
            negative = [negative]
        else:
            negative = list(negative)
        
        self.clear_repeated_items(positive, negative)

        self.positive = positive
        self.negative = negative

        self.theory = positive[0].theory

        if label == None:
            self.label = '+'.join(map(str, self.positive))
            if len(self.negative) > 0:
                self.label += '-'+'-'.join(map(str, self.negative))
        else:
            self.label=label

        self.datadict = dict()
    
    @cached_property
    def rdata(self):
        return self.positive[0].rdata

    def data(self, pole):
        if pole in self.datadict.keys():
            _xi, _xierr = self.datadict[pole]
        else:
            _npixels = len(self.positive[0].boxes)
            _nrdata = len(self.rdata)
            xis = np.zeros((_npixels, _nrdata))

            for _item in self.positive:
                xis += _item.xi_per_pixel(pole)
            for _item in self.negative:
                xis -= _item.xi_per_pixel(pole)

            _xi = xis.mean(axis=0)
            _xierr = xis.std(axis=0, ddof=1)/np.sqrt(_npixels)

            self.datadict[pole] = (_xi, _xierr)
        
        return _xi, _xierr

    def xi_per_pixel(self, pole):
        _npixels = len(self.positive[0].boxes)
        _nrdata = len(self.rdata)
        xis = np.zeros((_npixels, _nrdata))
        
        for _item in self.positive:
            xis += _item.xi_per_pixel(pole)
        for _item in self.negative:
            xis -= _item.xi_per_pixel(pole)

        return xis

    def model(self, pole):
        _modelxi = np.zeros_like(self.rmodel)
        
        for _item in self.positive:
            _modelxi += _item.model(pole)
        for _item in self.negative:
            _modelxi -= _item.model(pole)

        return _modelxi

    @staticmethod
    def clear_repeated_items(positive, negative):
        for _item in positive:
            _item_cleared = False
            while _item_cleared == False:
                if _item in negative:
                    positive.remove(_item)
                    negative.remove(_item)
                else:
                    _item_cleared=True

    @classmethod
    def mix_terms_from_array(cls, terms, positive_indices, negative_indices, label=None):
        positive = []
        negative = []

        [positive.append(terms[i]) for i in positive_indices]
        [negative.append(terms[i]) for i in negative_indices]

        return cls(positive, negative, label)
                    
class CombineTerms:
    @staticmethod
    def clear_repeated_items(positive, negative):
        for _item in positive:
            _item_cleared = False
            while _item_cleared == False:
                if _item in negative:
                    positive.remove(_item)
                    negative.remove(_item)
                else:
                    _item_cleared=True

    @classmethod
    def sum_data(cls, pole, positive, negative=[]):
        if type(positive) == Term:
            positive = [positive]
        else:
            positive = list(positive)
            
        if type(negative) == Term:
            negative = [negative]
        else:
            negative = list(negative)
        
        cls.clear_repeated_items(positive, negative)
        
        _dataxi = np.zeros_like(positive[0].rdata)
        _dataxierr = np.zeros_like(positive[0].rdata)
        
        for _item in positive:
            _xi, _xierr = _item.data(pole)
            _dataxi += _xi
            _dataxierr += _xierr**2
        for _item in negative:
            _xi, _xierr = _item.data(pole)
            _dataxi -= _xi
            _dataxierr += _xierr**2
        _dataxierr = np.sqrt(_dataxierr)
        
        return _dataxi, _dataxierr
    
    @classmethod
    def sum_models(cls, pole, positive, negative=[]):
        if type(positive) == Term:
            positive = [positive]
        else:
            positive = list(positive)
        
        if type(negative) == Term:
            negative = [negative]
        else:
            negative = list(negative)
            
        cls.clear_repeated_items(positive, negative)
        
        _modelxi = np.zeros_like(positive[0].rmodel)
        for _item in positive:
            _modelxi += _item.model(pole)
        for _item in negative:
            _modelxi -= _item.model(pole)
        
        return _modelxi
    
class AndreuTerms:
    @staticmethod
    def create_term(get_boxes_kwargs, get_theory_kwargs, label, source1, source2, rsd1, rsd2, bias1, bias2, rsd1_rev=False, rsd2_rev=False, basedir=Path(f'/global/cscratch1/sd/cramirez/NBodyKit/multibias2_0.5_4/')):
        term = Term(basedir, label, source1, source2, rsd1, rsd2, rsd1_rev, rsd2_rev, bias1, bias2)
        term.get_boxes(**get_boxes_kwargs)
        term.get_theory(**get_theory_kwargs)
        return term

    @classmethod
    def create_Andreu_terms(cls, basedir=Path(f'/global/cscratch1/sd/cramirez/NBodyKit/multibias2_0.5_4/'), 
                                         get_boxes_kwargs=dict(rmin=0.1, rmax=200, N_bins=41, zmin=0.5, zmax=0.7, nside=2, force_path=None),
                                         get_theory_kwargs=dict(
                                             box_path=Path('/global/cscratch1/sd/damonge/CoLoRe_sims/sim1000'),
                                             source=2,
                                             nz_filename=Path('/global/cscratch1/sd/cramirez/NBodyKit/hanyu_david_box/input_files/NzBlue.txt'),
                                             pk_filename=Path('/global/cscratch1/sd/cramirez/NBodyKit/hanyu_david_box/input_files/Pk_CAMB_test.dat'),
                                             param_cfg_filename=Path('/global/cscratch1/sd/damonge/CoLoRe_sims/sim1000/out_params.cfg'),
                                             bias_filename=Path('/global/cscratch1/sd/cramirez/NBodyKit/hanyu_david_box/input_files/BzBlue.txt'),
                                             apply_lognormal=True,
                                         ),
                                         biases_dict={1:0, 4:1, 5:2},
                           ):

        terms_pars = {
            1  : dict(source1=1, rsd1=True,  source2=None, rsd2=None,  rsd1_rev=False, rsd2_rev=False),
            2  : dict(source1=1, rsd1=True,  source2=4,    rsd2=False, rsd1_rev=False, rsd2_rev=False),
            3  : dict(source1=1, rsd1=True,  source2=4,    rsd2=True,  rsd1_rev=False, rsd2_rev=False),
            4  : dict(source1=4, rsd1=False, source2=None, rsd2=None,  rsd1_rev=False, rsd2_rev=False),
            5  : dict(source1=4, rsd1=False, source2=4,    rsd2=True,  rsd1_rev=False, rsd2_rev=False),
            6  : dict(source1=4, rsd1=True,  source2=None, rsd2=None,  rsd1_rev=False, rsd2_rev=False),
            7  : dict(source1=1, rsd1=True,  source2=4,    rsd2=True,  rsd1_rev=False, rsd2_rev=True),
            8  : dict(source1=4, rsd1=False, source2=4,    rsd2=True,  rsd1_rev=False, rsd2_rev=True),
            9  : dict(source1=4, rsd1=True,  source2=4,    rsd2=True,  rsd1_rev=False, rsd2_rev=True),
            10 : dict(source1=4, rsd1=True,  source2=None, rsd2=None,  rsd1_rev=True,  rsd2_rev=True),
        }

        terms = dict()
        for _key, _item in terms_pars.items():
            _item['bias1'] = biases_dict[_item['source1']]
            _item['bias2'] = biases_dict[_item['source2']] if _item['source2'] != None else None

            terms[_key] = cls.create_term(get_boxes_kwargs, get_theory_kwargs, label=_key, **_item, basedir=basedir)

        terms[11] = MixedTerm.mix_terms_from_array(terms, positive_indices=[3], negative_indices=[1,2], label=11)
        terms[12] = MixedTerm.mix_terms_from_array(terms, positive_indices=[5], negative_indices=[2, 4], label=12)
        terms[13] = MixedTerm.mix_terms_from_array(terms, positive_indices=[6, 2, 2, 4, 1], negative_indices=[5, 5, 3, 3], label=13)

        return terms

class TermsPlots:
    @staticmethod
    def plot_term(ax, term, poles=[0,2], c=['C0'], **kwargs):
        if type(term) != Term and type(term) != MixedTerm:
            raise TypeError('Term should be a Term or MixedTerm instance')

        if type(c) == str:
            c = [c for i in poles]
        elif len(poles) > len(c):
            [c.append(c[-1]) for i in range(len(c), len(poles))]
        elif len(poles) < len(c):
            c = c[:len(poles)]

        args = dict(label=term.label)
        args.update(kwargs)

        for pole, ci in zip(poles, c):
            _dataxi, _dataxierr = term.data(pole)
            _modelxi = term.model(pole)

            ax.plot(term.rmodel, term.rmodel**2*_modelxi, c=ci, **args)
            _label = args['label']
            args.pop('label')
            ax.errorbar(term.rdata, term.rdata**2*_dataxi, term.rdata**2*_dataxierr, c=ci, fmt='.', **args)
            args['label'] = _label


    @staticmethod
    def plot_sum(ax, terms, positive, negative=[], poles=[0,2], c=['C0'], **kwargs):
        positive = [terms[_item] for _item in positive]
        negative = [terms[_item] for _item in negative]
        if type(c) == str:
            c = [c for i in poles]
        elif len(poles) != len(c):
            c = [c[0] for i in poles]
            
        for pole, ci in zip(poles, c):
            _dataxi, _dataxierr = CombineTerms.sum_data(pole, positive, negative)
            _modelxi = CombineTerms.sum_models(pole, positive, negative)
            
            label = "+".join(map(str, positive))
            if len(negative) > 0:
                label += "-"+"-".join(map(str, negative))
            
            rmodel = positive[0].rmodel
            rdata = positive[0].rdata
            
            ax.plot(rmodel, rmodel**2*_modelxi, c=ci, label=label, **kwargs)
            ax.errorbar(rdata, rdata**2*_dataxi, rdata**2*_dataxierr, c=ci, fmt='.', **kwargs)