from CoLoRe_corrf_analysis.plot_methods import Plots
from CoLoRe_corrf_analysis.file_funcs import FileFuncs
from CoLoRe_corrf_analysis.fitter import Fitter
from CoLoRe_corrf_analysis.read_colore import ComputeModelsCoLoRe
from CoLoRe_corrf_analysis.rsd_antirsd_terms import Term, TermsPlots, AndreuTerms, CombineTerms

import argparse
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import sys

logger = logging.getLogger(__name__)

def getArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-files', type=Path, nargs=2, default=[Path('app1_RSD_terms_plots.pdf'), Path('app1_RSD_terms_estimates.pdf')], help='Output for the plots: 1. Plot of the pieces in the model. 2. Plot of the parts our model doesnt have')
    parser.add_argument('--log-level', default='WARNING', choices=['CRITICAL', 'ERROR', 
    'WARNING', 'INFO', 'DEBUG', 'NOTSET'])
    parser.add_argument('--show-plots', action='store_true')

    args = parser.parse_args()
    return args

def main(args=None):
    if args is None:
        args= getArgs()

    level = logging.getLevelName(args.log_level)
    logging.basicConfig(stream=sys.stdout, level=level, format='%(levelname)s:%(name)s:%(funcName)s:%(message)s')

    ###############
    # Possibility to hide warnings
    ###############
    if False:
        logger.info('Warnings will be disabled')
        import warnings
        warnings.filterwarnings('ignore')

    ###############
    # Generating all terms
    ###############
    logger.info("Generating terms (as defined in Andreu's PDF")
    terms = AndreuTerms.create_Andreu_terms()

    # Description for the terms we will use (to make labels)
    term_description = {
        1  : r'(a) $\langle \delta_A^s ~ \delta_A^s  \rangle$',
        4  : r'(b) $\langle \delta_B ~ \delta_B  \rangle$',
        6  : r'(c) $\langle \delta_B^s ~ \delta_B^s  \rangle$',
        2  : r'(d) $\langle \delta_A^s ~ \delta_B  \rangle$',
        3  : r'(e) $\langle \delta_A^s ~ \delta_B^s  \rangle$',
        5  : r'(f) $\langle \delta_B ~ \delta_B^s  \rangle$',
        11 : r'$\langle \eta ~ \epsilon \rangle$',
        12 : r'$\langle \delta_{\rm LN} ~ \epsilon \rangle$',
        13 : r'$\langle \epsilon ~ \epsilon \rangle$',
    }

    ##############
    # Make the fit for the clustered source (term 4)
    ##############
    logger.info('Making fit for the biased and clustered source')
    term = terms[4]
    term.fitter = Fitter(boxes=term.boxes, z=term.z, theory=term.theory, poles=[0], rsd=False, smooth_factor0=1.1, smooth_factor_rsd0=1, smooth_factor_cross0=1,
                            bias0=1)
    term.fitter.run_fit(free_params=['bias'])
    bias = term.fitter.out.params['bias'].value
    logger.info(f'Fitted bias: {bias}')
    
    ##############
    # Create terms again with correct bias
    ##############
    logger.info('Recreating terms with fitted bias')
    terms = AndreuTerms.create_Andreu_terms(biases_dict={1:0.001, 4:bias})

    ##############
    # Making first plot
    #############
    logger.info('Making first plot')
    matplotlib.rcParams.update({'font.size': 18})
    fig, axs = plt.subplots(ncols=2, nrows=3, sharex=True, sharey=True, figsize=(18, 15))
    # fig, axs = plt.subplots(ncols=2, nrows=3,sharex=True)

    axs = axs.reshape(-1)
    for iterm, ax, i in zip([1,4,6,2,3,5], axs, [1,2,3,4,5,6]):
        term = terms[iterm]
        poles = [0,2]# if term.rsd1 or term.rsd2 else [0]
        
        for pole, fill_c, c, label in zip(poles, ['#66a3ff','#ff6666'], ['b', 'r'], ['Monopole', 'Quadrupole']):
            xi, xierr = Plots.get_xi(pole, term.boxes)
        
            box = term.boxes[0]
            if True:
                
                ax.fill_between(term.rdata, term.rdata**2*(xi+xierr), term.rdata**2*(xi-xierr), facecolor=fill_c, alpha=0.7)
                ax.plot(term.rdata, term.rdata**2*(xi+xierr), c=fill_c, lw=1, alpha=0.7, label=label)
                ax.plot(term.rdata, term.rdata**2*(xi-xierr), c=fill_c, lw=1, alpha=0.7)
            else:
                ax.errorbar(box.savg, box.savg**2*xi, box.savg**2*xierr)

            _modelxi = term.model(pole)
            if iterm == 4:
                fitted_region = term.rmodel > 10
                ax.plot(term.rmodel, term.rmodel**2*_modelxi, c=c, lw=1, ls='--')
                ax.plot(term.rmodel[fitted_region], term.rmodel[fitted_region]**2*_modelxi[fitted_region], c=c, lw=1, ls='-')
            else:
                ax.plot(term.rmodel, term.rmodel**2*_modelxi, c=c, lw=1, ls='--')
            
        ax.set_xlim(-5, 200)
        ax.set_title(term_description[iterm], fontsize=25, pad=14)
        # ax.text(0.9, 0.9, f'P{i}', transform=ax.transAxes)
        ax.grid()
        
        fig.supylabel(r'$r^2 \xi(r) \, {\rm [Mpc/h]^2}$', x=0.03, fontsize=25)
        fig.supxlabel(r'$r \, {\rm [Mpc/h]}$', y=0.03, fontsize=25)
    axs[0].legend()
    plt.savefig(args.output_files[0])
    if args.show_plots:
        plt.show()

    ###############
    # Making second plot
    ###############
    matplotlib.rcParams.update({'font.size': 18})
    fig, axs = plt.subplots(nrows=2, ncols=3, sharey='row', sharex=True, figsize=(15, 10))

    for pole, axsi in zip([0,2], axs):
        if pole == 0:
            fill_c = '#66a3ff'
            c = 'b'
        elif pole == 2:
            fill_c = '#ff6666'
            c = 'r'
            
        for iterm, ax in zip([11, 12, 13], axsi):
            term = terms[iterm]

            # Plot Data
            # ----------
            xi, xierr = term.data(pole)
            
            ax.fill_between(term.rdata, term.rdata**2*(xi+xierr), term.rdata**2*(xi-xierr), facecolor=fill_c, alpha=0.7, label='Data')
            ax.plot(term.rdata, term.rdata**2*(xi+xierr), c=fill_c, lw=1, alpha=0.7)
            ax.plot(term.rdata, term.rdata**2*(xi-xierr), c=fill_c, lw=1, alpha=0.7)
            # ----------
                
                
            # Plot Full Model
            # ----------
            term = terms[6] # Plots of full model to compare      
            _modelxi = term.model(pole)
            ax.plot(term.rmodel, term.rmodel**2*_modelxi, c=c, lw=1, ls='--', label='Full model')
            # ----------
            
            # Plot Full Data
            if False: 
                fill_gray = '#AAAAAA'
                xi, xierr = term.data(pole)

                ax.fill_between(term.rdata, term.rdata**2*(xi+xierr), term.rdata**2*(xi-xierr), facecolor=fill_gray, alpha=0.7)
                ax.plot(term.rdata, term.rdata**2*(xi+xierr), c=fill_gray, lw=1, alpha=0.7)
                ax.plot(term.rdata, term.rdata**2*(xi-xierr), c=fill_gray, lw=1, alpha=0.7)
            
            ax.set_xlim(0, 200)
            if pole == 0:
                ax.set_title(term_description[iterm], fontsize=25, pad=8)
            ax.grid()
            
            
    axs[0][2].yaxis.set_label_position("right")
    axs[0][2].set_ylabel('Monopole', labelpad=28, rotation=-90, fontsize=22)
    axs[1][2].yaxis.set_label_position("right")
    axs[1][2].set_ylabel('Quadrupole', labelpad=28, rotation=-90, fontsize=22)
    # axs[0][0].legend()
    # axs[1][0].legend()
        
    fig.supylabel(r'$r^2 \xi(r) \, {\rm [Mpc/h]^2}$', x=0.03, fontsize=25)
    fig.supxlabel(r'$r \, {\rm [Mpc/h]}$', y=0.03, fontsize=25)
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.savefig(args.output_files[1])
    if args.show_plots:
        plt.show()

if __name__ == '__main__':
    main()