from CoLoRe_corrf_analysis.plot_methods import Plots
from CoLoRe_corrf_analysis.file_funcs import FileFuncs
from CoLoRe_corrf_analysis.fitter import Fitter
from CoLoRe_corrf_analysis.read_colore import ComputeModelsCoLoRe

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
    parser.add_argument('--output-file', type=Path, default=Path('multipoles_3d_clustering_norsd.pdf'), help='Output for the plot')
    parser.add_argument('--log-level', default='WARNING', choices=['CRITICAL', 'ERROR', 
    'WARNING', 'INFO', 'DEBUG', 'NOTSET'])
    parser.add_argument('--show-plot', action='store_true')

    args = parser.parse_args()
    return args

def main(args=None):
    if args is None:
        args= getArgs()

    level = logging.getLevelName(args.log_level)
    logging.basicConfig(stream=sys.stdout, level=level, format='%(levelname)s:%(name)s:%(funcName)s:%(message)s')

    basedir         = Path('/global/project/projectdirs/desi/users/cramirez/CoLoRe_analysis/NbodyKit/hanyu_david_box')
    theory_path     = Path('/global/cscratch1/sd/damonge/CoLoRe_sims/sim_bs_1000')
    bias_filename   = basedir / Path('input_files/BzBlue.txt')
    nz_filename     = basedir / Path('input_files/NzRed.txt')
    pk_filename     = basedir / Path('input_files/Pk_CAMB_test.dat')

    ###############
    # Instantiate model object (which is the same for all realizations)
    ###############
    logger.info('Instantiating theory model object')
    theory = ComputeModelsCoLoRe(
        box_path=theory_path,
        source=2,
        nz_filename=nz_filename,
        pk_filename=pk_filename,
        param_cfg_filename='/global/cscratch1/sd/damonge/CoLoRe_sims/sim1000/out_params.cfg', #I need this to read the cosmological parameters
        bias_filename=bias_filename,
        apply_lognormal=True,
        smooth_factor_analysis=0.35
    )

    ###############
    # Define a class to store the plot data
    ###############
    class empty:
        pass

    ###############
    # Create instances of that class for the data we want to plot
    ###############
    logger.info('Filling dummy classes to make plots')
    analyses_norsd = []

    _ = empty()
    _.zmin=0.5
    _.zmax=0.7
    _.rsd=False
    _.sub_boxes = get_boxes(basedir / 'analysis', zmin=_.zmin, zmax=_.zmax, rsd1=_.rsd)
    analyses_norsd.append(_)

    _ = empty()
    _.zmin=0.7
    _.zmax=0.9
    _.rsd=False
    _.sub_boxes = get_boxes(basedir / 'analysis', zmin=_.zmin, zmax=_.zmax, rsd1=_.rsd)
    analyses_norsd.append(_)

    analyses_rsd = []

    _ = empty()
    _.zmin=0.5
    _.zmax=0.7
    _.rsd=True
    _.sub_boxes = get_boxes(basedir / 'analysis', zmin=_.zmin, zmax=_.zmax, rsd1=_.rsd)
    analyses_rsd.append(_)

    _ = empty()
    _.zmin=0.7
    _.zmax=0.9
    _.rsd=True
    _.sub_boxes = get_boxes(basedir / 'analysis', zmin=_.zmin, zmax=_.zmax, rsd1=_.rsd)
    analyses_rsd.append(_)

    analyses = analyses_rsd + analyses_norsd

    ###############
    # Possibility to hide warnings
    ###############
    if False:
        logger.info('Warnings will be disabled')
        import warnings
        warnings.filterwarnings('ignore')

    ###############
    # Perform the fit for noRSD values
    ###############
    logger.info('Making fits of noRSD values')
    for analysis, analysis_rsd in zip(analyses_norsd, analyses_rsd):
        logger.info(f'Making fit for bin {analysis.zmin}-{analysis.zmax}')
        analysis.zeff = theory.get_zeff(analysis.zmin, analysis.zmax)
        analysis_rsd.zeff = analysis.zeff
        
        analysis.fitter = Fitter(boxes=analysis.sub_boxes, z=analysis.zeff,
                                theory=theory, poles=[0], rsd=analysis.rsd,
                                rmin={0:10, 2:200}, rmax={0:200, 2:200},
                                smooth_factor0=1.1, smooth_factor_rsd0=1, smooth_factor_cross0=1)
        
        analysis.fitter.run_fit(free_params=['bias'])

        # Mock RSD fitter to use the plot_best_fit method
        analysis_rsd.fitter = Fitter(boxes=analysis_rsd.sub_boxes, z=analysis_rsd.zeff,
                            theory=theory, poles=[], rsd=analysis_rsd.rsd,
                            rmin={0:200, 2:200}, rmax={0:200, 2:200},
                            smooth_factor0=1.1, smooth_factor_rsd0=1, smooth_factor_cross0=1,
                            bias0=analysis.fitter.out.params['bias'].value)
        
        # Use the same output fitted values (the parameters are exactly the same for both fits)
        analysis_rsd.fitter.out = analysis.fitter.out
  
        print(analysis.fitter.pars_tab())

    ############
    # Plot
    ############
    logger.info('Starting plot')
    matplotlib.rcParams.update({'font.size': 18})
    fig, axs = plt.subplots(2,2, sharey='row', sharex=True, figsize=(15,10))

    pole=0
    axs = axs.reshape(-1)
    for analysis, ax in zip(analyses_norsd+analyses_rsd, axs):
        analysis.zeff = theory.get_zeff(analysis.zmin, analysis.zmax)
        
        poles = [0,2] if analysis.rsd else [0]
        
        for pole in poles:
            if pole == 0:
                fill_c = '#66a3ff'
                c = 'b'
            elif pole == 2:
                fill_c = '#ff6666'
                c = 'r'
            
            # Plot data
            # --------------
            boxes = analysis.sub_boxes
            box = boxes[0]
            
            xi, xierr = Plots.get_xi(pole, boxes)
            xierr *= np.sqrt(10)
            
            ax.fill_between(box.savg, box.savg**2*(xi + xierr), box.savg**2*(xi-xierr), facecolor=fill_c, alpha=0.7)
            ax.plot(box.savg, box.savg**2*(xi+xierr), c=fill_c, lw=1, alpha=0.7)
            ax.plot(box.savg, box.savg**2*(xi-xierr), c=fill_c, lw=1, alpha=0.7)
            # ------------------
            
            
            # Plot model
            # ----------
            label = 'Monopole' if pole == 0 else 'Quadrupole'
            Plots.plot_best_fit(fitter=analysis.fitter, pole=pole, ax=ax,
                        plot_args=dict(c=c, lw=1, label=label), no_labels=True)
            # ----------

        ax.set_xlim(-5, 200)
        ax.grid(zorder=-1)

    #plt.ylabel(r'$r^2 \xi(r)$')
    axs[0].set_title('low-z')#r'$z \in ({}-{})$'.format(0.5, 0.7))
    axs[1].set_title('high-z')#r'$z \in ({}-{})$'.format(0.7, 0.9))

    axs[1].yaxis.set_label_position("right")
    axs[1].set_ylabel('Real space', labelpad=28, rotation=-90)
    axs[3].yaxis.set_label_position("right")
    axs[3].set_ylabel('Redshift space', labelpad=28, rotation=-90)

    handles, labels = axs[3].get_legend_handles_labels()
    axs[0].legend(handles, labels)

    fig.supylabel(r'$r^2 \, \xi(r) \, {\rm [Mpc/h]^2}$', x=0.03)
    fig.supxlabel(r'$r \, {\rm [Mpc/h]}$', y=0.03)
    plt.subplots_adjust(wspace=0.05, hspace=0.1)

    plt.savefig(args.output_file)

    if args.show_plot:
        plt.show()

def get_boxes(path, rsd1=False, rsd2=None,
                rmin=0.1, rmax=200, N_bins=41,
              zmin=0.5, zmax=0.7, nside=2):
    return FileFuncs.mix_sims(
        FileFuncs.get_full_path(path,
                                rsd=rsd1, rsd2=rsd2, 
                                rmin=rmin, rmax=rmax, N_bins=N_bins,
                                zmin=zmin, zmax=zmax, nside=2)
    )

if __name__ == '__main__':
    main()