Examples on how to perform the analysis are given in this directory.

- sim_bs_100_/show_analysis_available.ipynb:
    Notebook to show all the available analysis realizations that can be used in this type of notebooks.

- sim_bs_100_/s..:
    Analysis for particular cross or auto correlations. The combination of all these notebooks will check one or more of the terms in the Kaiser model. All of them seems to work except for s4_s5 both with RSD and s4(rsd), which shows the usual problem in the quadrupole.

- sim_bs_100_/check_bias_consistency..:
    Notebooks made to check that bias is not changing between different cross-correlation measurements for a given field.
    
- sim_bs_100_/sum_correlations.ipynb:
    Notebook to check that we can sum different terms together and we obtain the correct mixed term. It shows that the only combination that we cannot predict properly is (0 1 2 3), that means, all the terms together or simulations with bias and RSD on both tracers.

- 10_boxe_analysis:
    Notebook of the analysis of the initial 10 LSST boxes.

- LPT:
    Last analysis on LPT boxes.