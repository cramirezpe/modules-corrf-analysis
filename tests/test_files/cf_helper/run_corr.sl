#!/bin/bash -l

#SBATCH --partition regular
#SBATCH --nodes 1
#SBATCH --time 30
#SBATCH --job-name binned_corrf_david
#SBATCH --error /global/cscratch1/sd/cramirez/NBodyKit/hanyu_david_box/binned_analysis_with_error//nside_2/rsd/0.1_200/0.1_0.3//0/%x-%j.err
#SBATCH --output /global/cscratch1/sd/cramirez/NBodyKit/hanyu_david_box/binned_analysis_with_error//nside_2/rsd/0.1_200/0.1_0.3//0/%x-%j.out
#SBATCH -C haswell
#SBATCH -A desi
#SBATCH --cpus-per-task=64

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

module unload craype-hugepages2M

source activate corrf

srun python /global/homes/c/cramirez/Work/hanyu_lya_box/hanyu_correlation_enhanced.py --data /global/cscratch1/sd/cramirez/NBodyKit/hanyu_david_box/catalogues//colore_2_full_rsd.fits --randoms /global/cscratch1/sd/cramirez/NBodyKit/david_box/randoms_rsd.fits --out-dir /global/cscratch1/sd/cramirez/NBodyKit/hanyu_david_box/binned_analysis_with_error//nside_2/rsd/0.1_200/0.1_0.3//0 --log-level DEBUG --nthreads ${SLURM_CPUS_PER_TASK} --drq-format --zmin-covd 0.01 --zmax-covd 1.5 --zstep-covd 0.005 --zmin 0.1 --zmax 0.3 --nside 2 --pixel-mask 0  --min-bin 0.1 --max-bin 200
