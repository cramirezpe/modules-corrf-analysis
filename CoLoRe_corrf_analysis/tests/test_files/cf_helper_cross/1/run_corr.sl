#!/bin/bash -l

#SBATCH --partition regular
#SBATCH --nodes 1
#SBATCH --time 30
#SBATCH --job-name binned_corrf_david
#SBATCH --error /global/cscratch1/sd/cramirez/NBodyKit/cross_correlations/multibias/s1_s5/nside_2/rsd_norsd/0.1_200_41/0.5_0.7/1000//1/%x-%j.err
#SBATCH --output /global/cscratch1/sd/cramirez/NBodyKit/cross_correlations/multibias/s1_s5/nside_2/rsd_norsd/0.1_200_41/0.5_0.7/1000//1/%x-%j.out
#SBATCH -C haswell
#SBATCH -A desi
#SBATCH --cpus-per-task=64

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

module unload craype-hugepages2M

source activate corrf

srun python /global/u2/c/cramirez/Work/CoLoRe_multipoles/modules_corrf_analysis/hanyu_correlation_enhanced.py --data /global/cscratch1/sd/cramirez/NBodyKit/multibias/s1/catalogues/1000/rsd.fits --data2 /global/cscratch1/sd/cramirez/NBodyKit/multibias/s4/catalogues/1000/norsd.fits --randoms /global/cscratch1/sd/cramirez/NBodyKit/multibias/s1/catalogues/1000/rsd_rand.fits --out-dir /global/cscratch1/sd/cramirez/NBodyKit/cross_correlations/multibias/s1_s5/nside_2/rsd_norsd/0.1_200_41/0.5_0.7/1000//1 --log-level DEBUG --nthreads ${SLURM_CPUS_PER_TASK} --drq-format --zmin-covd 0.01 --zmax-covd 1.5 --zstep-covd 0.005 --zmin 0.5 --zmax 0.7 --nside 2 --pixel-mask 1 --min-bin 0.1 --max-bin 200 --n-bins 41
