#!/bin/bash
#SBATCH -J XY_op_run1
#SBATCH -p sapphire
#SBATCH -c 36
#SBATCH -t 0-12:00:00 
#SBATCH --mem=8G
#SBATCH -o py_%j.o 
#SBATCH -e py_%j.e 

module purge
module load python
eval "$(mamba shell hook --shell bash)"
mamba activate RSC_sim

cd $home
cd Raman_sideband_cooling/optimize_piecewise/XY
python XY_optimize_var_pop.py
