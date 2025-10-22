#!/bin/bash
#SBATCH -J XY_op_run1
#SBATCH -p sapphire
#SBATCH -c 36
#SBATCH -t 1-00:00:00 
#SBATCH --mem=16G
#SBATCH -o py_%j.o 
#SBATCH -e py_%j.e 

module purge
module load python
eval "$(mamba shell hook --shell bash)"
mamba activate RSC_sim

cd $home
cd Raman_sideband_cooling/optimize_piecewise/XYZ1
python XYZ1_optimize.py
