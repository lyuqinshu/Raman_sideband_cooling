#!/bin/bash
#SBATCH -J all_op_run1
#SBATCH -p sapphire
#SBATCH -c 72
#SBATCH -t 3-00:00:00 
#SBATCH --mem=16G
#SBATCH -o py_%j.o 
#SBATCH -e py_%j.e 

module purge
module load python
eval "$(mamba shell hook --shell bash)"
mamba activate RSC_sim

cd $home
cd Raman_sideband_cooling/optimize_all
python rsc_op.py
